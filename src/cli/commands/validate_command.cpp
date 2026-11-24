#include "luw/validate_commands.hpp"

#include "luw/deck_document.hpp"
#include "luw/preprocess_compute_config.hpp"

#include <QByteArray>
#include <QProcess>
#include <QString>
#include <QStringList>
#include <QVariant>

#include <vtkNew.h>
#include <vtkPolyData.h>
#include <vtkSTLReader.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace luw::cli {
namespace {

namespace fs = std::filesystem;

constexpr double kTolerance = 1.0e-3;
constexpr double kStlScaleBack = 1.0;

struct AxisRange {
    double minimum = 0.0;
    double maximum = 0.0;

    [[nodiscard]] double span() const noexcept { return maximum - minimum; }
};

struct Ranges {
    AxisRange x;
    AxisRange y;
    AxisRange z;
};

struct AxisErrors {
    double minimum = 0.0;
    double maximum = 0.0;
    double span = 0.0;
};

struct Comparison {
    bool passed = false;
    AxisErrors x;
    AxisErrors y;
    double decisionMaximum = 0.0;
    double reportedMaximum = 0.0;
};

class MissingFile final : public std::runtime_error {
public:
    explicit MissingFile(fs::path path)
        : std::runtime_error("file not found"), path_(std::move(path)) {}

    [[nodiscard]] const fs::path& path() const noexcept { return path_; }

private:
    fs::path path_;
};

QString qPath(const fs::path& path) {
#ifdef _WIN32
    return QString::fromStdWString(path.wstring());
#else
    const auto utf8 = path.u8string();
    return QString::fromUtf8(reinterpret_cast<const char*>(utf8.data()),
                             static_cast<qsizetype>(utf8.size()));
#endif
}

fs::path pathFromQString(const QString& path) {
#ifdef _WIN32
    return fs::path(path.toStdWString());
#else
    const QByteArray utf8 = path.toUtf8();
    return fs::path(std::string(utf8.constData(), static_cast<std::size_t>(utf8.size())));
#endif
}

fs::path pathFromUtf8(const std::string& path) {
    return pathFromQString(QString::fromUtf8(path.data(), static_cast<qsizetype>(path.size())));
}

std::string displayPath(const fs::path& path) {
    return qPath(path).toUtf8().toStdString();
}

fs::path expandUser(fs::path path) {
    const auto native = path.native();
    if (native.empty() || native.front() != fs::path("~").native().front()) return path;
    if (native.size() > 1
        && native.at(1) != fs::path::preferred_separator
        && native.at(1) != fs::path("/").native().front()) {
        return path;
    }

    QString home = qEnvironmentVariable("USERPROFILE");
    if (home.isEmpty()) home = qEnvironmentVariable("HOME");
    if (home.isEmpty()) return path;
    const fs::path homePath = pathFromQString(home);
    if (native.size() == 1) return homePath;
    return homePath / fs::path(native.substr(2));
}

fs::path absoluteNormalized(const std::string& raw) {
    std::error_code error;
    fs::path path = expandUser(pathFromUtf8(raw));
    path = fs::absolute(path, error);
    if (error) throw std::runtime_error(error.message());
    return path.lexically_normal();
}

std::string readUtf8ReplacingErrors(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw MissingFile(path);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::optional<std::int64_t> queryDefaultGpuMemory(std::ostream& out) {
    QProcess process;
    process.setProgram(QStringLiteral("nvidia-smi"));
    process.setArguments({QStringLiteral("--query-gpu=memory.total"),
                          QStringLiteral("--format=csv,noheader,nounits")});
    process.start();
    if (!process.waitForStarted() || !process.waitForFinished(-1)
        || process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        return std::nullopt;
    }

    const QString output = QString::fromLocal8Bit(process.readAllStandardOutput()).simplified();
    if (output.isEmpty()) return std::nullopt;
    bool ok = false;
    const qlonglong total = output.section(' ', 0, 0).toLongLong(&ok, 10);
    if (!ok) return std::nullopt;
    out << "    GPU0 Memory is " << total << " MiB\n";
    return static_cast<std::int64_t>(static_cast<double>(total) * 0.85);
}

DeckDocument ensureConfFields(const fs::path& confPath, std::ostream& out) {
    DeckDocument deck = [&] {
        if (fs::exists(confPath)) return DeckDocument::load(qPath(confPath));

        const fs::path templatePath = confPath.parent_path() / "template-conf.txt";
        if (fs::exists(templatePath)) {
            const std::string bytes = readUtf8ReplacingErrors(templatePath);
            out << "[!] deck file not found. Created from template-conf.txt\n";
            return DeckDocument::fromText(QString::fromUtf8(bytes.data(),
                                                            static_cast<qsizetype>(bytes.size())));
        }

        out << "[!] deck file not found. Created empty deck\n";
        return DeckDocument::fromText(QString());
    }();

    const auto datetime = deck.text(QStringLiteral("datetime"));
    if (!datetime || datetime->isEmpty()) {
        deck.setText(QStringLiteral("datetime"), QStringLiteral("20990101120000"));
        out << "[!] Field 'datetime' missing. Set default.\n";
    }

    if (deck.list(QStringLiteral("n_gpu")).isEmpty()) {
        deck.setList(QStringLiteral("n_gpu"), QVariantList{1, 1, 1});
        out << "[!] Field 'n_gpu' missing. Wrote default value.\n";
    }

    QString meshControl = deck.textOr(QStringLiteral("mesh_control")).trimmed().toLower();
    const auto cellSizeRaw = deck.raw(QStringLiteral("cell_size"));
    const bool hasCellSizeValue = cellSizeRaw && !cellSizeRaw->trimmed().isEmpty();
    if (meshControl.isEmpty()) {
        deck.setText(QStringLiteral("mesh_control"), QStringLiteral("gpu_memory"), true);
        meshControl = QStringLiteral("gpu_memory");
        out << "[!] Field 'mesh_control' missing. Wrote default value.\n";
    } else if (meshControl == QStringLiteral("cell_size") && !hasCellSizeValue) {
        deck.setText(QStringLiteral("mesh_control"), QStringLiteral("gpu_memory"), true);
        meshControl = QStringLiteral("gpu_memory");
        out << "[!] 'mesh_control' set to 'gpu_memory' because 'cell_size' is missing\n";
    }

    const auto legacyMemory = deck.text(QStringLiteral("memory_lbm"));
    if (legacyMemory && !legacyMemory->isEmpty()) {
        out << "[!] Removed legacy 'memory_lbm'\n";
    }

    if (meshControl == QStringLiteral("gpu_memory")
        && !deck.integer(QStringLiteral("gpu_memory"))) {
        std::optional<std::int64_t> migrated;
        if (legacyMemory && !legacyMemory->isEmpty()) {
            bool ok = false;
            const double value = legacyMemory->toDouble(&ok);
            if (ok && std::isfinite(value)) migrated = static_cast<std::int64_t>(value);
        }
        if (!migrated) migrated = queryDefaultGpuMemory(out).value_or(20000);
        deck.setInteger(QStringLiteral("gpu_memory"), *migrated);
        out << "[!] Ensured 'gpu_memory'\n";
    }

    if (!deck.has(QStringLiteral("cell_size"))) {
        deck.setRaw(QStringLiteral("cell_size"), QString());
        out << "[!] Inserted placeholder 'cell_size'\n";
    }

    deck.remove(QStringLiteral("memory_lbm"));
    deck.save(qPath(confPath));
    return deck;
}

void writeValidation(DeckDocument& deck, const fs::path& confPath, bool passed) {
    deck.setText(QStringLiteral("validation"),
                 passed ? QStringLiteral("pass") : QStringLiteral("error"));
    if (!deck.has(QStringLiteral("high_order"))) {
        deck.setBoolean(QStringLiteral("high_order"), true);
    }
    if (!deck.has(QStringLiteral("flux_correction"))) {
        deck.setBoolean(QStringLiteral("flux_correction"), true);
    }
    deck.save(qPath(confPath));
}

Ranges stlRanges(const fs::path& path) {
    if (!fs::is_regular_file(path)) throw MissingFile(path);

    vtkNew<vtkSTLReader> reader;
    const std::string filename = displayPath(path);
    reader->SetFileName(filename.c_str());
    reader->Update();
    vtkPolyData* mesh = reader->GetOutput();
    if (mesh == nullptr || mesh->GetNumberOfPoints() == 0 || mesh->GetNumberOfCells() == 0) {
        throw std::runtime_error("Failed to load STL mesh: " + filename);
    }

    double bounds[6]{};
    mesh->GetBounds(bounds);
    return {{bounds[0] * kStlScaleBack, bounds[1] * kStlScaleBack},
            {bounds[2] * kStlScaleBack, bounds[3] * kStlScaleBack},
            {bounds[4] * kStlScaleBack, bounds[5] * kStlScaleBack}};
}

bool readCsvRecord(std::istream& input, std::vector<std::string>& fields) {
    fields.clear();
    std::string field;
    bool inQuotes = false;
    bool sawAny = false;

    for (;;) {
        const int next = input.get();
        if (next == std::char_traits<char>::eof()) {
            if (!sawAny && field.empty() && fields.empty()) return false;
            fields.push_back(std::move(field));
            return true;
        }
        sawAny = true;
        const char character = static_cast<char>(next);
        if (inQuotes) {
            if (character == '"') {
                if (input.peek() == '"') {
                    input.get();
                    field.push_back('"');
                } else {
                    inQuotes = false;
                }
            } else {
                field.push_back(character);
            }
            continue;
        }

        if (character == '"' && field.empty()) {
            inQuotes = true;
        } else if (character == ',') {
            fields.push_back(std::move(field));
            field.clear();
        } else if (character == '\n') {
            if (!field.empty() && field.back() == '\r') field.pop_back();
            fields.push_back(std::move(field));
            return true;
        } else {
            field.push_back(character);
        }
    }
}

std::string_view trimAscii(std::string_view text) {
    constexpr std::string_view whitespace = " \t\r\n\f\v";
    const auto first = text.find_first_not_of(whitespace);
    if (first == std::string_view::npos) return {};
    const auto last = text.find_last_not_of(whitespace);
    return text.substr(first, last - first + 1);
}

bool isPandasMissing(std::string_view raw) {
    const std::string_view text = trimAscii(raw);
    static constexpr std::array<std::string_view, 18> missing = {
        "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
        "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null"
    };
    return std::find(missing.begin(), missing.end(), text) != missing.end();
}

double parseCsvNumber(const std::string& raw, const fs::path& path) {
    const std::string_view text = trimAscii(raw);
    double value = 0.0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value,
                                        std::chars_format::general);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::runtime_error("Unable to parse numeric CSV value '" + raw
                                 + "' in " + displayPath(path));
    }
    return value;
}

Ranges csvRanges(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw MissingFile(path);

    std::vector<std::string> fields;
    if (!readCsvRecord(input, fields)) {
        throw std::runtime_error("No columns to parse from file: " + displayPath(path));
    }
    if (!fields.empty() && fields.front().size() >= 3
        && static_cast<unsigned char>(fields.front()[0]) == 0xef
        && static_cast<unsigned char>(fields.front()[1]) == 0xbb
        && static_cast<unsigned char>(fields.front()[2]) == 0xbf) {
        fields.front().erase(0, 3);
    }

    std::array<std::size_t, 3> column{};
    for (std::size_t axis = 0; axis < column.size(); ++axis) {
        const std::string name(1, "XYZ"[axis]);
        const auto it = std::find(fields.begin(), fields.end(), name);
        if (it == fields.end()) {
            throw std::runtime_error("Usecols do not match columns, columns expected but not found: ['"
                                     + name + "']");
        }
        column[axis] = static_cast<std::size_t>(std::distance(fields.begin(), it));
    }

    std::array<double, 3> minimum{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    std::array<double, 3> maximum{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    std::array<bool, 3> seen{};

    while (readCsvRecord(input, fields)) {
        for (std::size_t axis = 0; axis < column.size(); ++axis) {
            const std::string empty;
            const std::string& raw = column[axis] < fields.size() ? fields[column[axis]] : empty;
            if (isPandasMissing(raw)) continue;
            const double value = parseCsvNumber(raw, path);
            if (std::isnan(value)) continue;
            minimum[axis] = std::min(minimum[axis], value);
            maximum[axis] = std::max(maximum[axis], value);
            seen[axis] = true;
        }
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t axis = 0; axis < seen.size(); ++axis) {
        if (!seen[axis]) minimum[axis] = maximum[axis] = nan;
    }
    return {{minimum[0], maximum[0]},
            {minimum[1], maximum[1]},
            {minimum[2], maximum[2]}};
}

AxisErrors compareAxis(const AxisRange& stl, const AxisRange& csv) {
    const double stlSpan = stl.span();
    const double denominator = stlSpan != 0.0
        ? std::abs(stlSpan)
        : std::max({std::abs(stl.minimum), std::abs(stl.maximum), 1.0});
    return {std::abs(stl.minimum - csv.minimum) / denominator,
            std::abs(stl.maximum - csv.maximum) / denominator,
            std::abs(stlSpan - csv.span()) / denominator};
}

Comparison compareXY(const Ranges& stl, const Ranges& csv) {
    Comparison comparison;
    comparison.x = compareAxis(stl.x, csv.x);
    comparison.y = compareAxis(stl.y, csv.y);
    // Python's built-in max starts from max_err=0.0 here; comparisons against
    // NaN are false, so preserve that edge-case behavior for empty CSV columns.
    for (const double value : {comparison.x.minimum, comparison.x.maximum, comparison.x.span,
                               comparison.y.minimum, comparison.y.maximum, comparison.y.span}) {
        if (value > comparison.decisionMaximum) comparison.decisionMaximum = value;
    }
    const auto pythonMaximum = [](const AxisErrors& errors) {
        double result = errors.minimum;
        if (errors.maximum > result) result = errors.maximum;
        if (errors.span > result) result = errors.span;
        return result;
    };
    const double xMaximum = pythonMaximum(comparison.x);
    const double yMaximum = pythonMaximum(comparison.y);
    comparison.reportedMaximum = yMaximum > xMaximum ? yMaximum : xMaximum;
    comparison.passed = comparison.decisionMaximum < kTolerance;
    return comparison;
}

void printRanges(std::ostream& out, const char* label, const Ranges& ranges) {
    out << label << " ranges:\n" << std::fixed << std::setprecision(3);
    const auto print = [&](char axis, const AxisRange& range) {
        out << "    " << axis << ": min=" << range.minimum << ", max=" << range.maximum
            << ", span=" << range.span() << '\n';
    };
    print('X', ranges.x);
    print('Y', ranges.y);
    print('Z', ranges.z);
}

void printMissingFile(std::ostream& out, const fs::path& path) {
    out << "ERROR: [Errno 2] No such file or directory: '" << displayPath(path) << "'\n";
}

} // namespace

int run_luwval(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    (void)err;
    out << "LUW Pre-run Validation Tool...\n";
    if (args.size() != 1) {
        out << "Usage: python prerun_validation.py <path-to-deck-file>\n";
        return 2;
    }

    fs::path confPath;
    try {
        confPath = absoluteNormalized(args.front());
        DeckDocument deck = ensureConfFields(confPath, out);
        const PreprocessComputeConfig computePolicy =
            resolvePreprocessComputeConfig(&deck).first;
        out << "[INFO] Preprocessing compute policy: "
            << preprocessComputePolicyText(computePolicy).toStdString() << '\n'
            << "[INFO] Pre-run validation has no GPU compute kernel; this stage "
               "runs on CPU.\n";
        const QString caseName = deck.textOr(QStringLiteral("casename"), QStringLiteral("example"));
        const QString datetime = deck.textOr(QStringLiteral("datetime"),
                                             QStringLiteral("20990101120000"));

        if (!deck.pair(QStringLiteral("cut_lon_manual"))
            || !deck.pair(QStringLiteral("cut_lat_manual"))) {
            out << "ERROR: conf \xE7\xBC\xBA\xE5\xB0\x91 cut_lon_manual/cut_lat_manual\n";
            writeValidation(deck, confPath, false);
            return 1;
        }

        const fs::path projectHome = confPath.parent_path();
        const fs::path projectTemp = projectHome / "proj_temp";
        const fs::path demPath = projectTemp
            / pathFromQString(caseName + QStringLiteral("_DEM.stl"));
        const fs::path stlPath = fs::exists(demPath)
            ? demPath
            : projectTemp / pathFromQString(caseName + QStringLiteral(".stl"));
        out << "Using STL file: " << displayPath(stlPath) << '\n';

        fs::path csvPath = projectTemp
            / pathFromQString(QStringLiteral("SurfData_") + datetime + QStringLiteral(".csv"));
        if (!fs::exists(csvPath)) {
            const fs::path latest = projectTemp / "SurfData_Latest.csv";
            if (fs::exists(latest)) csvPath = latest;
        }

        Ranges stl;
        Ranges csv;
        try {
            stl = stlRanges(stlPath);
            csv = csvRanges(csvPath);
        } catch (const MissingFile& missing) {
            printMissingFile(out, missing.path());
            writeValidation(deck, confPath, false);
            return 1;
        }

        printRanges(out, "STL", stl);
        printRanges(out, "CSV", csv);

        const Comparison comparison = compareXY(stl, csv);
        out << std::fixed << std::setprecision(6);
        if (comparison.passed) {
            out << "Validation passed. Maximum XY relative error "
                << comparison.reportedMaximum * 100.0 << "%\n";
        } else {
            const std::string border(60, '=');
            out << border << '\n';
            out << "WARNING: XY range mismatch exceeds 0.1%!\n";
            const auto print = [&](char axis, const AxisErrors& value) {
                out << "  Axis " << axis << ": min=" << value.minimum * 100.0
                    << "%, max=" << value.maximum * 100.0
                    << "%, span=" << value.span * 100.0 << "%\n";
            };
            print('x', comparison.x);
            print('y', comparison.y);
            out << border << '\n';
        }
        writeValidation(deck, confPath, comparison.passed);
        return 0;
    } catch (const MissingFile& missing) {
        printMissingFile(out, missing.path());
        return 1;
    } catch (const std::exception& exception) {
        out << "ERROR: " << exception.what() << '\n';
        return 1;
    }
}

} // namespace luw::cli
