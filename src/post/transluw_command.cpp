#include "luw/vtk_post_commands.hpp"

#include <QString>

#include <vtkAbstractArray.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCellDataToPointData.h>
#include <vtkCellType.h>
#include <vtkDataSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkDataSetReader.h>
#include <vtkDataSetWriter.h>
#include <vtkDoubleArray.h>
#include <vtkErrorCode.h>
#include <vtkIdTypeArray.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPointSet.h>
#include <vtkRectilinearGrid.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <queue>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace luw::cli {

namespace {

namespace fs = std::filesystem;

class ArgumentError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct Arguments final {
    fs::path inputPath;
    std::string tailname;
    bool absoluteOnly = false;
    bool help = false;
};

struct Vec2 final {
    double x = 0.0;
    double y = 0.0;
};

struct DemGrid final {
    std::vector<Vec2> points;
    std::vector<double> elevations;
    double xMaximum = 0.0;
    double yMaximum = 0.0;
    double zMinimum = std::numeric_limits<double>::quiet_NaN();
    double zMaximum = std::numeric_limits<double>::quiet_NaN();
};

std::string pathToUtf8(const fs::path& path) {
#ifdef _WIN32
    return QString::fromStdWString(path.wstring()).toUtf8().toStdString();
#else
    return path.string();
#endif
}

fs::path pathFromUtf8(const std::string& path) {
#ifdef _WIN32
    return fs::path(QString::fromUtf8(path.data(), static_cast<qsizetype>(path.size())).toStdWString());
#else
    return fs::path(path);
#endif
}

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n\f\v");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n\f\v");
    return value.substr(first, last - first + 1U);
}

fs::path absoluteNormalized(const fs::path& path) {
    std::error_code error;
    fs::path result = fs::weakly_canonical(path, error);
    if (!error) return result;
    result = fs::absolute(path, error);
    if (error) result = path;
    return result.lexically_normal();
}

bool isRegularFile(const fs::path& path) {
    std::error_code error;
    return fs::is_regular_file(path, error) && !error;
}

void printUsage(std::ostream& output) {
    output << "usage: transluw.py [-h] [--tailname TAILNAME] [--absonly] input_path\n\n"
              "Translate VTK datasets so that the minimum bounds move to the origin. Accepts a\n"
              "single input path plus an optional --tailname.\n\n"
              "positional arguments:\n"
              "  input_path            Path to a .vtk file or a text config file containing\n"
              "                        casename and datetime\n\n"
              "options:\n"
              "  -h, --help            show this help message and exit\n"
              "  --tailname TAILNAME   Optional tail string appended to the output filename\n"
              "  --absonly             Skip woDEM output when dem_grid.pkl exists\n";
}

void printShortUsage(std::ostream& output) {
    output << "usage: transluw.py [-h] [--tailname TAILNAME] [--absonly] input_path\n";
}

Arguments parseArguments(const std::vector<std::string>& arguments) {
    Arguments parsed;
    bool haveInput = false;
    bool positionalOnly = false;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];
        if (!positionalOnly && (argument == "-h" || argument == "--help")) {
            parsed.help = true;
            continue;
        }
        if (!positionalOnly && argument == "--") {
            positionalOnly = true;
            continue;
        }
        if (!positionalOnly && argument == "--absonly") {
            parsed.absoluteOnly = true;
            continue;
        }
        if (!positionalOnly && argument == "--tailname") {
            if (index + 1U >= arguments.size()) {
                throw ArgumentError("argument --tailname: expected one argument");
            }
            parsed.tailname = arguments[++index];
            continue;
        }
        if (!positionalOnly && argument.starts_with("--tailname=")) {
            parsed.tailname = argument.substr(std::string("--tailname=").size());
            continue;
        }
        if (!positionalOnly && !argument.empty() && argument.front() == '-') {
            throw ArgumentError("unrecognized arguments: " + argument);
        }
        if (haveInput) throw ArgumentError("unrecognized arguments: " + argument);
        parsed.inputPath = pathFromUtf8(argument);
        haveInput = true;
    }
    if (!parsed.help && !haveInput) {
        throw ArgumentError("the following arguments are required: input_path");
    }
    parsed.tailname = trim(std::move(parsed.tailname));
    return parsed;
}

std::string pythonFloat(double value) {
    char buffer[128]{};
    const auto converted = std::to_chars(
        std::begin(buffer), std::end(buffer), value, std::chars_format::general);
    std::string result;
    if (converted.ec == std::errc{}) {
        result.assign(buffer, converted.ptr);
    } else {
        std::ostringstream stream;
        stream << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
        result = stream.str();
    }
    if (std::isfinite(value) && result.find_first_of(".eE") == std::string::npos) result += ".0";
    return result;
}

std::string tuple3(const std::array<double, 3>& values) {
    return '(' + pythonFloat(values[0]) + ", " + pythonFloat(values[1]) + ", "
        + pythonFloat(values[2]) + ')';
}

vtkSmartPointer<vtkDataSet> readDataSet(const fs::path& path) {
    vtkNew<vtkDataSetReader> reader;
    const std::string filename = pathToUtf8(path);
    reader->SetFileName(filename.c_str());
    reader->ReadAllScalarsOn();
    reader->ReadAllVectorsOn();
    reader->ReadAllNormalsOn();
    reader->ReadAllTensorsOn();
    reader->ReadAllColorScalarsOn();
    reader->ReadAllTCoordsOn();
    reader->ReadAllFieldsOn();
    reader->Update();
    if (reader->GetErrorCode() != vtkErrorCode::NoError || reader->GetOutput() == nullptr) {
        throw std::runtime_error("Failed to read VTK file: " + pathToUtf8(path));
    }
    vtkSmartPointer<vtkDataSet> result =
        vtkSmartPointer<vtkDataSet>::Take(reader->GetOutput()->NewInstance());
    result->DeepCopy(reader->GetOutput());
    return result;
}

std::array<double, 3> translationOffset(vtkDataSet* mesh) {
    mesh->ComputeBounds();
    double bounds[6]{};
    mesh->GetBounds(bounds);
    if (mesh->GetNumberOfPoints() == 0 || !std::isfinite(bounds[0])
        || !std::isfinite(bounds[2]) || !std::isfinite(bounds[4])) {
        throw std::runtime_error("Cannot compute bounds for an empty or invalid VTK dataset");
    }
    return {-bounds[0], -bounds[2], -bounds[4] - 50.0};
}

vtkSmartPointer<vtkDataArray> shiftedCoordinates(vtkDataArray* input, double offset) {
    if (input == nullptr) throw std::runtime_error("Rectilinear grid is missing a coordinate axis");
    vtkSmartPointer<vtkDataArray> result;
    result.TakeReference(vtkDataArray::SafeDownCast(input->NewInstance()));
    if (result == nullptr) throw std::runtime_error("Unsupported rectilinear coordinate array");
    result->DeepCopy(input);
    for (vtkIdType tuple = 0; tuple < result->GetNumberOfTuples(); ++tuple) {
        result->SetComponent(tuple, 0, result->GetComponent(tuple, 0) + offset);
    }
    return result;
}

vtkSmartPointer<vtkDataSet> translateDataSet(
    vtkDataSet* input, const std::array<double, 3>& offset) {
    if (offset == std::array<double, 3>{0.0, 0.0, 0.0}) {
        return input;
    }
    vtkSmartPointer<vtkDataSet> result =
        vtkSmartPointer<vtkDataSet>::Take(input->NewInstance());
    result->DeepCopy(input);

    if (auto* image = vtkImageData::SafeDownCast(result)) {
        double origin[3]{};
        image->GetOrigin(origin);
        image->SetOrigin(origin[0] + offset[0], origin[1] + offset[1], origin[2] + offset[2]);
        return result;
    }
    if (auto* grid = vtkRectilinearGrid::SafeDownCast(result)) {
        grid->SetXCoordinates(shiftedCoordinates(grid->GetXCoordinates(), offset[0]));
        grid->SetYCoordinates(shiftedCoordinates(grid->GetYCoordinates(), offset[1]));
        grid->SetZCoordinates(shiftedCoordinates(grid->GetZCoordinates(), offset[2]));
        return result;
    }
    if (auto* pointSet = vtkPointSet::SafeDownCast(result)) {
        vtkPoints* points = pointSet->GetPoints();
        if (points == nullptr) throw std::runtime_error("VTK point set has no points");
        for (vtkIdType index = 0; index < points->GetNumberOfPoints(); ++index) {
            double point[3]{};
            points->GetPoint(index, point);
            point[0] += offset[0];
            point[1] += offset[1];
            point[2] += offset[2];
            points->SetPoint(index, point);
        }
        points->Modified();
        return result;
    }
    throw std::runtime_error(std::string("Unsupported VTK dataset type for translation: ")
                             + input->GetClassName());
}

std::string regexEscape(std::string_view value) {
    static const std::string special = R"(\.^$|()[]{}*+?)";
    std::string escaped;
    escaped.reserve(value.size() * 2U);
    for (const char character : value) {
        if (special.find(character) != std::string::npos) escaped.push_back('\\');
        escaped.push_back(character);
    }
    return escaped;
}

std::optional<fs::path> findLatest(const fs::path& folder,
                                   const std::string& datetime,
                                   std::string_view kind) {
    std::error_code error;
    if (!fs::is_directory(folder, error) || error) return std::nullopt;
    const std::regex pattern("^" + regexEscape(datetime) + "_raw_" + std::string(kind)
                                 + "-(\\d{9})\\.vtk$",
                             std::regex_constants::ECMAScript | std::regex_constants::icase);
    std::map<int, fs::path> candidates;
    for (fs::directory_iterator iterator(folder, error), end;
         !error && iterator != end; iterator.increment(error)) {
        std::error_code typeError;
        if (!iterator->is_regular_file(typeError) || typeError) continue;
        std::smatch match;
        const std::string name = pathToUtf8(iterator->path().filename());
        if (std::regex_match(name, match, pattern)) {
            candidates[std::stoi(match.str(1))] = iterator->path();
        }
    }
    if (candidates.empty()) return std::nullopt;
    return candidates.rbegin()->second;
}

std::pair<std::string, std::string> parseCaseText(const fs::path& path) {
    if (!isRegularFile(path)) throw std::runtime_error("Config file not found: " + pathToUtf8(path));
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Config file not found: " + pathToUtf8(path));
    const std::string content((std::istreambuf_iterator<char>(input)),
                              std::istreambuf_iterator<char>());
    const std::regex casePattern(R"(casename\s*=\s*([^\s]+))",
                                 std::regex_constants::ECMAScript | std::regex_constants::icase);
    const std::regex datetimePattern(R"(datetime\s*=\s*([^\s]+))",
                                     std::regex_constants::ECMAScript | std::regex_constants::icase);
    std::smatch caseMatch;
    std::smatch datetimeMatch;
    if (!std::regex_search(content, caseMatch, casePattern)
        || !std::regex_search(content, datetimeMatch, datetimePattern)) {
        throw std::runtime_error("Failed to parse 'casename' or 'datetime' from the config file");
    }
    return {caseMatch.str(1), datetimeMatch.str(1)};
}

struct DerivedName final {
    std::string casename;
    std::string datetime;
    std::string prefix;
};

DerivedName deriveSingleName(const fs::path& vtkPath) {
    const std::string casename = pathToUtf8(vtkPath.parent_path().filename());
    const std::string stem = pathToUtf8(vtkPath.stem());
    const std::regex uPattern(R"(^(.+?)_raw_u-\d{9}$)",
                              std::regex_constants::ECMAScript | std::regex_constants::icase);
    const std::regex rhoPattern(R"(^(.+?)_raw_rho-\d{9}$)",
                                std::regex_constants::ECMAScript | std::regex_constants::icase);
    std::smatch match;
    if (std::regex_match(stem, match, uPattern)) return {casename, match.str(1), "uvw"};
    if (std::regex_match(stem, match, rhoPattern)) return {casename, match.str(1), "rho"};
    return {casename, stem, "mesh"};
}

fs::path buildOutputPath(const fs::path& parent,
                         const std::string& prefix,
                         const std::string& casename,
                         const std::string& datetime,
                         const std::string& tailname) {
    const fs::path results = parent / "RESULTS";
    std::error_code error;
    fs::create_directories(results, error);
    if (error) {
        throw std::runtime_error("Cannot create RESULTS directory: " + pathToUtf8(results)
                                 + ": " + error.message());
    }
    std::string base = prefix + '-' + casename + '_' + datetime;
    if (!tailname.empty()) base += '_' + tailname;
    return results / (base + ".vtk");
}

void reserveNewOutput(const fs::path& path) {
#ifdef _WIN32
    int descriptor = -1;
    const int openError = _wsopen_s(&descriptor, path.c_str(),
                                    _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY,
                                    _SH_DENYRW, _S_IREAD | _S_IWRITE);
#else
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
    const int openError = descriptor < 0 ? errno : 0;
#endif
    if (descriptor < 0) {
        if (openError == EEXIST) {
            throw std::runtime_error("Output path already exists; refusing to overwrite: "
                                     + pathToUtf8(path));
        }
        throw std::runtime_error("Cannot reserve output path: " + pathToUtf8(path)
                                 + ": " + std::error_code(openError, std::generic_category()).message());
    }
#ifdef _WIN32
    if (_close(descriptor) != 0) {
#else
    if (::close(descriptor) != 0) {
#endif
        throw std::runtime_error("Cannot close newly reserved output path: " + pathToUtf8(path));
    }
}

void writeDataSetNew(const fs::path& path, vtkDataSet* data) {
    reserveNewOutput(path);
    vtkNew<vtkDataSetWriter> writer;
    const std::string filename = pathToUtf8(path);
    writer->SetFileName(filename.c_str());
    writer->SetInputData(data);
    writer->SetFileTypeToBinary();
    writer->SetHeader("LatticeUrbanWind transluw native output");
    if (writer->Write() == 0 || writer->GetErrorCode() != vtkErrorCode::NoError) {
        const char* detail = vtkErrorCode::GetStringFromErrorCode(writer->GetErrorCode());
        throw std::runtime_error("Failed to write VTK output: " + pathToUtf8(path)
                                 + (detail == nullptr ? std::string{} : ": " + std::string(detail)));
    }
}

void ensureSameTopology(vtkDataSet* left, vtkDataSet* right) {
    if (left->GetNumberOfPoints() != right->GetNumberOfPoints()) {
        throw std::runtime_error("Point count mismatch when merging: "
                                 + std::to_string(left->GetNumberOfPoints()) + " vs "
                                 + std::to_string(right->GetNumberOfPoints()));
    }
    if (left->GetNumberOfCells() != right->GetNumberOfCells()) {
        throw std::runtime_error("Cell count mismatch when merging: "
                                 + std::to_string(left->GetNumberOfCells()) + " vs "
                                 + std::to_string(right->GetNumberOfCells()));
    }
}

void copyArrays(vtkDataSetAttributes* source,
                vtkDataSetAttributes* destination,
                std::string_view suffix) {
    for (int index = 0; index < source->GetNumberOfArrays(); ++index) {
        vtkAbstractArray* array = source->GetAbstractArray(index);
        if (array == nullptr || array->GetName() == nullptr) continue;
        std::string name = array->GetName();
        if (destination->GetAbstractArray(name.c_str()) != nullptr) name += suffix;
        vtkSmartPointer<vtkAbstractArray> copy;
        copy.TakeReference(array->NewInstance());
        copy->DeepCopy(array);
        copy->SetName(name.c_str());
        destination->RemoveArray(name.c_str());
        destination->AddArray(copy);
    }
}

void copyPointAndCellData(vtkDataSet* source,
                          vtkDataSet* destination,
                          std::string_view suffix) {
    copyArrays(source->GetPointData(), destination->GetPointData(), suffix);
    copyArrays(source->GetCellData(), destination->GetCellData(), suffix);
}

// Restricted pickle VM for the exact NumPy ndarray/dict subset written by
// 1_buildBC.py (protocol 3) and recent NumPy (protocol 4). It never imports or
// executes a GLOBAL; REDUCE accepts only NumPy ndarray reconstruction and dtype.
struct PickleNode;
using PicklePtr = std::shared_ptr<PickleNode>;

struct PickleNode {
    enum class Kind { Null, Marker, Boolean, Integer, Floating, String, Bytes,
                      Tuple, Dictionary, Global, Array };
    Kind kind = Kind::Null;
    bool boolean = false;
    std::int64_t integer = 0;
    double floating = 0.0;
    std::string text;
    std::vector<std::uint8_t> bytes;
    std::vector<PicklePtr> sequence;
    std::map<std::string, PicklePtr> dictionary;
    std::vector<std::size_t> shape;
    std::string dtype = "f8";
    bool fortran = false;
};

PicklePtr pickleNode(PickleNode::Kind kind = PickleNode::Kind::Null) {
    auto node = std::make_shared<PickleNode>();
    node->kind = kind;
    return node;
}

class SafeNumpyPickleReader final {
public:
    explicit SafeNumpyPickleReader(std::vector<std::uint8_t> bytes)
        : bytes_(std::move(bytes)) {}

    PicklePtr read() {
        while (position_ < bytes_.size()) {
            const std::uint8_t opcode = byte();
            switch (opcode) {
            case 0x80: (void)byte(); break;
            case 0x95: position_ += checkedLength(8U); break;
            case '}': push(pickleNode(PickleNode::Kind::Dictionary)); break;
            case '(': push(pickleNode(PickleNode::Kind::Marker)); break;
            case ')': push(pickleNode(PickleNode::Kind::Tuple)); break;
            case 'N': push(pickleNode()); break;
            case 0x88: { auto n = pickleNode(PickleNode::Kind::Boolean); n->boolean = true; push(n); break; }
            case 0x89: { auto n = pickleNode(PickleNode::Kind::Boolean); n->boolean = false; push(n); break; }
            case 'K': pushInteger(byte()); break;
            case 'M': pushInteger(unsignedLittle(2U)); break;
            case 'J': pushInteger(signedLittle(4U)); break;
            case 0x8a: readLong1(); break;
            case 'G': readBigEndianDouble(); break;
            case 'X': readString(unsignedLittle(4U)); break;
            case 0x8c: readString(byte()); break;
            case 'B': readBytes(unsignedLittle(4U)); break;
            case 'C': readBytes(byte()); break;
            case 0x8e: readBytes(unsignedLittle(8U)); break;
            case 'c': readGlobal(); break;
            case 0x93: stackGlobal(); break;
            case 0x85: makeTuple(1U); break;
            case 0x86: makeTuple(2U); break;
            case 0x87: makeTuple(3U); break;
            case 't': makeMarkedTuple(); break;
            case 'R': reduce(); break;
            case 'b': build(); break;
            case 'q': memo_[byte()] = top(); break;
            case 'r': memo_[unsignedLittle(4U)] = top(); break;
            case 0x94: memo_[nextMemo_++] = top(); break;
            case 'h': pushMemo(byte()); break;
            case 'j': pushMemo(unsignedLittle(4U)); break;
            case 'u': setItems(); break;
            case 's': setItem(); break;
            case '.':
                if (stack_.empty()) throw std::runtime_error("Empty pickle result");
                return stack_.back();
            default:
                throw std::runtime_error("Unsupported pickle opcode 0x" + hex(opcode));
            }
        }
        throw std::runtime_error("Pickle ended before STOP opcode");
    }

private:
    std::uint8_t byte() {
        if (position_ >= bytes_.size()) throw std::runtime_error("Truncated pickle");
        return bytes_[position_++];
    }

    std::size_t checkedLength(std::size_t count) {
        if (count > bytes_.size() - position_) throw std::runtime_error("Truncated pickle payload");
        return count;
    }

    std::uint64_t unsignedLittle(std::size_t count) {
        if (count > 8U) throw std::runtime_error("Oversized pickle integer");
        checkedLength(count);
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < count; ++i) value |= std::uint64_t(byte()) << (i * 8U);
        return value;
    }

    std::int64_t signedLittle(std::size_t count) {
        const std::uint64_t raw = unsignedLittle(count);
        if (count < 8U && (raw & (std::uint64_t(1) << (count * 8U - 1U))) != 0U) {
            return static_cast<std::int64_t>(raw | (~std::uint64_t(0) << (count * 8U)));
        }
        return static_cast<std::int64_t>(raw);
    }

    void push(const PicklePtr& node) { stack_.push_back(node); }
    PicklePtr pop() {
        if (stack_.empty()) throw std::runtime_error("Malformed pickle stack");
        PicklePtr value = stack_.back();
        stack_.pop_back();
        return value;
    }
    PicklePtr top() const {
        if (stack_.empty()) throw std::runtime_error("Malformed pickle stack");
        return stack_.back();
    }

    static std::string hex(std::uint8_t value) {
        const char* digits = "0123456789abcdef";
        std::string result(2, '0');
        result[0] = digits[(value >> 4U) & 0xFU];
        result[1] = digits[value & 0xFU];
        return result;
    }

    void pushInteger(std::int64_t value) {
        auto node = pickleNode(PickleNode::Kind::Integer);
        node->integer = value;
        push(node);
    }

    void readLong1() {
        const std::size_t count = byte();
        pushInteger(signedLittle(count));
    }

    void readBigEndianDouble() {
        checkedLength(8U);
        std::uint64_t bits = 0;
        for (int i = 0; i < 8; ++i) bits = (bits << 8U) | byte();
        auto node = pickleNode(PickleNode::Kind::Floating);
        node->floating = std::bit_cast<double>(bits);
        push(node);
    }

    void readString(std::uint64_t length) {
        if (length > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("Pickle string too large");
        }
        const std::size_t count = static_cast<std::size_t>(length);
        checkedLength(count);
        auto node = pickleNode(PickleNode::Kind::String);
        node->text.assign(reinterpret_cast<const char*>(bytes_.data() + position_), count);
        position_ += count;
        push(node);
    }

    void readBytes(std::uint64_t length) {
        if (length > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("Pickle bytes too large");
        }
        const std::size_t count = static_cast<std::size_t>(length);
        checkedLength(count);
        auto node = pickleNode(PickleNode::Kind::Bytes);
        node->bytes.assign(bytes_.begin() + static_cast<std::ptrdiff_t>(position_),
                           bytes_.begin() + static_cast<std::ptrdiff_t>(position_ + count));
        position_ += count;
        push(node);
    }

    std::string line() {
        const std::size_t begin = position_;
        while (position_ < bytes_.size() && bytes_[position_] != '\n') ++position_;
        if (position_ >= bytes_.size()) throw std::runtime_error("Truncated pickle GLOBAL");
        std::string result(reinterpret_cast<const char*>(bytes_.data() + begin), position_ - begin);
        ++position_;
        return result;
    }

    void readGlobal() {
        auto node = pickleNode(PickleNode::Kind::Global);
        // Evaluate the two line reads explicitly. Operand evaluation order of
        // chained operator+ is not guaranteed, and MSVC otherwise consumes
        // the GLOBAL name before its module ("dtype numpy").
        const std::string module = line();
        const std::string name = line();
        node->text = module + " " + name;
        push(node);
    }

    void stackGlobal() {
        const PicklePtr name = pop();
        const PicklePtr module = pop();
        if (name->kind != PickleNode::Kind::String || module->kind != PickleNode::Kind::String) {
            throw std::runtime_error("Malformed STACK_GLOBAL");
        }
        auto node = pickleNode(PickleNode::Kind::Global);
        node->text = module->text + " " + name->text;
        push(node);
    }

    void makeTuple(std::size_t count) {
        if (stack_.size() < count) throw std::runtime_error("Malformed pickle tuple");
        auto tuple = pickleNode(PickleNode::Kind::Tuple);
        tuple->sequence.assign(stack_.end() - static_cast<std::ptrdiff_t>(count), stack_.end());
        stack_.resize(stack_.size() - count);
        push(tuple);
    }

    std::size_t markerIndex() const {
        for (std::size_t i = stack_.size(); i > 0U; --i) {
            if (stack_[i - 1U]->kind == PickleNode::Kind::Marker) return i - 1U;
        }
        throw std::runtime_error("Pickle MARK not found");
    }

    void makeMarkedTuple() {
        const std::size_t marker = markerIndex();
        auto tuple = pickleNode(PickleNode::Kind::Tuple);
        tuple->sequence.assign(stack_.begin() + static_cast<std::ptrdiff_t>(marker + 1U), stack_.end());
        stack_.resize(marker);
        push(tuple);
    }

    void reduce() {
        const PicklePtr args = pop();
        const PicklePtr callable = pop();
        if (callable->kind != PickleNode::Kind::Global || args->kind != PickleNode::Kind::Tuple) {
            throw std::runtime_error("Malformed pickle REDUCE");
        }
        if (callable->text.find("_reconstruct") != std::string::npos) {
            push(pickleNode(PickleNode::Kind::Array));
            return;
        }
        if (callable->text.find(" dtype") != std::string::npos) {
            auto dtype = pickleNode(PickleNode::Kind::Global);
            dtype->text = "dtype";
            if (!args->sequence.empty() && args->sequence.front()->kind == PickleNode::Kind::String) {
                dtype->text += ':' + args->sequence.front()->text;
            }
            push(dtype);
            return;
        }
        throw std::runtime_error("Disallowed pickle callable: " + callable->text);
    }

    static std::optional<std::int64_t> integerValue(const PicklePtr& node) {
        if (node && node->kind == PickleNode::Kind::Integer) return node->integer;
        return std::nullopt;
    }

    void build() {
        const PicklePtr state = pop();
        const PicklePtr instance = top();
        if (instance->kind == PickleNode::Kind::Global && instance->text.rfind("dtype", 0) == 0) {
            if (state->kind == PickleNode::Kind::Tuple && state->sequence.size() > 1U
                && state->sequence[1U]->kind == PickleNode::Kind::String) {
                const std::string base = instance->text.substr(instance->text.find(':') + 1U);
                instance->text = "dtype:" + state->sequence[1U]->text + base;
            }
            return;
        }
        if (instance->kind != PickleNode::Kind::Array || state->kind != PickleNode::Kind::Tuple
            || state->sequence.size() < 5U) {
            throw std::runtime_error("Unsupported pickle BUILD state");
        }
        const PicklePtr shape = state->sequence[1U];
        const PicklePtr dtype = state->sequence[2U];
        const PicklePtr fortran = state->sequence[3U];
        const PicklePtr data = state->sequence[4U];
        if (shape->kind != PickleNode::Kind::Tuple || data->kind != PickleNode::Kind::Bytes) {
            throw std::runtime_error("Malformed NumPy ndarray pickle state");
        }
        instance->shape.clear();
        for (const PicklePtr& extent : shape->sequence) {
            const auto value = integerValue(extent);
            if (!value || *value < 0) throw std::runtime_error("Invalid NumPy ndarray shape");
            instance->shape.push_back(static_cast<std::size_t>(*value));
        }
        if (dtype->kind == PickleNode::Kind::Global && dtype->text.rfind("dtype:", 0) == 0) {
            instance->dtype = dtype->text.substr(6U);
        }
        instance->fortran = fortran->kind == PickleNode::Kind::Boolean && fortran->boolean;
        instance->bytes = data->bytes;
    }

    void pushMemo(std::uint64_t index) {
        const auto found = memo_.find(index);
        if (found == memo_.end()) throw std::runtime_error("Invalid pickle memo reference");
        push(found->second);
    }

    void setItems() {
        const std::size_t marker = markerIndex();
        if (marker == 0U || (stack_.size() - marker - 1U) % 2U != 0U) {
            throw std::runtime_error("Malformed SETITEMS");
        }
        const PicklePtr dictionary = stack_[marker - 1U];
        if (dictionary->kind != PickleNode::Kind::Dictionary) {
            throw std::runtime_error("SETITEMS target is not dict");
        }
        for (std::size_t i = marker + 1U; i < stack_.size(); i += 2U) {
            if (stack_[i]->kind != PickleNode::Kind::String) {
                throw std::runtime_error("Non-string DEM pickle key");
            }
            dictionary->dictionary[stack_[i]->text] = stack_[i + 1U];
        }
        stack_.resize(marker);
    }

    void setItem() {
        const PicklePtr value = pop();
        const PicklePtr key = pop();
        const PicklePtr dictionary = top();
        if (dictionary->kind != PickleNode::Kind::Dictionary
            || key->kind != PickleNode::Kind::String) {
            throw std::runtime_error("Malformed SETITEM");
        }
        dictionary->dictionary[key->text] = value;
    }

    std::vector<std::uint8_t> bytes_;
    std::size_t position_ = 0;
    std::vector<PicklePtr> stack_;
    std::unordered_map<std::uint64_t, PicklePtr> memo_;
    std::uint64_t nextMemo_ = 0;
};

std::optional<double> pickleDouble(const PicklePtr& node) {
    if (!node) return std::nullopt;
    if (node->kind == PickleNode::Kind::Floating) return node->floating;
    if (node->kind == PickleNode::Kind::Integer) return static_cast<double>(node->integer);
    return std::nullopt;
}

std::vector<double> pickleArrayDoubles(const PicklePtr& node) {
    if (!node || node->kind != PickleNode::Kind::Array) {
        throw std::runtime_error("DEM pickle value is not ndarray");
    }
    std::size_t count = 1U;
    for (const std::size_t extent : node->shape) {
        if (extent != 0U && count > std::numeric_limits<std::size_t>::max() / extent) {
            throw std::runtime_error("DEM ndarray is too large");
        }
        count *= extent;
    }
    const bool isF8 = node->dtype.find("f8") != std::string::npos;
    const bool isF4 = node->dtype.find("f4") != std::string::npos;
    const std::size_t width = isF8 ? 8U : (isF4 ? 4U : 0U);
    if (width == 0U || node->bytes.size() != count * width) {
        throw std::runtime_error("Unsupported or inconsistent DEM ndarray dtype: " + node->dtype);
    }
    if (node->fortran) throw std::runtime_error("Fortran-ordered DEM ndarray is not supported");
    const bool bigEndian = !node->dtype.empty() && node->dtype.front() == '>';
    std::vector<double> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (isF8) {
            std::uint64_t bits = 0;
            if (bigEndian) {
                for (std::size_t j = 0; j < 8U; ++j) bits = (bits << 8U) | node->bytes[i * 8U + j];
            } else {
                for (std::size_t j = 0; j < 8U; ++j) bits |= std::uint64_t(node->bytes[i * 8U + j]) << (j * 8U);
            }
            result[i] = std::bit_cast<double>(bits);
        } else {
            std::uint32_t bits = 0;
            if (bigEndian) {
                for (std::size_t j = 0; j < 4U; ++j) bits = (bits << 8U) | node->bytes[i * 4U + j];
            } else {
                for (std::size_t j = 0; j < 4U; ++j) bits |= std::uint32_t(node->bytes[i * 4U + j]) << (j * 8U);
            }
            result[i] = static_cast<double>(std::bit_cast<float>(bits));
        }
    }
    return result;
}

double numpyMaximum(const std::vector<double>& values) {
    if (values.empty()) throw std::runtime_error("zero-size array reduction has no identity");
    if (std::any_of(values.begin(), values.end(), [](double value) { return std::isnan(value); })) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return *std::max_element(values.begin(), values.end());
}

DemGrid loadDemGrid(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open DEM pkl: " + pathToUtf8(path));
    input.seekg(0, std::ios::end);
    const std::streamoff length = input.tellg();
    if (length < 0) throw std::runtime_error("Cannot determine DEM pkl size");
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    input.read(reinterpret_cast<char*>(bytes.data()), length);
    if (!input) throw std::runtime_error("Cannot read DEM pkl");

    const PicklePtr root = SafeNumpyPickleReader(std::move(bytes)).read();
    if (!root || root->kind != PickleNode::Kind::Dictionary) {
        throw std::runtime_error("DEM pkl root is not a dict");
    }
    const auto required = [&](std::string_view key) -> PicklePtr {
        const auto found = root->dictionary.find(std::string(key));
        if (found == root->dictionary.end()) {
            throw std::runtime_error("DEM pkl missing key '" + std::string(key) + "'");
        }
        return found->second;
    };
    const PicklePtr demNode = required("dem_grid");
    std::vector<double> elevations = pickleArrayDoubles(demNode);
    const std::vector<double> xGrid = pickleArrayDoubles(required("x_grid"));
    const std::vector<double> yGrid = pickleArrayDoubles(required("y_grid"));
    const auto xMinimum = pickleDouble(required("x_min"));
    const auto yMinimum = pickleDouble(required("y_min"));
    if (!xMinimum || !yMinimum) throw std::runtime_error("DEM pkl x_min/y_min must be numeric");
    if (demNode->shape.size() != 2U || demNode->shape[0] != yGrid.size()
        || demNode->shape[1] != xGrid.size() || elevations.size() != xGrid.size() * yGrid.size()) {
        throw std::runtime_error("DEM pkl grid dimensions are inconsistent");
    }
    if (elevations.empty()) throw std::runtime_error("zero-size array reduction has no identity");

    DemGrid result;
    result.elevations = std::move(elevations);
    result.points.reserve(result.elevations.size());
    // The legacy code first adds x_min/y_min and immediately subtracts them
    // before interpolation, leaving these relative grid coordinates.
    for (const double y : yGrid) {
        for (const double x : xGrid) result.points.push_back({x, y});
    }
    result.xMaximum = numpyMaximum(xGrid);
    result.yMaximum = numpyMaximum(yGrid);
    for (const double value : result.elevations) {
        if (std::isnan(value)) continue;
        if (std::isnan(result.zMinimum)) {
            result.zMinimum = value;
            result.zMaximum = value;
        } else {
            result.zMinimum = std::min(result.zMinimum, value);
            result.zMaximum = std::max(result.zMaximum, value);
        }
    }
    return result;
}

class KdTree2D final {
public:
    explicit KdTree2D(const std::vector<Vec2>& points) : points_(points) {
        order_.resize(points.size());
        std::iota(order_.begin(), order_.end(), std::size_t(0));
        nodes_.reserve(points.size());
        root_ = build(0U, order_.size(), 0);
    }

    struct Neighbor final {
        double distance = 0.0;
        std::size_t index = 0;
    };

    std::vector<Neighbor> nearest(const Vec2& point, std::size_t count) const {
        count = std::min(count, points_.size());
        using Candidate = std::pair<double, std::size_t>;
        std::priority_queue<Candidate> heap;
        query(root_, point, count, heap);
        std::vector<Neighbor> result(heap.size());
        for (std::size_t i = heap.size(); i > 0U; --i) {
            const Candidate candidate = heap.top();
            heap.pop();
            result[i - 1U] = {std::sqrt(candidate.first), candidate.second};
        }
        return result;
    }

private:
    struct Node final {
        std::size_t point = 0;
        int axis = 0;
        int left = -1;
        int right = -1;
    };

    int build(std::size_t begin, std::size_t end, int depth) {
        if (begin >= end) return -1;
        const int axis = depth & 1;
        const std::size_t middle = begin + (end - begin) / 2U;
        std::nth_element(order_.begin() + static_cast<std::ptrdiff_t>(begin),
                         order_.begin() + static_cast<std::ptrdiff_t>(middle),
                         order_.begin() + static_cast<std::ptrdiff_t>(end),
                         [&](std::size_t left, std::size_t right) {
                             const double lv = axis == 0 ? points_[left].x : points_[left].y;
                             const double rv = axis == 0 ? points_[right].x : points_[right].y;
                             return lv == rv ? left < right : lv < rv;
                         });
        const int nodeIndex = static_cast<int>(nodes_.size());
        nodes_.push_back({order_[middle], axis, -1, -1});
        const int left = build(begin, middle, depth + 1);
        const int right = build(middle + 1U, end, depth + 1);
        nodes_[static_cast<std::size_t>(nodeIndex)].left = left;
        nodes_[static_cast<std::size_t>(nodeIndex)].right = right;
        return nodeIndex;
    }

    void query(int nodeIndex, const Vec2& point, std::size_t count,
               std::priority_queue<std::pair<double, std::size_t>>& heap) const {
        if (nodeIndex < 0 || count == 0U) return;
        const Node& node = nodes_[static_cast<std::size_t>(nodeIndex)];
        const Vec2& sample = points_[node.point];
        const double dx = point.x - sample.x;
        const double dy = point.y - sample.y;
        const double squared = dx * dx + dy * dy;
        if (heap.size() < count) {
            heap.emplace(squared, node.point);
        } else if (squared < heap.top().first
                   || (squared == heap.top().first && node.point < heap.top().second)) {
            heap.pop();
            heap.emplace(squared, node.point);
        }
        const double split = node.axis == 0 ? dx : dy;
        const int nearChild = split <= 0.0 ? node.left : node.right;
        const int farChild = split <= 0.0 ? node.right : node.left;
        query(nearChild, point, count, heap);
        const double threshold = heap.size() < count
            ? std::numeric_limits<double>::infinity() : heap.top().first;
        if (split * split <= threshold) query(farChild, point, count, heap);
    }

    const std::vector<Vec2>& points_;
    std::vector<std::size_t> order_;
    std::vector<Node> nodes_;
    int root_ = -1;
};

double idwPrediction(const std::vector<KdTree2D::Neighbor>& neighbors,
                     const std::vector<double>& elevations) {
    if (neighbors.empty()) return 0.0;
    if (neighbors.size() == 1U) return elevations[neighbors.front().index];
    double weighted = 0.0;
    double sum = 0.0;
    for (const auto& neighbor : neighbors) {
        const double distance = std::max(neighbor.distance, 1.0e-12);
        const double weight = 1.0 / (distance * distance);
        weighted += weight * elevations[neighbor.index];
        sum += weight;
    }
    return weighted / sum;
}

void preserveActiveAttributes(vtkPointData* source, vtkPointData* destination) {
    const auto activate = [&](vtkAbstractArray* array, auto setter) {
        if (array != nullptr && array->GetName() != nullptr
            && destination->GetAbstractArray(array->GetName()) != nullptr) {
            (destination->*setter)(array->GetName());
        }
    };
    activate(source->GetScalars(), &vtkPointData::SetActiveScalars);
    activate(source->GetVectors(), &vtkPointData::SetActiveVectors);
    activate(source->GetNormals(), &vtkPointData::SetActiveNormals);
    activate(source->GetTCoords(), &vtkPointData::SetActiveTCoords);
    activate(source->GetTensors(), &vtkPointData::SetActiveTensors);
}

vtkSmartPointer<vtkUnstructuredGrid> makeWoDemPointCloud(vtkDataSet* mesh,
                                                          const DemGrid& dem) {
    vtkNew<vtkCellDataToPointData> cellToPoint;
    cellToPoint->SetInputData(mesh);
    cellToPoint->PassCellDataOn();
    cellToPoint->Update();
    vtkDataSet* pointMesh = vtkDataSet::SafeDownCast(cellToPoint->GetOutput());
    if (pointMesh == nullptr) pointMesh = mesh;

    const vtkIdType pointCount = pointMesh->GetNumberOfPoints();
    std::vector<std::array<double, 3>> points(static_cast<std::size_t>(pointCount));
    for (vtkIdType index = 0; index < pointCount; ++index) {
        pointMesh->GetPoint(index, points[static_cast<std::size_t>(index)].data());
    }

    const KdTree2D tree(dem.points);
    const std::size_t neighborCount = std::min<std::size_t>(12U, dem.points.size());
    std::vector<double> demHeight(static_cast<std::size_t>(pointCount), 0.0);
    std::vector<double> newZ(static_cast<std::size_t>(pointCount), 0.0);
    std::vector<unsigned char> keep(static_cast<std::size_t>(pointCount), 0U);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t linear = 0; linear < static_cast<std::ptrdiff_t>(pointCount); ++linear) {
        const std::size_t index = static_cast<std::size_t>(linear);
        const auto& point = points[index];
        const bool inside = point[0] >= 0.0 && point[0] <= dem.xMaximum
            && point[1] >= 0.0 && point[1] <= dem.yMaximum;
        if (inside && neighborCount > 0U) {
            demHeight[index] = idwPrediction(tree.nearest({point[0], point[1]}, neighborCount),
                                              dem.elevations);
        }
        newZ[index] = point[2] - demHeight[index];
        keep[index] = newZ[index] >= 0.0 ? 1U : 0U;
    }

    std::vector<vtkIdType> kept;
    kept.reserve(static_cast<std::size_t>(pointCount));
    for (vtkIdType index = 0; index < pointCount; ++index) {
        if (keep[static_cast<std::size_t>(index)] != 0U) kept.push_back(index);
    }

    vtkNew<vtkPoints> outputPoints;
    outputPoints->SetDataTypeToDouble();
    outputPoints->SetNumberOfPoints(static_cast<vtkIdType>(kept.size()));
    for (vtkIdType outputIndex = 0; outputIndex < static_cast<vtkIdType>(kept.size()); ++outputIndex) {
        const vtkIdType sourceIndex = kept[static_cast<std::size_t>(outputIndex)];
        std::array<double, 3> point = points[static_cast<std::size_t>(sourceIndex)];
        point[2] = newZ[static_cast<std::size_t>(sourceIndex)];
        outputPoints->SetPoint(outputIndex, point.data());
    }

    vtkSmartPointer<vtkUnstructuredGrid> result = vtkSmartPointer<vtkUnstructuredGrid>::New();
    result->SetPoints(outputPoints);
    result->Allocate(static_cast<vtkIdType>(kept.size()));
    for (vtkIdType index = 0; index < static_cast<vtkIdType>(kept.size()); ++index) {
        result->InsertNextCell(VTK_VERTEX, 1, &index);
    }

    vtkNew<vtkIdTypeArray> originalIds;
    originalIds->SetName("vtkOriginalPointIds");
    originalIds->SetNumberOfTuples(static_cast<vtkIdType>(kept.size()));
    for (vtkIdType index = 0; index < static_cast<vtkIdType>(kept.size()); ++index) {
        originalIds->SetValue(index, index);
    }
    result->GetPointData()->AddArray(originalIds);

    vtkPointData* sourceData = pointMesh->GetPointData();
    vtkPointData* destinationData = result->GetPointData();
    for (int arrayIndex = 0; arrayIndex < sourceData->GetNumberOfArrays(); ++arrayIndex) {
        vtkAbstractArray* source = sourceData->GetAbstractArray(arrayIndex);
        if (source == nullptr || source->GetName() == nullptr
            || source->GetNumberOfTuples() != pointCount) {
            continue;
        }
        vtkSmartPointer<vtkAbstractArray> filtered;
        filtered.TakeReference(source->NewInstance());
        filtered->SetName(source->GetName());
        filtered->SetNumberOfComponents(source->GetNumberOfComponents());
        filtered->SetNumberOfTuples(static_cast<vtkIdType>(kept.size()));
        for (vtkIdType outputIndex = 0;
             outputIndex < static_cast<vtkIdType>(kept.size()); ++outputIndex) {
            filtered->SetTuple(outputIndex, kept[static_cast<std::size_t>(outputIndex)], source);
        }
        destinationData->RemoveArray(filtered->GetName());
        destinationData->AddArray(filtered);
    }

    vtkNew<vtkDoubleArray> demArray;
    demArray->SetName("dem_height");
    demArray->SetNumberOfTuples(static_cast<vtkIdType>(kept.size()));
    for (vtkIdType outputIndex = 0;
         outputIndex < static_cast<vtkIdType>(kept.size()); ++outputIndex) {
        const vtkIdType sourceIndex = kept[static_cast<std::size_t>(outputIndex)];
        demArray->SetValue(outputIndex, demHeight[static_cast<std::size_t>(sourceIndex)]);
    }
    destinationData->RemoveArray("dem_height");
    destinationData->AddArray(demArray);
    preserveActiveAttributes(sourceData, destinationData);
    return result;
}

void writeWoDem(vtkDataSet* baseMesh,
                const fs::path& demPath,
                const fs::path& outputPath,
                std::ostream& output) {
    const DemGrid dem = loadDemGrid(demPath);
    baseMesh->ComputeBounds();
    double bounds[6]{};
    baseMesh->GetBounds(bounds);
    output << std::fixed << std::setprecision(3)
           << "[INFO] VTK max corner after translation: (" << bounds[1] << ", "
           << bounds[3] << ", " << bounds[5] << ")\n"
           << "[INFO] PKL DEM XY range after subtracting x_min y_min: x=[0.000, "
           << dem.xMaximum << "], y=[0.000, " << dem.yMaximum << "]\n"
           << "[INFO] PKL DEM Z range: min " << dem.zMinimum << ", max "
           << dem.zMaximum << "\n" << std::defaultfloat;
    const vtkSmartPointer<vtkUnstructuredGrid> cloud = makeWoDemPointCloud(baseMesh, dem);
    writeDataSetNew(outputPath, cloud);
    output << "[OK] Wrote woDEM point-cloud VTK with DEM field to: "
           << pathToUtf8(outputPath) << '\n';
}

int processSingle(const fs::path& vtkPath,
                  const std::string& tailname,
                  std::ostream& output) {
    if (!isRegularFile(vtkPath)) {
        throw std::runtime_error("VTK file not found: " + pathToUtf8(vtkPath));
    }
    const DerivedName derived = deriveSingleName(vtkPath);
    const vtkSmartPointer<vtkDataSet> input = readDataSet(vtkPath);
    const std::array<double, 3> offset = translationOffset(input);
    const vtkSmartPointer<vtkDataSet> translated = translateDataSet(input, offset);
    const fs::path outputPath = buildOutputPath(vtkPath.parent_path(), derived.prefix,
                                                derived.casename, derived.datetime, tailname);
    writeDataSetNew(outputPath, translated);
    output << "[OK] Translated single VTK using offset " << tuple3(offset)
           << " and wrote: " << pathToUtf8(outputPath) << '\n';
    return 0;
}

int processConfig(const fs::path& configPath,
                  const std::string& tailname,
                  bool absoluteOnly,
                  std::ostream& output) {
    const auto [casename, datetime] = parseCaseText(configPath);
    const fs::path parent = configPath.parent_path();
    const fs::path searchDirectory = parent / "proj_temp" / "vtk";
    const std::optional<fs::path> vtkU = findLatest(searchDirectory, datetime, "u");
    const std::optional<fs::path> vtkRho = findLatest(searchDirectory, datetime, "rho");
    if (!vtkU && !vtkRho) {
        throw std::runtime_error("No matching VTK files found under: "
                                 + pathToUtf8(searchDirectory));
    }

    const fs::path basePath = vtkU ? *vtkU : *vtkRho;
    const vtkSmartPointer<vtkDataSet> baseInput = readDataSet(basePath);
    const std::array<double, 3> offset = translationOffset(baseInput);
    const vtkSmartPointer<vtkDataSet> base = translateDataSet(baseInput, offset);
    const std::string prefix = !vtkRho ? "uvw" : (vtkU ? "uvwrho" : "rho");

    if (vtkU && basePath != *vtkU) {
        const vtkSmartPointer<vtkDataSet> other = translateDataSet(readDataSet(*vtkU), offset);
        ensureSameTopology(base, other);
        copyPointAndCellData(other, base, "_u");
    }
    if (vtkRho && basePath != *vtkRho) {
        const vtkSmartPointer<vtkDataSet> other = translateDataSet(readDataSet(*vtkRho), offset);
        ensureSameTopology(base, other);
        copyPointAndCellData(other, base, "_rho");
    }

    const fs::path outputPath = buildOutputPath(parent, prefix, casename, datetime, tailname);
    writeDataSetNew(outputPath, base);

    const fs::path demPath = parent / "proj_temp" / "dem_grid.pkl";
    if (!absoluteOnly && isRegularFile(demPath)) {
        try {
            const fs::path woDemPath = buildOutputPath(
                parent, prefix, casename, "woDEM_" + datetime, tailname);
            writeWoDem(base, demPath, woDemPath, output);
        } catch (const std::exception& exception) {
            output << "[WARN] DEM pkl exists but woDEM generation failed: "
                   << exception.what() << '\n';
        }
    }

    std::vector<std::string> details;
    if (vtkU) details.push_back("u=" + pathToUtf8(vtkU->filename()));
    if (vtkRho) details.push_back("rho=" + pathToUtf8(vtkRho->filename()));
    output << "[OK] Translated with shared offset " << tuple3(offset) << "; merged ";
    for (std::size_t index = 0; index < details.size(); ++index) {
        if (index != 0U) output << " and ";
        output << details[index];
    }
    output << " into: " << pathToUtf8(outputPath) << '\n';
    return 0;
}

} // namespace

int run_transluw(const std::vector<std::string>& arguments,
                 std::ostream& output,
                 std::ostream& error) {
    try {
        const Arguments parsed = parseArguments(arguments);
        if (parsed.help) {
            printUsage(output);
            return 0;
        }
        const fs::path inputPath = absoluteNormalized(parsed.inputPath);
        output << "[INFO] Processing input: " << pathToUtf8(inputPath)
               << " with tailname: '" << parsed.tailname << "'\n";
        std::error_code existsError;
        if (!fs::exists(inputPath, existsError) || existsError) {
            error << "[ERROR] Input path does not exist: " << pathToUtf8(inputPath) << '\n';
            return 1;
        }
        if (lowerAscii(pathToUtf8(inputPath.extension())) == ".vtk") {
            return processSingle(inputPath, parsed.tailname, output);
        }
        return processConfig(inputPath, parsed.tailname, parsed.absoluteOnly, output);
    } catch (const ArgumentError& exception) {
        printShortUsage(error);
        error << "transluw.py: error: " << exception.what() << '\n';
        return 2;
    } catch (const std::exception& exception) {
        error << "[ERROR] " << exception.what() << '\n';
        return 1;
    }
}

} // namespace luw::cli
