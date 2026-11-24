#include "luw/validate_commands.hpp"

#include "luw/deck_document.hpp"
#include "luw/preprocess_compute_config.hpp"

#include <QByteArray>
#include <QString>

#include <vtkAppendPolyData.h>
#include <vtkCellArray.h>
#include <vtkCleanPolyData.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkRemoveDuplicatePolys.h>
#include <vtkSTLReader.h>
#include <vtkSTLWriter.h>
#include <vtkSmartPointer.h>
#include <vtkTriangle.h>
#include <vtkTriangleFilter.h>

#ifndef LUW_HAS_MANIFOLD
#define LUW_HAS_MANIFOLD 0
#endif

#if LUW_HAS_MANIFOLD
#include <manifold/manifold.h>
#endif

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iterator>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace luw::cli {
namespace {

namespace fs = std::filesystem;

class FatalError final : public std::exception {
public:
    FatalError(std::string message, int code)
        : message_(std::move(message)), code_(code) {}

    [[nodiscard]] const char* what() const noexcept override { return message_.c_str(); }
    [[nodiscard]] int code() const noexcept { return code_; }

private:
    std::string message_;
    int code_ = 1;
};

struct ConfigValues {
    QString caseName;
    double baseHeight = 0.0;
    double xExpansionRatio = 0.0;
    double yExpansionRatio = 0.0;
    double zLimit = 0.0;
    QString originalText;
};

struct Bounds {
    double xmin = 0.0;
    double xmax = 0.0;
    double ymin = 0.0;
    double ymax = 0.0;
    double zmin = 0.0;
    double zmax = 0.0;
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

std::string timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    std::ostringstream output;
    output << std::put_time(&local, "%Y-%m-%d %H:%M:%S");
    return output.str();
}

void debug(std::ostream& out, const std::string& message) {
    out << '[' << timestamp() << "] " << message << '\n';
}

[[noreturn]] void fatal(std::string message, int code = 1) {
    throw FatalError(std::move(message), code);
}

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool hasAllowedDeckExtension(const fs::path& path) {
    const std::string extension = lowerAscii(path.extension().string());
    return extension == ".luwdg" || extension == ".luwpf";
}

std::string pythonFloat(double value) {
    if (std::isnan(value)) return "nan";
    if (std::isinf(value)) return std::signbit(value) ? "-inf" : "inf";

    std::array<char, 128> storage{};
    const auto result = std::to_chars(storage.data(), storage.data() + storage.size(), value,
                                      std::chars_format::general);
    if (result.ec != std::errc{}) {
        std::ostringstream fallback;
        fallback << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
        return fallback.str();
    }
    std::string text(storage.data(), result.ptr);
    if (text.find_first_of(".eE") == std::string::npos) text += ".0";
    return text;
}

std::string pythonList(const std::vector<fs::path>& paths) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < paths.size(); ++index) {
        if (index != 0) output << ", ";
        std::string text = displayPath(paths[index]);
        std::string escaped;
        escaped.reserve(text.size());
        for (const char character : text) {
            if (character == '\\' || character == '\'') escaped.push_back('\\');
            escaped.push_back(character);
        }
        output << '\'' << escaped << '\'';
    }
    output << ']';
    return output.str();
}

std::string list3(double x, double y, double z) {
    return '[' + pythonFloat(x) + ", " + pythonFloat(y) + ", " + pythonFloat(z) + ']';
}

std::string fixedRange(double minimum, double maximum) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << '[' << minimum << ", " << maximum << ']';
    return output.str();
}

fs::path absoluteNormalized(const fs::path& input) {
    std::error_code error;
    fs::path result = fs::absolute(input, error);
    if (error) fatal(error.message());
    return result.lexically_normal();
}

std::vector<fs::path> findByExtensions(const fs::path& directory,
                                       const std::vector<std::string>& extensions) {
    std::vector<fs::path> result;
    std::error_code error;
    for (fs::directory_iterator it(directory, error), end; !error && it != end; it.increment(error)) {
        if (!it->is_regular_file(error) || error) continue;
        const std::string extension = lowerAscii(it->path().extension().string());
        if (std::find(extensions.begin(), extensions.end(), extension) != extensions.end()) {
            result.push_back(it->path());
        }
    }
    std::sort(result.begin(), result.end(), [](const fs::path& lhs, const fs::path& rhs) {
        return displayPath(lhs) < displayPath(rhs);
    });
    return result;
}

fs::path findInputDeck(const std::vector<std::string>& args, std::ostream& out) {
    const fs::path current = absoluteNormalized(fs::current_path());
    debug(out, "Current working directory: " + displayPath(current));

    if (!args.empty()) {
        const fs::path input = pathFromUtf8(args.front());
        debug(out, "Input argument provided: " + displayPath(input));
        if (!fs::is_regular_file(input)) {
            fatal("Input path does not exist or is not a file: " + displayPath(input));
        }
        if (!hasAllowedDeckExtension(input)) {
            fatal("Input file must end with .luwdg or .luwpf: " + displayPath(input));
        }
        return absoluteNormalized(input);
    }

    const fs::path confDg = current / "conf.luwdg";
    const fs::path confPf = current / "conf.luwpf";
    debug(out, "No input argument provided. Searching for conf.luwdg or conf.luwpf in current directory.");
    if (fs::is_regular_file(confDg)) {
        debug(out, "Found conf.luwdg: " + displayPath(confDg));
        return absoluteNormalized(confDg);
    }
    if (fs::is_regular_file(confPf)) {
        debug(out, "Found conf.luwpf: " + displayPath(confPf));
        return absoluteNormalized(confPf);
    }

    debug(out, "conf.luwdg/conf.luwpf not found. Searching for any *.luwdg or *.luwpf in current directory.");
    const std::vector<fs::path> candidates = findByExtensions(current, {".luwdg", ".luwpf"});
    if (!candidates.empty()) {
        debug(out, "Found candidate config files: " + pythonList(candidates));
        debug(out, "Using the first one (sorted): " + displayPath(candidates.front()));
        return absoluteNormalized(candidates.front());
    }

    fatal("No .luwdg/.luwpf file found in current directory, and no input argument was provided.");
}

QString readDeckText(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) fatal("Unable to read luwdg file: " + displayPath(path));
    const std::string bytes((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    return QString::fromUtf8(bytes.data(), static_cast<qsizetype>(bytes.size()));
}

ConfigValues parseDeck(const fs::path& path, std::ostream& out) {
    debug(out, "Reading luwdg file: " + displayPath(path));
    const QString originalText = readDeckText(path);
    const DeckDocument deck = DeckDocument::fromText(originalText);
    const PreprocessComputeConfig computePolicy =
        resolvePreprocessComputeConfig(&deck).first;
    debug(out, "Preprocessing compute policy: "
               + preprocessComputePolicyText(computePolicy).toStdString());
    debug(out, "Diagnostic-geometry preparation has no GPU compute kernel; "
               "this stage runs on CPU.");

    const auto caseName = deck.text(QStringLiteral("casename"));
    const auto baseHeight = deck.floating(QStringLiteral("base_height"));
    const auto xExpansion = deck.floating(QStringLiteral("x_exp_rat"));
    const auto yExpansion = deck.floating(QStringLiteral("y_exp_rat"));
    const auto zLimit = deck.floating(QStringLiteral("z_limit"));

    if (!caseName) fatal("casename not found in luwdg file.");
    if (!baseHeight) fatal("base_height not found in luwdg file.");
    if (*baseHeight <= 0.0) {
        fatal("base_height must be positive. Got: " + pythonFloat(*baseHeight));
    }
    if (!xExpansion) fatal("x_exp_rat not found in luwdg file.");
    if (!yExpansion) fatal("y_exp_rat not found in luwdg file.");
    if (*xExpansion <= 0.0) {
        fatal("x_exp_rat must be positive. Got: " + pythonFloat(*xExpansion));
    }
    if (*yExpansion <= 0.0) {
        fatal("y_exp_rat must be positive. Got: " + pythonFloat(*yExpansion));
    }
    if (!zLimit) fatal("z_limit not found in luwdg file.");
    if (*zLimit <= 0.0) fatal("z_limit must be positive. Got: " + pythonFloat(*zLimit));

    debug(out, "Parsed casename: " + caseName->toUtf8().toStdString());
    debug(out, "Parsed base_height (m): " + pythonFloat(*baseHeight));
    debug(out, "Parsed x_exp_rat: " + pythonFloat(*xExpansion));
    debug(out, "Parsed y_exp_rat: " + pythonFloat(*yExpansion));
    debug(out, "Parsed z_limit (m): " + pythonFloat(*zLimit));
    return {*caseName, *baseHeight, *xExpansion, *yExpansion, *zLimit, originalText};
}

fs::path pickStl(const fs::path& buildingDirectory, std::ostream& out) {
    const fs::path preferred = buildingDirectory / "rawbuildings.stl";
    debug(out, "Looking for STL in: " + displayPath(buildingDirectory));
    if (fs::is_regular_file(preferred)) {
        debug(out, "Found preferred STL: " + displayPath(preferred));
        return preferred;
    }

    debug(out, "Preferred rawbuildings.stl not found. Falling back to any *.stl.");
    const std::vector<fs::path> candidates = findByExtensions(buildingDirectory, {".stl"});
    if (candidates.empty()) fatal("No STL file found in " + displayPath(buildingDirectory));
    debug(out, "Found STL candidates: " + pythonList(candidates));
    debug(out, "Using the first one (sorted): " + displayPath(candidates.front()));
    return candidates.front();
}

Bounds boundsOf(vtkPolyData* mesh) {
    double raw[6]{};
    mesh->GetBounds(raw);
    return {raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]};
}

vtkSmartPointer<vtkPolyData> pointsToDouble(vtkPolyData* input) {
    vtkNew<vtkPoints> points;
    points->SetDataTypeToDouble();
    points->SetNumberOfPoints(input->GetNumberOfPoints());
    for (vtkIdType index = 0; index < input->GetNumberOfPoints(); ++index) {
        double point[3]{};
        input->GetPoint(index, point);
        points->SetPoint(index, point);
    }
    vtkSmartPointer<vtkPolyData> result = vtkSmartPointer<vtkPolyData>::New();
    result->DeepCopy(input);
    result->SetPoints(points);
    return result;
}

vtkSmartPointer<vtkPolyData> loadMesh(const fs::path& path, std::ostream& out) {
    debug(out, "Loading STL: " + displayPath(path));
    vtkNew<vtkSTLReader> reader;
    const std::string filename = displayPath(path);
    reader->SetFileName(filename.c_str());
    reader->MergingOn();
    reader->Update();
    vtkPolyData* loaded = reader->GetOutput();
    if (loaded == nullptr) fatal("Failed to load STL mesh (returned None).");

    vtkSmartPointer<vtkPolyData> mesh = pointsToDouble(loaded);
    debug(out, "Mesh vertices: " + std::to_string(mesh->GetNumberOfPoints()));
    debug(out, "Mesh faces: " + std::to_string(mesh->GetNumberOfPolys()));
    if (mesh->GetNumberOfPolys() == 0 || mesh->GetNumberOfPoints() == 0) {
        fatal("Mesh is empty.");
    }

    const Bounds bounds = boundsOf(mesh);
    debug(out, "Mesh bounds before base: min=" + list3(bounds.xmin, bounds.ymin, bounds.zmin)
               + ", max=" + list3(bounds.xmax, bounds.ymax, bounds.zmax));
    return mesh;
}

void addTriangle(vtkCellArray* cells, vtkIdType a, vtkIdType b, vtkIdType c) {
    vtkIdType ids[3] = {a, b, c};
    cells->InsertNextCell(3, ids);
}

vtkSmartPointer<vtkPolyData> createBase(const Bounds& bounds,
                                        double baseHeight,
                                        double xExpansion,
                                        double yExpansion,
                                        std::ostream& out) {
    const double dx = bounds.xmax - bounds.xmin;
    const double dy = bounds.ymax - bounds.ymin;
    if (dx <= 0.0 || dy <= 0.0) {
        fatal("Invalid mesh XY extents. dx=" + pythonFloat(dx) + ", dy=" + pythonFloat(dy));
    }
    if (xExpansion <= 0.0 || yExpansion <= 0.0) {
        fatal("Invalid expansion ratios. x_exp_rat=" + pythonFloat(xExpansion)
              + ", y_exp_rat=" + pythonFloat(yExpansion));
    }

    const double baseDx = xExpansion * dx;
    const double baseDy = yExpansion * dy;
    const double baseDz = baseHeight;
    const double centerX = (bounds.xmin + bounds.xmax) * 0.5;
    const double centerY = (bounds.ymin + bounds.ymax) * 0.5;
    const double centerZ = bounds.zmin - baseDz * 0.5;

    debug(out, "Creating base block with extents (m): x=" + pythonFloat(baseDx)
               + ", y=" + pythonFloat(baseDy) + ", z=" + pythonFloat(baseDz));
    debug(out, "Base block center (m): x=" + pythonFloat(centerX)
               + ", y=" + pythonFloat(centerY) + ", z=" + pythonFloat(centerZ));

    const double xmin = centerX - baseDx * 0.5;
    const double xmax = centerX + baseDx * 0.5;
    const double ymin = centerY - baseDy * 0.5;
    const double ymax = centerY + baseDy * 0.5;
    const double zmin = centerZ - baseDz * 0.5;
    const double zmax = centerZ + baseDz * 0.5;

    vtkNew<vtkPoints> points;
    points->SetDataTypeToDouble();
    points->InsertNextPoint(xmin, ymin, zmin); // 0
    points->InsertNextPoint(xmax, ymin, zmin); // 1
    points->InsertNextPoint(xmax, ymax, zmin); // 2
    points->InsertNextPoint(xmin, ymax, zmin); // 3
    points->InsertNextPoint(xmin, ymin, zmax); // 4
    points->InsertNextPoint(xmax, ymin, zmax); // 5
    points->InsertNextPoint(xmax, ymax, zmax); // 6
    points->InsertNextPoint(xmin, ymax, zmax); // 7

    vtkNew<vtkCellArray> triangles;
    addTriangle(triangles, 0, 2, 1);
    addTriangle(triangles, 0, 3, 2);
    addTriangle(triangles, 4, 5, 6);
    addTriangle(triangles, 4, 6, 7);
    addTriangle(triangles, 0, 1, 5);
    addTriangle(triangles, 0, 5, 4);
    addTriangle(triangles, 1, 2, 6);
    addTriangle(triangles, 1, 6, 5);
    addTriangle(triangles, 2, 3, 7);
    addTriangle(triangles, 2, 7, 6);
    addTriangle(triangles, 3, 0, 4);
    addTriangle(triangles, 3, 4, 7);

    vtkSmartPointer<vtkPolyData> base = vtkSmartPointer<vtkPolyData>::New();
    base->SetPoints(points);
    base->SetPolys(triangles);
    const Bounds baseBounds = boundsOf(base);
    debug(out, "Base bounds: min=" + list3(baseBounds.xmin, baseBounds.ymin, baseBounds.zmin)
               + ", max=" + list3(baseBounds.xmax, baseBounds.ymax, baseBounds.zmax));
    return base;
}

vtkSmartPointer<vtkPolyData> concatenateMeshes(vtkPolyData* mesh,
                                               vtkPolyData* base,
                                               std::ostream& out) {
    vtkNew<vtkAppendPolyData> append;
    append->AddInputData(mesh);
    append->AddInputData(base);
    append->Update();
    vtkPolyData* concatenated = append->GetOutput();
    debug(out, "Concatenated mesh vertices: " + std::to_string(concatenated->GetNumberOfPoints()));
    debug(out, "Concatenated mesh faces: " + std::to_string(concatenated->GetNumberOfPolys()));
    debug(out, "Running basic cleanup on concatenated mesh.");

    vtkNew<vtkCleanPolyData> clean;
    clean->SetInputConnection(append->GetOutputPort());
    clean->SetTolerance(0.0);
    clean->PointMergingOn();
    clean->ConvertLinesToPointsOff();
    clean->ConvertPolysToLinesOff();
    clean->ConvertStripsToPolysOff();

    vtkNew<vtkRemoveDuplicatePolys> removeDuplicates;
    removeDuplicates->SetInputConnection(clean->GetOutputPort());
    removeDuplicates->Update();

    vtkSmartPointer<vtkPolyData> merged = vtkSmartPointer<vtkPolyData>::New();
    merged->DeepCopy(removeDuplicates->GetOutput());
    debug(out, "After cleanup vertices: " + std::to_string(merged->GetNumberOfPoints()));
    debug(out, "After cleanup faces: " + std::to_string(merged->GetNumberOfPolys()));
    return merged;
}

struct MeshUnionResult {
    vtkSmartPointer<vtkPolyData> mesh;
    bool usedBoolean = false;
    std::string message;
};

#if LUW_HAS_MANIFOLD

std::string manifoldStatusName(manifold::Manifold::Error status) {
    using Error = manifold::Manifold::Error;
    switch (status) {
    case Error::NoError: return "NoError";
    case Error::NonFiniteVertex: return "NonFiniteVertex";
    case Error::NotManifold: return "NotManifold";
    case Error::VertexOutOfBounds: return "VertexOutOfBounds";
    case Error::PropertiesWrongLength: return "PropertiesWrongLength";
    case Error::MissingPositionProperties: return "MissingPositionProperties";
    case Error::MergeVectorsDifferentLengths: return "MergeVectorsDifferentLengths";
    case Error::MergeIndexOutOfBounds: return "MergeIndexOutOfBounds";
    case Error::TransformWrongLength: return "TransformWrongLength";
    case Error::RunIndexWrongLength: return "RunIndexWrongLength";
    case Error::FaceIDWrongLength: return "FaceIDWrongLength";
    case Error::InvalidConstruction: return "InvalidConstruction";
    case Error::ResultTooLarge: return "ResultTooLarge";
    }
    return "Unknown(" + std::to_string(static_cast<int>(status)) + ')';
}

manifold::MeshGL64 vtkPolyDataToManifold(vtkPolyData* input) {
    if (input == nullptr) throw std::runtime_error("input PolyData is null");

    vtkNew<vtkTriangleFilter> triangleFilter;
    triangleFilter->SetInputData(input);
    triangleFilter->PassLinesOff();
    triangleFilter->PassVertsOff();
    triangleFilter->Update();
    vtkPolyData* triangles = triangleFilter->GetOutput();
    if (triangles == nullptr || triangles->GetNumberOfPoints() == 0 ||
        triangles->GetNumberOfPolys() == 0) {
        throw std::runtime_error("triangulated PolyData is empty");
    }

    manifold::MeshGL64 result;
    result.numProp = 3;
    const vtkIdType pointCount = triangles->GetNumberOfPoints();
    result.vertProperties.reserve(static_cast<std::size_t>(pointCount) * 3U);
    for (vtkIdType index = 0; index < pointCount; ++index) {
        double point[3]{};
        triangles->GetPoint(index, point);
        result.vertProperties.push_back(point[0]);
        result.vertProperties.push_back(point[1]);
        result.vertProperties.push_back(point[2]);
    }

    vtkCellArray* polygons = triangles->GetPolys();
    if (polygons == nullptr) throw std::runtime_error("triangulated PolyData has no polygons");
    result.triVerts.reserve(
        static_cast<std::size_t>(triangles->GetNumberOfPolys()) * 3U);
    polygons->InitTraversal();
    vtkIdType cellPointCount = 0;
    const vtkIdType* cellPoints = nullptr;
    while (polygons->GetNextCell(cellPointCount, cellPoints) != 0) {
        if (cellPointCount != 3 || cellPoints == nullptr) {
            throw std::runtime_error("VTK triangle filter returned a non-triangle polygon");
        }
        for (vtkIdType corner = 0; corner < 3; ++corner) {
            const vtkIdType point = cellPoints[corner];
            if (point < 0 || point >= pointCount) {
                throw std::runtime_error("triangulated PolyData contains an invalid point index");
            }
            result.triVerts.push_back(static_cast<std::uint64_t>(point));
        }
    }
    if (result.triVerts.empty()) {
        throw std::runtime_error("triangulated PolyData contains no triangles");
    }
    return result;
}

std::uint64_t findRepresentative(std::vector<std::uint64_t>& parents,
                                 std::uint64_t vertex) {
    std::uint64_t root = vertex;
    while (parents[static_cast<std::size_t>(root)] != root) {
        root = parents[static_cast<std::size_t>(root)];
    }
    while (vertex != root) {
        const std::uint64_t next = parents[static_cast<std::size_t>(vertex)];
        parents[static_cast<std::size_t>(vertex)] = root;
        vertex = next;
    }
    return root;
}

vtkSmartPointer<vtkPolyData> manifoldToVtkPolyData(
    const manifold::MeshGL64& input) {
    if (input.numProp < 3 || input.vertProperties.size() % input.numProp != 0U ||
        input.triVerts.empty() || input.triVerts.size() % 3U != 0U) {
        throw std::runtime_error("boolean output MeshGL64 has invalid buffer lengths");
    }
    const std::uint64_t vertexCount = input.NumVert();
    if (vertexCount == 0U ||
        vertexCount > static_cast<std::uint64_t>(std::numeric_limits<vtkIdType>::max())) {
        throw std::runtime_error("boolean output MeshGL64 has an unsupported vertex count");
    }
    if (input.mergeFromVert.size() != input.mergeToVert.size()) {
        throw std::runtime_error("boolean output MeshGL64 has inconsistent merge vectors");
    }

    std::vector<std::uint64_t> parents(static_cast<std::size_t>(vertexCount));
    std::iota(parents.begin(), parents.end(), std::uint64_t{0});
    for (std::size_t index = 0; index < input.mergeFromVert.size(); ++index) {
        const std::uint64_t from = input.mergeFromVert[index];
        const std::uint64_t to = input.mergeToVert[index];
        if (from >= vertexCount || to >= vertexCount) {
            throw std::runtime_error("boolean output MeshGL64 merge index is out of range");
        }
        const std::uint64_t fromRoot = findRepresentative(parents, from);
        const std::uint64_t toRoot = findRepresentative(parents, to);
        if (fromRoot != toRoot) {
            parents[static_cast<std::size_t>(fromRoot)] = toRoot;
        }
    }

    vtkNew<vtkPoints> points;
    points->SetDataTypeToDouble();
    std::vector<vtkIdType> vtkPointForRoot(static_cast<std::size_t>(vertexCount), -1);
    for (std::uint64_t vertex = 0; vertex < vertexCount; ++vertex) {
        const std::uint64_t root = findRepresentative(parents, vertex);
        vtkIdType& vtkPoint = vtkPointForRoot[static_cast<std::size_t>(root)];
        if (vtkPoint >= 0) continue;
        const std::size_t offset = static_cast<std::size_t>(root * input.numProp);
        vtkPoint = points->InsertNextPoint(input.vertProperties[offset],
                                           input.vertProperties[offset + 1U],
                                           input.vertProperties[offset + 2U]);
    }

    vtkNew<vtkCellArray> triangles;
    for (std::size_t offset = 0; offset < input.triVerts.size(); offset += 3U) {
        vtkIdType cell[3]{};
        for (std::size_t corner = 0; corner < 3U; ++corner) {
            const std::uint64_t vertex = input.triVerts[offset + corner];
            if (vertex >= vertexCount) {
                throw std::runtime_error("boolean output MeshGL64 triangle index is out of range");
            }
            const std::uint64_t root = findRepresentative(parents, vertex);
            cell[corner] = vtkPointForRoot[static_cast<std::size_t>(root)];
        }
        triangles->InsertNextCell(3, cell);
    }

    vtkSmartPointer<vtkPolyData> result = vtkSmartPointer<vtkPolyData>::New();
    result->SetPoints(points);
    result->SetPolys(triangles);
    if (result->GetNumberOfPoints() == 0 || result->GetNumberOfPolys() == 0) {
        throw std::runtime_error("converted boolean PolyData is empty");
    }
    return result;
}

void requireValidManifold(const manifold::Manifold& value,
                          std::string_view label) {
    const manifold::Manifold::Error status = value.Status();
    if (status != manifold::Manifold::Error::NoError) {
        throw std::runtime_error(std::string(label) + " status is " +
                                 manifoldStatusName(status));
    }
    if (value.IsEmpty()) {
        throw std::runtime_error(std::string(label) + " is empty");
    }
}

#endif // LUW_HAS_MANIFOLD

MeshUnionResult unionMeshes(vtkPolyData* mesh,
                            vtkPolyData* base,
                            std::ostream& out) {
    debug(out, "Attempting boolean union of mesh and base block.");
    try {
#if LUW_HAS_MANIFOLD
        const manifold::Manifold building(vtkPolyDataToManifold(mesh));
        requireValidManifold(building, "building mesh");
        const manifold::Manifold baseBlock(vtkPolyDataToManifold(base));
        requireValidManifold(baseBlock, "base block");
        const manifold::Manifold booleanResult =
            building.Boolean(baseBlock, manifold::OpType::Add);
        requireValidManifold(booleanResult, "boolean union result");
        const manifold::MeshGL64 output = booleanResult.GetMeshGL64();
        vtkSmartPointer<vtkPolyData> converted = manifoldToVtkPolyData(output);
        const std::string message = "Boolean union succeeded.";
        debug(out, message);
        return {converted, true, message};
#else
        throw std::runtime_error("manifold boolean backend is unavailable in this build");
#endif
    } catch (const std::exception& exception) {
        const std::string message =
            "Boolean union failed. Falling back to mesh concatenation. Error: " +
            std::string(exception.what());
        debug(out, message);
        return {concatenateMeshes(mesh, base, out), false, message};
    } catch (...) {
        const std::string message =
            "Boolean union failed. Falling back to mesh concatenation. Error: "
            "unknown native exception";
        debug(out, message);
        return {concatenateMeshes(mesh, base, out), false, message};
    }
}

vtkSmartPointer<vtkPolyData> translateToTarget(vtkPolyData* input,
                                              double baseHeight,
                                              std::ostream& out) {
    const Bounds before = boundsOf(input);
    const double tx = -before.xmin;
    const double ty = -before.ymin;
    const double tz = -baseHeight - before.zmin;
    debug(out, "Translating mesh by (m): tx=" + pythonFloat(tx)
               + ", ty=" + pythonFloat(ty) + ", tz=" + pythonFloat(tz));

    vtkSmartPointer<vtkPolyData> result = vtkSmartPointer<vtkPolyData>::New();
    result->DeepCopy(input);
    vtkNew<vtkPoints> points;
    points->SetDataTypeToDouble();
    points->SetNumberOfPoints(input->GetNumberOfPoints());
    for (vtkIdType index = 0; index < input->GetNumberOfPoints(); ++index) {
        double point[3]{};
        input->GetPoint(index, point);
        points->SetPoint(index, point[0] + tx, point[1] + ty, point[2] + tz);
    }
    result->SetPoints(points);

    const Bounds after = boundsOf(result);
    debug(out, "Mesh bounds after translation: min=" + list3(after.xmin, after.ymin, after.zmin)
               + ", max=" + list3(after.xmax, after.ymax, after.zmax));
    debug(out, "Target min should be close to [0, 0, " + pythonFloat(-baseHeight)
               + "]. Actual min is " + list3(after.xmin, after.ymin, after.zmin));
    return result;
}

void ensureDirectory(const fs::path& path, std::ostream& out) {
    if (fs::is_directory(path)) return;
    debug(out, "Creating directory: " + displayPath(path));
    std::error_code error;
    fs::create_directories(path, error);
    if (error) fatal("Failed to create directory: " + displayPath(path) + ". Error: " + error.message());
}

void exportMesh(vtkPolyData* mesh, const fs::path& path, std::ostream& out) {
    debug(out, "Exporting STL to: " + displayPath(path));
    vtkNew<vtkTriangleFilter> triangles;
    triangles->SetInputData(mesh);
    triangles->PassLinesOff();
    triangles->PassVertsOff();

    vtkNew<vtkSTLWriter> writer;
    const std::string filename = displayPath(path);
    writer->SetFileName(filename.c_str());
    writer->SetFileTypeToBinary();
    writer->SetInputConnection(triangles->GetOutputPort());
    if (writer->Write() != 1) {
        fatal("Failed to export STL. Error: VTK STL writer failed");
    }
    debug(out, "Export completed.");
}

void updateDeckRanges(const fs::path& deckPath,
                      const ConfigValues& config,
                      const Bounds& bounds,
                      std::ostream& out) {
    DeckDocument deck = DeckDocument::fromText(config.originalText);
    deck.setPair(QStringLiteral("si_x_cfd"), {bounds.xmin, bounds.xmax}, 6);
    deck.setPair(QStringLiteral("si_y_cfd"), {bounds.ymin, bounds.ymax}, 6);
    deck.setPair(QStringLiteral("si_z_cfd"), {0.0, config.zLimit + config.baseHeight}, 6);

    debug(out, "Updating luwdg si_x_cfd to: si_x_cfd = "
               + fixedRange(bounds.xmin, bounds.xmax));
    debug(out, "Updating luwdg si_y_cfd to: si_y_cfd = "
               + fixedRange(bounds.ymin, bounds.ymax));
    debug(out, "Updating luwdg si_z_cfd to: si_z_cfd = "
               + fixedRange(0.0, config.zLimit + config.baseHeight));

    fs::path backupPath = deckPath;
    backupPath += ".bak";
    try {
        debug(out, "Writing backup luwdg to: " + displayPath(backupPath));
        std::ofstream backup(backupPath, std::ios::binary | std::ios::trunc);
        if (!backup) throw std::runtime_error("cannot open backup file");
        const QByteArray bytes = config.originalText.toUtf8();
        backup.write(bytes.constData(), bytes.size());
        if (!backup) throw std::runtime_error("failed while writing backup file");
    } catch (const std::exception& exception) {
        debug(out, "Backup write failed (continuing). Error: " + std::string(exception.what()));
    }

    debug(out, "Writing updated luwdg to: " + displayPath(deckPath));
    deck.save(qPath(deckPath));
    debug(out, "luwdg update completed.");
}

} // namespace

int run_dgprepare(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    (void)err;
    try {
        const fs::path deckPath = findInputDeck(args, out);
        const fs::path projectDirectory = deckPath.parent_path();
        debug(out, "Resolved luwdg path: " + displayPath(deckPath));
        debug(out, "Project directory ($projectDir): " + displayPath(projectDirectory));

        const std::string extension = lowerAscii(deckPath.extension().string());
        if (extension == ".luwdg") {
            out << "LatticeUrbanWind running in Dataset Generation Mode.\n";
        } else if (extension == ".luwpf") {
            out << "LatticeUrbanWind running in Profile Research Mode.\n";
        } else {
            fatal("Unsupported input file extension: " + displayPath(deckPath));
        }

        const ConfigValues config = parseDeck(deckPath, out);
        const fs::path buildingDirectory = projectDirectory / "building_db";
        if (!fs::is_directory(buildingDirectory)) {
            fatal("Missing directory: " + displayPath(buildingDirectory));
        }

        const fs::path inputStl = pickStl(buildingDirectory, out);
        vtkSmartPointer<vtkPolyData> mesh = loadMesh(inputStl, out);
        vtkSmartPointer<vtkPolyData> base = createBase(boundsOf(mesh), config.baseHeight,
                                                       config.xExpansionRatio,
                                                       config.yExpansionRatio, out);

        const MeshUnionResult unionResult = unionMeshes(mesh, base, out);
        debug(out, std::string("Union method: ") +
                   (unionResult.usedBoolean ? "boolean" : "concatenation"));
        debug(out, "Union message: " + unionResult.message);

        vtkSmartPointer<vtkPolyData> finalMesh =
            translateToTarget(unionResult.mesh, config.baseHeight, out);
        const fs::path outputDirectory = projectDirectory / "proj_temp";
        ensureDirectory(outputDirectory, out);
        const fs::path outputStl = outputDirectory
            / pathFromQString(config.caseName + QStringLiteral("_DG.stl"));
        exportMesh(finalMesh, outputStl, out);

        const Bounds finalBounds = boundsOf(finalMesh);
        debug(out, "Final STL XY bounds (m): x=[" + pythonFloat(finalBounds.xmin)
                   + ", " + pythonFloat(finalBounds.xmax) + "], y=["
                   + pythonFloat(finalBounds.ymin) + ", " + pythonFloat(finalBounds.ymax) + ']');
        updateDeckRanges(deckPath, config, finalBounds, out);
        debug(out, "All done.");
        return 0;
    } catch (const FatalError& exception) {
        out << '[' << timestamp() << "] [FATAL] " << exception.what() << '\n';
        return exception.code();
    } catch (const std::exception& exception) {
        out << '[' << timestamp() << "] [FATAL] " << exception.what() << '\n';
        return 1;
    }
}

} // namespace luw::cli
