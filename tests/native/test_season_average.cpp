#include "luw/legacy_vtk.hpp"
#include "luw/vtk_post_commands.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;
constexpr std::array<int, 3> dimensions{3, 2, 3};
constexpr std::size_t pointCount = 18;

std::string utf8Text(const fs::path& path) {
#if defined(__cpp_lib_char8_t)
    const auto text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
#else
    return path.u8string();
#endif
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void requireNear(double actual, double expected, double tolerance,
                 const std::string& message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::ostringstream detail;
        detail << message << ": expected " << std::setprecision(10) << expected
               << ", actual " << actual;
        throw std::runtime_error(detail.str());
    }
}

std::string readText(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "open text " + utf8Text(path));
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::vector<float> northWind() {
    std::vector<float> values(pointCount * 3U);
    for (std::size_t point = 0; point < pointCount; ++point) {
        values[point * 3U] = 1.0F + static_cast<float>(point);
        values[point * 3U + 1U] = 2.0F + 0.5F * static_cast<float>(point);
        values[point * 3U + 2U] = -1.0F + 0.25F * static_cast<float>(point);
    }
    return values;
}

std::array<std::vector<float>, 3> eastComponents() {
    std::array<std::vector<float>, 3> values;
    for (auto& component : values) component.resize(pointCount);
    for (std::size_t point = 0; point < pointCount; ++point) {
        values[0][point] = 10.0F + static_cast<float>(point);
        values[1][point] = -2.0F + 0.1F * static_cast<float>(point);
        values[2][point] = 3.0F - 0.2F * static_cast<float>(point);
    }
    return values;
}

std::vector<float> northTke() {
    std::vector<float> values(pointCount);
    for (std::size_t point = 0; point < pointCount; ++point) {
        values[point] = 2.0F + static_cast<float>(point);
    }
    return values;
}

std::vector<float> eastTke() {
    std::vector<float> values(pointCount);
    for (std::size_t point = 0; point < pointCount; ++point) {
        values[point] = 5.0F + 2.0F * static_cast<float>(point);
    }
    return values;
}

void writeBinaryNorth(const fs::path& path) {
    const std::vector<float> wind = northWind();
    const std::vector<float> tke = northTke();
    const std::array<luw::post::LegacyVtkFloatFieldView, 2> fields{{
        {"u_avg", 3, luw::post::LegacyVtkFieldKind::Vectors, wind},
        {"tke", 1, luw::post::LegacyVtkFieldKind::Scalars, tke},
    }};
    luw::post::writeLegacyStructuredPointsFloat(
        path, dimensions, {1.5, -2.0, 10.0}, {2.0, 3.0, 4.0}, fields,
        "season average north retained fixture");
}

void writeBinaryEast(const fs::path& path) {
    const auto wind = eastComponents();
    const std::vector<float> tke = eastTke();
    const std::array<luw::post::LegacyVtkFloatFieldView, 4> fields{{
        {"u", 1, luw::post::LegacyVtkFieldKind::Scalars, wind[0]},
        {"v", 1, luw::post::LegacyVtkFieldKind::Scalars, wind[1]},
        {"w", 1, luw::post::LegacyVtkFieldKind::Scalars, wind[2]},
        {"tke", 1, luw::post::LegacyVtkFieldKind::Scalars, tke},
    }};
    luw::post::writeLegacyStructuredPointsFloat(
        path, dimensions, {1.5, -2.0, 10.0}, {2.0, 3.0, 4.0}, fields,
        "season average east retained fixture");
}

void writeAsciiValues(std::ofstream& output, const std::vector<float>& values) {
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t index = 0; index < values.size(); ++index) {
        output << values[index] << ((index + 1U) % 8U == 0U ? '\n' : ' ');
    }
    if (values.size() % 8U != 0U) output << '\n';
}

void writeAsciiEast(const fs::path& path) {
    const auto wind = eastComponents();
    const std::vector<float> tke = eastTke();
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "create ASCII east fixture");
    output << "# vtk DataFile Version 3.0\n"
           << "season average east ASCII retained fixture\n"
           << "ASCII\n"
           << "DATASET STRUCTURED_POINTS\n"
           << "DIMENSIONS 3 2 3\n"
           << "ORIGIN 1.5 -2 10\n"
           << "SPACING 2 3 4\n\n"
           << "POINT_DATA 18\n";
    for (std::size_t component = 0; component < 3; ++component) {
        output << "SCALARS " << static_cast<char>('u' + component)
               << " float\nLOOKUP_TABLE default\n";
        writeAsciiValues(output, wind[component]);
    }
    output << "SCALARS tke float\nLOOKUP_TABLE default\n";
    writeAsciiValues(output, tke);
    output.flush();
    require(static_cast<bool>(output), "finish ASCII east fixture");
}

struct CasePaths final {
    fs::path root;
    fs::path config;
    fs::path vtkDirectory;
    fs::path outputVtk;
    fs::path summary;
    fs::path figures;
};

CasePaths writeCase(const fs::path& root, bool asciiEast) {
    const fs::path windBc = root / "wind_bc";
    const fs::path vtkDirectory = root / "RESULTS" / "custom_vtk";
    fs::create_directories(windBc);
    fs::create_directories(vtkDirectory);
    const fs::path config = root / "conf.luwpf";
    std::ofstream(config)
        << "crop_vis_dpi = 1\n"
        << "z_limit = 100\n"
        << "casename = season_fixture\n";
    std::ofstream(windBc / "profile.dat")
        << "# normalized z and U\n"
        << "0, 2\n"
        << "0.5; 6 // target level\n"
        << "1\t10\n";
    std::ofstream(windBc / "windrose_50m.csv")
        << "retained fixture metadata\n"
        << "Direction,C1_0_4,C2_4_8,C3_8_inf\n"
        << "N,10,20,0\n"
        << "E,0,10,10\n"
        << "ALL,100,100,100\n";
    writeBinaryNorth(vtkDirectory / "ANG_0_avg-10.vtk");
    writeBinaryEast(vtkDirectory / "ANG_90_avg-100.vtk");
    if (asciiEast) writeAsciiEast(vtkDirectory / "ANG_90_avg-1_cropped.vtk");
    else writeBinaryEast(vtkDirectory / "ANG_90_avg-1_cropped.vtk");
    // It is intentionally ignored by angle parsing.
    writeBinaryNorth(vtkDirectory / "unresolved_avg-999.vtk");
    return {root,
            config,
            vtkDirectory,
            root / "RESULTS" / "season_average" / "season_average.vtk",
            root / "RESULTS" / "season_average" / "season_average_summary.txt",
            root / "RESULTS" / "season_average" / "figures"};
}

std::vector<float> allFieldValues(const luw::post::LegacyStructuredPoints& vtk,
                                  std::string_view fieldName) {
    const auto* field = vtk.field(fieldName);
    require(field != nullptr, "output field exists: " + std::string(fieldName));
    luw::post::LegacyVtkPlaneReader reader(vtk);
    std::vector<float> result;
    for (int z = 0; z < vtk.dimensions[2]; ++z) {
        const std::vector<float> plane = reader.readPointPlane(*field, static_cast<std::size_t>(z));
        result.insert(result.end(), plane.begin(), plane.end());
    }
    return result;
}

void verifyNativeValues(const CasePaths& paths) {
    const auto vtk = luw::post::LegacyStructuredPoints::readHeader(paths.outputVtk);
    require(vtk.dimensions == dimensions, "native output dimensions match input");
    require(vtk.origin == std::array<double, 3>{1.5, -2.0, 10.0}, "native output origin");
    require(vtk.spacing == std::array<double, 3>{2.0, 3.0, 4.0}, "native output spacing");
    require(vtk.fields.size() == 6, "native output contains vector and five scalar fields");
    const auto* vector = vtk.field("u_avg");
    require(vector != nullptr && vector->components == 3, "u_avg vector output");
    const std::vector<float> vectorValues = allFieldValues(vtk, "u_avg");
    const std::vector<float> vm = allFieldValues(vtk, "vm");
    const std::vector<float> tke = allFieldValues(vtk, "tke");
    const std::vector<float> north = northWind();
    const auto east = eastComponents();
    const std::vector<float> northK = northTke();
    const std::vector<float> eastK = eastTke();
    const float northVelocityWeight = static_cast<float>(7.0 / 15.0);
    const float eastVelocityWeight = 0.6F;
    const float northTkeWeight = static_cast<float>(19.0 / 45.0);
    const float eastTkeWeight = 1.0F;
    for (std::size_t point = 0; point < pointCount; ++point) {
        for (std::size_t component = 0; component < 3; ++component) {
            const float expected = northVelocityWeight * north[point * 3U + component]
                + eastVelocityWeight * east[component][point];
            requireNear(vectorValues[point * 3U + component], expected, 2.0e-6,
                        "weighted vector point " + std::to_string(point));
        }
        const float northMagnitude = std::sqrt(
            north[point * 3U] * north[point * 3U]
            + north[point * 3U + 1U] * north[point * 3U + 1U]
            + north[point * 3U + 2U] * north[point * 3U + 2U]);
        const float eastMagnitude = std::sqrt(
            east[0][point] * east[0][point]
            + east[1][point] * east[1][point]
            + east[2][point] * east[2][point]);
        requireNear(vm[point], northVelocityWeight * northMagnitude
                                + eastVelocityWeight * eastMagnitude,
                    3.0e-6, "weighted speed magnitude point " + std::to_string(point));
        requireNear(tke[point], northTkeWeight * northK[point]
                                 + eastTkeWeight * eastK[point],
                    3.0e-6, "weighted TKE point " + std::to_string(point));
    }
    const std::string summary = readText(paths.summary);
    require(summary.find("Reference speed from profile: 6 m/s (exact sample)") != std::string::npos,
            "summary profile reference mode");
    require(summary.find("Normalized total probability used: 0.500000000") != std::string::npos,
            "summary scaled probability");
    require(summary.find("C3_8_inf: lower=8 m/s, upper=inf m/s, target=12 m/s") != std::string::npos,
            "summary infinite-bin target");
    require(summary.find("vel_weight=0.466666667") != std::string::npos,
            "summary north velocity weight");
    require(summary.find("tke_weight=1.000000000, vtk=ANG_90_avg-1_cropped.vtk") != std::string::npos,
            "summary preferred cropped east file");
}

void compareNativeOutputs(const fs::path& leftPath, const fs::path& rightPath) {
    const auto left = luw::post::LegacyStructuredPoints::readHeader(leftPath);
    const auto right = luw::post::LegacyStructuredPoints::readHeader(rightPath);
    require(left.dimensions == right.dimensions && left.origin == right.origin
                && left.spacing == right.spacing,
            "binary and ASCII source outputs have identical grids");
    for (const std::string_view field : {"u_avg", "u", "v", "w", "vm", "tke"}) {
        const std::vector<float> leftValues = allFieldValues(left, field);
        const std::vector<float> rightValues = allFieldValues(right, field);
        require(leftValues == rightValues,
                "binary and ASCII source outputs match exactly for " + std::string(field));
    }
}

std::pair<std::uint32_t, std::uint32_t> pngDimensions(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "open PNG " + utf8Text(path));
    std::array<unsigned char, 24> header{};
    input.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
    require(input.gcount() == static_cast<std::streamsize>(header.size()), "read PNG header");
    require(header[0] == 0x89U && header[1] == 'P' && header[2] == 'N' && header[3] == 'G',
            "PNG signature");
    auto number = [&](std::size_t offset) {
        return (static_cast<std::uint32_t>(header[offset]) << 24U)
            | (static_cast<std::uint32_t>(header[offset + 1U]) << 16U)
            | (static_cast<std::uint32_t>(header[offset + 2U]) << 8U)
            | static_cast<std::uint32_t>(header[offset + 3U]);
    };
    return {number(16), number(20)};
}

void testCommand(const fs::path& retainedRoot) {
    const CasePaths binary = writeCase(retainedRoot / "binary_case", false);
    std::ostringstream output;
    std::ostringstream error;
    int status = luw::cli::run_season_average(
        {utf8Text(binary.config), "--vtk-dir", "RESULTS/custom_vtk", "--skip-figures"},
        output, error);
    require(status == 0, "binary season_average succeeds: " + error.str() + output.str());
    require(output.str().find("Windrose CSV encoding used: utf-8-sig") != std::string::npos,
            "CSV encoding prompt");
    require(output.str().find("ignored 1 avg-like VTK file") != std::string::npos,
            "unresolved VTK prompt");
    require(output.str().find("Figure generation: skipped by --skip-figures") != std::string::npos,
            "skip figures prompt");
    verifyNativeValues(binary);

    const CasePaths ascii = writeCase(retainedRoot / "ascii_case", true);
    output.str({}); output.clear(); error.str({}); error.clear();
    status = luw::cli::run_season_average(
        {utf8Text(ascii.config), "--vtk-dir", "RESULTS/custom_vtk", "--skip-figures"},
        output, error);
    require(status == 0, "ASCII season_average succeeds: " + error.str() + output.str());
    compareNativeOutputs(binary.outputVtk, ascii.outputVtk);

    const CasePaths resampled = writeCase(retainedRoot / "resampled_case", false);
    output.str({}); output.clear(); error.str({}); error.clear();
    status = luw::cli::run_season_average(
        {utf8Text(resampled.config), "--vtk-dir", "RESULTS/custom_vtk",
         "--output-spacing", "2", "--skip-figures"}, output, error);
    require(status == 0, "one-value resampling succeeds: " + error.str() + output.str());
    const auto resampledVtk = luw::post::LegacyStructuredPoints::readHeader(resampled.outputVtk);
    require(resampledVtk.dimensions == std::array<int, 3>{3, 3, 5},
            "one-value output spacing dimensions");
    require(resampledVtk.spacing == std::array<double, 3>{2.0, 1.5, 2.0},
            "one-value output spacing preserves physical extent");

    const CasePaths defaults = writeCase(retainedRoot / "default_directory_case", false);
    const fs::path preferred = defaults.root / "RESULTS" / "crop" / "cropped_vtk";
    const fs::path fallback = defaults.root / "RESULTS" / "vtk";
    fs::create_directories(preferred);
    fs::create_directories(fallback);
    writeBinaryNorth(preferred / "ANG_0_avg-10_cropped.vtk");
    writeBinaryEast(preferred / "ANG_90_avg-10_cropped.vtk");
    writeBinaryNorth(fallback / "ANG_0_avg-999.vtk");
    writeBinaryEast(fallback / "ANG_90_avg-999.vtk");
    output.str({}); output.clear(); error.str({}); error.clear();
    status = luw::cli::run_season_average(
        {utf8Text(defaults.config), "--skip-figures"}, output, error);
    require(status == 0, "default VTK directory selection succeeds: "
                            + error.str() + output.str());
    require(output.str().find("Selected source directory: " + utf8Text(fs::weakly_canonical(preferred))
                              + " (preferred cropped_vtk)") != std::string::npos,
            "preferred cropped_vtk directory wins over fallback RESULTS/vtk");

    const CasePaths figures = writeCase(retainedRoot / "figure_case", false);
    output.str({}); output.clear(); error.str({}); error.clear();
    status = luw::cli::run_season_average(
        {utf8Text(figures.config), "--vtk-dir", "RESULTS/custom_vtk",
         "--output-spacing", "2", "3", "4"}, output, error);
    require(status == 0, "three-value spacing and figure generation succeeds: "
                            + error.str() + output.str());
    const auto figureVtk = luw::post::LegacyStructuredPoints::readHeader(figures.outputVtk);
    require(figureVtk.dimensions == dimensions
                && figureVtk.spacing == std::array<double, 3>{2.0, 3.0, 4.0},
            "three-value output spacing is applied independently per axis");
    for (const std::string_view name : {
             "season_average_u_9layers.png", "season_average_v_9layers.png",
             "season_average_w_9layers.png", "season_average_vm_9layers.png",
             "season_average_tke_9layers.png"}) {
        const auto [width, height] = pngDimensions(figures.figures / name);
        require(width == 1440U && height == 1280U,
                "native figure retains 18x16 inch times 80 DPI dimensions");
    }

    output.str({}); output.clear(); error.str({}); error.clear();
    require(luw::cli::run_season_average({"--help"}, output, error) == 0,
            "season_average help succeeds");
    require(output.str().find("--output-spacing") != std::string::npos,
            "season_average help documents output spacing");
    output.str({}); output.clear(); error.str({}); error.clear();
    require(luw::cli::run_season_average(
                {utf8Text(binary.config), "--output-spacing", "1", "2"}, output, error) == 2,
            "two-value output spacing is rejected");
    require(error.str().find("expects either one value or three values") != std::string::npos,
            "two-value output spacing error text");
}

} // namespace

int main() {
    try {
        const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const fs::path root = fs::path(LUW_SEASON_AVERAGE_ARTIFACT_ROOT)
            / ("run_" + std::to_string(nonce));
        fs::create_directories(root);
        testCommand(root);
        std::cout << "Native season_average retained compatibility tests passed.\n";
        std::cout << "Generated fixtures retained at: " << utf8Text(root) << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Native season_average compatibility test failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
