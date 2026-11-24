#include "luw/legacy_vtk.hpp"
#include "luw/vtk_post_commands.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

[[nodiscard]] std::string utf8(const fs::path& path) {
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

void requireContains(const std::string& text, std::string_view expected, const std::string& message) {
    require(text.find(expected) != std::string::npos,
            message + ": missing '" + std::string(expected) + "' in:\n" + text);
}

[[nodiscard]] std::uint32_t bigEndianU32(const std::array<unsigned char, 4>& value) {
    return (static_cast<std::uint32_t>(value[0]) << 24U)
        | (static_cast<std::uint32_t>(value[1]) << 16U)
        | (static_cast<std::uint32_t>(value[2]) << 8U)
        | static_cast<std::uint32_t>(value[3]);
}

[[nodiscard]] std::pair<std::uint32_t, std::uint32_t> pngDimensions(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "open PNG " + utf8(path));
    std::array<unsigned char, 24> header{};
    stream.read(reinterpret_cast<char*>(header.data()), header.size());
    require(stream.gcount() == static_cast<std::streamsize>(header.size()), "read PNG header");
    constexpr std::array<unsigned char, 8> signature{137,80,78,71,13,10,26,10};
    require(std::equal(signature.begin(), signature.end(), header.begin()), "PNG signature");
    require(header[12] == 'I' && header[13] == 'H' && header[14] == 'D' && header[15] == 'R',
            "PNG IHDR chunk");
    return {bigEndianU32({header[16], header[17], header[18], header[19]}),
            bigEndianU32({header[20], header[21], header[22], header[23]})};
}

void writeFixture(const fs::path& path) {
    constexpr std::array<int, 3> dimensions{7, 6, 91};
    constexpr std::size_t points = static_cast<std::size_t>(dimensions[0])
        * static_cast<std::size_t>(dimensions[1]) * static_cast<std::size_t>(dimensions[2]);
    std::vector<float> wind(points * 3U);
    std::vector<float> tke(points);
    for (int z = 0; z < dimensions[2]; ++z) for (int y = 0; y < dimensions[1]; ++y) {
        for (int x = 0; x < dimensions[0]; ++x) {
            const std::size_t point = (static_cast<std::size_t>(z) * dimensions[1]
                + static_cast<std::size_t>(y)) * dimensions[0] + static_cast<std::size_t>(x);
            wind[point * 3U] = 1.0F + 0.1F * x + 0.01F * z;
            wind[point * 3U + 1U] = 2.0F + 0.2F * y + 0.02F * z;
            wind[point * 3U + 2U] = 0.25F + 0.005F * z;
            tke[point] = 0.5F + 0.05F * x + 0.08F * y + 0.01F * z;
        }
    }
    const std::array<luw::post::LegacyVtkFloatFieldView, 2> fields{{
        {"u_avg", 3, luw::post::LegacyVtkFieldKind::Vectors, wind},
        {"tke", 1, luw::post::LegacyVtkFieldKind::Scalars, tke},
    }};
    luw::post::writeLegacyStructuredPointsFloat(
        path, dimensions, {0.0, 0.0, -50.0}, {10.0, 10.0, 10.0}, fields,
        "tke_viz retained generated fixture");
}

void verifyLocalCommand(const fs::path& root, const fs::path& vtkPath) {
    {
        std::ostringstream helpOutput;
        std::ostringstream helpError;
        require(luw::cli::run_cut_vis({"--help"}, helpOutput, helpError) == 0,
                "cut_vis alias help succeeds");
        requireContains(helpOutput.str(), "usage: cut_vis", "cut_vis keeps its invoked command name");
        require(helpError.str().empty(), "cut_vis help has no stderr");
    }
    const fs::path outputDirectory = root / "native_local";
    std::ostringstream output;
    std::ostringstream error;
    const int status = luw::cli::run_tke_viz({
        utf8(vtkPath), "--skip-config", "--local-xy",
        "--output-dir", utf8(outputDirectory), "--cell-size", "10",
        "--min-x", "10", "--max-x", "40", "--min-y", "10", "--max-y", "30",
        "--quiver-step", "2", "--dpi", "80", "--export-cropped-vtk"}, output, error);
    require(status == 0, "native tke_viz succeeds: " + error.str() + output.str());
    require(error.str().empty(), "native tke_viz has no stderr");
    requireContains(output.str(), "Config source: CLI only (--skip-config)", "config source prompt");
    requireContains(output.str(), "X_idx=[1, 5), Y_idx=[1, 4), size=4x3", "local crop prompt");
    requireContains(output.str(), "800m -> z= 85 (OK)", "fixed height mapping prompt");

    const std::string stem = utf8(vtkPath.stem());
    const fs::path range = outputDirectory / (stem + "_range_diagnostic.png");
    const fs::path wind = outputDirectory / (stem + "_wind_arrows_9layers.png");
    const fs::path tkeImage = outputDirectory / (stem + "_tke_9layers.png");
    const fs::path cropped = outputDirectory / (stem + "_cropped.vtk");
    require(pngDimensions(range) == std::pair<std::uint32_t,std::uint32_t>{720,640},
            "range PNG matches Matplotlib 9x8 inch DPI dimensions");
    require(pngDimensions(wind) == std::pair<std::uint32_t,std::uint32_t>{1440,1280},
            "wind PNG matches Matplotlib 18x16 inch DPI dimensions");
    require(pngDimensions(tkeImage) == std::pair<std::uint32_t,std::uint32_t>{1440,1280},
            "TKE PNG matches Matplotlib 18x16 inch DPI dimensions");

    const auto croppedVtk = luw::post::LegacyStructuredPoints::readHeader(cropped);
    require(croppedVtk.dimensions == std::array<int,3>{4,3,91}, "cropped VTK dimensions");
    require(croppedVtk.origin == std::array<double,3>{10.0,10.0,-50.0}, "cropped VTK origin");
    const auto* windField = croppedVtk.field("u_avg");
    const auto* tkeField = croppedVtk.field("tke");
    require(windField != nullptr && windField->components == 3, "cropped wind field metadata");
    require(tkeField != nullptr && tkeField->components == 1, "cropped TKE field metadata");
    const auto firstWindPlane = croppedVtk.readPointPlane(*windField, 0);
    const auto firstTkePlane = croppedVtk.readPointPlane(*tkeField, 0);
    require(std::abs(firstWindPlane[0] - 1.1F) < 1.0e-6F, "cropped first u value");
    require(std::abs(firstWindPlane[1] - 2.2F) < 1.0e-6F, "cropped first v value");
    require(std::abs(firstTkePlane[0] - 0.63F) < 1.0e-6F, "cropped first TKE value");
}

void verifyBatchCommand(const fs::path& root, const fs::path& vtkPath) {
    const fs::path caseDirectory = root / "batch_case";
    const fs::path inputDirectory = caseDirectory / "RESULTS" / "vtk";
    fs::create_directories(inputDirectory);
    const fs::path batchInput = inputDirectory / "tiny_avg-20.vtk";
    fs::copy_file(vtkPath, batchInput, fs::copy_options::overwrite_existing);
    const fs::path configPath = caseDirectory / "conf.luwpf";
    std::ofstream(configPath)
        << "target_crs = EPSG:32631\n"
        << "crop_min_lon = 3.000000\n"
        << "crop_max_lon = 3.000539\n"
        << "crop_min_lat = 0.000000\n"
        << "crop_max_lat = 0.000452\n"
        << "cut_lon_manual = [3.000000, 3.000539]\n"
        << "cut_lat_manual = [0.000000, 0.000452]\n"
        << "utm_crs = EPSG:32631\n"
        << "rotate_deg = 0\n"
        << "crop_cell_size = 10\n"
        << "crop_vis_dpi = 80\n"
        << "crop_z_max = 50\n"
        << "crop_debug_file_glob = \"*_avg-*.vtk\"\n";
    std::ostringstream output;
    std::ostringstream error;
    const int status = luw::cli::run_batch_tke_geo_viz({utf8(configPath)}, output, error);
    require(status == 0, "native batch_tke_geo_viz succeeds: " + error.str() + output.str());
    require(error.str().empty(), "batch command has no stderr");
    requireContains(output.str(), "Config type: luwpf", "LUW-family config type prompt");
    requireContains(output.str(), "Input directory source: auto:RESULTS/vtk", "input discovery prompt");
    requireContains(output.str(), "Wind field selected for visualization: u_avg", "wind selection prompt");
    requireContains(output.str(), "TKE field selected for visualization: tke", "TKE selection prompt");
    requireContains(output.str(), "Processed files: 1", "completion prompt");
    const fs::path cropRoot = caseDirectory / "RESULTS" / "crop";
    const fs::path cropped = cropRoot / "cropped_vtk" / "tiny_avg-20_cropped.vtk";
    const fs::path figures = cropRoot / "figures";
    const auto outputVtk = luw::post::LegacyStructuredPoints::readHeader(cropped);
    require(outputVtk.dimensions[0] >= 5 && outputVtk.dimensions[0] <= 7,
            "geographic crop x extent is mapped to the local VTK domain");
    require(outputVtk.dimensions[1] >= 4 && outputVtk.dimensions[1] <= 6,
            "geographic crop y extent is mapped to the local VTK domain");
    require(outputVtk.dimensions[2] == 11, "batch z crop retains z <= 50 m");
    require(pngDimensions(figures / "tiny_avg-20_range_diagnostic.png")
                == std::pair<std::uint32_t,std::uint32_t>{720,640}, "batch range image dimensions");
    require(pngDimensions(figures / "tiny_avg-20_wind_arrows_9layers.png")
                == std::pair<std::uint32_t,std::uint32_t>{1440,1280}, "batch wind image dimensions");
    require(pngDimensions(figures / "tiny_avg-20_tke_9layers.png")
                == std::pair<std::uint32_t,std::uint32_t>{1440,1280}, "batch TKE image dimensions");
}

} // namespace

int main() {
    try {
        const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const fs::path root = fs::path(LUW_TKE_VIZ_ARTIFACT_ROOT) / ("run_" + std::to_string(nonce));
        fs::create_directories(root);
        const fs::path vtkPath = root / "tiny_avg-10.vtk";
        writeFixture(vtkPath);
        verifyLocalCommand(root, vtkPath);
        verifyBatchCommand(root, vtkPath);
        std::cout << "Native TKE/cut/batch visualization retained tests passed.\n"
                  << "Generated fixtures retained at: " << utf8(root) << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Native TKE/cut/batch visualization retained test failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
