#include "luw/vtk_post_commands.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void writeBigEndianFloat(std::ostream& stream, float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::array<unsigned char, 4> bytes{
        static_cast<unsigned char>((bits >> 24U) & 0xffU),
        static_cast<unsigned char>((bits >> 16U) & 0xffU),
        static_cast<unsigned char>((bits >> 8U) & 0xffU),
        static_cast<unsigned char>(bits & 0xffU)};
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
}

void writeFixture(const fs::path& path) {
    fs::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(stream), "cannot create LES VTK fixture");
    stream << "# vtk DataFile Version 3.0\n"
           << "native les spectra fixture\n"
           << "BINARY\n"
           << "DATASET STRUCTURED_POINTS\n"
           << "DIMENSIONS 8 6 13\n"
           << "ORIGIN 0 0 0\n"
           << "SPACING 10 10 20\n"
           << "POINT_DATA 624\n"
           << "SCALARS uvwrho float 4\n"
           << "LOOKUP_TABLE default\n";
    for (int z = 0; z < 13; ++z) {
        for (int y = 0; y < 6; ++y) {
            for (int x = 0; x < 8; ++x) {
                if (z == 0 && y == 0 && x == 0) {
                    for (int component = 0; component < 4; ++component) writeBigEndianFloat(stream, 0.0F);
                    continue;
                }
                writeBigEndianFloat(stream, static_cast<float>(z + 1) + 0.25F * x + 0.1F * y);
                writeBigEndianFloat(stream, static_cast<float>((x % 3) - 1) + 0.2F * z + 0.05F * y);
                writeBigEndianFloat(stream, 0.5F * y - 0.3F * z + 0.02F * x);
                writeBigEndianFloat(stream, 1.0F + 0.01F * x);
            }
        }
    }
    require(static_cast<bool>(stream), "cannot finalize LES VTK fixture");
}

[[nodiscard]] std::string readText(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "cannot read expected output: " + path.string());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

[[nodiscard]] std::pair<std::uint32_t, std::uint32_t> pngDimensions(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "cannot read PNG: " + path.string());
    std::array<unsigned char, 24> header{};
    stream.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
    require(stream.gcount() == static_cast<std::streamsize>(header.size()), "short PNG header");
    static constexpr std::array<unsigned char, 8> signature{137,80,78,71,13,10,26,10};
    require(std::equal(signature.begin(), signature.end(), header.begin()), "invalid PNG signature");
    const auto u32 = [&](std::size_t offset) {
        return (static_cast<std::uint32_t>(header[offset]) << 24U)
             | (static_cast<std::uint32_t>(header[offset + 1]) << 16U)
             | (static_cast<std::uint32_t>(header[offset + 2]) << 8U)
             | static_cast<std::uint32_t>(header[offset + 3]);
    };
    return {u32(16), u32(20)};
}

void requirePng(const fs::path& path, std::uint32_t width, std::uint32_t height) {
    require(fs::is_regular_file(path), "missing PNG: " + path.string());
    require(pngDimensions(path) == std::pair(width, height),
            "unexpected PNG dimensions: " + path.string());
}

void testQuickRun(const fs::path& root) {
    const fs::path vtk = root / "les_fixture.vtk";
    const fs::path outputDirectory = root / "native-output";
    writeFixture(vtk);
    std::ostringstream output;
    std::ostringstream error;
    const int code = luw::cli::run_les_spectra({
        vtk.string(), "--output-dir", outputDirectory.string(), "--quick-test",
        "--test-height-count", "3", "--test-3d-layers", "4", "--workers", "1"},
        output, error);
    require(code == 0, "les_spectra failed: " + error.str());
    require(error.str().empty(), "les_spectra emitted stderr: " + error.str());
    const std::string messages = output.str();
    require(messages.find("Parsed grid: nx=8, ny=6, nz=13, dx=10.0, dy=10.0, dz=20.0")
                != std::string::npos,
            "parsed-grid message differs");
    require(messages.find("Selected 3 target height layers") != std::string::npos,
            "selected-layer message missing");
    require(messages.find("Computing horizontal spectrum at z-index 2 (40.00 m)")
                != std::string::npos,
            "ties-to-even target mapping was not preserved");
    require(messages.find("3D spectrum subvolume: z-index 1:5 (20.00 to 80.00 m)")
                != std::string::npos,
            "quick 3D subvolume message differs");
    require(messages.find("3D FFT component 3/3") != std::string::npos,
            "3D component progress message missing");
    require(messages.find("Outputs written to:") != std::string::npos,
            "completion message missing");

    requirePng(outputDirectory / "layers_kxky" / "kxky_z0040m.png", 1364, 1188);
    requirePng(outputDirectory / "layers_kxky" / "kxky_z0100m.png", 1364, 1188);
    requirePng(outputDirectory / "layers_kxky" / "kxky_z0160m.png", 1364, 1188);
    requirePng(outputDirectory / "kxky_overview.png", 4136, 880);
    requirePng(outputDirectory / "kxky_overview_100_200_300_400_500_600_largefont.png",
               4290, 2640);
    requirePng(outputDirectory / "isotropic_spectrum.png", 1892, 1276);
    requirePng(outputDirectory / "isotropic_spectrum_plain.png", 1760, 1188);

    const std::string layerCsv = readText(outputDirectory / "kxky_layer_metadata.csv");
    require(layerCsv.starts_with(
        "target_height_m,actual_height_m,z_index,valid_fraction,valid_points,output_png\r\n"),
        "layer CSV header/newline differs");
    require(layerCsv.find("50.0,40.0,2,1.0,48,") != std::string::npos,
            "layer CSV target/actual mapping differs");
    require(layerCsv.find("100.0,100.0,5,1.0,48,") != std::string::npos,
            "layer CSV valid count differs");

    const std::string isotropicCsv = readText(outputDirectory / "isotropic_spectrum.csv");
    require(isotropicCsv.starts_with(
        "k_rad_per_m,E_k,k_5_3_E_k,shell_energy,shell_mode_weight\r\n"),
        "isotropic CSV header/newline differs");
    require(isotropicCsv.find(",0.0\r\n") == std::string::npos,
            "isotropic shell weights unexpectedly vanish");

    const std::string summary = readText(outputDirectory / "run_summary.json");
    require(summary.find("\"quick_test\": true") != std::string::npos,
            "summary quick-test flag differs");
    require(summary.find("\"dimensions_xyz\": [8, 6, 13]") != std::string::npos,
            "summary dimensions differ");
    require(summary.find("\"z_start_index\": 1") != std::string::npos,
            "summary 3D start differs");
    require(summary.find("\"z_stop_index\": 5") != std::string::npos,
            "summary 3D stop differs");
    require(summary.find("\"selected_overview_png\": null") == std::string::npos,
            "selected overview unexpectedly null");
}

void testArgumentBehavior() {
    {
        std::ostringstream output;
        std::ostringstream error;
        const int code = luw::cli::run_les_spectra({"--help"}, output, error);
        require(code == 0 && error.str().empty(), "--help did not exit cleanly");
        require(output.str().find("Compute horizontal kx-ky spectra every 50 m") != std::string::npos,
                "help description missing");
    }
    {
        std::ostringstream output;
        std::ostringstream error;
        const int code = luw::cli::run_les_spectra({"--not-an-option"}, output, error);
        require(code == 2, "unknown option did not use argparse exit status 2");
        require(error.str().find("les_spectra.py: error: unrecognized arguments: --not-an-option")
                    != std::string::npos,
                "unknown-option diagnostic differs");
    }
}

} // namespace

int main() {
    try {
        const fs::path root = fs::path(LUW_LES_SPECTRA_ARTIFACT_ROOT);
        fs::create_directories(root);
        testQuickRun(root);
        testArgumentBehavior();
        std::cout << "Retained LES spectra fixtures: " << root.string() << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
}
