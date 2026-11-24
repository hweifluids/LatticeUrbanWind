#include "luw/vtk_post_commands.hpp"

#include <vtkCellData.h>
#include <vtkCellType.h>
#include <vtkDataArray.h>
#include <vtkDataSet.h>
#include <vtkDataSetReader.h>
#include <vtkDataSetWriter.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkIntArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

fs::path uniqueRoot() {
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const fs::path root = fs::path(LUW_TRANSLUW_ARTIFACT_ROOT)
        / ("run-" + std::to_string(stamp));
    require(fs::create_directories(root), "failed to create unique retained fixture root");
    return fs::absolute(root).lexically_normal();
}

void writeText(const fs::path& path, std::string_view text) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "failed to create fixture text file");
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    require(static_cast<bool>(output), "failed to write fixture text file");
}

std::vector<std::uint8_t> decodeHex(std::string_view text) {
    require(text.size() % 2U == 0U, "invalid embedded hex fixture");
    const auto nibble = [](char value) -> unsigned int {
        if (value >= '0' && value <= '9') return static_cast<unsigned int>(value - '0');
        if (value >= 'a' && value <= 'f') return 10U + static_cast<unsigned int>(value - 'a');
        if (value >= 'A' && value <= 'F') return 10U + static_cast<unsigned int>(value - 'A');
        throw std::runtime_error("invalid embedded hex digit");
    };
    std::vector<std::uint8_t> bytes(text.size() / 2U);
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>(
            (nibble(text[index * 2U]) << 4U) | nibble(text[index * 2U + 1U]));
    }
    return bytes;
}

void writeBytes(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "failed to create binary fixture");
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    require(static_cast<bool>(output), "failed to write binary fixture");
}

std::vector<std::uint8_t> validDemPickle() {
    // pickle.dumps(..., protocol=3) for float64 dem_grid[2,3], x_grid[3],
    // y_grid[2], x_min=10.0 and y_min=20.0. This is a retained protocol-3
    // byte fixture, not a runtime Python dependency.
    return decodeHex(
        "80037d710028580800000064656d5f677269647101636e756d70792e5f636f72652e6d756c746961727261790a5f7265636f6e7374727563740a7102636e756d70790a6e6461727261790a71034b008571044301627105877106527107284b014b024b03867108636e756d70790a64747970650a710958020000006638710a898887710b52710c284b0358010000003c710d4e4e4e4affffffff4affffffff4b0074710e62894330000000000000f03f00000000000000400000000000000840000000000000004000000000000008400000000000001040710f747110625806000000785f677269647111680268034b008571126805877113527114284b014b03857115680c8943180000000000000000000000000000144000000000000024407116747117625806000000795f677269647118680268034b00857119680587711a52711b284b014b0285711c680c89431000000000000000000000000000001440711d74711e625805000000785f6d696e711f4740240000000000005805000000795f6d696e7120474034000000000000752e");
}

std::vector<std::uint8_t> symmetricSixteenPointDemPickle() {
    // Protocol-3 NumPy pickle for a 4x4 grid with x=y=[-1,1,3,5] and
    // elevations 1..16. Query (0,0) has four candidates tied at the k=12
    // boundary, exercising cKDTree's boundary-neighbor set semantics.
    return decodeHex(
        "80037d710028580800000064656d5f677269647101636e756d70792e5f636f72652e6d756c746961727261790a5f7265636f6e7374727563740a7102636e756d70790a6e6461727261790a71034b008571044301627105877106527107284b014b044b04867108636e756d70790a64747970650a710958020000006638710a898887710b52710c284b0358010000003c710d4e4e4e4affffffff4affffffff4b0074710e62894380000000000000f03f000000000000004000000000000008400000000000001040000000000000144000000000000018400000000000001c40000000000000204000000000000022400000000000002440000000000000264000000000000028400000000000002a400000000000002c400000000000002e400000000000003040710f747110625806000000785f677269647111680268034b008571126805877113527114284b014b04857115680c894320000000000000f0bf000000000000f03f000000000000084000000000000014407116747117625806000000795f677269647118680268034b00857119680587711a52711b284b014b0485711c680c894320000000000000f0bf000000000000f03f00000000000008400000000000001440711d74711e625805000000785f6d696e711f4740240000000000005805000000795f6d696e7120474034000000000000752e");
}

std::vector<std::uint8_t> maliciousPickle(const std::string& command) {
    require(command.size() <= 0xffffffffULL, "malicious fixture command is too large");
    std::vector<std::uint8_t> bytes{0x80U, 0x03U, 'c', 'o', 's', '\n',
                                    's', 'y', 's', 't', 'e', 'm', '\n', 'X'};
    const std::uint32_t length = static_cast<std::uint32_t>(command.size());
    for (unsigned int shift = 0; shift < 32U; shift += 8U) {
        bytes.push_back(static_cast<std::uint8_t>((length >> shift) & 0xffU));
    }
    bytes.insert(bytes.end(), command.begin(), command.end());
    bytes.push_back(0x85U); // TUPLE1
    bytes.push_back('R');   // REDUCE: must be rejected, never executed
    bytes.push_back('.');
    return bytes;
}

void addPointArrays(vtkImageData* image, bool rho, int generation) {
    const vtkIdType count = image->GetNumberOfPoints();
    if (rho) {
        vtkNew<vtkDoubleArray> density;
        density->SetName("density");
        density->SetNumberOfTuples(count);
        for (vtkIdType index = 0; index < count; ++index) {
            density->SetValue(index, generation * 100.0 + static_cast<double>(index));
        }
        image->GetPointData()->SetScalars(density);
    } else {
        vtkNew<vtkFloatArray> velocity;
        velocity->SetName("velocity");
        velocity->SetNumberOfComponents(3);
        velocity->SetNumberOfTuples(count);
        for (vtkIdType index = 0; index < count; ++index) {
            const float base = static_cast<float>(generation * 100 + index);
            velocity->SetTuple3(index, base, base + 0.25F, base + 0.5F);
        }
        image->GetPointData()->SetVectors(velocity);
    }

    vtkNew<vtkIntArray> collision;
    collision->SetName("collision");
    collision->SetNumberOfTuples(count);
    for (vtkIdType index = 0; index < count; ++index) {
        collision->SetValue(index, (rho ? 2000 : 1000) + generation * 10
                                      + static_cast<int>(index));
    }
    image->GetPointData()->AddArray(collision);
}

void addCellArrays(vtkImageData* image, bool rho, int generation) {
    const vtkIdType count = image->GetNumberOfCells();
    vtkNew<vtkFloatArray> specific;
    specific->SetName(rho ? "rho_cell" : "u_cell");
    specific->SetNumberOfTuples(count);
    for (vtkIdType index = 0; index < count; ++index) {
        specific->SetValue(index, static_cast<float>(generation * 10 + index));
    }
    image->GetCellData()->SetScalars(specific);

    vtkNew<vtkIntArray> shared;
    shared->SetName("shared_cell");
    shared->SetNumberOfTuples(count);
    for (vtkIdType index = 0; index < count; ++index) {
        shared->SetValue(index, (rho ? 4000 : 3000) + generation * 10
                                   + static_cast<int>(index));
    }
    image->GetCellData()->AddArray(shared);
}

void writeImage(const fs::path& path, bool rho, int generation) {
    fs::create_directories(path.parent_path());
    vtkNew<vtkImageData> image;
    image->SetDimensions(3, 2, 2);
    image->SetOrigin(10.0, 20.0, 100.0);
    image->SetSpacing(5.0, 5.0, 70.0);
    addPointArrays(image, rho, generation);
    addCellArrays(image, rho, generation);

    vtkNew<vtkDataSetWriter> writer;
    writer->SetFileName(path.string().c_str());
    writer->SetInputData(image);
    writer->SetFileTypeToBinary();
    require(writer->Write() != 0, "failed to write VTK fixture");
}

void createConfigFixture(const fs::path& directory,
                         const std::vector<std::uint8_t>& pickle) {
    writeText(directory / "conf.luw",
              "# case-insensitive raw-text parsing fixture\n"
              "CASENAME = CompatCase\n"
              "DateTime = 20260719_230000\n");
    const fs::path vtk = directory / "proj_temp" / "vtk";
    writeImage(vtk / "20260719_230000_raw_u-000000001.vtk", false, 1);
    writeImage(vtk / "20260719_230000_raw_u-000000010.vtk", false, 10);
    writeImage(vtk / "20260719_230000_raw_rho-000000002.vtk", true, 2);
    writeImage(vtk / "20260719_230000_raw_rho-000000009.vtk", true, 9);
    writeImage(vtk / "ignore_raw_u-999999999.vtk", false, 99);
    writeBytes(directory / "proj_temp" / "dem_grid.pkl", pickle);
}

vtkSmartPointer<vtkDataSet> readDataSet(const fs::path& path) {
    vtkNew<vtkDataSetReader> reader;
    reader->SetFileName(path.string().c_str());
    reader->ReadAllScalarsOn();
    reader->ReadAllVectorsOn();
    reader->ReadAllFieldsOn();
    reader->Update();
    require(reader->GetOutput() != nullptr, "failed to read generated VTK");
    vtkSmartPointer<vtkDataSet> output =
        vtkSmartPointer<vtkDataSet>::Take(reader->GetOutput()->NewInstance());
    output->DeepCopy(reader->GetOutput());
    return output;
}

void requireNear(double actual, double expected, double tolerance, std::string_view message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(std::string(message) + ": expected "
                                 + std::to_string(expected) + ", got "
                                 + std::to_string(actual));
    }
}

std::vector<std::uint8_t> readBytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "failed to read generated bytes");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void verifyMain(const fs::path& path) {
    const vtkSmartPointer<vtkDataSet> mesh = readDataSet(path);
    require(mesh->GetNumberOfPoints() == 12, "main output point count mismatch");
    require(mesh->GetNumberOfCells() == 2, "main output cell count mismatch");
    double bounds[6]{};
    mesh->GetBounds(bounds);
    const std::array<double, 6> expected{0.0, 10.0, 0.0, 5.0, -50.0, 20.0};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        requireNear(bounds[index], expected[index], 1.0e-12, "main translated bounds mismatch");
    }

    vtkDataArray* velocity = mesh->GetPointData()->GetArray("velocity");
    vtkDataArray* density = mesh->GetPointData()->GetArray("density");
    vtkDataArray* collision = mesh->GetPointData()->GetArray("collision");
    vtkDataArray* collisionRho = mesh->GetPointData()->GetArray("collision_rho");
    require(velocity != nullptr && velocity->GetNumberOfComponents() == 3,
            "latest u velocity array missing");
    require(density != nullptr, "latest rho density array missing");
    require(collision != nullptr && collisionRho != nullptr,
            "point collision suffix merge missing");
    requireNear(velocity->GetComponent(0, 0), 1000.0, 0.0, "latest u step not selected");
    requireNear(density->GetComponent(0, 0), 900.0, 0.0, "latest rho step not selected");
    requireNear(collision->GetComponent(0, 0), 1100.0, 0.0, "u collision changed");
    requireNear(collisionRho->GetComponent(0, 0), 2090.0, 0.0,
                "rho collision suffix value mismatch");

    require(mesh->GetCellData()->GetArray("u_cell") != nullptr,
            "u cell array missing");
    require(mesh->GetCellData()->GetArray("rho_cell") != nullptr,
            "rho cell array missing");
    require(mesh->GetCellData()->GetArray("shared_cell") != nullptr,
            "base shared cell array missing");
    require(mesh->GetCellData()->GetArray("shared_cell_rho") != nullptr,
            "rho shared cell suffix missing");
}

void verifyWoDem(const fs::path& path) {
    const vtkSmartPointer<vtkDataSet> mesh = readDataSet(path);
    auto* cloud = vtkUnstructuredGrid::SafeDownCast(mesh);
    require(cloud != nullptr, "woDEM output is not an UnstructuredGrid");
    require(cloud->GetNumberOfPoints() == 6, "woDEM kept point count mismatch");
    require(cloud->GetNumberOfCells() == 6, "woDEM vertex cell count mismatch");
    for (vtkIdType index = 0; index < cloud->GetNumberOfCells(); ++index) {
        require(cloud->GetCellType(index) == VTK_VERTEX, "woDEM cell is not VTK_VERTEX");
    }
    vtkDataArray* dem = cloud->GetPointData()->GetArray("dem_height");
    vtkDataArray* ids = cloud->GetPointData()->GetArray("vtkOriginalPointIds");
    require(dem != nullptr, "woDEM dem_height missing");
    require(ids != nullptr, "woDEM vtkOriginalPointIds missing");
    require(cloud->GetPointData()->GetArray("u_cell") != nullptr,
            "cell-to-point u_cell missing");
    require(cloud->GetPointData()->GetArray("rho_cell") != nullptr,
            "cell-to-point rho_cell missing");
    const std::array<double, 6> expectedDem{1.0, 2.0, 3.0, 2.0, 3.0, 4.0};
    const std::array<double, 6> expectedZ{19.0, 18.0, 17.0, 18.0, 17.0, 16.0};
    for (vtkIdType index = 0; index < 6; ++index) {
        requireNear(dem->GetComponent(index, 0), expectedDem[static_cast<std::size_t>(index)],
                    1.0e-9, "woDEM IDW value mismatch");
        double point[3]{};
        cloud->GetPoint(index, point);
        requireNear(point[2], expectedZ[static_cast<std::size_t>(index)], 1.0e-9,
                    "woDEM z value mismatch");
        requireNear(ids->GetComponent(index, 0), static_cast<double>(index), 0.0,
                    "woDEM original point id mismatch");
    }
}

void testConfigAndNoOverwrite(const fs::path& root) {
    const fs::path nativeCase = root / "native_case";
    const fs::path oracleCase = root / "oracle_case";
    createConfigFixture(nativeCase, validDemPickle());
    createConfigFixture(oracleCase, validDemPickle());

    std::ostringstream output;
    std::ostringstream error;
    const std::vector<std::string> arguments{
        (nativeCase / "conf.luw").string(), "--tailname", "  compare  "};
    require(luw::cli::run_transluw(arguments, output, error) == 0,
            "config transluw failed");
    require(error.str().empty(), "config transluw wrote an error");
    require(output.str().find("with tailname: 'compare'") != std::string::npos,
            "tailname trimming prompt mismatch");
    require(output.str().find("u=20260719_230000_raw_u-000000010.vtk and "
                              "rho=20260719_230000_raw_rho-000000009.vtk")
                != std::string::npos,
            "latest source detail prompt mismatch");

    const fs::path main = nativeCase / "RESULTS"
        / "uvwrho-CompatCase_20260719_230000_compare.vtk";
    const fs::path wodem = nativeCase / "RESULTS"
        / "uvwrho-CompatCase_woDEM_20260719_230000_compare.vtk";
    require(fs::is_regular_file(main), "main config output missing");
    if (!fs::is_regular_file(wodem)) {
        throw std::runtime_error("woDEM config output missing; console was:\n" + output.str());
    }
    verifyMain(main);
    verifyWoDem(wodem);

    const std::vector<std::uint8_t> before = readBytes(main);
    std::ostringstream secondOutput;
    std::ostringstream secondError;
    require(luw::cli::run_transluw(arguments, secondOutput, secondError) == 1,
            "existing output was not rejected");
    require(secondError.str().find("refusing to overwrite") != std::string::npos,
            "existing output rejection prompt missing");
    require(readBytes(main) == before, "existing main output bytes changed");

    std::cout << "ORACLE_CASE=" << fs::absolute(oracleCase / "conf.luw").string() << '\n';
    std::cout << "NATIVE_MAIN=" << fs::absolute(main).string() << '\n';
    std::cout << "NATIVE_WODEM=" << fs::absolute(wodem).string() << '\n';
}

void testSingleModes(const fs::path& root) {
    const fs::path uSource = root / "single_u" / "stamp_raw_u-000000007.vtk";
    writeImage(uSource, false, 7);
    std::ostringstream output;
    std::ostringstream error;
    require(luw::cli::run_transluw({uSource.string(), "--tailname=tag"}, output, error) == 0,
            "single u mode failed");
    require(fs::is_regular_file(root / "single_u" / "RESULTS"
                                / "uvw-single_u_stamp_tag.vtk"),
            "single u output naming mismatch");

    const fs::path rhoSource = root / "single_rho" / "STAMP_raw_rho-000000008.VTK";
    writeImage(rhoSource, true, 8);
    output.str({});
    output.clear();
    error.str({});
    error.clear();
    require(luw::cli::run_transluw({rhoSource.string()}, output, error) == 0,
            "single rho case-insensitive suffix mode failed");
    require(fs::is_regular_file(root / "single_rho" / "RESULTS"
                                / "rho-single_rho_STAMP.vtk"),
            "single rho output naming mismatch");

    const fs::path fallback = root / "single_fallback" / "odd.mesh.vtk";
    writeImage(fallback, false, 3);
    output.str({});
    output.clear();
    error.str({});
    error.clear();
    require(luw::cli::run_transluw({fallback.string()}, output, error) == 0,
            "single fallback mode failed");
    require(fs::is_regular_file(root / "single_fallback" / "RESULTS"
                                / "mesh-single_fallback_odd.mesh.vtk"),
            "single fallback output naming mismatch");
}

void testAbsoluteOnly(const fs::path& root) {
    const fs::path directory = root / "absolute_only";
    createConfigFixture(directory, validDemPickle());
    std::ostringstream output;
    std::ostringstream error;
    require(luw::cli::run_transluw(
                {(directory / "conf.luw").string(), "--absonly", "--tailname", "abs"},
                output, error) == 0,
            "--absonly config mode failed");
    require(fs::is_regular_file(directory / "RESULTS"
                                / "uvwrho-CompatCase_20260719_230000_abs.vtk"),
            "--absonly main output missing");
    require(!fs::exists(directory / "RESULTS"
                        / "uvwrho-CompatCase_woDEM_20260719_230000_abs.vtk"),
            "--absonly unexpectedly wrote woDEM output");
}

void testTwelveNeighborTieBoundary(const fs::path& root) {
    const fs::path directory = root / "symmetric_k12_native";
    const fs::path oracleDirectory = root / "symmetric_k12_oracle";
    createConfigFixture(directory, symmetricSixteenPointDemPickle());
    createConfigFixture(oracleDirectory, symmetricSixteenPointDemPickle());
    std::ostringstream output;
    std::ostringstream error;
    require(luw::cli::run_transluw(
                {(directory / "conf.luw").string(), "--tailname", "k12"}, output, error) == 0,
            "symmetric k=12 fixture failed");
    const fs::path wodem = directory / "RESULTS"
        / "uvwrho-CompatCase_woDEM_20260719_230000_k12.vtk";
    const vtkSmartPointer<vtkDataSet> cloud = readDataSet(wodem);
    vtkDataArray* dem = cloud->GetPointData()->GetArray("dem_height");
    require(dem != nullptr && dem->GetNumberOfTuples() == 6,
            "symmetric k=12 dem_height shape mismatch");
    // Full SciPy cKDTree(k=12, power=2) oracle values for the six retained
    // translated top-layer points; x=10 lies outside the inclusive DEM box.
    const std::array<double, 6> expected{
        4.462433510638299, 6.404327497854278, 0.0,
        12.405454731902612, 16.0, 0.0};
    for (vtkIdType index = 0; index < static_cast<vtkIdType>(expected.size()); ++index) {
        requireNear(dem->GetComponent(index, 0), expected[static_cast<std::size_t>(index)],
                    1.0e-12, "symmetric k=12 SciPy oracle mismatch");
    }
    std::cout << "ORACLE_K12_CASE="
              << fs::absolute(oracleDirectory / "conf.luw").string() << '\n';
    std::cout << "NATIVE_K12_WODEM=" << fs::absolute(wodem).string() << '\n';
}

void testMaliciousPickle(const fs::path& root) {
    const std::string probeName = "transluw_pickle_probe_"
        + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count())
        + ".txt";
    const fs::path probe = fs::current_path() / probeName;
    require(!fs::exists(probe), "malicious pickle probe unexpectedly pre-exists");
    const fs::path directory = root / "malicious_pickle";
    createConfigFixture(directory, maliciousPickle("echo owned>" + probeName));

    std::ostringstream output;
    std::ostringstream error;
    require(luw::cli::run_transluw(
                {(directory / "conf.luw").string(), "--tailname", "safe"}, output, error) == 0,
            "malicious pickle should warn after writing safe main output");
    require(output.str().find("Disallowed pickle callable: os system") != std::string::npos,
            "malicious GLOBAL/REDUCE rejection warning missing");
    require(!fs::exists(probe), "malicious pickle callable was executed");
    require(fs::is_regular_file(directory / "RESULTS"
                                / "uvwrho-CompatCase_20260719_230000_safe.vtk"),
            "malicious pickle path lost the safe main output");
}

void testExistingWoDemIsPreserved(const fs::path& root) {
    const fs::path directory = root / "existing_wodem";
    createConfigFixture(directory, validDemPickle());
    const fs::path wodem = directory / "RESULTS"
        / "uvwrho-CompatCase_woDEM_20260719_230000_preserve.vtk";
    const std::string sentinel = "pre-existing woDEM sentinel\n";
    writeText(wodem, sentinel);

    std::ostringstream output;
    std::ostringstream error;
    require(luw::cli::run_transluw(
                {(directory / "conf.luw").string(), "--tailname", "preserve"}, output, error) == 0,
            "existing woDEM should be isolated as a warning");
    require(error.str().empty(), "existing woDEM unexpectedly wrote stderr");
    require(output.str().find("[WARN] DEM pkl exists but woDEM generation failed:")
                != std::string::npos
            && output.str().find("refusing to overwrite") != std::string::npos,
            "existing woDEM warning mismatch");
    require(readBytes(wodem) == std::vector<std::uint8_t>(sentinel.begin(), sentinel.end()),
            "existing woDEM bytes changed");
    require(fs::is_regular_file(directory / "RESULTS"
                                / "uvwrho-CompatCase_20260719_230000_preserve.vtk"),
            "safe main output missing when woDEM already exists");
}

void testCliEdges(const fs::path& root) {
    std::ostringstream output;
    std::ostringstream error;
    require(luw::cli::run_transluw({"--help"}, output, error) == 0,
            "--help failed");
    require(output.str().find("--absonly") != std::string::npos,
            "--help text incomplete");
    require(error.str().empty(), "--help wrote an error");

    output.str({});
    output.clear();
    error.str({});
    error.clear();
    require(luw::cli::run_transluw({"--unknown"}, output, error) == 2,
            "unknown option did not return argparse status 2");
    require(error.str().find("transluw.py: error: unrecognized arguments: --unknown")
                != std::string::npos,
            "unknown option error mismatch");

    const fs::path missing = root / "never-created-input.luw";
    output.str({});
    output.clear();
    error.str({});
    error.clear();
    require(luw::cli::run_transluw({missing.string()}, output, error) == 1,
            "missing input did not return status 1");
    require(output.str().find("[INFO] Processing input:") != std::string::npos,
            "missing input processing prompt missing");
    require(error.str().find("[ERROR] Input path does not exist:") != std::string::npos,
            "missing input error prompt missing");
}

} // namespace

int main() {
    try {
        const fs::path root = uniqueRoot();
        testConfigAndNoOverwrite(root);
        testSingleModes(root);
        testAbsoluteOnly(root);
        testTwelveNeighborTieBoundary(root);
        testMaliciousPickle(root);
        testExistingWoDemIsPreserved(root);
        testCliEdges(root);
        std::cout << "TRANSluw retained fixture root: " << root.string() << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "transluw native compatibility test failed: " << exception.what() << '\n';
        return 1;
    }
}
