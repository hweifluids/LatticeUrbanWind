#include "luw/deck_document.hpp"
#include "luw/validate_commands.hpp"

#include <QString>

#include <vtkFeatureEdges.h>
#include <vtkMassProperties.h>
#include <vtkNew.h>
#include <vtkPolyData.h>
#include <vtkSTLReader.h>
#include <vtkTriangleFilter.h>

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

#ifndef LUW_HAS_MANIFOLD
#define LUW_HAS_MANIFOLD 0
#endif

namespace {

namespace fs = std::filesystem;

QString qPath(const fs::path& path) {
#ifdef _WIN32
    return QString::fromStdWString(path.wstring());
#else
    const auto utf8 = path.u8string();
    return QString::fromUtf8(reinterpret_cast<const char*>(utf8.data()),
                             static_cast<qsizetype>(utf8.size()));
#endif
}

void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void expectNear(double actual, double expected, double tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": expected " + std::to_string(expected)
                                 + ", got " + std::to_string(actual));
    }
}

void writeText(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write test fixture: " + path.string());
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output) throw std::runtime_error("failed writing test fixture: " + path.string());
}

std::string readText(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read test output: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string triangleStl() {
    return "solid fixture\n"
           "  facet normal 0 0 1\n"
           "    outer loop\n"
           "      vertex 0 0 0\n"
           "      vertex 10 0 5\n"
           "      vertex 0 20 0\n"
           "    endloop\n"
           "  endfacet\n"
           "endsolid fixture\n";
}

std::string tetrahedronStl() {
    return "solid buildings\n"
           "facet normal 0 0 -1\nouter loop\nvertex 10 30 5\nvertex 20 30 5\nvertex 10 50 5\nendloop\nendfacet\n"
           "facet normal 0 -1 1\nouter loop\nvertex 10 30 5\nvertex 20 30 5\nvertex 10 30 15\nendloop\nendfacet\n"
           "facet normal 1 1 1\nouter loop\nvertex 20 30 5\nvertex 10 50 5\nvertex 10 30 15\nendloop\nendfacet\n"
           "facet normal -1 0 1\nouter loop\nvertex 10 50 5\nvertex 10 30 5\nvertex 10 30 15\nendloop\nendfacet\n"
           "endsolid buildings\n";
}

std::string cubeStl() {
    return R"(solid cube
facet normal 0 0 -1
outer loop
vertex 10 30 5
vertex 20 50 5
vertex 20 30 5
endloop
endfacet
facet normal 0 0 -1
outer loop
vertex 10 30 5
vertex 10 50 5
vertex 20 50 5
endloop
endfacet
facet normal 0 0 1
outer loop
vertex 10 30 15
vertex 20 30 15
vertex 20 50 15
endloop
endfacet
facet normal 0 0 1
outer loop
vertex 10 30 15
vertex 20 50 15
vertex 10 50 15
endloop
endfacet
facet normal 0 -1 0
outer loop
vertex 10 30 5
vertex 20 30 5
vertex 20 30 15
endloop
endfacet
facet normal 0 -1 0
outer loop
vertex 10 30 5
vertex 20 30 15
vertex 10 30 15
endloop
endfacet
facet normal 1 0 0
outer loop
vertex 20 30 5
vertex 20 50 5
vertex 20 50 15
endloop
endfacet
facet normal 1 0 0
outer loop
vertex 20 30 5
vertex 20 50 15
vertex 20 30 15
endloop
endfacet
facet normal 0 1 0
outer loop
vertex 20 50 5
vertex 10 50 5
vertex 10 50 15
endloop
endfacet
facet normal 0 1 0
outer loop
vertex 20 50 5
vertex 10 50 15
vertex 20 50 15
endloop
endfacet
facet normal -1 0 0
outer loop
vertex 10 50 5
vertex 10 30 5
vertex 10 30 15
endloop
endfacet
facet normal -1 0 0
outer loop
vertex 10 50 5
vertex 10 30 15
vertex 10 50 15
endloop
endfacet
endsolid cube
)";
}

struct StlSummary {
    vtkIdType polygonCount = 0;
    vtkIdType boundaryOrNonManifoldEdgeCount = 0;
    std::array<double, 6> bounds{};
    double volume = 0.0;
};

StlSummary inspectStl(const fs::path& path) {
    vtkNew<vtkSTLReader> reader;
    const std::string filename = path.string();
    reader->SetFileName(filename.c_str());
    reader->MergingOn();
    reader->Update();

    vtkNew<vtkTriangleFilter> triangles;
    triangles->SetInputConnection(reader->GetOutputPort());
    triangles->PassLinesOff();
    triangles->PassVertsOff();
    triangles->Update();

    vtkNew<vtkFeatureEdges> openEdges;
    openEdges->SetInputConnection(triangles->GetOutputPort());
    openEdges->BoundaryEdgesOn();
    openEdges->NonManifoldEdgesOn();
    openEdges->FeatureEdgesOff();
    openEdges->ManifoldEdgesOff();
    openEdges->Update();

    vtkNew<vtkMassProperties> massProperties;
    massProperties->SetInputConnection(triangles->GetOutputPort());
    massProperties->Update();

    StlSummary result;
    result.polygonCount = triangles->GetOutput()->GetNumberOfPolys();
    result.boundaryOrNonManifoldEdgeCount =
        openEdges->GetOutput()->GetNumberOfCells();
    triangles->GetOutput()->GetBounds(result.bounds.data());
    result.volume = massProperties->GetVolume();
    return result;
}

void expectOracleBounds(const StlSummary& summary, const std::string& label) {
    expectNear(summary.bounds[0], 0.0, 1.0e-5, label + " x minimum");
    expectNear(summary.bounds[1], 15.0, 1.0e-5, label + " x maximum");
    expectNear(summary.bounds[2], 0.0, 1.0e-5, label + " y minimum");
    expectNear(summary.bounds[3], 40.0, 1.0e-5, label + " y maximum");
    expectNear(summary.bounds[4], -2.0, 1.0e-5, label + " z minimum");
    expectNear(summary.bounds[5], 10.0, 1.0e-5, label + " z maximum");
}

fs::path newScratchRoot() {
    const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const fs::path root = fs::temp_directory_path()
        / ("luw_validate_dg_native_" + std::to_string(nonce));
    fs::create_directories(root);
    return root;
}

void testValidationPassAndMismatch(const fs::path& scratch) {
    const fs::path project = scratch / "validation";
    const fs::path deckPath = project / "conf.luw";
    writeText(deckPath,
              "// validation fixture\n"
              "casename = \"fixture\"\n"
              "datetime = 20260102030405\n"
              "cut_lon_manual = [0, 1]\n"
              "cut_lat_manual = [0, 1]\n"
              "mesh_control = \"cell_size\"\n"
              "cell_size = 1.0\n");
    writeText(project / "proj_temp" / "fixture.stl", triangleStl());
    const fs::path csv = project / "proj_temp" / "SurfData_20260102030405.csv";
    writeText(csv, "X,Y,Z\n0,0,0\n10,20,5\n");

    std::ostringstream output;
    std::ostringstream errors;
    const int passedCode = luw::cli::run_luwval({deckPath.string()}, output, errors);
    expect(passedCode == 0, "luwval pass fixture returned non-zero");
    expect(output.str().find("Validation passed. Maximum XY relative error 0.000000%")
               != std::string::npos,
           "luwval did not report matching ranges");
    luw::DeckDocument deck = luw::DeckDocument::load(qPath(deckPath));
    expect(deck.textOr(QStringLiteral("validation")) == QStringLiteral("pass"),
           "luwval did not write validation=pass");
    expect(deck.boolean(QStringLiteral("high_order")).value_or(false),
           "luwval did not default high_order=true");
    expect(deck.boolean(QStringLiteral("flux_correction")).value_or(false),
           "luwval did not default flux_correction=true");
    expect(deck.list(QStringLiteral("n_gpu")).size() == 3,
           "luwval did not default n_gpu");

    writeText(csv, "X,Y,Z\n0,0,0\n10.02,20,5\n");
    output.str({});
    output.clear();
    const int mismatchCode = luw::cli::run_luwval({deckPath.string()}, output, errors);
    expect(mismatchCode == 0, "luwval mismatch must retain the Python tool's zero exit code");
    expect(output.str().find("WARNING: XY range mismatch exceeds 0.1%!") != std::string::npos,
           "luwval did not report range mismatch");
    deck = luw::DeckDocument::load(qPath(deckPath));
    expect(deck.textOr(QStringLiteral("validation")) == QStringLiteral("error"),
           "luwval did not write validation=error");
}

void testValidationRequiredRanges(const fs::path& scratch) {
    const fs::path project = scratch / "validation_missing_ranges";
    const fs::path deckPath = project / "conf.luw";
    writeText(deckPath,
              "casename = fixture\n"
              "datetime = 20260102030405\n"
              "n_gpu = [1, 1, 1]\n"
              "mesh_control = cell_size\n"
              "cell_size = 1\n");
    std::ostringstream output;
    std::ostringstream errors;
    const int code = luw::cli::run_luwval({deckPath.string()}, output, errors);
    expect(code == 1, "luwval missing manual ranges must return one");
    expect(output.str().find("cut_lon_manual/cut_lat_manual") != std::string::npos,
           "luwval missing ranges message was not emitted");
    const luw::DeckDocument deck = luw::DeckDocument::load(qPath(deckPath));
    expect(deck.textOr(QStringLiteral("validation")) == QStringLiteral("error"),
           "luwval missing ranges did not write validation=error");
}

void testDatasetGeometryFallback(const fs::path& scratch) {
    const fs::path project = scratch / "dataset_generation_fallback";
    const fs::path deckPath = project / "conf.luwdg";
    const std::string originalDeck =
        "// dataset fixture\n"
        "casename = \"sample\"\n"
        "base_height = 2\n"
        "x_exp_rat = 1.5\n"
        "y_exp_rat = 2\n"
        "z_limit = 50\n"
        "si_x_cfd = [-1, -1]\n"
        "si_y_cfd = [-1, -1]\n"
        "si_z_cfd = [-1, -1]\n";
    writeText(deckPath, originalDeck);
    writeText(project / "building_db" / "rawbuildings.stl", tetrahedronStl());

    std::ostringstream output;
    std::ostringstream errors;
    const int code = luw::cli::run_dgprepare({deckPath.string()}, output, errors);
    expect(code == 0, "dgprepare returned non-zero");
    expect(output.str().find("LatticeUrbanWind running in Dataset Generation Mode.")
               != std::string::npos,
           "dgprepare did not report dataset mode");
    expect(output.str().find(
               "Boolean union failed. Falling back to mesh concatenation. Error:")
               != std::string::npos,
           "dgprepare did not report the legacy boolean failure fallback");
    expect(output.str().find("Union method: concatenation") != std::string::npos,
           "dgprepare did not report the concatenation fallback");

    fs::path backup = deckPath;
    backup += ".bak";
    expect(fs::is_regular_file(backup), "dgprepare did not retain .bak");
    expect(readText(backup) == originalDeck, "dgprepare backup differs from original deck");

    const luw::DeckDocument deck = luw::DeckDocument::load(qPath(deckPath));
    const auto x = deck.pair(QStringLiteral("si_x_cfd"));
    const auto y = deck.pair(QStringLiteral("si_y_cfd"));
    const auto z = deck.pair(QStringLiteral("si_z_cfd"));
    expect(x && y && z, "dgprepare did not write CFD ranges");
    expectNear(x->first, 0.0, 1.0e-6, "dgprepare x minimum");
    expectNear(x->second, 15.0, 1.0e-6, "dgprepare x maximum");
    expectNear(y->first, 0.0, 1.0e-6, "dgprepare y minimum");
    expectNear(y->second, 40.0, 1.0e-6, "dgprepare y maximum");
    expectNear(z->first, 0.0, 1.0e-6, "dgprepare z minimum");
    expectNear(z->second, 52.0, 1.0e-6, "dgprepare z maximum");

    const fs::path outputStl = project / "proj_temp" / "sample_DG.stl";
    expect(fs::is_regular_file(outputStl), "dgprepare did not write output STL");
    const StlSummary summary = inspectStl(outputStl);
    expect(summary.polygonCount > 0, "dgprepare fallback output STL is empty");
    expectOracleBounds(summary, "dgprepare fallback output STL");
}

void testDatasetGeometryBoolean(const fs::path& scratch) {
    const fs::path project = scratch / "dataset_generation_boolean";
    const fs::path deckPath = project / "conf.luwdg";
    writeText(deckPath,
              "// closed-volume boolean fixture\n"
              "casename = \"cube\"\n"
              "base_height = 2\n"
              "x_exp_rat = 1.5\n"
              "y_exp_rat = 2\n"
              "z_limit = 50\n"
              "si_x_cfd = [-1, -1]\n"
              "si_y_cfd = [-1, -1]\n"
              "si_z_cfd = [-1, -1]\n");
    writeText(project / "building_db" / "rawbuildings.stl", cubeStl());

    std::ostringstream output;
    std::ostringstream errors;
    const int code = luw::cli::run_dgprepare({deckPath.string()}, output, errors);
    expect(code == 0, "dgprepare closed-volume fixture returned non-zero");
#if LUW_HAS_MANIFOLD
    expect(output.str().find("Boolean union succeeded.") != std::string::npos,
           "dgprepare did not report the successful boolean union");
    expect(output.str().find("Union method: boolean") != std::string::npos,
           "dgprepare did not report the boolean method");
    expect(output.str().find("Union message: Boolean union succeeded.")
               != std::string::npos,
           "dgprepare boolean success message differs from the legacy oracle");
#else
    expect(output.str().find(
               "Boolean union failed. Falling back to mesh concatenation. Error: "
               "manifold boolean backend is unavailable in this build")
               != std::string::npos,
           "dependency-light dgprepare did not report its safe fallback");
    expect(output.str().find("Union method: concatenation") != std::string::npos,
           "dependency-light dgprepare did not use concatenation fallback");
#endif

    const luw::DeckDocument deck = luw::DeckDocument::load(qPath(deckPath));
    const auto x = deck.pair(QStringLiteral("si_x_cfd"));
    const auto y = deck.pair(QStringLiteral("si_y_cfd"));
    const auto z = deck.pair(QStringLiteral("si_z_cfd"));
    expect(x && y && z, "dgprepare boolean fixture did not write CFD ranges");
    expectNear(x->first, 0.0, 1.0e-6, "dgprepare boolean x minimum");
    expectNear(x->second, 15.0, 1.0e-6, "dgprepare boolean x maximum");
    expectNear(y->first, 0.0, 1.0e-6, "dgprepare boolean y minimum");
    expectNear(y->second, 40.0, 1.0e-6, "dgprepare boolean y maximum");
    expectNear(z->first, 0.0, 1.0e-6, "dgprepare boolean z minimum");
    expectNear(z->second, 52.0, 1.0e-6, "dgprepare boolean z maximum");

    const fs::path outputStl = project / "proj_temp" / "cube_DG.stl";
    expect(fs::is_regular_file(outputStl),
           "dgprepare closed-volume fixture did not write output STL");
    const StlSummary summary = inspectStl(outputStl);
    expect(summary.polygonCount > 0,
           "dgprepare closed-volume output STL is empty");
    expectOracleBounds(summary, "dgprepare closed-volume output STL");
    expect(summary.boundaryOrNonManifoldEdgeCount == 0,
           "dgprepare closed-volume output STL is not watertight");
    expectNear(summary.volume, 3200.0, 1.0e-4,
               "dgprepare closed-volume output STL volume");
}

void testDatasetValidationError(const fs::path& scratch) {
    const fs::path project = scratch / "dataset_invalid";
    const fs::path deckPath = project / "conf.luwdg";
    writeText(deckPath,
              "casename = invalid\n"
              "base_height = 0\n"
              "x_exp_rat = 1\n"
              "y_exp_rat = 1\n"
              "z_limit = 20\n");
    std::ostringstream output;
    std::ostringstream errors;
    const int code = luw::cli::run_dgprepare({deckPath.string()}, output, errors);
    expect(code == 1, "dgprepare invalid base height must return one");
    expect(output.str().find("[FATAL] base_height must be positive. Got: 0.0")
               != std::string::npos,
           "dgprepare invalid base height message differs");
    expect(!fs::exists(project / "proj_temp"),
           "dgprepare created outputs after an early validation error");
}

} // namespace

int main() {
    try {
        const fs::path scratch = newScratchRoot();
        testValidationPassAndMismatch(scratch);
        testValidationRequiredRanges(scratch);
        testDatasetGeometryFallback(scratch);
        testDatasetGeometryBoolean(scratch);
        testDatasetValidationError(scratch);
        std::cout << "validate/dgprepare compatibility tests passed; artifacts retained at "
                  << scratch.string() << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "validate/dgprepare compatibility test failure: " << exception.what() << '\n';
        return 1;
    }
}
