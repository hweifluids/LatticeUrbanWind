#include "test_support.hpp"

#include "luw/deck_document.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include <cmath>
#include <stdexcept>

namespace {

void boolTokensAreFuzzyAndQuoted(TestContext& context) {
    const luw::DeckDocument deck = luw::DeckDocument::fromText(R"(
            // Physics
            buoyancy = "yes"
            coriolis_term = t
            ibm_enabler = n
            enable_top_sponge = 0
            enable_buffer_nudging = 2
        )");
    context.expect(deck.boolean("buoyancy").value_or(false), "quoted yes is true");
    context.expect(deck.boolean("coriolis_term").value_or(false), "t is true");
    context.expect(!deck.boolean("ibm_enabler").value_or(true), "n is false");
    context.expect(!deck.boolean("enable_top_sponge").value_or(true), "0 is false");
    context.expect(deck.boolean("enable_buffer_nudging").value_or(false), "non-zero number is true");
}

void aliasesNormalize(TestContext& context) {
    const luw::DeckDocument deck = luw::DeckDocument::fromText(R"(
            // Turbulence inflow
            vk-inlet-enable = "y"
            vk inlet anisotropy scale = [1.0, 2.0, 3.0]
        )");
    context.expect(deck.has("turb_inflow_enable"), "boolean alias canonicalized");
    context.expect(deck.boolean("turb_inflow_enable").value_or(false), "aliased bool parsed");
    const QVector<double> values = deck.floatList("vk_inlet_anisotropy");
    context.expect(values == QVector<double>({1.0, 2.0, 3.0}), "list alias canonicalized");
}

void renderRepairsOrderAndPreservesUnknowns(TestContext& context) {
    luw::DeckDocument deck = luw::DeckDocument::fromText(R"(
            custom_note = alpha
            probes =
            // CFD control
            gpu_memory = 24000
            vk_inlet_enable = yes
            mystery-key = 42
            // Domain
            cut_lon_manual = [121.7, 121.3]
            cut_lat_manual = [31.4, 31.1]
        )");
    deck.setBoolean("flux_correction", true);
    const QString rendered = deck.render();
    context.expect(rendered.contains("// Domain"), "domain section rendered");
    context.expect(rendered.contains("// CFD Controls"), "CFD section rendered");
    context.expect(rendered.contains("// Output & Probes"), "output section rendered");
    context.expect(rendered.contains("probes ="), "empty known value preserved");
    context.expect(rendered.contains("turb_inflow_enable = true"), "known alias rendered canonically");
    context.expect(rendered.contains("mystery_key = 42"), "unknown key preserved");
    context.expect(rendered.indexOf("// Domain") < rendered.indexOf("// CFD Controls"), "section order domain/CFD");
    context.expect(rendered.indexOf("// CFD Controls") < rendered.indexOf("// Output & Probes"), "section order CFD/output");
}

void terrainKeysRoundTrip(TestContext& context) {
    const luw::DeckDocument deck = luw::DeckDocument::fromText(R"(
            // Domain
            terr_voxel_height_field = HEIGHT_M
            terr_voxel_ignore_under = 3.500000
            terr_voxel_approach = kriging
            terr_voxel_grid_resolution = 25.000000
            terr_voxel_idw_sigma = 0.500000
            terr_voxel_idw_power = 1.500000
            terr_voxel_idw_neighbors = 8
            // Preprocessing
            preprocess_use_gpu = true
            preprocess_gpu_device = 1
        )");
    context.equal(deck.textOr("terr_voxel_height_field"), QStringLiteral("HEIGHT_M"), "height field");
    context.expect(deck.floating("terr_voxel_ignore_under") == 3.5, "ignore threshold");
    context.equal(deck.textOr("terr_voxel_approach"), QStringLiteral("kriging"), "approach");
    context.expect(deck.floating("terr_voxel_grid_resolution") == 25.0, "grid resolution");
    context.expect(deck.floating("terr_voxel_idw_sigma") == 0.5, "sigma");
    context.expect(deck.floating("terr_voxel_idw_power") == 1.5, "power");
    context.expect(deck.integer("terr_voxel_idw_neighbors") == 8, "neighbors");
    context.expect(deck.boolean("preprocess_use_gpu") == true, "preprocess GPU switch");
    context.expect(deck.integer("preprocess_gpu_device") == 1, "preprocess GPU device");
    const QString rendered = deck.render();
    context.expect(rendered.contains("terr_voxel_approach = kriging"), "approach round-trip");
    context.expect(rendered.contains("terr_voxel_idw_neighbors = 8"), "neighbors round-trip");
    context.expect(rendered.contains("preprocess_use_gpu = true"), "GPU switch round-trip");
}

void surfaceIbmKeysAreKnownAndRoundTrip(TestContext& context) {
    const luw::DeckDocument deck = luw::DeckDocument::fromText(R"(
            // Physics
            ibm_enabler = true
            culbm_surface_ibm_delta_kernel = roma3_point
            culbm_surface_ibm_geometry_policy = exterior_union
            culbm_surface_ibm_marker_spacing_lu = 0.75
            culbm_surface_ibm_forcing_iterations = 3
            culbm_surface_ibm_relaxation = 1.0
            culbm_surface_ibm_residual_tolerance = 0.001
            culbm_surface_ibm_retraction_lu = 0.0
            culbm_surface_ibm_support_thickness_lu = 0.75
            culbm_surface_ibm_max_markers = 50000000
            culbm_surface_ibm_strict_geometry = false
            culbm_surface_ibm_diagnostics_interval = 0
            // CFD Controls
            cfd_solver = culbm
        )");
    context.expect(deck.boolean("ibm_enabler").value_or(false), "Surface IBM enable switch");
    context.equal(deck.textOr("culbm_surface_ibm_delta_kernel"),
                  QStringLiteral("roma3_point"), "Surface IBM delta kernel");
    context.equal(deck.textOr("culbm_surface_ibm_geometry_policy"),
                  QStringLiteral("exterior_union"), "Surface IBM geometry policy");
    context.expect(deck.floating("culbm_surface_ibm_marker_spacing_lu") == 0.75,
                   "Surface IBM marker spacing");
    context.expect(deck.integer("culbm_surface_ibm_forcing_iterations") == 3,
                   "Surface IBM forcing iterations");
    context.expect(deck.floating("culbm_surface_ibm_relaxation") == 1.0,
                   "Surface IBM relaxation");
    context.expect(deck.floating("culbm_surface_ibm_residual_tolerance") == 0.001,
                   "Surface IBM residual tolerance");
    context.expect(deck.floating("culbm_surface_ibm_retraction_lu") == 0.0,
                   "Surface IBM retraction");
    context.expect(deck.floating("culbm_surface_ibm_support_thickness_lu") == 0.75,
                   "Surface IBM support thickness");
    context.expect(deck.integer("culbm_surface_ibm_max_markers") == 50000000,
                   "Surface IBM marker cap");
    context.expect(!deck.boolean("culbm_surface_ibm_strict_geometry").value_or(true),
                   "Surface IBM strict geometry switch");
    context.expect(deck.integer("culbm_surface_ibm_diagnostics_interval") == 0,
                   "Surface IBM diagnostics interval");
    context.equal(deck.textOr("cfd_solver"), QStringLiteral("culbm"),
                  "CULBM solver selection");

    const QString rendered = deck.render();
    context.expect(rendered.contains("ibm_enabler = true"), "Surface IBM switch round-trip");
    context.expect(rendered.contains("culbm_surface_ibm_geometry_policy = exterior_union"),
                   "Surface IBM geometry policy round-trip");
    context.expect(rendered.contains("culbm_surface_ibm_max_markers = 50000000"),
                   "Surface IBM marker cap round-trip");
    context.expect(rendered.contains("cfd_solver = \"culbm\""),
                   "solver selection is schema-owned and rendered quoted");
    context.expect(!rendered.contains("// Custom"), "Surface IBM keys are schema-owned");
}

void surfaceIbmProjectTemplateDefaultsMatchRuntimeContract(TestContext& context) {
    const QString fixtureRoot = QStringLiteral(LUW_TEST_FIXTURE_ROOT);
    const QString templatePath = QDir::cleanPath(
        QDir(fixtureRoot).filePath(QStringLiteral("../../project_template/conf.luw")));
    const luw::DeckDocument deck = luw::DeckDocument::load(templatePath);

    context.equal(deck.textOr("cfd_solver"), QStringLiteral("culbm"),
                  "project template selects CULBM by default");
    context.expect(!deck.boolean("ibm_enabler").value_or(true), "Surface IBM disabled by default");
    context.equal(deck.textOr("culbm_surface_ibm_delta_kernel"),
                  QStringLiteral("roma3_point"), "template delta kernel default");
    context.equal(deck.textOr("culbm_surface_ibm_geometry_policy"),
                  QStringLiteral("exterior_union"), "template geometry policy default");
    context.expect(deck.floating("culbm_surface_ibm_marker_spacing_lu") == 0.75,
                   "template marker spacing default");
    context.expect(deck.integer("culbm_surface_ibm_forcing_iterations") == 8,
                   "template forcing iterations default");
    context.expect(deck.floating("culbm_surface_ibm_relaxation") == 1.0,
                   "template relaxation default");
    context.expect(deck.floating("culbm_surface_ibm_residual_tolerance") == 0.001,
                   "template residual tolerance default");
    context.expect(deck.floating("culbm_surface_ibm_retraction_lu") == 0.5,
                   "template retraction default");
    context.expect(deck.floating("culbm_surface_ibm_support_thickness_lu") == 0.75,
                   "template support thickness default");
    context.expect(deck.integer("culbm_surface_ibm_max_markers") == 5000000,
                   "template marker cap default");
    context.expect(!deck.boolean("culbm_surface_ibm_strict_geometry").value_or(true),
                   "template strict geometry default");
    context.expect(deck.integer("culbm_surface_ibm_diagnostics_interval") == 0,
                   "template diagnostics interval default");
}

void edgeContracts(TestContext& context) {
    const luw::DeckDocument deck = luw::DeckDocument::fromText(
        "// Project\rquoted = \"https://example.invalid/a//b\" // tail\r"
        "duplicate = first\rduplicate = second\r"
        "reverse_pair = [9, -2]\rstrict_int = 1.0\r");
    context.equal(deck.rawOr("quoted"), QStringLiteral("\"https://example.invalid/a//b\""),
                  "comment markers inside quotes are preserved");
    context.equal(deck.textOr("duplicate"), QStringLiteral("second"), "duplicate is last-wins");
    context.expect(deck.duplicateKeys() == QStringList{QStringLiteral("duplicate")}, "duplicate is recorded");
    context.expect(!deck.integer("strict_int"), "decimal text is not an integer");
    const auto pair = deck.pair("reverse_pair");
    context.expect(pair && pair->first == -2.0 && pair->second == 9.0, "pair is sorted");
    bool threw = false;
    try {
        static_cast<void>(luw::DeckDocument::fromText("x=1\nx=2\n", true));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    context.expect(threw, "strict duplicate mode throws");
}

void goldenRenderMatchesPythonOracle(TestContext& context) {
    const QString path = QDir(QStringLiteral(LUW_TEST_FIXTURE_ROOT))
                             .filePath(QStringLiteral("deck_compatibility.luw"));
    const QString actual = luw::DeckDocument::load(path).render();
    const QString expected = QString::fromUtf8(
        "This preamble is intentionally preserved.\n"
        "\n"
        "// Domain\n"
        "cut_lon_manual = [121.700000, 121.300000]\n"
        "cut_lat_manual = [31.4, 31.1]\n"
        "utm_crs = \"EPSG:32651\"\n"
        "url_value = \"https://example.invalid/a//b\" // comment\n"
        "\n"
        "// CFD Controls\n"
        "gpu_memory = 20000\n"
        "mystery_key = second\n"
        "\n"
        "// Output & Probes\n"
        "probes =\n"
        "\n"
        "// Turbulence inflow\n"
        "turb_inflow_enable = true // legacy alias\n");
    context.equal(actual, expected, "rendered deck exactly matches Python oracle bytes");
}

void invalidUtf8IsIgnoredWithoutDroppingARealReplacementCharacter(TestContext& context) {
    QTemporaryDir temporary;
    context.expect(temporary.isValid(), "temporary directory for UTF-8 fixture");
    const QString path = temporary.filePath(QStringLiteral("invalid-utf8.luw"));
    QFile file(path);
    context.expect(file.open(QIODevice::WriteOnly), "open invalid UTF-8 fixture");
    QByteArray bytes("custom_value = before");
    bytes.append(char(0xff));
    bytes.append("after");
    bytes.append("\xe2\x28\xa1");
    bytes.append("-\xef\xbf\xbd-end\n");
    file.write(bytes);
    file.close();

    const luw::DeckDocument deck = luw::DeckDocument::load(path);
    context.equal(deck.textOr(QStringLiteral("custom_value")),
                  QString::fromUtf8("beforeafter(-\xef\xbf\xbd-end"),
                  "invalid UTF-8 bytes are ignored and valid U+FFFD is preserved");
}

void numericTokensMatchLegacyDeckParsing(TestContext& context) {
    const luw::DeckDocument deck = luw::DeckDocument::fromText(R"(
        integer_value = +1_000
        float_value = +1_000.5
        exponent_value = 1_2.5e+1_0
        infinity_value = -Infinity
        nan_value = NaN
        numeric_bool = 1_0
        nonfinite_bool = inf
        numeric_list = [1_0, -inf, nan]
    )");
    context.expect(deck.integer("integer_value") == 1000, "underscored integer token");
    context.expect(deck.floating("float_value") == 1000.5, "underscored float token");
    context.expect(deck.floating("exponent_value") == 12.5e10, "underscored exponent token");
    context.expect(deck.floating("infinity_value")
                       && std::isinf(*deck.floating("infinity_value"))
                       && *deck.floating("infinity_value") < 0.0,
                   "signed infinity token");
    context.expect(!deck.floating("nan_value"), "scalar NaN uses the default path");
    context.expect(deck.boolean("numeric_bool").value_or(false), "underscored numeric bool");
    context.expect(!deck.boolean("nonfinite_bool"), "non-finite numeric bool is rejected");
    const QVector<double> list = deck.floatList("numeric_list");
    context.expect(list.size() == 3 && list.at(0) == 10.0
                       && std::isinf(list.at(1)) && list.at(1) < 0.0
                       && std::isnan(list.at(2)),
                   "float list retains infinity and NaN");
}

} // namespace

int main(int argc, char** argv) {
    QCoreApplication application(argc, argv);
    TestContext context;
    boolTokensAreFuzzyAndQuoted(context);
    aliasesNormalize(context);
    renderRepairsOrderAndPreservesUnknowns(context);
    terrainKeysRoundTrip(context);
    surfaceIbmKeysAreKnownAndRoundTrip(context);
    surfaceIbmProjectTemplateDefaultsMatchRuntimeContract(context);
    edgeContracts(context);
    goldenRenderMatchesPythonOracle(context);
    invalidUtf8IsIgnoredWithoutDroppingARealReplacementCharacter(context);
    numericTokensMatchLegacyDeckParsing(context);
    runTerrainVoxelConfigTests(context);
    if (context.failures == 0) {
        std::cout << "All native deck/config compatibility tests passed.\n";
    }
    return context.failures == 0 ? 0 : 1;
}
