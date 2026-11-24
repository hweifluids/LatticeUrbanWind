#include "test_support.hpp"

#include "luw/deck_document.hpp"
#include "luw/preprocess_compute_config.hpp"
#include "luw/terrain_voxel_config.hpp"

#include <QStringList>

void runTerrainVoxelConfigTests(TestContext& context) {
    {
        const auto [config, sources] = luw::resolvePreprocessComputeConfig(nullptr);
        context.expect(!config.useGpu && config.gpuDevice == 0,
                       "preprocess compute defaults are CPU/device 0");
        context.equal(sources.value("use_gpu"), QStringLiteral("default"),
                      "default GPU policy source");
        context.equal(luw::preprocessComputePolicyText(config), QStringLiteral("CPU"),
                      "CPU policy text");
    }
    {
        const luw::DeckDocument deck = luw::DeckDocument::fromText(R"(
            preprocess_use_gpu = true
            preprocess_gpu_device = 2
        )");
        const auto [config, sources] = luw::resolvePreprocessComputeConfig(&deck);
        context.expect(config.useGpu && config.gpuDevice == 2,
                       "deck preprocessing policy values");
        context.equal(sources.value("gpu_device"), QStringLiteral("deck"),
                      "deck GPU device source");

        luw::PreprocessComputeOverrides invocation;
        invocation.useGpu = false;
        invocation.gpuDevice = 1;
        {
            const luw::PreprocessComputeOverrideScope scope(invocation);
            const auto [scoped, scopedSources] = luw::resolvePreprocessComputeConfig(&deck);
            context.expect(!scoped.useGpu && scoped.gpuDevice == 1,
                           "invocation scope overrides deck policy");
            context.equal(scopedSources.value("use_gpu"), QStringLiteral("cli"),
                          "scoped GPU policy source");

            luw::PreprocessComputeOverrides direct;
            direct.useGpu = true;
            direct.gpuDevice = 3;
            const auto [explicitConfig, explicitSources] =
                luw::resolvePreprocessComputeConfig(&deck, direct);
            context.expect(explicitConfig.useGpu && explicitConfig.gpuDevice == 3,
                           "direct CLI overrides invocation scope");
            context.equal(explicitSources.value("gpu_device"), QStringLiteral("cli"),
                          "direct GPU device source");
        }
        const auto [restored, restoredSources] = luw::resolvePreprocessComputeConfig(&deck);
        context.expect(restored.useGpu && restored.gpuDevice == 2,
                       "invocation scope restores prior policy");
        context.equal(restoredSources.value("use_gpu"), QStringLiteral("deck"),
                      "restored GPU source");
    }
    {
        for (const QString& text : {QStringLiteral("preprocess_use_gpu = perhaps\n"),
                                    QStringLiteral("preprocess_gpu_device = -1\n")}) {
            bool rejected = false;
            try {
                const luw::DeckDocument deck = luw::DeckDocument::fromText(text);
                (void)luw::resolvePreprocessComputeConfig(&deck);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            context.expect(rejected, "invalid preprocessing deck value is fatal");
        }
    }
    {
        const luw::DeckDocument deck = luw::DeckDocument::fromText(R"(
            // Domain
            terr_voxel_height_field = BUILD_H
            terr_voxel_ignore_under = 4.0
            terr_voxel_approach = kriging
            terr_voxel_grid_resolution = 20.0
            terr_voxel_idw_sigma = 0.25
            terr_voxel_idw_power = 1.5
            terr_voxel_idw_neighbors = 6
        )");
        const auto [config, sources] = luw::resolveTerrainVoxelConfig(&deck);
        context.equal(config.heightField, QStringLiteral("BUILD_H"), "deck height field");
        context.expect(config.ignoreUnder == 4.0 && config.approach == "kriging", "deck scalar values");
        context.expect(config.gridResolution == 20.0 && config.idwSigma == 0.25, "deck interpolation values");
        context.expect(config.idwPower == 1.5 && config.idwNeighbors == 6, "deck power/neighbors");
        context.equal(sources.value("approach"), QStringLiteral("deck"), "deck approach source");
    }
    {
        const luw::DeckDocument deck = luw::DeckDocument::fromText(R"(
            terr_voxel_height_field = DECK_HEIGHT
            terr_voxel_ignore_under = 2
            terr_voxel_approach = idw
            terr_voxel_grid_resolution = 50
            terr_voxel_idw_sigma = 1
            terr_voxel_idw_power = 2
            terr_voxel_idw_neighbors = 12
        )");
        luw::TerrainVoxelOverrides overrides;
        overrides.heightField = QStringLiteral("CLI_HEIGHT");
        overrides.ignoreUnder = 5.0;
        overrides.approach = QStringLiteral("kriging");
        overrides.gridResolution = 15.0;
        overrides.idwSigma = 0.1;
        overrides.idwPower = 1.2;
        overrides.idwNeighbors = 4;
        const auto [config, sources] = luw::resolveTerrainVoxelConfig(&deck, overrides);
        context.expect(config.heightField == "CLI_HEIGHT" && config.ignoreUnder == 5.0,
                       "CLI overrides string/float");
        context.expect(config.approach == "kriging" && config.gridResolution == 15.0,
                       "CLI overrides choice/grid");
        context.expect(config.idwSigma == 0.1 && config.idwPower == 1.2 && config.idwNeighbors == 4,
                       "CLI overrides interpolation");
        for (auto it = sources.cbegin(); it != sources.cend(); ++it) {
            context.equal(it.value(), QStringLiteral("cli"), "all sources are CLI");
        }
    }
    {
        const luw::DeckDocument deck = luw::DeckDocument::fromText("terr_voxel_height_field = Inferred\n");
        const auto [config, sources] = luw::resolveTerrainVoxelConfig(&deck);
        context.equal(config.heightField, QStringLiteral("auto"), "inferred normalizes to auto");
        context.equal(sources.value("height_field"), QStringLiteral("deck"), "height source remains deck");
    }
    {
        const luw::DeckDocument deck = luw::DeckDocument::fromText(R"(
            terr_voxel_grid_resolution = 0
            terr_voxel_idw_sigma = -1
            terr_voxel_idw_power = 0
            terr_voxel_idw_neighbors = 0
        )");
        QStringList warnings;
        const auto [config, sources] = luw::resolveTerrainVoxelConfig(
            &deck, {}, [&warnings](const QString& message) { warnings.push_back(message); });
        context.expect(config.approach == "idw" && config.gridResolution == 50.0,
                       "invalid grid uses default");
        context.expect(config.idwSigma == 1.0 && config.idwPower == 2.0 && config.idwNeighbors == 12,
                       "invalid interpolation values use defaults");
        context.equal(sources.value("approach"), QStringLiteral("default"), "default approach source");
        context.expect(warnings.size() >= 4, "invalid values emit warnings");
    }
    {
        const luw::DeckDocument deck = luw::DeckDocument::fromText(
            "terr_voxel_approach = kriging_gpu\n");
        bool rejected = false;
        try {
            (void)luw::resolveTerrainVoxelConfig(&deck);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        context.expect(rejected, "removed terrain approach is rejected instead of remapped");
    }
    {
        const luw::DeckDocument deck = luw::DeckDocument::fromText(R"(
            terr_voxel_approach = kriging
            terr_voxel_idw_neighbors = 17
        )");
        bool rejected = false;
        try {
            (void)luw::resolveTerrainVoxelConfig(&deck);
        } catch (const std::invalid_argument& exception) {
            rejected = std::string(exception.what()).find(
                           "identical for CPU and GPU") != std::string::npos;
        }
        context.expect(rejected,
                       "kriging neighbor domain is backend-independent");
    }
}
