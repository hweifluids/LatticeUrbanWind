#pragma once

#include <QHash>
#include <QString>

#include <functional>
#include <optional>
#include <utility>

namespace luw {

class DeckDocument;

struct TerrainVoxelConfig {
    QString approach = QStringLiteral("idw");
    QString heightField = QStringLiteral("auto");
    double ignoreUnder = 0.0;
    double gridResolution = 50.0;
    double idwSigma = 1.0;
    double idwPower = 2.0;
    int idwNeighbors = 12;
};

struct TerrainVoxelOverrides {
    std::optional<QString> approach;
    std::optional<QString> heightField;
    std::optional<double> ignoreUnder;
    std::optional<double> gridResolution;
    std::optional<double> idwSigma;
    std::optional<double> idwPower;
    std::optional<int> idwNeighbors;
};

using WarningSink = std::function<void(const QString&)>;

std::pair<TerrainVoxelConfig, QHash<QString, QString>> resolveTerrainVoxelConfig(
    const DeckDocument* deck,
    const TerrainVoxelOverrides& overrides = {},
    const WarningSink& warn = {});

} // namespace luw

