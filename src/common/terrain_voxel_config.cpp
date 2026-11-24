#include "luw/terrain_voxel_config.hpp"

#include "luw/deck_document.hpp"

#include <QStringList>

#include <cmath>
#include <stdexcept>

namespace luw {

namespace {

void warning(const WarningSink& sink, const QString& message) {
    if (sink) sink(message);
}

template <typename T>
std::pair<T, QString> chooseOptional(const std::optional<T>& cli,
                                     const std::optional<T>& deck,
                                     const T& fallback) {
    if (cli) return {*cli, QStringLiteral("cli")};
    if (deck) return {*deck, QStringLiteral("deck")};
    return {fallback, QStringLiteral("default")};
}

std::pair<QString, QString> resolveChoice(const std::optional<QString>& cli,
                                          const std::optional<QString>& deck,
                                          const QString& fallback,
                                          const QStringList& allowed,
                                          const QString& label,
                                          const WarningSink& warn) {
    for (const auto& candidate : {std::pair{QStringLiteral("CLI"), cli},
                                  std::pair{QStringLiteral("deck"), deck}}) {
        if (!candidate.second) continue;
        const QString normalized = candidate.second->trimmed().toLower();
        if (allowed.contains(normalized)) return {normalized, candidate.first.toLower()};
        QStringList sorted = allowed;
        sorted.sort();
        const QString message = label + QStringLiteral(" from ") + candidate.first
            + QStringLiteral(" has unsupported value '") + *candidate.second
            + QStringLiteral("'. Allowed values: ") + sorted.join(QStringLiteral(", "))
            + QStringLiteral(".");
        warning(warn, message);
        throw std::invalid_argument(message.toStdString());
    }
    return {fallback, QStringLiteral("default")};
}

std::pair<QString, QString> resolveString(const std::optional<QString>& cli,
                                          const std::optional<QString>& deck,
                                          const QString& fallback,
                                          const QString& label,
                                          const WarningSink& warn) {
    for (const auto& candidate : {std::pair{QStringLiteral("CLI"), cli},
                                  std::pair{QStringLiteral("deck"), deck}}) {
        if (!candidate.second) continue;
        const QString value = candidate.second->trimmed();
        if (!value.isEmpty()) return {value, candidate.first.toLower()};
        warning(warn, label + QStringLiteral(" from ") + candidate.first
                          + QStringLiteral(" is empty. Falling back to the next source."));
    }
    return {fallback, QStringLiteral("default")};
}

std::pair<double, QString> resolveDouble(const std::optional<double>& cli,
                                         const std::optional<double>& deck,
                                         double fallback,
                                         const QString& label,
                                         double minimum,
                                         bool inclusive,
                                         const WarningSink& warn) {
    for (const auto& candidate : {std::pair{QStringLiteral("CLI"), cli},
                                  std::pair{QStringLiteral("deck"), deck}}) {
        if (!candidate.second) continue;
        const double value = *candidate.second;
        if (!std::isfinite(value)) {
            warning(warn, label + QStringLiteral(" from ") + candidate.first
                              + QStringLiteral(" is not finite. Falling back to the next source."));
            continue;
        }
        const bool valid = inclusive ? value >= minimum : value > minimum;
        if (!valid) {
            warning(warn, label + QStringLiteral(" from ") + candidate.first
                              + QStringLiteral(" must be ") + (inclusive ? QStringLiteral(">= ") : QStringLiteral("> "))
                              + QString::number(minimum) + QStringLiteral(". Got ") + QString::number(value)
                              + QStringLiteral(". Falling back to the next source."));
            continue;
        }
        return {value, candidate.first.toLower()};
    }
    return {fallback, QStringLiteral("default")};
}

std::pair<int, QString> resolveInteger(const std::optional<int>& cli,
                                       const std::optional<std::int64_t>& deck,
                                       int fallback,
                                       const QString& label,
                                       int minimum,
                                       const WarningSink& warn) {
    if (cli) {
        if (*cli >= minimum) return {*cli, QStringLiteral("cli")};
        warning(warn, label + QStringLiteral(" from CLI must be >= ") + QString::number(minimum)
                          + QStringLiteral(". Got ") + QString::number(*cli)
                          + QStringLiteral(". Falling back to the next source."));
    }
    if (deck) {
        if (*deck >= minimum) return {static_cast<int>(*deck), QStringLiteral("deck")};
        warning(warn, label + QStringLiteral(" from deck must be >= ") + QString::number(minimum)
                          + QStringLiteral(". Got ") + QString::number(*deck)
                          + QStringLiteral(". Falling back to the next source."));
    }
    return {fallback, QStringLiteral("default")};
}

} // namespace

std::pair<TerrainVoxelConfig, QHash<QString, QString>> resolveTerrainVoxelConfig(
    const DeckDocument* deck,
    const TerrainVoxelOverrides& overrides,
    const WarningSink& warn) {
    const auto deckText = [deck](const QString& key) -> std::optional<QString> {
        return deck ? deck->text(key) : std::nullopt;
    };
    const auto deckFloat = [deck](const QString& key) -> std::optional<double> {
        return deck ? deck->floating(key) : std::nullopt;
    };

    auto [approach, approachSource] = resolveChoice(
        overrides.approach, deckText(QStringLiteral("terr_voxel_approach")),
        QStringLiteral("idw"), {QStringLiteral("idw"), QStringLiteral("kriging")},
        QStringLiteral("terr_voxel_approach"), warn);
    auto [heightField, heightSource] = resolveString(
        overrides.heightField, deckText(QStringLiteral("terr_voxel_height_field")),
        QStringLiteral("auto"), QStringLiteral("terr_voxel_height_field"), warn);
    if (heightField.trimmed().compare(QStringLiteral("inferred"), Qt::CaseInsensitive) == 0) {
        heightField = QStringLiteral("auto");
    }
    auto [ignoreUnder, ignoreSource] = resolveDouble(
        overrides.ignoreUnder, deckFloat(QStringLiteral("terr_voxel_ignore_under")),
        0.0, QStringLiteral("terr_voxel_ignore_under"), 0.0, true, warn);
    auto [gridResolution, gridSource] = resolveDouble(
        overrides.gridResolution, deckFloat(QStringLiteral("terr_voxel_grid_resolution")),
        50.0, QStringLiteral("terr_voxel_grid_resolution"), 0.0, false, warn);
    auto [idwSigma, sigmaSource] = resolveDouble(
        overrides.idwSigma, deckFloat(QStringLiteral("terr_voxel_idw_sigma")),
        1.0, QStringLiteral("terr_voxel_idw_sigma"), 0.0, true, warn);
    auto [idwPower, powerSource] = resolveDouble(
        overrides.idwPower, deckFloat(QStringLiteral("terr_voxel_idw_power")),
        2.0, QStringLiteral("terr_voxel_idw_power"), 0.0, false, warn);
    auto [idwNeighbors, neighborsSource] = resolveInteger(
        overrides.idwNeighbors,
        deck ? deck->integer(QStringLiteral("terr_voxel_idw_neighbors")) : std::nullopt,
        12, QStringLiteral("terr_voxel_idw_neighbors"), 1, warn);
    if (approach == QStringLiteral("kriging") && idwNeighbors > 16) {
        const QString message = QStringLiteral(
            "terr_voxel_idw_neighbors must be <= 16 when "
            "terr_voxel_approach is kriging; this limit is identical for CPU "
            "and GPU backends. Got %1.")
                                    .arg(idwNeighbors);
        warning(warn, message);
        throw std::invalid_argument(message.toStdString());
    }

    TerrainVoxelConfig config{approach, heightField, ignoreUnder, gridResolution,
                              idwSigma, idwPower, idwNeighbors};
    QHash<QString, QString> sources{
        {QStringLiteral("approach"), approachSource},
        {QStringLiteral("height_field"), heightSource},
        {QStringLiteral("ignore_under"), ignoreSource},
        {QStringLiteral("grid_resolution"), gridSource},
        {QStringLiteral("idw_sigma"), sigmaSource},
        {QStringLiteral("idw_power"), powerSource},
        {QStringLiteral("idw_neighbors"), neighborsSource},
    };
    return {config, sources};
}

} // namespace luw
