#include "luw/preprocess_compute_config.hpp"

#include "luw/deck_document.hpp"

#include <limits>
#include <stdexcept>

namespace luw {

namespace {

thread_local PreprocessComputeOverrides invocationOverrides;

void warning(const PreprocessWarningSink& sink, const QString& message) {
    if (sink) sink(message);
}

std::optional<bool> deckBoolean(const DeckDocument* deck,
                                const QString& key,
                                const PreprocessWarningSink& warn) {
    if (!deck || !deck->has(key)) return std::nullopt;
    const auto value = deck->boolean(key);
    if (!value) {
        const QString message = key + QStringLiteral(" from deck is not a valid boolean.");
        warning(warn, message);
        throw std::invalid_argument(message.toStdString());
    }
    return value;
}

std::optional<int> deckDevice(const DeckDocument* deck,
                              const PreprocessWarningSink& warn) {
    const QString key = QStringLiteral("preprocess_gpu_device");
    if (!deck || !deck->has(key)) return std::nullopt;
    const auto value = deck->integer(key);
    if (!value) {
        const QString message = key + QStringLiteral(" from deck is not a valid integer.");
        warning(warn, message);
        throw std::invalid_argument(message.toStdString());
    }
    if (*value < 0 || *value > std::numeric_limits<int>::max()) {
        const QString message = key
            + QStringLiteral(" from deck must be a non-negative 32-bit integer. Got ")
            + QString::number(*value) + QStringLiteral(".");
        warning(warn, message);
        throw std::invalid_argument(message.toStdString());
    }
    return static_cast<int>(*value);
}

} // namespace

std::pair<PreprocessComputeConfig, QHash<QString, QString>> resolvePreprocessComputeConfig(
    const DeckDocument* deck,
    const PreprocessComputeOverrides& overrides,
    const PreprocessWarningSink& warn) {
    const PreprocessComputeOverrides invocation = currentPreprocessComputeOverrides();

    PreprocessComputeConfig config;
    QHash<QString, QString> sources;

    if (overrides.useGpu) {
        config.useGpu = *overrides.useGpu;
        sources.insert(QStringLiteral("use_gpu"), QStringLiteral("cli"));
    } else if (invocation.useGpu) {
        config.useGpu = *invocation.useGpu;
        sources.insert(QStringLiteral("use_gpu"), QStringLiteral("cli"));
    } else if (const auto value = deckBoolean(deck, QStringLiteral("preprocess_use_gpu"), warn)) {
        config.useGpu = *value;
        sources.insert(QStringLiteral("use_gpu"), QStringLiteral("deck"));
    } else {
        sources.insert(QStringLiteral("use_gpu"), QStringLiteral("default"));
    }

    const auto applyDevice = [&](const std::optional<int>& value, const QString& source) -> bool {
        if (!value) return false;
        if (*value < 0) {
            const QString message = QStringLiteral("preprocess_gpu_device from ")
                + source.toUpper() + QStringLiteral(" must be >= 0. Got ")
                + QString::number(*value) + QStringLiteral(".");
            warning(warn, message);
            throw std::invalid_argument(message.toStdString());
        }
        config.gpuDevice = *value;
        sources.insert(QStringLiteral("gpu_device"), source);
        return true;
    };

    if (!applyDevice(overrides.gpuDevice, QStringLiteral("cli"))
        && !applyDevice(invocation.gpuDevice, QStringLiteral("cli"))) {
        const auto fromDeck = deckDevice(deck, warn);
        if (!applyDevice(fromDeck, QStringLiteral("deck"))) {
            sources.insert(QStringLiteral("gpu_device"), QStringLiteral("default"));
        }
    }

    return {config, sources};
}

PreprocessComputeOverrides currentPreprocessComputeOverrides() {
    return invocationOverrides;
}

PreprocessComputeOverrideScope::PreprocessComputeOverrideScope(
    const PreprocessComputeOverrides& overrides)
    : previous_(invocationOverrides) {
    invocationOverrides = overrides;
}

PreprocessComputeOverrideScope::~PreprocessComputeOverrideScope() {
    invocationOverrides = previous_;
}

QString preprocessComputePolicyText(const PreprocessComputeConfig& config) {
    return config.useGpu
        ? QStringLiteral("OpenCL GPU device %1").arg(config.gpuDevice)
        : QStringLiteral("CPU");
}

} // namespace luw
