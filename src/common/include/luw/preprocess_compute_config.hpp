#pragma once

#include <QHash>
#include <QString>

#include <functional>
#include <optional>
#include <utility>

namespace luw {

class DeckDocument;

struct PreprocessComputeConfig {
    bool useGpu = false;
    int gpuDevice = 0;
};

struct PreprocessComputeOverrides {
    std::optional<bool> useGpu;
    std::optional<int> gpuDevice;
};

using PreprocessWarningSink = std::function<void(const QString&)>;

// Resolves one shared preprocessing policy. Explicit command overrides take
// precedence over the invocation-wide CLI scope, then deck values, then the
// CPU/device-0 defaults.
std::pair<PreprocessComputeConfig, QHash<QString, QString>> resolvePreprocessComputeConfig(
    const DeckDocument* deck,
    const PreprocessComputeOverrides& overrides = {},
    const PreprocessWarningSink& warn = {});

// The command dispatcher installs a thread-local scope after consuming the
// common CLI flags. Native preprocessing commands can therefore share the same
// policy without each command owning a second parser.
PreprocessComputeOverrides currentPreprocessComputeOverrides();

class PreprocessComputeOverrideScope final {
public:
    explicit PreprocessComputeOverrideScope(const PreprocessComputeOverrides& overrides);
    ~PreprocessComputeOverrideScope();

    PreprocessComputeOverrideScope(const PreprocessComputeOverrideScope&) = delete;
    PreprocessComputeOverrideScope& operator=(const PreprocessComputeOverrideScope&) = delete;

private:
    PreprocessComputeOverrides previous_;
};

QString preprocessComputePolicyText(const PreprocessComputeConfig& config);

} // namespace luw
