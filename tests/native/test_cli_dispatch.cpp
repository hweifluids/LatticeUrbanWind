#include "luw/command_registry.hpp"
#include "luw/preprocess_compute_config.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void touch(const QString& path) {
    QFile file(path);
    require(file.open(QIODevice::WriteOnly), "failed to create temporary deck");
}

class CurrentDirectoryGuard final {
public:
    CurrentDirectoryGuard() : original_(QDir::currentPath()) {}
    ~CurrentDirectoryGuard() { QDir::setCurrent(original_); }

private:
    QString original_;
};

void testConventionalDeckWins() {
    QTemporaryDir directory;
    require(directory.isValid(), "failed to create temporary directory");
    touch(directory.filePath(QStringLiteral("conf.luw")));
    touch(directory.filePath(QStringLiteral("other.luw")));
    CurrentDirectoryGuard guard;
    require(QDir::setCurrent(directory.path()), "failed to enter temporary directory");

    std::string received;
    luw::cli::CommandRegistry registry;
    registry.add({"probe", "", "", luw::cli::DeckArgumentPolicy::Required,
                  [&](const luw::cli::CommandArguments& args, std::ostream&, std::ostream&) {
                      received = args.at(0);
                      return 7;
                  }});
    std::ostringstream output;
    std::ostringstream error;
    require(registry.invoke("probe", {}, output, error) == 7, "handler exit code was not preserved");
    require(QFileInfo(QString::fromStdString(received)).fileName() == QStringLiteral("conf.luw"),
            "conf.luw did not take precedence");
    require(QFileInfo(QString::fromStdString(received)).isAbsolute(), "deck path was not absolute");
}

void testUniqueDeckDiscovery() {
    QTemporaryDir directory;
    require(directory.isValid(), "failed to create temporary directory");
    touch(directory.filePath(QStringLiteral("unique.luw")));
    CurrentDirectoryGuard guard;
    require(QDir::setCurrent(directory.path()), "failed to enter temporary directory");

    std::string received;
    luw::cli::CommandRegistry registry;
    registry.add({"probe", "", "", luw::cli::DeckArgumentPolicy::Required,
                  [&](const luw::cli::CommandArguments& args, std::ostream&, std::ostream&) {
                      received = args.at(0);
                      return 0;
                  }});
    std::ostringstream output;
    std::ostringstream error;
    require(registry.invoke("probe", {}, output, error) == 0, "unique deck discovery failed");
    require(QFileInfo(QString::fromStdString(received)).fileName() == QStringLiteral("unique.luw"),
            "wrong unique deck discovered");
}

void testAmbiguousDeckRejected() {
    QTemporaryDir directory;
    require(directory.isValid(), "failed to create temporary directory");
    touch(directory.filePath(QStringLiteral("first.luw")));
    touch(directory.filePath(QStringLiteral("second.luw")));
    CurrentDirectoryGuard guard;
    require(QDir::setCurrent(directory.path()), "failed to enter temporary directory");

    bool called = false;
    luw::cli::CommandRegistry registry;
    registry.add({"probe", "", "", luw::cli::DeckArgumentPolicy::Required,
                  [&](const luw::cli::CommandArguments&, std::ostream&, std::ostream&) {
                      called = true;
                      return 0;
                  }});
    std::ostringstream output;
    std::ostringstream error;
    require(registry.invoke("probe", {}, output, error) == 2, "ambiguous deck returned wrong code");
    require(!called, "handler ran for ambiguous deck");
    require(error.str().find("no unique *.luw found") != std::string::npos,
            "ambiguous deck diagnostic changed");
}

void testExplicitMissingDeckRejected() {
    QTemporaryDir directory;
    require(directory.isValid(), "failed to create temporary directory");
    CurrentDirectoryGuard guard;
    require(QDir::setCurrent(directory.path()), "failed to enter temporary directory");

    luw::cli::CommandRegistry registry;
    registry.add({"probe", "", "", luw::cli::DeckArgumentPolicy::Required,
                  [](const luw::cli::CommandArguments&, std::ostream&, std::ostream&) { return 0; }});
    std::ostringstream output;
    std::ostringstream error;
    require(registry.invoke("probe", {"missing.luw"}, output, error) == 2,
            "missing explicit deck returned wrong code");
    require(error.str().find("ERROR: Conf file not found:") == 0,
            "missing deck diagnostic changed");
}

void testAutoWhenMissingPreservesExplicitInputs() {
    QTemporaryDir directory;
    require(directory.isValid(), "failed to create temporary directory");
    touch(directory.filePath(QStringLiteral("conf.luw")));
    CurrentDirectoryGuard guard;
    require(QDir::setCurrent(directory.path()), "failed to enter temporary directory");

    std::string received;
    luw::cli::CommandRegistry registry;
    registry.add({"probe", "", "", luw::cli::DeckArgumentPolicy::AutoWhenMissing,
                  [&](const luw::cli::CommandArguments& args, std::ostream&, std::ostream&) {
                      received = args.at(0);
                      return 11;
                  }});
    std::ostringstream output;
    std::ostringstream error;
    require(registry.invoke("probe", {"missing-input.nc"}, output, error) == 11,
            "explicit command-specific input was rejected by dispatcher");
    require(received == "missing-input.nc", "explicit command-specific input was rewritten");
    require(error.str().empty(), "explicit command-specific input produced a dispatcher error");

    received.clear();
    require(registry.invoke("probe", {}, output, error) == 11,
            "AutoWhenMissing deck discovery failed");
    require(QFileInfo(QString::fromStdString(received)).fileName() == QStringLiteral("conf.luw"),
            "AutoWhenMissing did not discover conf.luw");
}

void testPreprocessOptionsAreExtractedBeforeDeckResolution() {
    QTemporaryDir directory;
    require(directory.isValid(), "failed to create temporary directory");
    const QString deckPath = directory.filePath(QStringLiteral("conf.luw"));
    touch(deckPath);

    luw::PreprocessComputeOverrides receivedOverrides;
    luw::cli::CommandArguments receivedArguments;
    luw::cli::CommandRegistry registry;
    registry.add({"probe", "", "", luw::cli::DeckArgumentPolicy::Required,
                  [&](const luw::cli::CommandArguments& args, std::ostream&, std::ostream&) {
                      receivedArguments = args;
                      receivedOverrides = luw::currentPreprocessComputeOverrides();
                      return 13;
                  },
                  true});

    std::ostringstream output;
    std::ostringstream error;
    const int code = registry.invoke(
        "probe",
        {"--preprocess-use-gpu=true", deckPath.toStdString(),
         "--preprocess-gpu-device", "3"},
        output, error);
    require(code == 13 && error.str().empty(),
            "preprocess options before/after deck were not accepted");
    require(receivedArguments.size() == 1U
                && QFileInfo(QString::fromStdString(receivedArguments.front())).isAbsolute(),
            "preprocess options leaked into the command-specific parser");
    require(receivedOverrides.useGpu == std::optional<bool>(true)
                && receivedOverrides.gpuDevice == std::optional<int>(3),
            "preprocess override scope did not reach the command handler");
    const luw::PreprocessComputeOverrides restored =
        luw::currentPreprocessComputeOverrides();
    require(!restored.useGpu && !restored.gpuDevice,
            "preprocess override scope leaked after command invocation");

    receivedOverrides = {};
    receivedArguments.clear();
    output.str(std::string{});
    output.clear();
    error.str(std::string{});
    error.clear();
    require(registry.invoke(
                "probe",
                {deckPath.toStdString(), "--preprocess-use-gpu", "false",
                 "--preprocess-gpu-device=7"},
                output, error) == 13,
            "split boolean or equals-form device option was rejected");
    require(receivedOverrides.useGpu == std::optional<bool>(false)
                && receivedOverrides.gpuDevice == std::optional<int>(7),
            "split/equals preprocess values were parsed incorrectly");
}

void testInvalidPreprocessOptionsFailBeforeHandler() {
    QTemporaryDir directory;
    require(directory.isValid(), "failed to create temporary directory");
    const QString deckPath = directory.filePath(QStringLiteral("conf.luw"));
    touch(deckPath);

    bool called = false;
    luw::cli::CommandRegistry registry;
    registry.add({"probe", "", "", luw::cli::DeckArgumentPolicy::Required,
                  [&](const luw::cli::CommandArguments&, std::ostream&, std::ostream&) {
                      called = true;
                      return 0;
                  },
                  true});

    const auto rejected = [&](luw::cli::CommandArguments arguments,
                              const std::string& diagnostic) {
        called = false;
        std::ostringstream output;
        std::ostringstream error;
        require(registry.invoke("probe", std::move(arguments), output, error) == 2,
                "invalid preprocess option returned the wrong status");
        require(!called, "handler ran after invalid preprocess option");
        require(error.str().find(diagnostic) != std::string::npos,
                "invalid preprocess option diagnostic changed: " + error.str());
    };

    rejected({deckPath.toStdString(), "--preprocess-use-gpu", "maybe"},
             "invalid boolean value: 'maybe'");
    rejected({deckPath.toStdString(), "--preprocess-use-gpu"},
             "expected one argument");
    rejected({deckPath.toStdString(), "--preprocess-gpu-device", "-1"},
             "invalid non-negative integer value: '-1'");
    rejected({deckPath.toStdString(), "--preprocess-gpu-device=2147483648"},
             "invalid non-negative integer value: '2147483648'");
}

void testPreprocessOptionBoundaryAndOptIn() {
    luw::cli::CommandArguments acceptedArguments;
    luw::PreprocessComputeOverrides acceptedOverrides;
    luw::cli::CommandRegistry accepted;
    accepted.add({"probe", "", "", luw::cli::DeckArgumentPolicy::None,
                  [&](const luw::cli::CommandArguments& args, std::ostream&, std::ostream&) {
                      acceptedArguments = args;
                      acceptedOverrides = luw::currentPreprocessComputeOverrides();
                      return 0;
                  },
                  true});
    std::ostringstream output;
    std::ostringstream error;
    require(accepted.invoke(
                "probe", {"--", "--preprocess-use-gpu", "true"}, output, error) == 0,
            "-- delimiter invocation failed");
    require(acceptedArguments == luw::cli::CommandArguments(
                {"--", "--preprocess-use-gpu", "true"}),
            "preprocess option after -- was consumed");
    require(!acceptedOverrides.useGpu && !acceptedOverrides.gpuDevice,
            "preprocess option after -- entered the invocation scope");

    output.str(std::string{});
    output.clear();
    error.str(std::string{});
    error.clear();
    require(accepted.invoke("probe", {"--help"}, output, error) == 0 &&
                output.str().find("--preprocess-use-gpu BOOL") !=
                    std::string::npos &&
                output.str().find("--preprocess-gpu-device DEVICE") !=
                    std::string::npos,
            "dispatcher did not inject the shared preprocessing help footer");

    luw::cli::CommandArguments ordinaryArguments;
    luw::cli::CommandRegistry ordinary;
    ordinary.add({"probe", "", "", luw::cli::DeckArgumentPolicy::None,
                  [&](const luw::cli::CommandArguments& args, std::ostream&, std::ostream&) {
                      ordinaryArguments = args;
                      const auto overrides = luw::currentPreprocessComputeOverrides();
                      require(!overrides.useGpu && !overrides.gpuDevice,
                              "non-preprocess command received a preprocess scope");
                      return 0;
                  }});
    require(ordinary.invoke(
                "probe", {"--preprocess-use-gpu", "true"}, output, error) == 0,
            "ordinary command invocation failed");
    require(ordinaryArguments == luw::cli::CommandArguments(
                {"--preprocess-use-gpu", "true"}),
            "ordinary command lost options it owns");
}

} // namespace

int main() {
    try {
        testConventionalDeckWins();
        testUniqueDeckDiscovery();
        testAmbiguousDeckRejected();
        testExplicitMissingDeckRejected();
        testAutoWhenMissingPreservesExplicitInputs();
        testPreprocessOptionsAreExtractedBeforeDeckResolution();
        testInvalidPreprocessOptionsFailBeforeHandler();
        testPreprocessOptionBoundaryAndOptIn();
        std::cout << "CLI dispatch compatibility tests passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "CLI dispatch compatibility test failed: " << exception.what() << '\n';
        return 1;
    }
}
