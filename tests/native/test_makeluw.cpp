#include "luw/makeluw_command.hpp"
#include "luw/preprocess_compute_config.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

QByteArray readAll(const QString& path) {
    QFile file(path);
    require(file.open(QIODevice::ReadOnly), "failed to read workflow log");
    return file.readAll();
}

QByteArray fakeStepLine(const QByteArray& log, const QByteArray& command) {
    const QByteArray prefix = QByteArray("FAKE_NATIVE_STEP=") + command;
    for (const QByteArray& line : log.split('\n')) {
        if (line.startsWith(prefix)) return line;
    }
    return {};
}

} // namespace

int main(int argc, char* argv[]) {
    QCoreApplication application(argc, argv);
    try {
        QTemporaryDir project;
        require(project.isValid(), "failed to create temporary project");
        const QString deckPath = project.filePath(QStringLiteral("conf.luw"));
        QFile deck(deckPath);
        require(deck.open(QIODevice::WriteOnly), "failed to create temporary deck");
        deck.close();

        std::ostringstream output;
        std::ostringstream error;
        luw::PreprocessComputeOverrides preprocessOverrides;
        preprocessOverrides.useGpu = true;
        preprocessOverrides.gpuDevice = 4;
        const luw::PreprocessComputeOverrideScope preprocessScope(preprocessOverrides);
        const int code = luw::cli::run_makeluw(
            {deckPath.toStdString()}, output, error);
        require(code == 1, "makeluw did not propagate a child-step failure");
        require(error.str().empty(), "makeluw unexpectedly wrote to stderr");

        const std::array<const char*, 6> displays{{
            "cdfInspect.py", "shpInspect.py", "1_buildBC.py",
            "2_shpCutter.py", "3_voxelization.py", "prerunValidate.py",
        }};
        std::size_t position = 0;
        for (const char* display : displays) {
            const std::size_t found = output.str().find(std::string("Running ") + display, position);
            require(found != std::string::npos, std::string("missing or reordered workflow step: ") + display);
            position = found + 1;
        }
        require(output.str().find("2_shpCutter.py failed (exit 9)") != std::string::npos,
                "failed child was not reported");
        require(output.str().find("Workflow finished with errors.") != std::string::npos,
                "workflow failure completion prompt missing");

        QDir logs(project.filePath(QStringLiteral("proj_temp")));
        const QStringList files = logs.entryList({QStringLiteral("*.log")}, QDir::Files);
        require(files.size() == 1, "workflow did not create exactly one log");
        const QByteArray log = readAll(logs.filePath(files.front()));
        for (const char* command : {"cdfinspect", "shpinspect", "luwbc", "luwcut", "luwvox", "luwval"}) {
            require(log.contains(QByteArray("FAKE_NATIVE_STEP=") + command),
                    std::string("native child output missing from log: ") + command);
        }
        for (const char* command : {"cdfinspect", "shpinspect"}) {
            const QByteArray line = fakeStepLine(log, command);
            require(!line.contains("--preprocess-use-gpu")
                        && !line.contains("--preprocess-gpu-device"),
                    std::string("inspector received compute-only overrides: ") + command);
        }
        for (const char* command : {"luwbc", "luwcut", "luwvox", "luwval"}) {
            const QByteArray line = fakeStepLine(log, command);
            require(line.contains("|ARG=--preprocess-use-gpu|ARG=true")
                        && line.contains("|ARG=--preprocess-gpu-device|ARG=4"),
                    std::string("compute step did not receive explicit overrides: ") + command);
        }
        require(log.contains("--- 2_shpCutter.py exited with code 9 ---"),
                "legacy failure footer missing from log");

        require(qputenv("LUW_MAKELUW_FAKE_ALL_SUCCESS", "1"),
                "failed to configure successful fake workflow");
        QTemporaryDir successfulProject;
        require(successfulProject.isValid(),
                "failed to create successful temporary project");
        const QString successfulDeckPath =
            successfulProject.filePath(QStringLiteral("conf.luw"));
        QFile successfulDeck(successfulDeckPath);
        require(successfulDeck.open(QIODevice::WriteOnly),
                "failed to create successful temporary deck");
        successfulDeck.close();
        std::ostringstream successfulOutput;
        std::ostringstream successfulError;
        const int successfulCode = luw::cli::run_makeluw(
            {successfulDeckPath.toStdString()}, successfulOutput,
            successfulError);
        qunsetenv("LUW_MAKELUW_FAKE_ALL_SUCCESS");
        require(successfulCode == 0 && successfulError.str().empty(),
                "makeluw all-success workflow did not return status 0");
        require(successfulOutput.str().find("All task finished as expected!") !=
                    std::string::npos &&
                    successfulOutput.str().find("Workflow finished with errors.") ==
                    std::string::npos,
                "makeluw all-success completion prompt changed");

        std::cout << "makeluw orchestration compatibility test passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "makeluw orchestration compatibility test failed: " << exception.what() << '\n';
        return 1;
    }
}
