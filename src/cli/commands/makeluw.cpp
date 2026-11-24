#include "luw/makeluw_command.hpp"

#include "luw/preprocess_compute_config.hpp"
#include "luw/progress.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QProcess>
#include <QStringList>

#include <array>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace luw::cli {

namespace {

namespace fs = std::filesystem;

struct WorkflowStep {
    const char* command;
    // The .py spelling is deliberately display-only.  Existing console/log
    // consumers use these legacy labels; no script is located or executed.
    const char* legacyDisplayName;
    const char* progressStem;
    const char* title;
    bool acceptsPreprocessOptions;
};

constexpr std::array<WorkflowStep, 6> kWorkflowSteps{{
    {"cdfinspect", "cdfInspect.py", "cdfInspect", "Inspect Wind Climate Inputs", false},
    {"shpinspect", "shpInspect.py", "shpInspect", "Inspect Building Footprints", false},
    {"luwbc", "1_buildBC.py", "1_buildBC", "Generate Boundary Conditions", true},
    {"luwcut", "2_shpCutter.py", "2_shpCutter", "Crop Geometry Domain", true},
    {"luwvox", "3_voxelization.py", "3_voxelization", "Generate Voxel Domain", true},
    {"luwval", "prerunValidate.py", "prerunValidate", "Validate Case Setup", true},
}};

std::string utf8(const QString& value) {
    return value.toUtf8().toStdString();
}

fs::path pathFromUtf8(const std::string& value) {
#ifdef _WIN32
    return fs::path(QString::fromStdString(value).toStdWString());
#else
    return fs::path(value);
#endif
}

std::string pathToUtf8(const fs::path& value) {
#ifdef _WIN32
    return QString::fromStdWString(value.wstring()).toUtf8().toStdString();
#else
    return value.string();
#endif
}

std::tm localTime(std::time_t value) {
    std::tm result{};
#ifdef _WIN32
    localtime_s(&result, &value);
#else
    localtime_r(&value, &result);
#endif
    return result;
}

std::string fileTimestamp() {
    const std::time_t now = std::time(nullptr);
    const std::tm local = localTime(now);
    std::ostringstream rendered;
    rendered << std::put_time(&local, "%Y%m%d%H%M%S");
    return rendered.str();
}

std::string lineTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t wall = std::chrono::system_clock::to_time_t(now);
    const std::tm local = localTime(wall);
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    std::ostringstream rendered;
    rendered << std::put_time(&local, "%Y-%m-%d %H:%M:%S")
             << '.' << std::setfill('0') << std::setw(3) << milliseconds.count();
    return rendered.str();
}

int processId() {
#ifdef _WIN32
    return _getpid();
#else
    return static_cast<int>(getpid());
#endif
}

void printBanner(std::ostream& output) {
    output
        << "|-------------------------------------------------------------------|\n"
        << "|  Project:   WRFcpLBM - toolbox for WRF-FluidX3D coupling          |\n"
        << "|  Module:    MAKELUW - THE MAKING OF LUW                           |\n"
        << "|  Author:    Huanxia Wei                                           |\n"
        << "|  Email:     huanxia.wei@u.nus.edu                                 |\n"
        << "|  Version:   <20251031A-GPU>                                       |\n"
        << "|  License:   Customized License.                                   |\n"
        << "|-------------------------------------------------------------------|\n";
    output.flush();
}

class WorkflowLog final {
public:
    WorkflowLog(const fs::path& deckPath, std::ostream& terminal, std::ostream& error)
        : terminal_(terminal) {
        const fs::path directory = deckPath.parent_path() / "proj_temp";
        std::error_code createError;
        fs::create_directories(directory, createError);
        std::string name = fileTimestamp();
        if (name.empty()) name = std::to_string(processId());
        path_ = directory / (name + ".log");
        file_.open(path_, std::ios::out | std::ios::app | std::ios::binary);
        if (!file_) {
            error << "ERROR: Cannot open workflow log: " << pathToUtf8(path_) << '\n';
        }
    }

    [[nodiscard]] bool isOpen() const noexcept { return file_.is_open(); }

    void blank() {
        terminal_ << '\n';
        terminal_.flush();
        file_ << '\n';
        file_.flush();
    }

    void line(const std::string& message) {
        const std::string rendered = '[' + lineTimestamp() + "] " + message + '\n';
        terminal_ << rendered;
        terminal_.flush();
        file_ << rendered;
        file_.flush();
    }

    void childBytes(const QByteArray& bytes) {
        if (bytes.isEmpty()) return;
        terminal_.write(bytes.constData(), static_cast<std::streamsize>(bytes.size()));
        terminal_.flush();
        file_.write(bytes.constData(), static_cast<std::streamsize>(bytes.size()));
        file_.flush();
    }

    void footer(const std::string& displayName, int exitCode) {
        file_ << "\n--- " << displayName << " exited with code " << exitCode << " ---\n";
        file_.flush();
    }

private:
    std::ostream& terminal_;
    fs::path path_;
    std::ofstream file_;
};

QString canonicalNativeLauncher() {
    const QString suffix =
#ifdef _WIN32
        QStringLiteral(".exe");
#else
        QString();
#endif
    const QString sibling = QDir(QCoreApplication::applicationDirPath())
                                .filePath(QStringLiteral("luw") + suffix);
    if (QFileInfo(sibling).isFile()) return QFileInfo(sibling).absoluteFilePath();

    const QFileInfo current(QCoreApplication::applicationFilePath());
    if (current.completeBaseName().compare(QStringLiteral("luw"), Qt::CaseInsensitive) == 0
        && current.isFile()) {
        return current.absoluteFilePath();
    }
    return {};
}

int runNativeStep(const QString& launcher,
                  const WorkflowStep& step,
                  const QString& deckPath,
                  const QStringList& preprocessArguments,
                  WorkflowLog& log) {
    QProcess process;
    process.setProgram(launcher);
    QStringList arguments{QString::fromLatin1(step.command), deckPath};
    if (step.acceptsPreprocessOptions) arguments.append(preprocessArguments);
    process.setArguments(arguments);
    process.setProcessChannelMode(QProcess::MergedChannels);
    process.setInputChannelMode(QProcess::ForwardedInputChannel);
    process.start();
    if (!process.waitForStarted(30000)) {
        const std::string message = "ERROR: Failed to start native command "
            + std::string(step.command) + ": " + utf8(process.errorString()) + '\n';
        log.childBytes(QByteArray::fromStdString(message));
        return 127;
    }

    while (process.state() != QProcess::NotRunning) {
        process.waitForReadyRead(100);
        log.childBytes(process.readAll());
    }
    log.childBytes(process.readAll());
    return process.exitStatus() == QProcess::NormalExit ? process.exitCode() : 1;
}

} // namespace

int run_makeluw(const CommandArguments& arguments,
                std::ostream& output,
                std::ostream& error) {
    printBanner(output);
    if (arguments.empty()) {
        error << "Usage: makeluw <deck_file_path>\n";
        return 1;
    }

    const fs::path deckPath = pathFromUtf8(arguments.front());
    WorkflowLog log(deckPath, output, error);
    if (!log.isOpen()) return 1;

    const QString launcher = canonicalNativeLauncher();
    if (launcher.isEmpty()) {
        log.line("[ERROR] Native LUW launcher not found beside the current executable.");
        return 1;
    }

    ProgressEmitter progress(QStringLiteral("Running Workflow"));
    const QString deck = QString::fromStdString(arguments.front());
    const PreprocessComputeOverrides overrides = currentPreprocessComputeOverrides();
    QStringList preprocessArguments;
    if (overrides.useGpu) {
        preprocessArguments << QStringLiteral("--preprocess-use-gpu")
                            << (*overrides.useGpu ? QStringLiteral("true")
                                                  : QStringLiteral("false"));
    }
    if (overrides.gpuDevice) {
        preprocessArguments << QStringLiteral("--preprocess-gpu-device")
                            << QString::number(*overrides.gpuDevice);
    }
    bool anyStepFailed = false;
    for (std::size_t index = 0; index < kWorkflowSteps.size(); ++index) {
        const WorkflowStep& step = kWorkflowSteps[index];
        progress.stage(
            QStringLiteral("workflow_") + QString::fromLatin1(step.progressStem),
            QStringLiteral("Workflow step %1/%2")
                .arg(static_cast<qulonglong>(index + 1))
                .arg(static_cast<qulonglong>(kWorkflowSteps.size())),
            QString::fromLatin1(step.title));

        log.blank();
        log.line(std::string(84, '-') + " Running " + step.legacyDisplayName + ' ');
        const int exitCode = runNativeStep(
            launcher, step, deck, preprocessArguments, log);
        log.line(std::string(84, '-') + ' ' + step.legacyDisplayName
                 + " exited with code " + std::to_string(exitCode) + ' ');
        log.footer(step.legacyDisplayName, exitCode);
        if (exitCode != 0) {
            anyStepFailed = true;
            log.line("[ERROR] " + std::string(step.legacyDisplayName)
                     + " failed (exit " + std::to_string(exitCode) + ").");
        }
    }

    log.blank();
    log.line(std::string(84, '-')
             + (anyStepFailed ? " Workflow finished with errors. "
                              : " All task finished as expected! "));
    log.blank();
    return anyStepFailed ? 1 : 0;
}

} // namespace luw::cli
