#include "luwgui/CommandRunner.h"

#include <QDir>
#include <QFileInfo>
#include <QProcessEnvironment>

namespace luwgui {

QString commandPresetTitle(CommandPreset preset, RunMode mode) {
    Q_UNUSED(mode)
    switch (preset) {
    case CommandPreset::FullWorkflow:
        return "Full Workflow";
    case CommandPreset::CdfInspect:
        return "Inspect Wind Climate Inputs";
    case CommandPreset::ShpInspect:
        return "Inspect Building Footprints";
    case CommandPreset::BuildBoundaryConditions:
        return "Generate Boundary Conditions";
    case CommandPreset::CutGeometry:
        return "Crop Geometry Domain";
    case CommandPreset::Voxelize:
        return "Generate Voxel Domain";
    case CommandPreset::Validate:
        return "Validate Case Setup";
    case CommandPreset::PrepareBatchGeometry:
        return "Build Batch Geometry";
    case CommandPreset::Solve:
        return "Solve";
    case CommandPreset::VisLuw:
        return "Generate Visualization Data";
    case CommandPreset::Vtk2Nc:
        return "Export NetCDF";
    case CommandPreset::CutVis:
        return "Generate Cut Visuals";
    case CommandPreset::Clean:
        return "Clean Project Outputs";
    }
    return "command";
}

CommandRunner::CommandRunner(QObject* parent)
    : QObject(parent)
    , process_(new QProcess(this)) {
    process_->setProcessChannelMode(QProcess::MergedChannels);

    connect(process_, &QProcess::started, this, [this] {
        emit started(activeTitle_);
    });
    connect(process_, &QProcess::readyReadStandardOutput, this, [this] {
        emit outputReady(QString::fromLocal8Bit(process_->readAllStandardOutput()));
    });
    connect(process_, &QProcess::readyReadStandardError, this, [this] {
        emit errorText(QString::fromLocal8Bit(process_->readAllStandardError()));
    });
    connect(process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        Q_UNUSED(error)
        emit errorText(process_->errorString() + "\n");
    });
    connect(process_, &QProcess::finished, this, [this](int exitCode, QProcess::ExitStatus exitStatus) {
        emit finished(activeTitle_, exitCode, exitStatus);
        activeTitle_.clear();
    });
}

void CommandRunner::setDocument(ConfigDocument* document) {
    document_ = document;
}

ConfigDocument* CommandRunner::document() const {
    return document_;
}

bool CommandRunner::isRunning() const {
    return process_->state() != QProcess::NotRunning;
}

QString CommandRunner::pythonExecutable() const {
    return findPython();
}

CommandSpec CommandRunner::buildPreset(CommandPreset preset, const QStringList& extraArguments) const {
    const RunMode mode = document_ ? document_->mode() : RunMode::Luw;
    const QString deck = deckPath();
    const QString python = findPython();
    const QString workDir = projectDirectory();

    CommandSpec spec;
    spec.title = commandPresetTitle(preset, mode);
    spec.workingDirectory = workDir;

    auto usePythonScript = [&](const QString& relativePath, QStringList args = {}) {
        spec.program = python;
        spec.arguments = {scriptPath(relativePath)};
        spec.arguments += args;
    };

    switch (preset) {
    case CommandPreset::FullWorkflow:
        if (mode == RunMode::Luw) {
            usePythonScript("core/tools_core/makeluw.py", {deck});
        } else {
            usePythonScript("core/datagen_core/dgPrepare_stlinput.py", {deck});
        }
        break;
    case CommandPreset::CdfInspect:
        usePythonScript("core/tools_core/cdfInspect.py", {deck});
        break;
    case CommandPreset::ShpInspect:
        usePythonScript("core/tools_core/shpInspect.py", {deck});
        break;
    case CommandPreset::BuildBoundaryConditions:
        usePythonScript("core/bridge_core/1_buildBC.py", {deck});
        break;
    case CommandPreset::CutGeometry:
        usePythonScript("core/bridge_core/2_shpCutter.py", {deck});
        break;
    case CommandPreset::Voxelize:
        usePythonScript("core/bridge_core/3_voxelization.py", {deck});
        break;
    case CommandPreset::Validate:
        usePythonScript("core/tools_core/prerunValidate.py", {deck});
        break;
    case CommandPreset::PrepareBatchGeometry:
        usePythonScript("core/datagen_core/dgPrepare_stlinput.py", {deck});
        break;
    case CommandPreset::Solve:
        spec.program = solverExecutable();
        spec.arguments = {deck};
        break;
    case CommandPreset::VisLuw:
        usePythonScript("core/tools_core/visluw.py", {deck});
        break;
    case CommandPreset::Vtk2Nc:
        usePythonScript("core/tools_core/vtk2nc.py", {deck});
        break;
    case CommandPreset::CutVis:
        usePythonScript("core/tools_core/cut_vis.py", extraArguments);
        break;
    case CommandPreset::Clean:
        usePythonScript("core/tools_core/cleanluw.py", {deck});
        break;
    }

    if (preset != CommandPreset::CutVis) {
        spec.arguments += extraArguments;
    }

    return spec;
}

bool CommandRunner::startPreset(CommandPreset preset, const QStringList& extraArguments) {
    return startCommand(buildPreset(preset, extraArguments));
}

bool CommandRunner::startCommand(const CommandSpec& spec) {
    if (spec.program.isEmpty()) {
        emit errorText("Command program is empty.\n");
        return false;
    }
    if (isRunning()) {
        emit errorText("Another command is already running.\n");
        return false;
    }

    activeTitle_ = spec.title;
    process_->setProgram(spec.program);
    process_->setArguments(spec.arguments);
    process_->setWorkingDirectory(spec.workingDirectory);

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("LUW_HOME", repoRoot());
    QString path = env.value("PATH");
    const QString binPath = QDir(repoRoot()).filePath("bin");
    if (!path.contains(binPath, Qt::CaseInsensitive)) {
        path.prepend(binPath + QDir::listSeparator());
    }
    env.insert("PATH", path);
    process_->setProcessEnvironment(env);
    process_->start();
    return process_->waitForStarted(2000);
}

void CommandRunner::stop() {
    if (!isRunning()) {
        return;
    }

    const qint64 processId = process_->processId();
#ifdef Q_OS_WIN
    if (processId > 0) {
        QProcess::execute("taskkill", {"/PID", QString::number(processId), "/T", "/F"});
    }
#else
    if (processId > 0) {
        QProcess::execute("pkill", {"-TERM", "-P", QString::number(processId)});
    }
#endif

    process_->terminate();
    if (!process_->waitForFinished(1500)) {
        process_->kill();
        process_->waitForFinished(1500);
    }
}

QString CommandRunner::repoRoot() const {
    if (document_) {
        return document_->repoRoot();
    }
    return QDir::currentPath();
}

QString CommandRunner::deckPath() const {
    if (document_ && !document_->filePath().isEmpty()) {
        return document_->filePath();
    }
    const RunMode mode = document_ ? document_->mode() : RunMode::Luw;
    return QDir(projectDirectory()).filePath(defaultDeckName(mode));
}

QString CommandRunner::projectDirectory() const {
    if (document_) {
        return document_->projectDirectory();
    }
    return repoRoot();
}

QString CommandRunner::findPython() const {
    const QString root = repoRoot();
    const QStringList candidates = {
        QDir(root).filePath(".venv/Scripts/python.exe"),
        QDir(root).filePath(".venv/bin/python"),
        "python",
        "python3"
    };

    for (const QString& candidate : candidates) {
        if (candidate.contains('/') || candidate.contains('\\')) {
            if (QFileInfo::exists(candidate)) {
                return candidate;
            }
        } else {
            return candidate;
        }
    }
    return "python";
}

QString CommandRunner::solverExecutable() const {
    const QString root = repoRoot();
    const QStringList candidates = {
        QDir(root).filePath("core/cfd_core/FluidX3D/bin/FluidX3D.exe"),
        QDir(root).filePath("core/cfd_core/FluidX3D/bin/FluidX3D"),
        "FluidX3D"
    };
    for (const QString& candidate : candidates) {
        if (candidate.contains('/') || candidate.contains('\\')) {
            if (QFileInfo::exists(candidate)) {
                return candidate;
            }
        } else {
            return candidate;
        }
    }
    return {};
}

QString CommandRunner::scriptPath(const QString& relativePath) const {
    return QDir(repoRoot()).filePath(relativePath);
}

}
