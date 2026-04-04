#include "luwgui/CommandRunner.h"
#include "luwgui/RuntimePaths.h"

#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcessEnvironment>

namespace luwgui {

QString commandPresetTitle(CommandPreset preset, RunMode mode) {
    Q_UNUSED(mode)
    switch (preset) {
    case CommandPreset::FullWorkflow:
        return "Run preprocessing workflow";
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
        pendingOutput_.clear();
        pendingCarriageReturn_ = false;
        emit started(activeTitle_);
    });
    connect(process_, &QProcess::readyReadStandardOutput, this, [this] {
        processMergedOutput(process_->readAllStandardOutput());
    });
    connect(process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        Q_UNUSED(error)
        emit errorText(process_->errorString() + "\n");
    });
    connect(process_, &QProcess::finished, this, [this](int exitCode, QProcess::ExitStatus exitStatus) {
        flushPendingOutput();
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

QString CommandRunner::activeTitle() const {
    return activeTitle_;
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
    env.insert("LUW_PROGRESS_MODE", "gui");
    process_->setProcessEnvironment(env);
    process_->start();
    return process_->waitForStarted(2000);
}

void CommandRunner::processMergedOutput(const QByteArray& data) {
    QString chunk = QString::fromLocal8Bit(data);
    int index = 0;

    if (pendingCarriageReturn_) {
        if (!chunk.isEmpty() && chunk.front() == '\n') {
            dispatchCompletedLine(pendingOutput_);
            pendingOutput_.clear();
            index = 1;
        } else {
            handleStandaloneCarriageReturn();
        }
        pendingCarriageReturn_ = false;
    }

    for (; index < chunk.size(); ++index) {
        const QChar ch = chunk.at(index);
        if (ch == '\r') {
            if (index + 1 < chunk.size()) {
                if (chunk.at(index + 1) == '\n') {
                    dispatchCompletedLine(pendingOutput_);
                    pendingOutput_.clear();
                    ++index;
                } else {
                    handleStandaloneCarriageReturn();
                }
            } else {
                pendingCarriageReturn_ = true;
            }
            continue;
        }

        if (ch == '\n') {
            dispatchCompletedLine(pendingOutput_);
            pendingOutput_.clear();
            continue;
        }

        pendingOutput_.append(ch);
    }
}

void CommandRunner::flushPendingOutput() {
    if (pendingCarriageReturn_) {
        handleStandaloneCarriageReturn();
        pendingCarriageReturn_ = false;
    }
    if (pendingOutput_.isEmpty()) {
        return;
    }
    dispatchCompletedLine(pendingOutput_);
    pendingOutput_.clear();
}

void CommandRunner::dispatchCompletedLine(const QString& line) {
    if (!tryHandleProgressLine(line)) {
        emit outputReady(line + "\n");
    }
}

void CommandRunner::handleStandaloneCarriageReturn() {
    const QString line = pendingOutput_.trimmed();
    pendingOutput_.clear();
    if (line.isEmpty()) {
        return;
    }
    tryHandleProgressLine(line);
}

bool CommandRunner::tryHandleProgressLine(const QString& line) {
    static const QString kProgressPrefix = QStringLiteral("[[LUW_PROGRESS]]");
    if (!line.startsWith(kProgressPrefix)) {
        return false;
    }

    const QByteArray payload = line.mid(kProgressPrefix.size()).trimmed().toUtf8();
    QJsonParseError parseError{};
    const QJsonDocument document = QJsonDocument::fromJson(payload, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        return false;
    }

    const QJsonObject object = document.object();
    const QString stage = object.value("stage").toString();
    const QString summary = object.value("label").toString(stage.isEmpty() ? QStringLiteral("CFD progress") : stage);
    const QString detail = object.value("detail").toString();
    const qint64 current = object.value("current").toInteger(-1);
    const qint64 total = object.value("total").toInteger(-1);
    const bool indeterminate = object.value("indeterminate").toBool(current < 0 || total <= 0);

    emit progressUpdated(summary, detail, current, total, indeterminate);
    return true;
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
    return detectRepoRoot();
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
    return findPythonExecutable(repoRoot());
}

QString CommandRunner::solverExecutable() const {
    return findSolverExecutable(repoRoot());
}

QString CommandRunner::scriptPath(const QString& relativePath) const {
    return resolveRepoFilePath(repoRoot(), relativePath);
}

}
