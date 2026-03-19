#pragma once

#include "luwgui/ConfigDocument.h"

#include <QObject>
#include <QProcess>

namespace luwgui {

enum class CommandPreset {
    FullWorkflow,
    CdfInspect,
    ShpInspect,
    BuildBoundaryConditions,
    CutGeometry,
    Voxelize,
    Validate,
    PrepareBatchGeometry,
    Solve,
    VisLuw,
    Vtk2Nc,
    CutVis,
    Clean
};

struct CommandSpec {
    QString title;
    QString program;
    QStringList arguments;
    QString workingDirectory;
};

class CommandRunner : public QObject {
    Q_OBJECT

public:
    explicit CommandRunner(QObject* parent = nullptr);

    void setDocument(ConfigDocument* document);
    ConfigDocument* document() const;

    bool isRunning() const;
    QString pythonExecutable() const;

    CommandSpec buildPreset(CommandPreset preset, const QStringList& extraArguments = {}) const;
    bool startPreset(CommandPreset preset, const QStringList& extraArguments = {});
    bool startCommand(const CommandSpec& spec);
    void stop();

signals:
    void started(const QString& title);
    void finished(const QString& title, int exitCode, QProcess::ExitStatus exitStatus);
    void outputReady(const QString& text);
    void errorText(const QString& text);

private:
    QString repoRoot() const;
    QString deckPath() const;
    QString projectDirectory() const;
    QString findPython() const;
    QString solverExecutable() const;
    QString scriptPath(const QString& relativePath) const;

    ConfigDocument* document_ = nullptr;
    QProcess* process_ = nullptr;
    QString activeTitle_;
};

QString commandPresetTitle(CommandPreset preset, RunMode mode);

}
