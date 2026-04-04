#pragma once

#include <QList>
#include <QString>

namespace luwgui {

struct PythonImportCheck {
    QString packageName;
    QString moduleName;
    QString errorText;
    bool success = false;
};

struct StartupCheckResult {
    QString repoRoot;
    QString pythonExecutable;
    QString pythonVersion;
    QString requirementsPath;
    QString processError;
    QString rawStdout;
    QString rawStderr;
    QList<PythonImportCheck> packageChecks;
    bool pythonResolved = false;
    bool requirementsResolved = false;

    bool hasWarnings() const;
    QString warningText() const;
};

StartupCheckResult runPythonEnvironmentSelfCheck(const QString& repoRoot, int timeoutMs = 120000);

}
