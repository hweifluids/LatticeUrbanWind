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

struct AcceleratorDeviceCheck {
    QString name;
    QString vendor;
    QString version;
    QString driverVersion;
    QString computeCapability;
    int computeUnits = 0;
};

struct AcceleratorCheck {
    QString summary;
    QString errorText;
    QString provider;
    QString sourceRoot;
    QString runtimeRoot;
    QString cudaHome;
    QString runtimeVersion;
    QString driverVersion;
    QString numbaVersion;
    QString version;
    QList<AcceleratorDeviceCheck> devices;
    bool configured = false;
    bool prepared = false;
    bool available = false;
    bool warning = false;
};

struct StartupCheckResult {
    QString repoRoot;
    QString pythonExecutable;
    QString pythonVersion;
    QString requirementsPath;
    QString hostPlatform;
    QString pythonSummary;
    QString processError;
    QString rawStdout;
    QString rawStderr;
    QList<PythonImportCheck> packageChecks;
    AcceleratorCheck cuda;
    AcceleratorCheck opencl;
    bool pythonResolved = false;
    bool requirementsResolved = false;

    bool hasWarnings() const;
    QString warningText() const;
};

StartupCheckResult runPythonEnvironmentSelfCheck(const QString& repoRoot, int timeoutMs = 120000);

}
