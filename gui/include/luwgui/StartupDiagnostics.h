#pragma once

#include <QList>
#include <QString>
#include <QtCore/qtypes.h>

namespace luwgui {

struct AcceleratorDeviceCheck {
    QString name;
    QString vendor;
    QString type;
    QString version;
    QString driverVersion;
    QString computeCapability;
    int computeUnits = 0;
    int maxClockMHz = 0;
    quint64 globalMemoryBytes = 0;
};

struct AcceleratorCheck {
    QString summary;
    QString errorText;
    QString version;
    QList<AcceleratorDeviceCheck> devices;
    bool available = false;
    bool warning = false;
};

struct StartupCheckResult {
    QString repoRoot;
    QString nativeExecutable;
    QString nativeVersion;
    QString culbmExecutable;
    QString fluidX3DExecutable;
    QString schemaPath;
    QString hostPlatform;
    QString processError;
    QString rawStdout;
    QString rawStderr;
    AcceleratorCheck cuda;
    AcceleratorCheck opencl;
    bool nativeResolved = false;
    bool culbmResolved = false;
    bool fluidX3DResolved = false;
    bool schemaResolved = false;

    bool hasWarnings() const;
    QString warningText() const;
};

StartupCheckResult runNativeEnvironmentSelfCheck(const QString& repoRoot, int timeoutMs = 10000);

}
