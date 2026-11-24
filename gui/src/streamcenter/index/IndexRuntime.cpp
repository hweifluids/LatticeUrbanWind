#include "index/IndexRuntime.h"

#include <QDir>
#include <QFileInfo>
#include <QLibrary>
#include <QProcessEnvironment>
#include <QStringList>

#ifndef STREAMCENTERPLUS_ENABLE_NVIDIA_INDEX
#define STREAMCENTERPLUS_ENABLE_NVIDIA_INDEX 0
#endif

namespace Streamcenter::Index {
namespace {

QStringList runtimeCandidates() {
    QStringList candidates;
    const QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    const QStringList roots{
        env.value(QStringLiteral("STREAMCENTERPLUS_NVIDIA_INDEX_ROOT")),
        env.value(QStringLiteral("NVINDEX_ROOT"))
    };

    for (const QString& root : roots) {
        if (root.trimmed().isEmpty()) {
            continue;
        }
        const QDir dir(root);
#if defined(Q_OS_WIN)
        candidates << dir.absoluteFilePath(QStringLiteral("bin/nvindex.dll"))
                   << dir.absoluteFilePath(QStringLiteral("bin/libnvindex.dll"))
                   << dir.absoluteFilePath(QStringLiteral("lib/nvindex.dll"))
                   << dir.absoluteFilePath(QStringLiteral("lib/libnvindex.dll"));
#else
        candidates << dir.absoluteFilePath(QStringLiteral("lib/libnvindex.so"))
                   << dir.absoluteFilePath(QStringLiteral("lib64/libnvindex.so"));
#endif
    }

#if defined(Q_OS_WIN)
    candidates << QStringLiteral("nvindex") << QStringLiteral("libnvindex");
#else
    candidates << QStringLiteral("nvindex") << QStringLiteral("libnvindex.so");
#endif
    candidates.removeDuplicates();
    return candidates;
}

RuntimeStatus probeRuntime() {
#if STREAMCENTERPLUS_ENABLE_NVIDIA_INDEX
    QStringList errors;
    for (const QString& candidate : runtimeCandidates()) {
        QLibrary library(candidate);
        if (!library.load()) {
            errors << QStringLiteral("%1: %2").arg(candidate, library.errorString());
            continue;
        }
        if (library.resolve("nv_factory") == nullptr) {
            errors << QStringLiteral("%1: nv_factory entry point was not found").arg(candidate);
            library.unload();
            continue;
        }

        RuntimeStatus status;
        status.available = true;
        status.message = QStringLiteral("NVIDIA IndeX runtime was found: %1").arg(candidate);
        return status;
    }

    RuntimeStatus status;
    status.available = false;
    status.message = QStringLiteral("NVIDIA IndeX was enabled at build time, but libnvindex could not be loaded. %1")
                         .arg(errors.join(QStringLiteral("; ")));
    return status;
#else
    RuntimeStatus status;
    status.available = false;
    status.message = QStringLiteral("NVIDIA IndeX backend is not enabled in this build. Reconfigure GUI with STREAMCENTERPLUS_ENABLE_NVIDIA_INDEX=ON and provide STREAMCENTERPLUS_NVIDIA_INDEX_ROOT.");
    return status;
#endif
}

}  // namespace

RuntimeStatus IndexRuntime::status() {
    static const RuntimeStatus cached = probeRuntime();
    return cached;
}

}  // namespace Streamcenter::Index
