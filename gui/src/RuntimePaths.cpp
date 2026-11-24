#include "luwgui/RuntimePaths.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>

namespace luwgui {

namespace {

QString findCfdExecutable(const QString& repoRoot,
                          const QString& packageName,
                          const QString& executableBaseName) {
#ifdef Q_OS_WIN
    const QString executableName = executableBaseName + QStringLiteral(".exe");
#else
    const QString executableName = executableBaseName;
#endif
    const QStringList candidates = {
        QDir(QCoreApplication::applicationDirPath()).filePath(executableName),
        QDir(repoRoot).filePath(
            QStringLiteral("core/cfd_core/") + packageName
            + QStringLiteral("/bin/") + executableName)
    };

    for (const QString& candidate : candidates) {
        const QFileInfo info(QDir::cleanPath(candidate));
        if (info.isFile()) {
            return info.absoluteFilePath();
        }
    }
    return QStandardPaths::findExecutable(executableBaseName);
}

} // namespace

QString detectRepoRoot() {
    QDir dir(QCoreApplication::applicationDirPath());
    for (int depth = 0; depth < 8; ++depth) {
        if (dir.exists("core") && dir.exists("gui")) {
            return dir.absolutePath();
        }
        if (!dir.cdUp()) {
            break;
        }
    }
    return QDir::currentPath();
}

QString findNativeExecutable(const QString& repoRoot) {
#ifdef Q_OS_WIN
    const QString executableName = QStringLiteral("luw.exe");
#else
    const QString executableName = QStringLiteral("luw");
#endif

    QStringList candidates = {
        QDir(QCoreApplication::applicationDirPath()).filePath(executableName),
        QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("../bin/") + executableName),
        QDir(repoRoot).filePath(QStringLiteral("bin/") + executableName)
    };
    const QFileInfoList buildDirectories = QDir(repoRoot).entryInfoList(
        {QStringLiteral("build*")}, QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
    for (const QFileInfo& buildDirectory : buildDirectories) {
        const QDir build(buildDirectory.absoluteFilePath());
        candidates << build.filePath(QStringLiteral("bin/Release/") + executableName)
                   << build.filePath(QStringLiteral("bin/RelWithDebInfo/") + executableName)
                   << build.filePath(QStringLiteral("bin/Debug/") + executableName)
                   << build.filePath(QStringLiteral("bin/") + executableName);
    }

    for (const QString& candidate : candidates) {
        const QFileInfo info(QDir::cleanPath(candidate));
        if (info.isFile()) {
            return info.absoluteFilePath();
        }
    }
    return QStandardPaths::findExecutable(QStringLiteral("luw"));
}

QString findCulbmExecutable(const QString& repoRoot) {
    return findCfdExecutable(repoRoot, QStringLiteral("CULBM"), QStringLiteral("CULBM"));
}

QString findFluidX3DExecutable(const QString& repoRoot) {
    return findCfdExecutable(
        repoRoot, QStringLiteral("FluidX3D"), QStringLiteral("FluidX3D"));
}

QString resolveRepoFilePath(const QString& repoRoot, const QString& relativePath) {
    return QDir(repoRoot).filePath(relativePath);
}

}
