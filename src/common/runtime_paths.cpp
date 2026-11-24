#include "luw/runtime_paths.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QProcessEnvironment>
#include <QtGlobal>

namespace luw {

QString executableDirectory() {
    if (QCoreApplication::instance()) {
        return QCoreApplication::applicationDirPath();
    }
    return QDir::currentPath();
}

namespace {

bool looksLikeInstallRoot(const QDir& directory) {
    return directory.exists("core/deck_schema.json")
        && (directory.exists("core/cfd_core/CULBM")
            || directory.exists("core/cfd_core/FluidX3D"));
}

} // namespace

QString detectInstallRoot() {
    const QString configured = QProcessEnvironment::systemEnvironment()
                                   .value(QStringLiteral("LUW_HOME"))
                                   .trimmed();
    if (!configured.isEmpty()) {
        const QDir directory(configured);
        if (looksLikeInstallRoot(directory)) {
            return directory.absolutePath();
        }
    }

    QDir directory(executableDirectory());
    for (int depth = 0; depth < 10; ++depth) {
        if (looksLikeInstallRoot(directory)) {
            return directory.absolutePath();
        }
        if (!directory.cdUp()) {
            break;
        }
    }

#ifdef LUW_SOURCE_ROOT
    const QDir sourceDirectory(QStringLiteral(LUW_SOURCE_ROOT));
    if (looksLikeInstallRoot(sourceDirectory)) {
        return sourceDirectory.absolutePath();
    }
#endif
    return QDir::currentPath();
}

QString resolveInstallFile(const QString& relativePath) {
    return QDir(detectInstallRoot()).filePath(relativePath);
}

void configureBundledDataPaths() {
    const QDir installRoot(detectInstallRoot());
    const QString gdalData = installRoot.filePath(QStringLiteral("share/gdal"));
    if (!qEnvironmentVariableIsSet("GDAL_DATA")
        && QFileInfo(gdalData).isDir()) {
        qputenv("GDAL_DATA", QDir::toNativeSeparators(gdalData).toUtf8());
    }

    const QString projData = installRoot.filePath(QStringLiteral("share/proj"));
    if (QFileInfo(projData + QStringLiteral("/proj.db")).isFile()) {
        const QByteArray encodedPath = QDir::toNativeSeparators(projData).toUtf8();
        if (!qEnvironmentVariableIsSet("PROJ_DATA")) {
            qputenv("PROJ_DATA", encodedPath);
        }
        if (!qEnvironmentVariableIsSet("PROJ_LIB")) {
            qputenv("PROJ_LIB", encodedPath);
        }
    }
}

} // namespace luw
