#include "luwgui/RuntimePaths.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>

namespace luwgui {

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

QString findPythonExecutable(const QString& repoRoot) {
    const QStringList candidates = {
        QDir(repoRoot).filePath(".venv/Scripts/python.exe"),
        QDir(repoRoot).filePath(".venv/bin/python"),
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

QString findSolverExecutable(const QString& repoRoot) {
    const QStringList candidates = {
        QDir(repoRoot).filePath("core/cfd_core/FluidX3D/bin/FluidX3D.exe"),
        QDir(repoRoot).filePath("core/cfd_core/FluidX3D/bin/FluidX3D"),
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

QString resolveRepoFilePath(const QString& repoRoot, const QString& relativePath) {
    return QDir(repoRoot).filePath(relativePath);
}

}
