#pragma once

#include <QString>

namespace luwgui {

QString detectRepoRoot();
QString findPythonExecutable(const QString& repoRoot);
QString findSolverExecutable(const QString& repoRoot);
QString resolveRepoFilePath(const QString& repoRoot, const QString& relativePath);

}
