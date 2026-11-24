#pragma once

#include <QString>

namespace luwgui {

QString detectRepoRoot();
QString findNativeExecutable(const QString& repoRoot);
QString findCulbmExecutable(const QString& repoRoot);
QString findFluidX3DExecutable(const QString& repoRoot);
QString resolveRepoFilePath(const QString& repoRoot, const QString& relativePath);

}
