#pragma once

#include <QString>

namespace luw {

QString executableDirectory();
QString detectInstallRoot();
QString resolveInstallFile(const QString& relativePath);
void configureBundledDataPaths();

} // namespace luw
