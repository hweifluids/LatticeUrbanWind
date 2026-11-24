#include "luwgui/BuildInfo.h"
#include "luwgui/MainWindow.h"
#include "luwgui/Preferences.h"
#include "luwgui/RuntimePaths.h"
#include "luwgui/StartupDiagnostics.h"
#include "luwgui/StartupSplash.h"
#include "luwgui/Theme.h"

#include <QApplication>
#include <QDir>
#include <QGuiApplication>
#include <QLibraryInfo>
#include <QMessageBox>
#include <QSurfaceFormat>
#include <QTextStream>
#include <QTimer>

#include <QVTKOpenGLNativeWidget.h>

namespace {

QString startupAssetPath(const QString& repoRoot) {
    return luwgui::resolveRepoFilePath(repoRoot, QStringLiteral("gui/src/assets/img/startup_default.jpg"));
}

QString startupMessage(const char8_t* utf8Text) {
    return QString::fromUtf8(reinterpret_cast<const char*>(utf8Text));
}

} // namespace

int main(int argc, char* argv[]) {
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    QApplication app(argc, argv);
    app.setApplicationName("LatticeUrbanWind Studio");
    app.setApplicationVersion(QLatin1String(luwgui::buildinfo::kStudioVersion));
    app.setOrganizationName("LatticeUrbanWind");

    if (QCoreApplication::arguments().contains(
            QStringLiteral("--runtime-self-check"))) {
        QTextStream output(stdout);
        const QString pluginRoot =
            QLibraryInfo::path(QLibraryInfo::PluginsPath);
        if (!QDir(pluginRoot).exists(QStringLiteral("platforms"))) {
            output << "LatticeUrbanWind Studio runtime self-check: FAILED\n"
                   << "Qt platform plugin directory was not found: "
                   << QDir::toNativeSeparators(pluginRoot) << '\n';
            return 2;
        }
        output << "LatticeUrbanWind Studio runtime self-check: OK\n"
               << "Qt platform: " << QGuiApplication::platformName() << '\n'
               << "Qt plugins: " << QDir::toNativeSeparators(pluginRoot) << '\n';
        return 0;
    }

    const QString repoRoot = luwgui::detectRepoRoot();
    luwgui::StartupSplash splash(startupAssetPath(repoRoot));
    splash.showCentered();
    splash.setStatusMessage(startupMessage(u8"Locating the repository root, startup image asset, and current runtime directory..."));

    splash.setStatusMessage(startupMessage(u8"Resolving the native command runner, CFD solver, and deck schema..."));
    splash.setStatusMessage(startupMessage(u8"Validating the native preprocessing and post-processing runtime..."));
    const luwgui::StartupCheckResult startupCheck = luwgui::runNativeEnvironmentSelfCheck(repoRoot);

    splash.setStatusMessage(startupMessage(u8"Loading interface preferences, theme settings, and font presets..."));
    const luwgui::AppPreferences preferences = luwgui::loadPreferences();
    luwgui::applyTheme(app, preferences.themeMode, preferences.fontSizePreset);

    splash.setStatusMessage(startupMessage(u8"Building the main window, parameter editors, progress panel, console, and visualization widgets..."));
    luwgui::MainWindow window(preferences);

    splash.setStatusMessage(startupMessage(u8"Finalizing startup diagnostics and preparing the main workspace for display..."));
    window.show();
    splash.close();
    window.appendStartupReport(startupCheck);

    if (startupCheck.hasWarnings()) {
        QTimer::singleShot(0, &window, [startupCheck, &window] {
            QMessageBox::warning(&window,
                                 startupMessage(u8"Native Runtime Self-Check"),
                                 startupCheck.warningText());
        });
    }

    return app.exec();
}
