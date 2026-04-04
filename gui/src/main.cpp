#include "luwgui/MainWindow.h"
#include "luwgui/Preferences.h"
#include "luwgui/RuntimePaths.h"
#include "luwgui/StartupDiagnostics.h"
#include "luwgui/StartupSplash.h"
#include "luwgui/Theme.h"

#include <QApplication>
#include <QMessageBox>
#include <QSurfaceFormat>
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
    app.setOrganizationName("LatticeUrbanWind");

    const QString repoRoot = luwgui::detectRepoRoot();
    luwgui::StartupSplash splash(startupAssetPath(repoRoot));
    splash.showCentered();
    splash.setStatusMessage(startupMessage(u8"Locating the repository root, startup image asset, and current runtime directory..."));

    splash.setStatusMessage(startupMessage(u8"Resolving the Python interpreter, virtual environment candidates, and backend script entry points..."));
    splash.setStatusMessage(startupMessage(u8"Importing and validating Python packages for post-processing, geometry preparation, and data analysis; first startup may take a little longer..."));
    const luwgui::StartupCheckResult startupCheck = luwgui::runPythonEnvironmentSelfCheck(repoRoot);

    splash.setStatusMessage(startupMessage(u8"Loading interface preferences, theme settings, and font presets..."));
    const luwgui::AppPreferences preferences = luwgui::loadPreferences();
    luwgui::applyTheme(app, preferences.themeMode, preferences.fontSizePreset);

    splash.setStatusMessage(startupMessage(u8"Building the main window, parameter editors, progress panel, console, and visualization widgets..."));
    luwgui::MainWindow window(preferences);

    splash.setStatusMessage(startupMessage(u8"Finalizing startup diagnostics and preparing the main workspace for display..."));
    window.show();
    splash.close();

    if (startupCheck.hasWarnings()) {
        QTimer::singleShot(0, &window, [startupCheck, &window] {
            QMessageBox::warning(&window,
                                 startupMessage(u8"Python Environment Self-Check"),
                                 startupCheck.warningText());
        });
    }

    return app.exec();
}
