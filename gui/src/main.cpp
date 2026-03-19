#include "luwgui/MainWindow.h"
#include "luwgui/Preferences.h"
#include "luwgui/Theme.h"

#include <QApplication>
#include <QSurfaceFormat>

#include <QVTKOpenGLNativeWidget.h>

int main(int argc, char* argv[]) {
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    QApplication app(argc, argv);
    app.setApplicationName("LUW Studio");
    app.setOrganizationName("LatticeUrbanWind");

    const luwgui::AppPreferences preferences = luwgui::loadPreferences();
    luwgui::applyTheme(app, preferences.themeMode);

    luwgui::MainWindow window(preferences);
    window.show();
    return app.exec();
}
