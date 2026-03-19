#pragma once

#include "luwgui/Theme.h"

#include <QString>

namespace luwgui {

struct AppPreferences {
    ThemeMode themeMode = ThemeMode::Light;
};

QString preferencesFilePath();
AppPreferences loadPreferences(QString* warningMessage = nullptr);
bool savePreferences(const AppPreferences& preferences, QString* errorMessage = nullptr);

}
