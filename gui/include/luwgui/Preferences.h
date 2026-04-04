#pragma once

#include "luwgui/Theme.h"

#include <QString>
#include <QStringList>

namespace luwgui {

inline constexpr int kDefaultRecentProjectsLimit = 10;
inline constexpr int kMaxRecentProjectsLimit = 25;

struct AppPreferences {
    ThemeMode themeMode = ThemeMode::LightDefault;
    FontSizePreset fontSizePreset = FontSizePreset::Normal;
    QString defaultProjectLocation;
    int recentProjectsLimit = kDefaultRecentProjectsLimit;
    QStringList recentProjectFiles;
};

QString preferencesFilePath();
AppPreferences loadPreferences(QString* warningMessage = nullptr);
bool savePreferences(const AppPreferences& preferences, QString* errorMessage = nullptr);

}
