#pragma once

#include <QApplication>
#include <QString>

namespace luwgui {

enum class ThemeMode {
    Dark,
    Light
};

QString themeModeDisplayName(ThemeMode mode);
ThemeMode themeModeFromString(const QString& text);

void applyTheme(QApplication& app, ThemeMode mode);

}
