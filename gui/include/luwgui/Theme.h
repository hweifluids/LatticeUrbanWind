#pragma once

#include <QApplication>
#include <QString>
#include <QVector>

namespace luwgui {

enum class ThemeMode {
    LightDefault,
    DarkDefault,
    White,
    Black,
    Pink,
    Green,
    Siemens,
    Frieren,
    Himmel
};

enum class FontSizePreset {
    PerfectForGenZ,
    Small,
    Normal,
    Large
};

QVector<ThemeMode> availableThemeModes();
QString themeModeDisplayName(ThemeMode mode);
QString themeModeStorageKey(ThemeMode mode);
ThemeMode themeModeFromString(const QString& text);

QString fontSizePresetDisplayName(FontSizePreset preset);
QString fontSizePresetStorageKey(FontSizePreset preset);
FontSizePreset fontSizePresetFromString(const QString& text);
int interfaceFontPixelSize(FontSizePreset preset);
int consoleFontPixelSize(FontSizePreset preset);

void applyTheme(QApplication& app, ThemeMode mode, FontSizePreset fontSizePreset = FontSizePreset::Normal);

}
