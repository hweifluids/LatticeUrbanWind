#include "luwgui/Preferences.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>

namespace luwgui {

namespace {

QString detectRepoRoot() {
    QDir dir(QCoreApplication::applicationDirPath());
    for (int depth = 0; depth < 8; ++depth) {
        if (dir.exists("core") && dir.exists("gui")) {
            return dir.absolutePath();
        }
        if (!dir.cdUp()) {
            break;
        }
    }
    return QDir::currentPath();
}

} // namespace

QString preferencesFilePath() {
    return QDir(detectRepoRoot()).filePath("gui/preferences.json");
}

AppPreferences loadPreferences(QString* warningMessage) {
    AppPreferences preferences;
    QFile file(preferencesFilePath());
    if (!file.exists()) {
        return preferences;
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (warningMessage) {
            *warningMessage = file.errorString();
        }
        return preferences;
    }

    const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
    if (!document.isObject()) {
        if (warningMessage) {
            *warningMessage = "Preference file is not a JSON object. Using defaults.";
        }
        return preferences;
    }

    const QJsonObject object = document.object();
    preferences.themeMode = themeModeFromString(object.value("theme_mode").toString("light"));
    return preferences;
}

bool savePreferences(const AppPreferences& preferences, QString* errorMessage) {
    const QString path = preferencesFilePath();
    QDir dir = QFileInfo(path).absoluteDir();
    if (!dir.exists() && !dir.mkpath(".")) {
        if (errorMessage) {
            *errorMessage = "Failed to create preferences directory.";
        }
        return false;
    }

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
        if (errorMessage) {
            *errorMessage = file.errorString();
        }
        return false;
    }

    QJsonObject object;
    object.insert("theme_mode", preferences.themeMode == ThemeMode::Light ? "light" : "dark");
    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    return true;
}

}
