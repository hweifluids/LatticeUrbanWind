#include "luwgui/Preferences.h"
#include "luwgui/RuntimePaths.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <algorithm>

namespace luwgui {

namespace {

int clampRecentProjectsLimit(int value) {
    return std::clamp(value, 1, kMaxRecentProjectsLimit);
}

QString normalizeRecentProjectPath(QString path) {
    path = path.trimmed();
    if (path.isEmpty()) {
        return {};
    }
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

Qt::CaseSensitivity projectPathCaseSensitivity() {
#ifdef Q_OS_WIN
    return Qt::CaseInsensitive;
#else
    return Qt::CaseSensitive;
#endif
}

bool sameProjectPath(const QString& lhs, const QString& rhs) {
    return lhs.compare(rhs, projectPathCaseSensitivity()) == 0;
}

QStringList normalizeRecentProjectFiles(const QStringList& rawFiles, int limit) {
    QStringList normalizedFiles;
    normalizedFiles.reserve(std::min(static_cast<int>(rawFiles.size()), limit));
    for (const QString& rawFile : rawFiles) {
        const QString normalizedPath = normalizeRecentProjectPath(rawFile);
        if (normalizedPath.isEmpty()) {
            continue;
        }

        bool duplicate = false;
        for (const QString& existingPath : normalizedFiles) {
            if (sameProjectPath(existingPath, normalizedPath)) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }

        normalizedFiles.push_back(normalizedPath);
        if (normalizedFiles.size() >= limit) {
            break;
        }
    }
    return normalizedFiles;
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
    preferences.themeMode = themeModeFromString(object.value("theme_mode").toString("light_default"));
    preferences.fontSizePreset = fontSizePresetFromString(object.value("font_size_preset").toString("normal"));
    preferences.defaultProjectLocation = object.value("default_project_location").toString().trimmed();
    preferences.recentProjectsLimit = clampRecentProjectsLimit(
        object.value("recent_project_limit").toInt(kDefaultRecentProjectsLimit));

    QStringList recentProjectFiles;
    const QJsonArray recentProjectArray = object.value("recent_project_files").toArray();
    recentProjectFiles.reserve(recentProjectArray.size());
    for (const QJsonValue& value : recentProjectArray) {
        if (value.isString()) {
            recentProjectFiles.push_back(value.toString());
        }
    }
    preferences.recentProjectFiles = normalizeRecentProjectFiles(
        recentProjectFiles,
        preferences.recentProjectsLimit);
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
    object.insert("theme_mode", themeModeStorageKey(preferences.themeMode));
    object.insert("font_size_preset", fontSizePresetStorageKey(preferences.fontSizePreset));
    object.insert("default_project_location", preferences.defaultProjectLocation.trimmed());
    const int recentProjectsLimit = clampRecentProjectsLimit(preferences.recentProjectsLimit);
    object.insert("recent_project_limit", recentProjectsLimit);

    QJsonArray recentProjectArray;
    for (const QString& filePath : normalizeRecentProjectFiles(preferences.recentProjectFiles, recentProjectsLimit)) {
        recentProjectArray.push_back(filePath);
    }
    object.insert("recent_project_files", recentProjectArray);
    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    return true;
}

}
