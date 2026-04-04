#pragma once

#include "luwgui/ConfigSchema.h"

#include <QHash>
#include <QObject>
#include <QString>
#include <QVariant>
#include <QVector>

class QFileSystemWatcher;
class QTimer;

namespace luwgui {

class ConfigDocument : public QObject {
    Q_OBJECT

public:
    explicit ConfigDocument(QObject* parent = nullptr);

    bool loadFromFile(const QString& filePath, QString* errorMessage = nullptr);
    bool reloadFromDisk(QString* errorMessage = nullptr);
    bool save(QString* errorMessage = nullptr);
    bool saveAs(const QString& filePath, QString* errorMessage = nullptr);
    bool applyRawText(const QString& text, QString* errorMessage = nullptr);
    void clearFilePath();

    QString filePath() const;
    QString projectDirectory() const;
    QString repoRoot() const;
    QString resultsDirectory() const;
    QString rawText() const;
    QString renderedText() const;

    RunMode mode() const;
    void setMode(RunMode mode);

    bool hasKey(const QString& key) const;
    QString rawValue(const QString& key) const;
    QVariant typedValue(const QString& key) const;

    void setRawValue(const QString& key, const QString& value);
    void setTypedValue(const QString& key, const QVariant& value);

    QHash<QString, QString> rawValues() const;
    QStringList unknownKeys() const;

signals:
    void documentReloaded();
    void changed();
    void keyChanged(const QString& key);
    void modeChanged(luwgui::RunMode mode);
    void externalFileReloaded(const QString& filePath);
    void externalFileReloadFailed(const QString& filePath, const QString& error);

private:
    struct DeckEntry {
        QString key;
        QString value;
        QString sectionId;
        QString comment;
        bool known = false;
    };

    static int commentIndex(const QString& line);
    static QString normalizeText(const QString& text);
    static QString normalizeKey(const QString& key);
    static QVariant parseValue(const FieldSpec& spec, const QString& rawValue);
    static QString serializeValue(const FieldSpec& spec, const QVariant& value);
    static QString renderEntry(const DeckEntry& entry);

    bool readTextFile(const QString& filePath, QString* text, QString* errorMessage) const;
    void rewriteCanonicalFileIfNeeded(const QString& originalText);
    void syncWatchPaths();
    void scheduleExternalReload();
    bool parseDocument(const QString& text, bool strictDuplicates, QString* errorMessage);
    void setTextInternal(const QString& text);
    QStringList renderSection(const QString& sectionId) const;
    void rememberUnknownKey(const QString& sectionId, const QString& key);
    void forgetUnknownKey(const QString& key);

    QString filePath_;
    QString repoRoot_;
    QString rawText_;
    RunMode mode_ = RunMode::Luw;
    QStringList preambleLines_;
    QHash<QString, QStringList> sectionLooseLines_;
    QHash<QString, DeckEntry> entries_;
    QHash<QString, QStringList> unknownOrder_;
    QStringList duplicateKeys_;
    QHash<QString, QString> values_;
    QFileSystemWatcher* watcher_ = nullptr;
    QTimer* externalReloadTimer_ = nullptr;
};

}
