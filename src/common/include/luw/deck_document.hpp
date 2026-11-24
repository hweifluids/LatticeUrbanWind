#pragma once

#include <QHash>
#include <QString>
#include <QStringList>
#include <QVariant>
#include <QVector>

#include <cstdint>
#include <optional>
#include <utility>

namespace luw {

struct DeckEntry {
    QString key;
    QString value;
    QString sectionId;
    QString comment;
    bool known = true;
};

class DeckDocument final {
public:
    static DeckDocument fromText(const QString& text, bool strictDuplicates = false);
    static DeckDocument load(const QString& path, bool strictDuplicates = false);

    bool has(const QString& key) const;
    std::optional<QString> raw(const QString& key) const;
    QString rawOr(const QString& key, const QString& fallback = {}) const;
    std::optional<QString> text(const QString& key) const;
    QString textOr(const QString& key, const QString& fallback = {}) const;
    std::optional<std::int64_t> integer(const QString& key) const;
    std::optional<double> floating(const QString& key) const;
    std::optional<bool> boolean(const QString& key) const;
    QStringList list(const QString& key) const;
    QVector<double> floatList(const QString& key) const;
    std::optional<std::pair<double, double>> pair(const QString& key) const;

    void setRaw(const QString& key,
                const QString& value,
                const QString& sectionId = {},
                const std::optional<QString>& comment = std::nullopt);
    void setText(const QString& key,
                 const QString& value,
                 bool quoted = false,
                 const QString& sectionId = {},
                 const std::optional<QString>& comment = std::nullopt);
    void setInteger(const QString& key, std::int64_t value, const QString& sectionId = {});
    void setFloat(const QString& key, double value, int precision = 6, const QString& sectionId = {});
    void setBoolean(const QString& key, bool value, const QString& sectionId = {});
    void setList(const QString& key, const QVariantList& values, const QString& sectionId = {});
    void setPair(const QString& key,
                 const std::pair<double, double>& values,
                 int precision = 6,
                 const QString& sectionId = {});
    void remove(const QString& key);

    QString render() const;
    void save(const QString& path = {});
    QString path() const;
    QStringList duplicateKeys() const;
    QHash<QString, QString> values() const;

private:
    static qsizetype commentIndex(const QString& line);
    static QStringList parseListItems(const QString& raw);
    QStringList renderSection(const QString& sectionId) const;
    static QString renderEntry(const DeckEntry& entry);

    QString path_;
    QStringList preambleLines_;
    QHash<QString, QStringList> sectionLooseLines_;
    QHash<QString, DeckEntry> entries_;
    QHash<QString, QStringList> unknownOrder_;
    QHash<QString, QStringList> duplicates_;
};

} // namespace luw

