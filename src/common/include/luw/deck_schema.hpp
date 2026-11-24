#pragma once

#include <QHash>
#include <QString>
#include <QStringList>
#include <QVector>

#include <optional>

namespace luw {

enum class FieldKind {
    String,
    Integer,
    Float,
    Boolean,
    Enum,
    FloatPair,
    FloatTriplet,
    UIntTriplet,
    FloatList,
    TokenList,
    Multiline,
};

struct SectionSpec {
    QString id;
    QString title;
    QString description;
    QStringList aliases;
};

struct FieldSpec {
    QString key;
    QString label;
    QString sectionId;
    QString help;
    FieldKind kind = FieldKind::String;
    QStringList enumValues;
    QStringList aliases;
    int modeMask = 0x7;
    bool quoted = false;
    bool readOnly = false;
};

class DeckSchema final {
public:
    static const DeckSchema& instance();

    const QVector<SectionSpec>& sections() const noexcept;
    const QVector<FieldSpec>& fields() const noexcept;
    const FieldSpec* field(const QString& key) const;
    QString normalizeKey(const QString& raw) const;
    QString sectionTitle(const QString& sectionId) const;
    QStringList fieldOrder(const QString& sectionId) const;
    QString matchSectionHeader(const QString& strippedLine) const;

private:
    DeckSchema();

    QVector<SectionSpec> sections_;
    QVector<FieldSpec> fields_;
    QHash<QString, int> fieldIndex_;
    QHash<QString, QString> aliases_;
    QHash<QString, int> sectionIndex_;
};

QString stripQuotes(const QString& raw);
QString sanitizeDeckKey(QString raw);
std::optional<double> parseDeckFloatToken(const QString& raw);
std::optional<bool> parseBoolToken(const QString& raw);

} // namespace luw
