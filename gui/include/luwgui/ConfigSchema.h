#pragma once

#include <QHash>
#include <QMetaType>
#include <QString>
#include <QStringList>
#include <QVector>

namespace luwgui {

enum class RunMode {
    Luw,
    Luwdg,
    Luwpf
};

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
    Multiline
};

constexpr int ModeMaskLuw = 1 << 0;
constexpr int ModeMaskLuwdg = 1 << 1;
constexpr int ModeMaskLuwpf = 1 << 2;
constexpr int ModeMaskAll = ModeMaskLuw | ModeMaskLuwdg | ModeMaskLuwpf;

struct SectionSpec {
    QString id;
    QString title;
    QString description;
};

struct FieldSpec {
    QString key;
    QString label;
    QString sectionId;
    QString help;
    FieldKind kind = FieldKind::String;
    QStringList enumValues;
    int modeMask = ModeMaskAll;
    bool quoted = false;
    bool readOnly = false;
};

QString runModeDisplayName(RunMode mode);
QString runModeSuffix(RunMode mode);
RunMode runModeFromPath(const QString& filePath);
QString defaultDeckName(RunMode mode);

const QVector<SectionSpec>& sectionSpecs();
const QVector<FieldSpec>& fieldSpecs();
const QHash<QString, FieldSpec>& fieldSpecMap();
const FieldSpec* findFieldSpec(const QString& key);
QVector<FieldSpec> fieldsForSection(const QString& sectionId, RunMode mode);
QStringList knownKeys();

}

Q_DECLARE_METATYPE(luwgui::RunMode)
