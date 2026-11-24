#include "luwgui/ConfigSchema.h"
#include "luwgui/RuntimePaths.h"

#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>

#include <cmath>

namespace luwgui {

namespace {

QString sanitizeDeckKey(QString key) {
    key = key.trimmed().toLower();
    key.replace(QRegularExpression(R"([\s\-]+)"), "_");
    key.replace(QRegularExpression(R"(_+)"), "_");
    while (key.startsWith('_')) {
        key.remove(0, 1);
    }
    while (key.endsWith('_')) {
        key.chop(1);
    }
    return key;
}

FieldKind parseFieldKind(const QString& text) {
    const QString normalized = text.trimmed().toLower();
    if (normalized == "integer") {
        return FieldKind::Integer;
    }
    if (normalized == "float") {
        return FieldKind::Float;
    }
    if (normalized == "boolean") {
        return FieldKind::Boolean;
    }
    if (normalized == "enum") {
        return FieldKind::Enum;
    }
    if (normalized == "float_pair") {
        return FieldKind::FloatPair;
    }
    if (normalized == "float_triplet") {
        return FieldKind::FloatTriplet;
    }
    if (normalized == "uint_triplet") {
        return FieldKind::UIntTriplet;
    }
    if (normalized == "float_list") {
        return FieldKind::FloatList;
    }
    if (normalized == "token_list") {
        return FieldKind::TokenList;
    }
    if (normalized == "multiline") {
        return FieldKind::Multiline;
    }
    return FieldKind::String;
}

int parseModeMask(const QJsonValue& value) {
    if (!value.isArray()) {
        return ModeMaskAll;
    }

    int mask = 0;
    const QJsonArray modes = value.toArray();
    for (const QJsonValue& item : modes) {
        const QString mode = item.toString().trimmed().toLower();
        if (mode == "luw") {
            mask |= ModeMaskLuw;
        } else if (mode == "luwdg") {
            mask |= ModeMaskLuwdg;
        } else if (mode == "luwpf") {
            mask |= ModeMaskLuwpf;
        }
    }
    return mask == 0 ? ModeMaskAll : mask;
}

QStringList jsonStringList(const QJsonValue& value) {
    QStringList out;
    if (!value.isArray()) {
        return out;
    }
    const QJsonArray array = value.toArray();
    out.reserve(array.size());
    for (const QJsonValue& item : array) {
        out.push_back(item.toString());
    }
    return out;
}

struct SchemaData {
    QVector<SectionSpec> sections;
    QVector<FieldSpec> fields;
    QHash<QString, FieldSpec> canonicalFieldMap;
    QHash<QString, QString> keyAliasMap;
};

SchemaData loadSchemaData() {
    SchemaData data;
    const QString path = resolveRepoFilePath(detectRepoRoot(), "core/deck_schema.json");
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return data;
    }

    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    if (!doc.isObject()) {
        return data;
    }

    const QJsonObject root = doc.object();
    const QJsonArray sections = root.value("sections").toArray();
    data.sections.reserve(sections.size());
    for (const QJsonValue& value : sections) {
        const QJsonObject object = value.toObject();
        SectionSpec spec;
        spec.id = object.value("id").toString();
        spec.title = object.value("title").toString(spec.id);
        spec.description = object.value("description").toString();
        spec.aliases = jsonStringList(object.value("aliases"));
        data.sections.push_back(spec);
    }

    const QJsonArray fields = root.value("fields").toArray();
    data.fields.reserve(fields.size());
    for (const QJsonValue& value : fields) {
        const QJsonObject object = value.toObject();
        FieldSpec spec;
        spec.key = sanitizeDeckKey(object.value("key").toString());
        spec.label = object.value("label").toString(spec.key);
        spec.sectionId = object.value("section").toString("custom");
        spec.help = object.value("help").toString();
        spec.kind = parseFieldKind(object.value("kind").toString("string"));
        spec.enumValues = jsonStringList(object.value("enum_values"));
        spec.aliases = jsonStringList(object.value("aliases"));
        spec.modeMask = parseModeMask(object.value("modes"));
        spec.quoted = object.value("quoted").toBool(false);
        spec.readOnly = object.value("read_only").toBool(false);

        data.fields.push_back(spec);
        data.canonicalFieldMap.insert(spec.key, spec);
        data.keyAliasMap.insert(spec.key, spec.key);
        for (const QString& alias : spec.aliases) {
            data.keyAliasMap.insert(sanitizeDeckKey(alias), spec.key);
        }
    }

    return data;
}

const SchemaData& schemaData() {
    static const SchemaData data = loadSchemaData();
    return data;
}

} // namespace

QString runModeDisplayName(RunMode mode) {
    switch (mode) {
    case RunMode::Luw:
        return "LUW";
    case RunMode::Luwdg:
        return "LUWDG";
    case RunMode::Luwpf:
        return "LUWPF";
    }
    return "LUW";
}

QString runModeSuffix(RunMode mode) {
    switch (mode) {
    case RunMode::Luw:
        return ".luw";
    case RunMode::Luwdg:
        return ".luwdg";
    case RunMode::Luwpf:
        return ".luwpf";
    }
    return ".luw";
}

RunMode runModeFromPath(const QString& filePath) {
    const QString ext = QFileInfo(filePath).suffix().toLower();
    if (ext == "luwdg") {
        return RunMode::Luwdg;
    }
    if (ext == "luwpf") {
        return RunMode::Luwpf;
    }
    return RunMode::Luw;
}

QString defaultDeckName(RunMode mode) {
    return "conf" + runModeSuffix(mode);
}

const QVector<SectionSpec>& sectionSpecs() {
    return schemaData().sections;
}

const QVector<FieldSpec>& fieldSpecs() {
    return schemaData().fields;
}

const QHash<QString, FieldSpec>& fieldSpecMap() {
    return schemaData().canonicalFieldMap;
}

QString normalizeDeckKey(const QString& key) {
    const QString sanitized = sanitizeDeckKey(key);
    return schemaData().keyAliasMap.value(sanitized, sanitized);
}

const FieldSpec* findFieldSpec(const QString& key) {
    const QString canonical = normalizeDeckKey(key);
    auto it = fieldSpecMap().constFind(canonical);
    if (it == fieldSpecMap().cend()) {
        return nullptr;
    }
    return &it.value();
}

QVector<FieldSpec> fieldsForSection(const QString& sectionId, RunMode mode) {
    const int flag = (mode == RunMode::Luw) ? ModeMaskLuw : (mode == RunMode::Luwdg ? ModeMaskLuwdg : ModeMaskLuwpf);
    QVector<FieldSpec> out;
    for (const FieldSpec& spec : fieldSpecs()) {
        if (spec.sectionId == sectionId && (spec.modeMask & flag) != 0) {
            out.push_back(spec);
        }
    }
    return out;
}

QStringList knownKeys() {
    QStringList out;
    out.reserve(fieldSpecs().size());
    for (const FieldSpec& spec : fieldSpecs()) {
        out.push_back(spec.key);
    }
    out.sort();
    return out;
}

bool tryParseDeckBool(const QString& raw, bool* value) {
    QString normalized = raw.trimmed().toLower();
    if ((normalized.startsWith('"') && normalized.endsWith('"'))
        || (normalized.startsWith('\'') && normalized.endsWith('\''))) {
        normalized = normalized.mid(1, normalized.size() - 2).trimmed();
    }
    if (normalized.isEmpty()) {
        return false;
    }

    static const QStringList trueTokens = {
        "1", "true", "t", "yes", "y", "on", "enable", "enabled"
    };
    static const QStringList falseTokens = {
        "0", "false", "f", "no", "n", "off", "disable", "disabled"
    };

    bool parsed = false;
    if (trueTokens.contains(normalized)) {
        parsed = true;
    } else if (falseTokens.contains(normalized)) {
        parsed = false;
    } else {
        bool ok = false;
        const double numeric = normalized.toDouble(&ok);
        if (!ok || !std::isfinite(numeric)) {
            return false;
        }
        parsed = numeric != 0.0;
    }

    if (value) {
        *value = parsed;
    }
    return true;
}

} // namespace luwgui
