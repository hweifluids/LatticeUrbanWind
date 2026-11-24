#include "luw/deck_schema.hpp"
#include "luw/runtime_paths.hpp"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace luw {

namespace {

FieldKind parseFieldKind(const QString& raw) {
    const QString text = raw.trimmed().toLower();
    if (text == "integer") return FieldKind::Integer;
    if (text == "float") return FieldKind::Float;
    if (text == "boolean") return FieldKind::Boolean;
    if (text == "enum") return FieldKind::Enum;
    if (text == "float_pair") return FieldKind::FloatPair;
    if (text == "float_triplet") return FieldKind::FloatTriplet;
    if (text == "uint_triplet") return FieldKind::UIntTriplet;
    if (text == "float_list") return FieldKind::FloatList;
    if (text == "token_list") return FieldKind::TokenList;
    if (text == "multiline") return FieldKind::Multiline;
    return FieldKind::String;
}

QStringList jsonStringList(const QJsonValue& value) {
    QStringList result;
    for (const QJsonValue& item : value.toArray()) {
        result.push_back(item.toString());
    }
    return result;
}

int parseModeMask(const QJsonValue& value) {
    if (!value.isArray()) return 0x7;
    int mask = 0;
    for (const QJsonValue& item : value.toArray()) {
        const QString mode = item.toString().trimmed().toLower();
        if (mode == "luw") mask |= 0x1;
        else if (mode == "luwdg") mask |= 0x2;
        else if (mode == "luwpf") mask |= 0x4;
    }
    return mask == 0 ? 0x7 : mask;
}

QString normalizeSectionLabel(QString text) {
    text = text.trimmed().toLower();
    if (text.startsWith('[')) {
        const qsizetype closing = text.indexOf(']');
        if (closing >= 0) text = text.mid(1, closing - 1);
    }
    return text.simplified();
}

} // namespace

QString stripQuotes(const QString& raw) {
    const QString text = raw.trimmed();
    if (text.size() >= 2) {
        const QChar first = text.front();
        const QChar last = text.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            return text.mid(1, text.size() - 2).trimmed();
        }
    }
    return text;
}

QString sanitizeDeckKey(QString text) {
    static const QRegularExpression separators(QStringLiteral(R"([\s\-]+)"));
    static const QRegularExpression repeats(QStringLiteral(R"(_+)"));
    text = text.trimmed().toLower();
    text.replace(separators, QStringLiteral("_"));
    text.replace(repeats, QStringLiteral("_"));
    while (text.startsWith('_')) text.remove(0, 1);
    while (text.endsWith('_')) text.chop(1);
    return text;
}

std::optional<double> parseDeckFloatToken(const QString& raw) {
    const QString text = stripQuotes(raw).trimmed();
    if (text.isEmpty()) return std::nullopt;

    const QString lower = text.toLower();
    static const QRegularExpression special(
        QStringLiteral(R"(^([+-]?)(inf(?:inity)?|nan)$)"));
    const QRegularExpressionMatch specialMatch = special.match(lower);
    if (specialMatch.hasMatch()) {
        const bool negative = specialMatch.captured(1) == QStringLiteral("-");
        if (specialMatch.captured(2) == QStringLiteral("nan")) {
            const double value = std::numeric_limits<double>::quiet_NaN();
            return negative ? -value : value;
        }
        const double value = std::numeric_limits<double>::infinity();
        return negative ? -value : value;
    }

    static const QRegularExpression numeric(QStringLiteral(
        R"(^[+-]?(?:(?:[0-9](?:_?[0-9])*)(?:\.(?:[0-9](?:_?[0-9])*)?)?|\.(?:[0-9](?:_?[0-9])*))(?:[eE][+-]?[0-9](?:_?[0-9])*)?$)"));
    if (!numeric.match(text).hasMatch()) return std::nullopt;

    QString cleaned = text;
    cleaned.remove('_');
    const QByteArray bytes = cleaned.toLatin1();
    char* end = nullptr;
    const double value = std::strtod(bytes.constData(), &end);
    if (end != bytes.constData() + bytes.size()) return std::nullopt;
    return value;
}

std::optional<bool> parseBoolToken(const QString& raw) {
    const QString text = stripQuotes(raw).trimmed().toLower();
    if (text.isEmpty()) return std::nullopt;
    static const QStringList trueTokens = {
        "1", "true", "t", "yes", "y", "on", "enable", "enabled"
    };
    static const QStringList falseTokens = {
        "0", "false", "f", "no", "n", "off", "disable", "disabled"
    };
    if (trueTokens.contains(text)) return true;
    if (falseTokens.contains(text)) return false;
    const auto value = parseDeckFloatToken(text);
    if (!value || !std::isfinite(*value)) return std::nullopt;
    return *value != 0.0;
}

const DeckSchema& DeckSchema::instance() {
    static const DeckSchema schema;
    return schema;
}

DeckSchema::DeckSchema() {
    const QString path = resolveInstallFile(QStringLiteral("core/deck_schema.json"));
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        throw std::runtime_error(("Cannot open deck schema: " + path).toStdString());
    }
    QJsonParseError error{};
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError || !document.isObject()) {
        throw std::runtime_error(("Invalid deck schema: " + error.errorString()).toStdString());
    }

    const QJsonObject root = document.object();
    for (const QJsonValue& value : root.value(QStringLiteral("sections")).toArray()) {
        const QJsonObject object = value.toObject();
        SectionSpec spec;
        spec.id = object.value(QStringLiteral("id")).toString();
        spec.title = object.value(QStringLiteral("title")).toString(spec.id);
        spec.description = object.value(QStringLiteral("description")).toString();
        spec.aliases = jsonStringList(object.value(QStringLiteral("aliases")));
        sectionIndex_.insert(spec.id, sections_.size());
        sections_.push_back(std::move(spec));
    }

    for (const QJsonValue& value : root.value(QStringLiteral("fields")).toArray()) {
        const QJsonObject object = value.toObject();
        FieldSpec spec;
        spec.key = sanitizeDeckKey(object.value(QStringLiteral("key")).toString());
        spec.label = object.value(QStringLiteral("label")).toString(spec.key);
        spec.sectionId = object.value(QStringLiteral("section")).toString(QStringLiteral("custom"));
        spec.help = object.value(QStringLiteral("help")).toString();
        spec.kind = parseFieldKind(object.value(QStringLiteral("kind")).toString());
        spec.enumValues = jsonStringList(object.value(QStringLiteral("enum_values")));
        spec.aliases = jsonStringList(object.value(QStringLiteral("aliases")));
        spec.modeMask = parseModeMask(object.value(QStringLiteral("modes")));
        spec.quoted = object.value(QStringLiteral("quoted")).toBool(false);
        spec.readOnly = object.value(QStringLiteral("read_only")).toBool(false);

        const int index = fields_.size();
        fieldIndex_.insert(spec.key, index);
        aliases_.insert(spec.key, spec.key);
        for (const QString& alias : spec.aliases) {
            aliases_.insert(sanitizeDeckKey(alias), spec.key);
        }
        fields_.push_back(std::move(spec));
    }
}

const QVector<SectionSpec>& DeckSchema::sections() const noexcept { return sections_; }
const QVector<FieldSpec>& DeckSchema::fields() const noexcept { return fields_; }

const FieldSpec* DeckSchema::field(const QString& key) const {
    const auto it = fieldIndex_.constFind(normalizeKey(key));
    return it == fieldIndex_.cend() ? nullptr : &fields_.at(it.value());
}

QString DeckSchema::normalizeKey(const QString& raw) const {
    const QString sanitized = sanitizeDeckKey(raw);
    return aliases_.value(sanitized, sanitized);
}

QString DeckSchema::sectionTitle(const QString& sectionId) const {
    const auto it = sectionIndex_.constFind(sectionId);
    return it == sectionIndex_.cend() ? sectionId : sections_.at(it.value()).title;
}

QStringList DeckSchema::fieldOrder(const QString& sectionId) const {
    QStringList result;
    for (const FieldSpec& spec : fields_) {
        if (spec.sectionId == sectionId) result.push_back(spec.key);
    }
    return result;
}

QString DeckSchema::matchSectionHeader(const QString& strippedLine) const {
    if (strippedLine.isEmpty()
        || (!strippedLine.startsWith(QStringLiteral("//")) && !strippedLine.startsWith('#'))) {
        return {};
    }
    const QString label = strippedLine.startsWith(QStringLiteral("//"))
        ? strippedLine.mid(2).trimmed()
        : strippedLine.mid(1).trimmed();
    const QString normalized = normalizeSectionLabel(label);
    for (const SectionSpec& spec : sections_) {
        if (normalized == normalizeSectionLabel(spec.id)
            || normalized == normalizeSectionLabel(spec.title)) {
            return spec.id;
        }
        for (const QString& alias : spec.aliases) {
            if (normalized == normalizeSectionLabel(alias)) return spec.id;
        }
    }
    return {};
}

} // namespace luw
