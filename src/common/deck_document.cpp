#include "luw/deck_document.hpp"

#include "luw/deck_schema.hpp"

#include <QDir>
#include <QByteArray>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QTextStream>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace luw {

namespace {

QString formatScalar(const QVariant& value) {
    if (value.metaType().id() == QMetaType::Bool) {
        return value.toBool() ? QStringLiteral("true") : QStringLiteral("false");
    }
    switch (value.metaType().id()) {
    case QMetaType::Int:
    case QMetaType::UInt:
    case QMetaType::LongLong:
    case QMetaType::ULongLong:
        return value.toString();
    case QMetaType::Float:
    case QMetaType::Double:
        return QString::number(value.toDouble(), 'f', 6);
    default:
        return value.toString().trimmed();
    }
}

QString normalizeNewlines(QString text) {
    text.replace(QStringLiteral("\r\n"), QStringLiteral("\n"));
    text.replace('\r', '\n');
    return text;
}

bool isUtf8Continuation(unsigned char value) {
    return (value & 0xc0U) == 0x80U;
}

QString decodeUtf8IgnoringErrors(const QByteArray& source) {
    QByteArray valid;
    valid.reserve(source.size());
    qsizetype index = 0;
    while (index < source.size()) {
        const auto first = static_cast<unsigned char>(source.at(index));
        qsizetype length = 0;
        if (first <= 0x7fU) {
            length = 1;
        } else if (first >= 0xc2U && first <= 0xdfU
                   && index + 1 < source.size()
                   && isUtf8Continuation(static_cast<unsigned char>(source.at(index + 1)))) {
            length = 2;
        } else if (first >= 0xe0U && first <= 0xefU && index + 2 < source.size()) {
            const auto second = static_cast<unsigned char>(source.at(index + 1));
            const auto third = static_cast<unsigned char>(source.at(index + 2));
            const bool validSecond = (first == 0xe0U && second >= 0xa0U && second <= 0xbfU)
                || (first == 0xedU && second >= 0x80U && second <= 0x9fU)
                || (((first >= 0xe1U && first <= 0xecU) || (first >= 0xeeU && first <= 0xefU))
                    && isUtf8Continuation(second));
            if (validSecond && isUtf8Continuation(third)) {
                length = 3;
            }
        } else if (first >= 0xf0U && first <= 0xf4U && index + 3 < source.size()) {
            const auto second = static_cast<unsigned char>(source.at(index + 1));
            const auto third = static_cast<unsigned char>(source.at(index + 2));
            const auto fourth = static_cast<unsigned char>(source.at(index + 3));
            const bool validSecond = (first == 0xf0U && second >= 0x90U && second <= 0xbfU)
                || (first == 0xf4U && second >= 0x80U && second <= 0x8fU)
                || (first >= 0xf1U && first <= 0xf3U && isUtf8Continuation(second));
            if (validSecond && isUtf8Continuation(third) && isUtf8Continuation(fourth)) {
                length = 4;
            }
        }

        if (length == 0) {
            ++index;
            continue;
        }
        valid.append(source.constData() + index, length);
        index += length;
    }
    return QString::fromUtf8(valid);
}

std::optional<std::int64_t> parseDeckIntegerToken(const QString& raw) {
    const QString text = stripQuotes(raw).trimmed();
    static const QRegularExpression integer(
        QStringLiteral(R"(^[+-]?[0-9](?:_?[0-9])*$)"));
    if (!integer.match(text).hasMatch()) return std::nullopt;
    QString cleaned = text;
    cleaned.remove('_');
    bool ok = false;
    const qlonglong value = cleaned.toLongLong(&ok, 10);
    return ok ? std::optional<std::int64_t>(value) : std::nullopt;
}

} // namespace

DeckDocument DeckDocument::fromText(const QString& source, bool strictDuplicates) {
    DeckDocument document;
    const DeckSchema& schema = DeckSchema::instance();
    const QString normalized = normalizeNewlines(source);
    QString currentSection;
    bool sawStructuredContent = false;

    const QStringList lines = normalized.split('\n', Qt::KeepEmptyParts);
    for (const QString& line : lines) {
        const QString stripped = line.trimmed();
        const QString headerSection = schema.matchSectionHeader(stripped);
        if (!headerSection.isEmpty()) {
            currentSection = headerSection;
            sawStructuredContent = true;
            continue;
        }

        const qsizetype index = commentIndex(line);
        const QString content = index < 0 ? line : line.left(index);
        const QString comment = index < 0 ? QString() : line.mid(index).trimmed();
        const qsizetype equals = content.indexOf('=');
        if (equals >= 0) {
            const QString key = schema.normalizeKey(content.left(equals));
            if (key.isEmpty()) continue;
            const QString value = content.mid(equals + 1).trimmed();
            const FieldSpec* field = schema.field(key);
            const QString sectionId = field
                ? field->sectionId
                : (currentSection.isEmpty() ? QStringLiteral("custom") : currentSection);
            if (document.entries_.contains(key)) {
                QStringList& duplicates = document.duplicates_[key];
                if (duplicates.isEmpty()) duplicates.push_back(document.entries_.value(key).value);
                duplicates.push_back(value);
            }
            document.entries_.insert(key, DeckEntry{key, value, sectionId, comment, field != nullptr});
            if (!field) {
                QStringList& order = document.unknownOrder_[sectionId];
                if (!order.contains(key)) order.push_back(key);
            }
            sawStructuredContent = true;
            continue;
        }

        if (stripped.isEmpty()) {
            if (!sawStructuredContent && currentSection.isEmpty()) {
                document.preambleLines_.push_back(QString());
            }
            continue;
        }
        if (!sawStructuredContent && currentSection.isEmpty()) {
            document.preambleLines_.push_back(line);
        } else {
            const QString target = currentSection.isEmpty() ? QStringLiteral("custom") : currentSection;
            document.sectionLooseLines_[target].push_back(line);
        }
    }

    if (strictDuplicates && !document.duplicates_.isEmpty()) {
        QStringList keys = document.duplicates_.keys();
        keys.sort();
        throw std::runtime_error(
            ("Duplicate deck keys are not allowed: " + keys.join(QStringLiteral(", "))).toStdString());
    }
    return document;
}

DeckDocument DeckDocument::load(const QString& path, bool strictDuplicates) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error(("Cannot read deck: " + QFileInfo(path).absoluteFilePath()).toStdString());
    }
    const QString text = decodeUtf8IgnoringErrors(file.readAll());
    DeckDocument document = fromText(text, strictDuplicates);
    document.path_ = QFileInfo(path).absoluteFilePath();
    return document;
}

bool DeckDocument::has(const QString& key) const {
    return entries_.contains(DeckSchema::instance().normalizeKey(key));
}

std::optional<QString> DeckDocument::raw(const QString& key) const {
    const auto it = entries_.constFind(DeckSchema::instance().normalizeKey(key));
    if (it == entries_.cend()) return std::nullopt;
    return it->value;
}

QString DeckDocument::rawOr(const QString& key, const QString& fallback) const {
    const auto value = raw(key);
    return value ? *value : fallback;
}

std::optional<QString> DeckDocument::text(const QString& key) const {
    const auto value = raw(key);
    if (!value) return std::nullopt;
    return stripQuotes(*value);
}

QString DeckDocument::textOr(const QString& key, const QString& fallback) const {
    const auto value = text(key);
    return value ? *value : fallback;
}

std::optional<std::int64_t> DeckDocument::integer(const QString& key) const {
    const auto value = text(key);
    if (!value || value->isEmpty()) return std::nullopt;
    return parseDeckIntegerToken(*value);
}

std::optional<double> DeckDocument::floating(const QString& key) const {
    const auto value = text(key);
    if (!value || value->isEmpty()) return std::nullopt;
    const auto parsed = parseDeckFloatToken(*value);
    if (!parsed || std::isnan(*parsed)) return std::nullopt;
    return parsed;
}

std::optional<bool> DeckDocument::boolean(const QString& key) const {
    const auto value = text(key);
    return value ? parseBoolToken(*value) : std::nullopt;
}

QStringList DeckDocument::list(const QString& key) const {
    const auto value = raw(key);
    return value ? parseListItems(*value) : QStringList{};
}

QVector<double> DeckDocument::floatList(const QString& key) const {
    QVector<double> result;
    for (const QString& item : list(key)) {
        const auto value = parseDeckFloatToken(item);
        if (!value) return {};
        result.push_back(*value);
    }
    return result;
}

std::optional<std::pair<double, double>> DeckDocument::pair(const QString& key) const {
    const QVector<double> values = floatList(key);
    if (values.size() != 2) return std::nullopt;
    return std::minmax(values.at(0), values.at(1));
}

void DeckDocument::setRaw(const QString& rawKey,
                          const QString& value,
                          const QString& requestedSection,
                          const std::optional<QString>& comment) {
    const DeckSchema& schema = DeckSchema::instance();
    const QString key = schema.normalizeKey(rawKey);
    const auto existing = entries_.constFind(key);
    const FieldSpec* field = schema.field(key);
    QString sectionId = requestedSection;
    if (sectionId.isEmpty() && field) sectionId = field->sectionId;
    if (sectionId.isEmpty() && existing != entries_.cend()) sectionId = existing->sectionId;
    if (sectionId.isEmpty()) sectionId = QStringLiteral("custom");
    const QString renderedComment = comment
        ? comment->trimmed()
        : (existing == entries_.cend() ? QString() : existing->comment);
    entries_.insert(key, DeckEntry{key, value.trimmed(), sectionId, renderedComment, field != nullptr});
    duplicates_.remove(key);
    if (!field) {
        QStringList& order = unknownOrder_[sectionId];
        if (!order.contains(key)) order.push_back(key);
    }
}

void DeckDocument::setText(const QString& key,
                           const QString& value,
                           bool quoted,
                           const QString& sectionId,
                           const std::optional<QString>& comment) {
    const QString text = value.trimmed();
    setRaw(key, quoted ? QStringLiteral("\"") + text + QStringLiteral("\"") : text,
           sectionId, comment);
}

void DeckDocument::setInteger(const QString& key, std::int64_t value, const QString& sectionId) {
    setRaw(key, QString::number(value), sectionId);
}

void DeckDocument::setFloat(const QString& key, double value, int precision, const QString& sectionId) {
    setRaw(key, QString::number(value, 'f', precision), sectionId);
}

void DeckDocument::setBoolean(const QString& key, bool value, const QString& sectionId) {
    setRaw(key, value ? QStringLiteral("true") : QStringLiteral("false"), sectionId);
}

void DeckDocument::setList(const QString& key, const QVariantList& values, const QString& sectionId) {
    QStringList parts;
    parts.reserve(values.size());
    for (const QVariant& value : values) parts.push_back(formatScalar(value));
    setRaw(key, QStringLiteral("[") + parts.join(QStringLiteral(", ")) + QStringLiteral("]"), sectionId);
}

void DeckDocument::setPair(const QString& key,
                           const std::pair<double, double>& values,
                           int precision,
                           const QString& sectionId) {
    setRaw(key,
           QStringLiteral("[") + QString::number(values.first, 'f', precision)
               + QStringLiteral(", ") + QString::number(values.second, 'f', precision)
               + QStringLiteral("]"),
           sectionId);
}

void DeckDocument::remove(const QString& key) {
    const QString normalized = DeckSchema::instance().normalizeKey(key);
    entries_.remove(normalized);
    duplicates_.remove(normalized);
}

QString DeckDocument::render() const {
    QStringList lines;
    if (!preambleLines_.isEmpty()) {
        lines.append(preambleLines_);
        while (!lines.isEmpty() && lines.back().isEmpty()) lines.removeLast();
        if (!lines.isEmpty()) lines.push_back(QString());
    } else {
        lines.push_back(QStringLiteral("// LUW deck"));
        lines.push_back(QString());
    }

    for (const SectionSpec& section : DeckSchema::instance().sections()) {
        const QStringList sectionLines = renderSection(section.id);
        if (sectionLines.isEmpty()) continue;
        lines.append(sectionLines);
        lines.push_back(QString());
    }
    while (!lines.isEmpty() && lines.back().isEmpty()) lines.removeLast();
    return lines.join('\n') + QStringLiteral("\n");
}

void DeckDocument::save(const QString& requestedPath) {
    const QString target = requestedPath.isEmpty() ? path_ : QFileInfo(requestedPath).absoluteFilePath();
    if (target.isEmpty()) throw std::runtime_error("No target path provided for deck save.");
    QFile file(target);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
        throw std::runtime_error(("Cannot write deck: " + target).toStdString());
    }
    QTextStream output(&file);
    output.setEncoding(QStringConverter::Utf8);
    output << render();
    output.flush();
    if (output.status() != QTextStream::Ok) {
        throw std::runtime_error(("Failed while writing deck: " + target).toStdString());
    }
    path_ = QFileInfo(target).absoluteFilePath();
}

QString DeckDocument::path() const { return path_; }

QStringList DeckDocument::duplicateKeys() const {
    QStringList keys = duplicates_.keys();
    keys.sort();
    return keys;
}

QHash<QString, QString> DeckDocument::values() const {
    QHash<QString, QString> result;
    for (auto it = entries_.cbegin(); it != entries_.cend(); ++it) result.insert(it.key(), it->value);
    return result;
}

qsizetype DeckDocument::commentIndex(const QString& line) {
    bool inSingle = false;
    bool inDouble = false;
    for (qsizetype index = 0; index + 1 < line.size(); ++index) {
        const QChar character = line.at(index);
        if (character == '\'' && !inDouble) {
            inSingle = !inSingle;
            continue;
        }
        if (character == '"' && !inSingle) {
            inDouble = !inDouble;
            continue;
        }
        if (!inSingle && !inDouble && character == '/' && line.at(index + 1) == '/') return index;
    }
    return -1;
}

QStringList DeckDocument::parseListItems(const QString& raw) {
    QString text = raw.trimmed();
    if (text.startsWith('[') && text.endsWith(']')) text = text.mid(1, text.size() - 2);
    if (text.trimmed().isEmpty()) return {};
    QStringList result;
    for (const QString& part : text.split(',', Qt::KeepEmptyParts)) {
        const QString item = part.trimmed();
        if (!item.isEmpty()) result.push_back(item);
    }
    return result;
}

QStringList DeckDocument::renderSection(const QString& sectionId) const {
    QStringList known;
    for (const QString& key : DeckSchema::instance().fieldOrder(sectionId)) {
        const auto it = entries_.constFind(key);
        if (it != entries_.cend() && it->sectionId == sectionId) known.push_back(key);
    }
    QStringList unknown;
    for (const QString& key : unknownOrder_.value(sectionId)) {
        const auto it = entries_.constFind(key);
        if (it != entries_.cend() && it->sectionId == sectionId) unknown.push_back(key);
    }
    QStringList loose;
    for (const QString& line : sectionLooseLines_.value(sectionId)) {
        if (!line.trimmed().isEmpty()) loose.push_back(line);
    }
    if (known.isEmpty() && unknown.isEmpty() && loose.isEmpty()) return {};
    QStringList lines{QStringLiteral("// ") + DeckSchema::instance().sectionTitle(sectionId)};
    lines.append(loose);
    for (const QString& key : known) lines.push_back(renderEntry(entries_.value(key)));
    for (const QString& key : unknown) lines.push_back(renderEntry(entries_.value(key)));
    return lines;
}

QString DeckDocument::renderEntry(const DeckEntry& entry) {
    QString value = entry.value;
    const FieldSpec* spec = entry.known ? DeckSchema::instance().field(entry.key) : nullptr;
    if (spec) {
        if (spec->kind == FieldKind::Boolean) {
            if (const auto parsed = parseBoolToken(value)) value = *parsed ? "true" : "false";
        } else if (spec->kind == FieldKind::FloatPair
                   || spec->kind == FieldKind::FloatTriplet
                   || spec->kind == FieldKind::UIntTriplet
                   || spec->kind == FieldKind::FloatList
                   || spec->kind == FieldKind::TokenList) {
            if (!value.trimmed().isEmpty()) {
                value = QStringLiteral("[") + parseListItems(value).join(QStringLiteral(", ")) + QStringLiteral("]");
            }
        } else if (spec->quoted && !value.trimmed().isEmpty()) {
            const QString trimmed = value.trimmed();
            const bool matchingQuotes = trimmed.size() >= 2
                && ((trimmed.front() == '"' && trimmed.back() == '"')
                    || (trimmed.front() == '\'' && trimmed.back() == '\''));
            value = QStringLiteral("\"")
                + (matchingQuotes ? trimmed.mid(1, trimmed.size() - 2).trimmed() : trimmed)
                + QStringLiteral("\"");
        }
    }
    QString line = entry.key + QStringLiteral(" =");
    if (!value.trimmed().isEmpty()) line += QStringLiteral(" ") + value.trimmed();
    if (!entry.comment.isEmpty()) line += QStringLiteral(" ") + entry.comment;
    return line.trimmed();
}

} // namespace luw
