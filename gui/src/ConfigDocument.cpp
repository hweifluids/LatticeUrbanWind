#include "luwgui/ConfigDocument.h"
#include "luwgui/RuntimePaths.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFileSystemWatcher>
#include <QRegularExpression>
#include <QStringBuilder>
#include <QTextStream>
#include <QTimer>

namespace luwgui {

namespace {

QString normalizeTextForCompare(QString text) {
    text.replace("\r\n", "\n");
    text.replace('\r', '\n');
    if (!text.isEmpty() && !text.endsWith('\n')) {
        text += '\n';
    }
    return text;
}

QString trimQuotes(const QString& raw) {
    const QString s = raw.trimmed();
    if (s.size() >= 2) {
        const QChar first = s.front();
        const QChar last = s.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            return s.mid(1, s.size() - 2).trimmed();
        }
    }
    return s;
}

QStringList splitList(const QString& raw) {
    QString s = raw.trimmed();
    if (s.startsWith('[') && s.endsWith(']')) {
        s = s.mid(1, s.size() - 2);
    }
    const QStringList parts = s.split(',', Qt::SkipEmptyParts);
    QStringList out;
    out.reserve(parts.size());
    for (const QString& part : parts) {
        out.push_back(part.trimmed());
    }
    return out;
}

QVariantList parseNumberList(const QString& raw, bool integerMode, int exactCount = 0) {
    QVariantList out;
    const QStringList parts = splitList(raw);
    if (exactCount > 0 && parts.size() != exactCount) {
        return {};
    }
    for (const QString& part : parts) {
        bool ok = false;
        if (integerMode) {
            const int value = part.toInt(&ok);
            if (!ok) {
                return {};
            }
            out.push_back(value);
        } else {
            const double value = part.toDouble(&ok);
            if (!ok) {
                return {};
            }
            out.push_back(value);
        }
    }
    return out;
}

QString renderList(const QVariantList& values, bool integerMode) {
    QStringList parts;
    parts.reserve(values.size());
    for (const QVariant& value : values) {
        if (integerMode) {
            parts.push_back(QString::number(value.toInt()));
        } else {
            parts.push_back(QString::number(value.toDouble(), 'f', 6));
        }
    }
    return "[" % parts.join(",   ") % "]";
}

QString normalizeSectionLabel(QString text) {
    text = text.trimmed().toLower();
    if (text.startsWith('[')) {
        const int endIndex = text.indexOf(']');
        if (endIndex > 0) {
            text = text.mid(1, endIndex - 1);
        }
    }
    return text.simplified();
}

QStringList sectionOrder() {
    return {
        "project",
        "domain",
        "generated",
        "cfd",
        "output",
        "physics",
        "vk",
        "batch",
        "custom",
    };
}

QString sectionTitle(const QString& sectionId) {
    if (sectionId == "project") {
        return "Project";
    }
    if (sectionId == "domain") {
        return "Domain";
    }
    if (sectionId == "generated") {
        return "Generated";
    }
    if (sectionId == "cfd") {
        return "CFD Controls";
    }
    if (sectionId == "output") {
        return "Output & Probes";
    }
    if (sectionId == "physics") {
        return "Physics";
    }
    if (sectionId == "vk") {
        return "Turbulence inflow";
    }
    if (sectionId == "batch") {
        return "Batch";
    }
    if (sectionId == "custom") {
        return "Custom";
    }
    return sectionId;
}

QStringList sectionAliases(const QString& sectionId) {
    if (sectionId == "project") {
        return {"project", "project info", "case"};
    }
    if (sectionId == "domain") {
        return {"domain", "projected si range after rotation", "wrf data range in lon/lat"};
    }
    if (sectionId == "generated") {
        return {"generated", "generated info", "volume-mean uvw and downstream boundary with yaw angle"};
    }
    if (sectionId == "cfd") {
        return {"cfd control", "cfd controls"};
    }
    if (sectionId == "output") {
        return {"output", "output and probes", "output & probes"};
    }
    if (sectionId == "physics") {
        return {"physics"};
    }
    if (sectionId == "vk") {
        return {"turbulence inflow", "vk inlet", "von karman inlet"};
    }
    if (sectionId == "batch") {
        return {"batch", "batch modes", "dataset generation", "inflow directions"};
    }
    if (sectionId == "custom") {
        return {"custom"};
    }
    return {};
}

QString orderedSectionForKey(const QString& key) {
    const FieldSpec* spec = findFieldSpec(key);
    if (spec) {
        return spec->sectionId;
    }
    return "custom";
}

QString matchSectionHeader(const QString& strippedLine) {
    if (strippedLine.isEmpty()) {
        return {};
    }
    if (!strippedLine.startsWith("//") && !strippedLine.startsWith('#')) {
        return {};
    }

    QString label = strippedLine.startsWith("//")
        ? strippedLine.mid(2).trimmed()
        : strippedLine.mid(1).trimmed();
    const QString normalized = normalizeSectionLabel(label);

    for (const QString& sectionId : sectionOrder()) {
        if (normalized == normalizeSectionLabel(sectionId)) {
            return sectionId;
        }
        if (normalized == normalizeSectionLabel(sectionTitle(sectionId))) {
            return sectionId;
        }
        for (const QString& alias : sectionAliases(sectionId)) {
            if (normalized == normalizeSectionLabel(alias)) {
                return sectionId;
            }
        }
    }

    return {};
}

QStringList orderedKnownKeysForSection(const QString& sectionId) {
    QStringList keys;
    for (const FieldSpec& spec : fieldSpecs()) {
        if (spec.sectionId == sectionId) {
            keys.push_back(spec.key.toLower());
        }
    }
    return keys;
}

} // namespace

ConfigDocument::ConfigDocument(QObject* parent)
    : QObject(parent)
    , watcher_(new QFileSystemWatcher(this))
    , externalReloadTimer_(new QTimer(this)) {
    repoRoot_ = detectRepoRoot();
    externalReloadTimer_->setSingleShot(true);
    externalReloadTimer_->setInterval(150);

    connect(watcher_, &QFileSystemWatcher::fileChanged, this, [this](const QString&) {
        scheduleExternalReload();
    });
    connect(watcher_, &QFileSystemWatcher::directoryChanged, this, [this](const QString&) {
        scheduleExternalReload();
    });
    connect(externalReloadTimer_, &QTimer::timeout, this, [this] {
        QString error;
        if (!reloadFromDisk(&error) && !filePath_.isEmpty() && !error.trimmed().isEmpty()) {
            emit externalFileReloadFailed(filePath_, error);
        }
    });
}

bool ConfigDocument::loadFromFile(const QString& filePath, QString* errorMessage) {
    QString text;
    if (!readTextFile(filePath, &text, errorMessage)) {
        return false;
    }

    filePath_ = QFileInfo(filePath).absoluteFilePath();
    mode_ = runModeFromPath(filePath_);
    if (!parseDocument(text, false, errorMessage)) {
        return false;
    }
    rewriteCanonicalFileIfNeeded(text);
    syncWatchPaths();
    emit modeChanged(mode_);
    emit documentReloaded();
    emit changed();
    return true;
}

bool ConfigDocument::reloadFromDisk(QString* errorMessage) {
    if (filePath_.isEmpty()) {
        if (errorMessage) {
            *errorMessage = "No source file path set.";
        }
        return false;
    }

    QString text;
    if (!readTextFile(filePath_, &text, errorMessage)) {
        syncWatchPaths();
        return false;
    }

    syncWatchPaths();
    if (normalizeText(text) == normalizeText(rawText_)) {
        return true;
    }

    const RunMode nextMode = runModeFromPath(filePath_);
    const bool modeChangedNow = nextMode != mode_;
    mode_ = nextMode;
    if (!parseDocument(text, false, errorMessage)) {
        return false;
    }
    rewriteCanonicalFileIfNeeded(text);
    if (modeChangedNow) {
        emit modeChanged(mode_);
    }
    emit documentReloaded();
    emit changed();
    emit externalFileReloaded(filePath_);
    return true;
}

bool ConfigDocument::save(QString* errorMessage) {
    if (filePath_.isEmpty()) {
        if (errorMessage) {
            *errorMessage = "No target file path set.";
        }
        return false;
    }
    return saveAs(filePath_, errorMessage);
}

bool ConfigDocument::saveAs(const QString& filePath, QString* errorMessage) {
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
        if (errorMessage) {
            *errorMessage = file.errorString();
        }
        return false;
    }

    QTextStream out(&file);
    out.setEncoding(QStringConverter::Utf8);
    const QString text = renderedText();
    out << text;
    filePath_ = QFileInfo(filePath).absoluteFilePath();
    mode_ = runModeFromPath(filePath_);
    rawText_ = text;
    syncWatchPaths();
    emit modeChanged(mode_);
    emit changed();
    return true;
}

bool ConfigDocument::applyRawText(const QString& text, QString* errorMessage) {
    if (!parseDocument(text, true, errorMessage)) {
        return false;
    }
    emit documentReloaded();
    emit changed();
    return true;
}

QString ConfigDocument::filePath() const {
    return filePath_;
}

QString ConfigDocument::projectDirectory() const {
    if (!filePath_.isEmpty()) {
        return QFileInfo(filePath_).absolutePath();
    }
    return repoRoot_;
}

QString ConfigDocument::repoRoot() const {
    return repoRoot_;
}

QString ConfigDocument::resultsDirectory() const {
    return QDir(projectDirectory()).filePath("RESULTS");
}

QString ConfigDocument::rawText() const {
    return rawText_;
}

QString ConfigDocument::renderEntry(const DeckEntry& entry) {
    QString line = entry.key % " = " % entry.value;
    if (!entry.comment.trimmed().isEmpty()) {
        line += " " % entry.comment.trimmed();
    }
    return line;
}

QStringList ConfigDocument::renderSection(const QString& sectionId) const {
    const QStringList looseLines = sectionLooseLines_.value(sectionId);
    const QStringList knownKeys = orderedKnownKeysForSection(sectionId);
    const QStringList unknownKeys = unknownOrder_.value(sectionId);

    QStringList lines;
    for (const QString& looseLine : looseLines) {
        if (!looseLine.trimmed().isEmpty()) {
            lines.push_back(looseLine);
        }
    }

    for (const QString& key : knownKeys) {
        const auto it = entries_.constFind(key);
        if (it == entries_.cend() || it.value().sectionId != sectionId) {
            continue;
        }
        lines.push_back(renderEntry(it.value()));
    }

    for (const QString& key : unknownKeys) {
        const auto it = entries_.constFind(key);
        if (it == entries_.cend() || it.value().sectionId != sectionId || it.value().known) {
            continue;
        }
        lines.push_back(renderEntry(it.value()));
    }

    if (lines.isEmpty()) {
        return {};
    }

    QStringList out;
    out.push_back("// " + sectionTitle(sectionId));
    out.append(lines);
    return out;
}

QString ConfigDocument::renderedText() const {
    QStringList out;

    if (!preambleLines_.isEmpty()) {
        out = preambleLines_;
        while (!out.isEmpty() && out.back().trimmed().isEmpty()) {
            out.removeLast();
        }
        if (!out.isEmpty()) {
            out.push_back("");
        }
    } else {
        out << "// LUW deck" << "";
    }

    for (const QString& sectionId : sectionOrder()) {
        const QStringList sectionLines = renderSection(sectionId);
        if (sectionLines.isEmpty()) {
            continue;
        }
        out.append(sectionLines);
        out.push_back("");
    }

    while (!out.isEmpty() && out.back().trimmed().isEmpty()) {
        out.removeLast();
    }

    QString text = out.join('\n');
    if (!text.endsWith('\n')) {
        text += '\n';
    }
    return text;
}

void ConfigDocument::clearFilePath() {
    filePath_.clear();
    syncWatchPaths();
    emit changed();
}

RunMode ConfigDocument::mode() const {
    return mode_;
}

void ConfigDocument::setMode(RunMode mode) {
    if (mode_ == mode) {
        return;
    }
    mode_ = mode;
    emit modeChanged(mode_);
    emit changed();
}

bool ConfigDocument::hasKey(const QString& key) const {
    return values_.contains(normalizeKey(key));
}

QString ConfigDocument::rawValue(const QString& key) const {
    return values_.value(normalizeKey(key));
}

QVariant ConfigDocument::typedValue(const QString& key) const {
    const QString normalized = normalizeKey(key);
    const FieldSpec* spec = findFieldSpec(normalized);
    if (!spec) {
        return rawValue(normalized);
    }
    return parseValue(*spec, rawValue(normalized));
}

void ConfigDocument::rememberUnknownKey(const QString& sectionId, const QString& key) {
    if (findFieldSpec(key)) {
        return;
    }
    QStringList& keys = unknownOrder_[sectionId];
    if (!keys.contains(key)) {
        keys.push_back(key);
    }
}

void ConfigDocument::forgetUnknownKey(const QString& key) {
    for (auto it = unknownOrder_.begin(); it != unknownOrder_.end(); ++it) {
        it.value().removeAll(key);
    }
}

void ConfigDocument::setRawValue(const QString& key, const QString& value) {
    const QString normalized = normalizeKey(key);
    const QString trimmed = value.trimmed();
    const auto existing = entries_.constFind(normalized);

    QString sectionId = orderedSectionForKey(normalized);
    QString comment;
    if (existing != entries_.cend()) {
        sectionId = existing.value().sectionId;
        comment = existing.value().comment;
    }

    const bool known = findFieldSpec(normalized) != nullptr;
    if (known) {
        forgetUnknownKey(normalized);
    } else {
        rememberUnknownKey(sectionId, normalized);
    }

    DeckEntry entry;
    entry.key = normalized;
    entry.value = trimmed;
    entry.sectionId = sectionId;
    entry.comment = comment;
    entry.known = known;

    entries_.insert(normalized, entry);
    values_[normalized] = trimmed;
    duplicateKeys_.removeAll(normalized);
    rawText_ = renderedText();
    emit keyChanged(normalized);
    emit changed();
}

void ConfigDocument::setTypedValue(const QString& key, const QVariant& value) {
    const QString normalized = normalizeKey(key);
    const FieldSpec* spec = findFieldSpec(normalized);
    if (!spec) {
        setRawValue(normalized, value.toString());
        return;
    }
    setRawValue(normalized, serializeValue(*spec, value));
}

QHash<QString, QString> ConfigDocument::rawValues() const {
    return values_;
}

QStringList ConfigDocument::unknownKeys() const {
    QStringList out;
    out.reserve(values_.size());
    for (auto it = values_.cbegin(); it != values_.cend(); ++it) {
        if (!findFieldSpec(it.key())) {
            out.push_back(it.key());
        }
    }
    out.sort();
    return out;
}

int ConfigDocument::commentIndex(const QString& line) {
    bool inSingleQuote = false;
    bool inDoubleQuote = false;

    for (int i = 0; i < line.size() - 1; ++i) {
        const QChar c = line[i];
        const QChar next = line[i + 1];
        if (c == '\'' && !inDoubleQuote) {
            inSingleQuote = !inSingleQuote;
        } else if (c == '"' && !inSingleQuote) {
            inDoubleQuote = !inDoubleQuote;
        } else if (!inSingleQuote && !inDoubleQuote && c == '/' && next == '/') {
            return i;
        }
    }

    return -1;
}

QString ConfigDocument::normalizeText(const QString& text) {
    return normalizeTextForCompare(text);
}

QString ConfigDocument::normalizeKey(const QString& key) {
    const QString normalized = key.trimmed().toLower();
    if (normalized == "vk_inlet_enable") {
        return "turb_inflow_enable";
    }
    return normalized;
}

QVariant ConfigDocument::parseValue(const FieldSpec& spec, const QString& rawValue) {
    const QString raw = rawValue.trimmed();
    switch (spec.kind) {
    case FieldKind::String:
    case FieldKind::Enum:
    case FieldKind::Multiline:
        return trimQuotes(raw);
    case FieldKind::Integer: {
        bool ok = false;
        const int value = trimQuotes(raw).toInt(&ok);
        return ok ? QVariant(value) : QVariant(trimQuotes(raw));
    }
    case FieldKind::Float: {
        bool ok = false;
        const double value = trimQuotes(raw).toDouble(&ok);
        return ok ? QVariant(value) : QVariant(trimQuotes(raw));
    }
    case FieldKind::Boolean: {
        const QString normalized = trimQuotes(raw).toLower();
        if (normalized == "true" || normalized == "1") {
            return true;
        }
        if (normalized == "false" || normalized == "0") {
            return false;
        }
        return QVariant(false);
    }
    case FieldKind::FloatPair:
        return parseNumberList(raw, false, 2);
    case FieldKind::FloatTriplet:
        return parseNumberList(raw, false, 3);
    case FieldKind::UIntTriplet:
        return parseNumberList(raw, true, 3);
    case FieldKind::FloatList:
        return parseNumberList(raw, false);
    case FieldKind::TokenList:
        return splitList(raw);
    }
    return trimQuotes(raw);
}

QString ConfigDocument::serializeValue(const FieldSpec& spec, const QVariant& value) {
    switch (spec.kind) {
    case FieldKind::String:
    case FieldKind::Multiline: {
        const QString text = value.toString().trimmed();
        return spec.quoted ? ("\"" % text % "\"") : text;
    }
    case FieldKind::Enum: {
        const QString text = value.toString().trimmed();
        return spec.quoted ? ("\"" % text % "\"") : text;
    }
    case FieldKind::Integer:
        return QString::number(value.toInt());
    case FieldKind::Float:
        return QString::number(value.toDouble(), 'f', 6);
    case FieldKind::Boolean:
        return value.toBool() ? "true" : "false";
    case FieldKind::FloatPair:
    case FieldKind::FloatTriplet:
    case FieldKind::FloatList:
        return renderList(value.toList(), false);
    case FieldKind::UIntTriplet:
        return renderList(value.toList(), true);
    case FieldKind::TokenList: {
        QStringList parts;
        if (value.metaType().id() == QMetaType::QStringList) {
            parts = value.toStringList();
        } else {
            const QVariantList rawList = value.toList();
            for (const QVariant& entry : rawList) {
                parts.push_back(entry.toString().trimmed());
            }
        }
        return "[" % parts.join(", ") % "]";
    }
    }
    return value.toString().trimmed();
}

bool ConfigDocument::readTextFile(const QString& filePath, QString* text, QString* errorMessage) const {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (errorMessage) {
            *errorMessage = file.errorString();
        }
        return false;
    }

    QTextStream in(&file);
    in.setEncoding(QStringConverter::Utf8);
    if (text) {
        *text = in.readAll();
    }
    return true;
}

void ConfigDocument::rewriteCanonicalFileIfNeeded(const QString& originalText) {
    if (filePath_.isEmpty()) {
        return;
    }
    if (normalizeText(originalText) == normalizeText(rawText_)) {
        return;
    }

    QFile file(filePath_);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
        return;
    }

    QTextStream out(&file);
    out.setEncoding(QStringConverter::Utf8);
    out << rawText_;
}

void ConfigDocument::syncWatchPaths() {
    if (!watcher_) {
        return;
    }

    const QStringList currentFiles = watcher_->files();
    for (const QString& path : currentFiles) {
        if (filePath_.isEmpty() || QFileInfo(path).absoluteFilePath() != filePath_) {
            watcher_->removePath(path);
        }
    }

    const QString watchedDir = filePath_.isEmpty() ? QString() : QFileInfo(filePath_).absolutePath();
    const QStringList currentDirs = watcher_->directories();
    for (const QString& path : currentDirs) {
        if (watchedDir.isEmpty() || QFileInfo(path).absoluteFilePath() != watchedDir) {
            watcher_->removePath(path);
        }
    }

    if (filePath_.isEmpty()) {
        return;
    }

    if (!watchedDir.isEmpty() && QFileInfo(watchedDir).exists() && !watcher_->directories().contains(watchedDir)) {
        watcher_->addPath(watchedDir);
    }
    if (QFileInfo(filePath_).exists() && !watcher_->files().contains(filePath_)) {
        watcher_->addPath(filePath_);
    }
}

void ConfigDocument::scheduleExternalReload() {
    if (filePath_.isEmpty() || !externalReloadTimer_) {
        return;
    }
    syncWatchPaths();
    externalReloadTimer_->start();
}

bool ConfigDocument::parseDocument(const QString& text, bool strictDuplicates, QString* errorMessage) {
    QStringList preambleLines;
    QHash<QString, QStringList> sectionLooseLines;
    QHash<QString, DeckEntry> entries;
    QHash<QString, QStringList> unknownOrder;
    QStringList duplicateKeys;
    QHash<QString, QString> values;

    const auto rememberUnknownKey = [&](const QString& sectionId, const QString& key) {
        if (findFieldSpec(key)) {
            return;
        }
        QStringList& keys = unknownOrder[sectionId];
        if (!keys.contains(key)) {
            keys.push_back(key);
        }
    };
    const auto forgetUnknownKey = [&](const QString& key) {
        for (auto it = unknownOrder.begin(); it != unknownOrder.end(); ++it) {
            it.value().removeAll(key);
        }
    };

    QString normalized = text;
    normalized.replace("\r\n", "\n");
    normalized.replace('\r', '\n');
    QStringList rawLines = normalized.split('\n');
    if (!rawLines.isEmpty() && rawLines.back().isEmpty()) {
        rawLines.removeLast();
    }

    const QRegularExpression kvPattern(R"(^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$)");
    QString currentSection;
    bool sawStructuredContent = false;

    for (const QString& originalLine : rawLines) {
        const QString stripped = originalLine.trimmed();
        const QString headerSection = matchSectionHeader(stripped);
        if (!headerSection.isEmpty()) {
            currentSection = headerSection;
            sawStructuredContent = true;
            continue;
        }

        const int cmtIndex = commentIndex(originalLine);
        const QString content = (cmtIndex >= 0) ? originalLine.left(cmtIndex) : originalLine;
        const QString comment = (cmtIndex >= 0) ? originalLine.mid(cmtIndex).trimmed() : QString();
        const QRegularExpressionMatch match = kvPattern.match(content);
        if (match.hasMatch()) {
            const QString key = normalizeKey(match.captured(1));
            const QString value = match.captured(2).trimmed();
            const bool known = findFieldSpec(key) != nullptr;

            if (entries.contains(key) && !duplicateKeys.contains(key)) {
                duplicateKeys.push_back(key);
            }
            if (entries.contains(key) && strictDuplicates) {
                if (errorMessage) {
                    *errorMessage = "Duplicate deck keys are not allowed: " + duplicateKeys.join(", ");
                }
                return false;
            }

            forgetUnknownKey(key);

            DeckEntry entry;
            entry.key = key;
            entry.value = value;
            entry.sectionId = known ? orderedSectionForKey(key) : (currentSection.isEmpty() ? "custom" : currentSection);
            entry.comment = comment;
            entry.known = known;

            entries.insert(key, entry);
            values.insert(key, value);
            if (!known) {
                rememberUnknownKey(entry.sectionId, key);
            }
            sawStructuredContent = true;
            continue;
        }

        if (stripped.isEmpty()) {
            if (!sawStructuredContent && currentSection.isEmpty()) {
                preambleLines.push_back("");
            }
            continue;
        }

        if (!sawStructuredContent && currentSection.isEmpty()) {
            preambleLines.push_back(originalLine);
        } else {
            const QString targetSection = currentSection.isEmpty() ? "custom" : currentSection;
            sectionLooseLines[targetSection].push_back(originalLine);
        }
    }

    preambleLines_ = preambleLines;
    sectionLooseLines_ = sectionLooseLines;
    entries_ = entries;
    unknownOrder_ = unknownOrder;
    duplicateKeys_ = duplicateKeys;
    values_ = values;
    rawText_ = renderedText();
    return true;
}

void ConfigDocument::setTextInternal(const QString& text) {
    parseDocument(text, false, nullptr);
}

} // namespace luwgui
