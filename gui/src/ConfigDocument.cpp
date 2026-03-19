#include "luwgui/ConfigDocument.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFileSystemWatcher>
#include <QRegularExpression>
#include <QStringBuilder>
#include <QTimer>
#include <QTextStream>

namespace luwgui {

namespace {

QString detectRepoRoot() {
    QDir dir(QCoreApplication::applicationDirPath());
    for (int depth = 0; depth < 8; ++depth) {
        if (dir.exists("core") && dir.exists("bin")) {
            return dir.absolutePath();
        }
        if (!dir.cdUp()) {
            break;
        }
    }
    return QDir::currentPath();
}

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
    setTextInternal(text);
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
    setTextInternal(text);
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
    Q_UNUSED(errorMessage)
    setTextInternal(text);
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

QString ConfigDocument::renderedText() const {
    QStringList out;
    out.reserve(lines_.size());
    for (const ParsedLine& line : lines_) {
        if (line.type == ParsedLine::Type::Raw) {
            out.push_back(line.raw);
            continue;
        }
        QString rendered = line.key % " = " % line.value;
        if (!line.comment.trimmed().isEmpty()) {
            rendered += " " % line.comment.trimmed();
        }
        out.push_back(rendered);
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

void ConfigDocument::setRawValue(const QString& key, const QString& value) {
    const QString normalized = normalizeKey(key);
    values_[normalized] = value.trimmed();
    upsertLine(normalized, value.trimmed());
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
    return key.trimmed().toLower();
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

void ConfigDocument::rebuildIndex() {
    keyToLine_.clear();
    values_.clear();
    for (int i = 0; i < lines_.size(); ++i) {
        const ParsedLine& line = lines_[i];
        if (line.type != ParsedLine::Type::KeyValue) {
            continue;
        }
        keyToLine_.insert(line.key, i);
        values_.insert(line.key, line.value);
    }
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

void ConfigDocument::setTextInternal(const QString& text) {
    rawText_ = text;
    lines_.clear();

    QString normalized = text;
    normalized.replace("\r\n", "\n");
    normalized.replace('\r', '\n');
    QStringList rawLines = normalized.split('\n');
    if (!rawLines.isEmpty() && rawLines.back().isEmpty()) {
        rawLines.removeLast();
    }

    const QRegularExpression kvPattern(R"(^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$)");

    for (const QString& originalLine : rawLines) {
        ParsedLine parsed;
        const int cmtIndex = commentIndex(originalLine);
        const QString content = (cmtIndex >= 0) ? originalLine.left(cmtIndex) : originalLine;
        const QString comment = (cmtIndex >= 0) ? originalLine.mid(cmtIndex) : QString();
        const QRegularExpressionMatch match = kvPattern.match(content);
        if (!match.hasMatch()) {
            parsed.type = ParsedLine::Type::Raw;
            parsed.raw = originalLine;
            lines_.push_back(parsed);
            continue;
        }

        parsed.type = ParsedLine::Type::KeyValue;
        parsed.key = normalizeKey(match.captured(1));
        parsed.value = match.captured(2).trimmed();
        parsed.comment = comment.trimmed();
        lines_.push_back(parsed);
    }

    rebuildIndex();
}

int ConfigDocument::ensureManagedSection() {
    static const QString marker = "// GUI managed additions";
    for (int i = 0; i < lines_.size(); ++i) {
        if (lines_[i].type == ParsedLine::Type::Raw && lines_[i].raw.trimmed() == marker) {
            return i;
        }
    }

    if (!lines_.isEmpty()) {
        ParsedLine blank;
        blank.type = ParsedLine::Type::Raw;
        blank.raw = "";
        lines_.push_back(blank);
    }

    ParsedLine markerLine;
    markerLine.type = ParsedLine::Type::Raw;
    markerLine.raw = marker;
    lines_.push_back(markerLine);
    return lines_.size() - 1;
}

void ConfigDocument::upsertLine(const QString& key, const QString& value) {
    const auto it = keyToLine_.constFind(key);
    if (it != keyToLine_.cend()) {
        ParsedLine& line = lines_[it.value()];
        line.type = ParsedLine::Type::KeyValue;
        line.key = key;
        line.value = value;
        return;
    }

    const int managedIndex = ensureManagedSection();
    ParsedLine entry;
    entry.type = ParsedLine::Type::KeyValue;
    entry.key = key;
    entry.value = value;
    lines_.insert(managedIndex + 1, entry);
    rebuildIndex();
}

}
