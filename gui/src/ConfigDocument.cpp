#include "luwgui/ConfigDocument.h"
#include "luwgui/RuntimePaths.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFileSystemWatcher>
#include <QProcess>
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

QString renderFloatingPointValue(double value) {
    QString text = QString::number(value, 'f', 15);
    while (text.contains('.') && text.endsWith('0')) {
        text.chop(1);
    }
    if (text.endsWith('.')) {
        text.chop(1);
    }
    if (text == "-0") {
        text = "0";
    }
    return text;
}

QString renderList(const QVariantList& values, bool integerMode) {
    QStringList parts;
    parts.reserve(values.size());
    for (const QVariant& value : values) {
        if (integerMode) {
            parts.push_back(QString::number(value.toInt()));
        } else {
            parts.push_back(renderFloatingPointValue(value.toDouble()));
        }
    }
    return "[" % parts.join(", ") % "]";
}

int defaultGpuMemoryMiB() {
    QProcess process;
    process.start(QStringLiteral("nvidia-smi"),
                  {QStringLiteral("--query-gpu=memory.total"),
                   QStringLiteral("--format=csv,noheader,nounits")});
    if (!process.waitForStarted(1000) || !process.waitForFinished(3000)) {
        return 20000;
    }
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        return 20000;
    }

    const QString output = QString::fromUtf8(process.readAllStandardOutput()).trimmed();
    const QString firstLine = output.section('\n', 0, 0).trimmed();
    bool ok = false;
    const int totalMemoryMiB = firstLine.toInt(&ok);
    if (!ok || totalMemoryMiB <= 0) {
        return 20000;
    }

    const int recommended = (totalMemoryMiB * 85) / 100;
    return recommended > 0 ? recommended : 20000;
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
    QStringList out;
    const QVector<SectionSpec>& specs = sectionSpecs();
    out.reserve(specs.size());
    for (const SectionSpec& spec : specs) {
        out.push_back(spec.id);
    }
    return out;
}

QString sectionTitle(const QString& sectionId) {
    for (const SectionSpec& spec : sectionSpecs()) {
        if (spec.id == sectionId) {
            return spec.title;
        }
    }
    return sectionId;
}

QStringList sectionAliases(const QString& sectionId) {
    for (const SectionSpec& spec : sectionSpecs()) {
        if (spec.id == sectionId) {
            return spec.aliases;
        }
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

bool ConfigDocument::repairRawText(const QString& text,
                                   QString* repairedText,
                                   QStringList* operations,
                                   QString* errorMessage) const {
    ConfigDocument temp;
    temp.mode_ = mode_;

    if (!temp.parseDocument(text, false, errorMessage)) {
        return false;
    }

    QStringList localOperations;
    const QString canonicalText = temp.renderedText();
    if (!temp.duplicateKeys_.isEmpty()) {
        localOperations.push_back(
            QStringLiteral("Collapsed duplicate keys by keeping the last value: %1.")
                .arg(temp.duplicateKeys_.join(QStringLiteral(", "))));
    }
    if (normalizeText(text) != normalizeText(canonicalText)) {
        localOperations.push_back(QStringLiteral("Reordered deck into canonical section/key order."));
    }

    const QString datetimeValue = temp.typedValue(QStringLiteral("datetime")).toString().trimmed();
    if (datetimeValue.isEmpty()) {
        temp.setTypedValue(QStringLiteral("datetime"), QStringLiteral("20990101120000"));
        localOperations.push_back(QStringLiteral("Added missing 'datetime' with default 20990101120000."));
    }

    const QVariantList nGpu = temp.typedValue(QStringLiteral("n_gpu")).toList();
    if (nGpu.size() != 3) {
        temp.setTypedValue(QStringLiteral("n_gpu"), QVariantList{1, 1, 1});
        localOperations.push_back(QStringLiteral("Added missing 'n_gpu' with default [1, 1, 1]."));
    }

    QString meshControl = temp.typedValue(QStringLiteral("mesh_control")).toString().trimmed().toLower();
    const QString cellSizeRaw = temp.rawValue(QStringLiteral("cell_size")).trimmed();
    const bool hasCellSizeValue = !cellSizeRaw.isEmpty();
    if (meshControl.isEmpty()) {
        temp.setTypedValue(QStringLiteral("mesh_control"), QStringLiteral("gpu_memory"));
        meshControl = QStringLiteral("gpu_memory");
        localOperations.push_back(QStringLiteral("Added missing 'mesh_control' with default \"gpu_memory\"."));
    } else if (meshControl == QStringLiteral("cell_size") && !hasCellSizeValue) {
        temp.setTypedValue(QStringLiteral("mesh_control"), QStringLiteral("gpu_memory"));
        meshControl = QStringLiteral("gpu_memory");
        localOperations.push_back(
            QStringLiteral("Changed 'mesh_control' to \"gpu_memory\" because 'cell_size' is empty."));
    }

    const QString legacyMemoryKey = normalizeKey(QStringLiteral("memory_lbm"));
    const bool hadLegacyMemory = temp.values_.contains(legacyMemoryKey);
    const QString legacyMemory = trimQuotes(temp.values_.value(legacyMemoryKey));
    if (meshControl == QStringLiteral("gpu_memory")) {
        bool gpuMemoryOk = false;
        temp.rawValue(QStringLiteral("gpu_memory")).trimmed().toInt(&gpuMemoryOk);
        if (!gpuMemoryOk) {
            int ensuredMemory = defaultGpuMemoryMiB();
            if (!legacyMemory.isEmpty()) {
                bool legacyOk = false;
                const int migrated = static_cast<int>(legacyMemory.toDouble(&legacyOk));
                if (legacyOk && migrated > 0) {
                    ensuredMemory = migrated;
                }
            }
            temp.setTypedValue(QStringLiteral("gpu_memory"), ensuredMemory);
            localOperations.push_back(
                QStringLiteral("Added missing 'gpu_memory' with default %1.").arg(ensuredMemory));
        }
    }

    if (!temp.hasKey(QStringLiteral("cell_size"))) {
        temp.setRawValue(QStringLiteral("cell_size"), QString());
        localOperations.push_back(QStringLiteral("Inserted placeholder 'cell_size'."));
    }

    if (!temp.hasKey(QStringLiteral("high_order"))) {
        temp.setTypedValue(QStringLiteral("high_order"), true);
        localOperations.push_back(QStringLiteral("Added missing 'high_order' with default true."));
    }
    if (!temp.hasKey(QStringLiteral("flux_correction"))) {
        temp.setTypedValue(QStringLiteral("flux_correction"), true);
        localOperations.push_back(QStringLiteral("Added missing 'flux_correction' with default true."));
    }

    if (temp.mode_ != RunMode::Luwdg) {
        if (!temp.hasKey(QStringLiteral("terr_voxel_height_field"))) {
            temp.setTypedValue(QStringLiteral("terr_voxel_height_field"), QStringLiteral("auto"));
            localOperations.push_back(QStringLiteral("Added missing 'terr_voxel_height_field' with default \"auto\"."));
        }
        if (!temp.hasKey(QStringLiteral("terr_voxel_ignore_under"))) {
            temp.setTypedValue(QStringLiteral("terr_voxel_ignore_under"), 0.0);
            localOperations.push_back(QStringLiteral("Added missing 'terr_voxel_ignore_under' with default 0.0."));
        }
        if (!temp.hasKey(QStringLiteral("terr_voxel_approach"))) {
            temp.setTypedValue(QStringLiteral("terr_voxel_approach"), QStringLiteral("idw"));
            localOperations.push_back(QStringLiteral("Added missing 'terr_voxel_approach' with default \"idw\"."));
        }
        if (!temp.hasKey(QStringLiteral("terr_voxel_grid_resolution"))) {
            temp.setTypedValue(QStringLiteral("terr_voxel_grid_resolution"), 50.0);
            localOperations.push_back(QStringLiteral("Added missing 'terr_voxel_grid_resolution' with default 50.0."));
        }
        if (!temp.hasKey(QStringLiteral("terr_voxel_idw_sigma"))) {
            temp.setTypedValue(QStringLiteral("terr_voxel_idw_sigma"), 1.0);
            localOperations.push_back(QStringLiteral("Added missing 'terr_voxel_idw_sigma' with default 1.0."));
        }
        if (!temp.hasKey(QStringLiteral("terr_voxel_idw_power"))) {
            temp.setTypedValue(QStringLiteral("terr_voxel_idw_power"), 2.0);
            localOperations.push_back(QStringLiteral("Added missing 'terr_voxel_idw_power' with default 2.0."));
        }
        if (!temp.hasKey(QStringLiteral("terr_voxel_idw_neighbors"))) {
            temp.setTypedValue(QStringLiteral("terr_voxel_idw_neighbors"), 12);
            localOperations.push_back(QStringLiteral("Added missing 'terr_voxel_idw_neighbors' with default 12."));
        }
    }

    if (hadLegacyMemory) {
        temp.entries_.remove(legacyMemoryKey);
        temp.values_.remove(legacyMemoryKey);
        temp.duplicateKeys_.removeAll(legacyMemoryKey);
        temp.forgetUnknownKey(legacyMemoryKey);
        temp.rawText_ = temp.renderedText();
        localOperations.push_back(QStringLiteral("Removed legacy 'memory_lbm'."));
    }

    if (repairedText) {
        *repairedText = temp.renderedText();
    }
    if (operations) {
        *operations = localOperations;
    }
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
    QString renderedValue = entry.value.trimmed();
    if (entry.known) {
        const FieldSpec* spec = findFieldSpec(entry.key);
        bool parsed = false;
        if (spec) {
            if (spec->kind == FieldKind::Boolean && tryParseDeckBool(renderedValue, &parsed)) {
                renderedValue = parsed ? "true" : "false";
            } else if (!renderedValue.isEmpty() && (spec->kind == FieldKind::FloatPair
                       || spec->kind == FieldKind::FloatTriplet
                       || spec->kind == FieldKind::UIntTriplet
                       || spec->kind == FieldKind::FloatList
                       || spec->kind == FieldKind::TokenList)) {
                renderedValue = "[" % splitList(renderedValue).join(", ") % "]";
            } else if (spec->quoted && !renderedValue.isEmpty()) {
                renderedValue = "\"" % trimQuotes(renderedValue) % "\"";
            }
        }
    }

    QString line = entry.key % " =";
    if (!renderedValue.isEmpty()) {
        line += " " % renderedValue;
    }
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
    return normalizeDeckKey(key);
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
        bool parsed = false;
        if (tryParseDeckBool(raw, &parsed)) {
            return parsed;
        }
        return QVariant();
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
        return renderFloatingPointValue(value.toDouble());
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
        const int eqIndex = content.indexOf('=');
        if (eqIndex >= 0) {
            const QString key = normalizeKey(content.left(eqIndex));
            const QString value = content.mid(eqIndex + 1).trimmed();
            if (key.isEmpty()) {
                continue;
            }
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
