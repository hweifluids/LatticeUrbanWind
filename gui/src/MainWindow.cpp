#include "luwgui/MainWindow.h"
#include "luwgui/BuildInfo.h"

#include <QAbstractAnimation>
#include <QAbstractItemView>
#include <QAbstractSpinBox>
#include <QAction>
#include <QClipboard>
#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleValidator>
#include <QFileDialog>
#include <QFile>
#include <QFileInfo>
#include <QFocusEvent>
#include <QFormLayout>
#include <QFrame>
#include <QFont>
#include <QGuiApplication>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QIntValidator>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPlainTextEdit>
#include <QPropertyAnimation>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QSizePolicy>
#include <QSplitter>
#include <QStackedWidget>
#include <QStatusBar>
#include <QStyle>
#include <QStyleOptionComboBox>
#include <QStylePainter>
#include <QTabWidget>
#include <QToolTip>
#include <QToolButton>
#include <QTreeView>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QUrl>
#include <QVBoxLayout>
#include <QWindow>

#ifdef Q_OS_WIN
#include <windows.h>
#include <windowsx.h>
#endif

#include <algorithm>
#include <functional>
#include <limits>
#include <set>

namespace luwgui {

namespace {

QColor blendWidgetColors(const QColor& a, const QColor& b, qreal t) {
    const qreal u = std::clamp(t, 0.0, 1.0);
    return QColor::fromRgbF(
        a.redF() + (b.redF() - a.redF()) * u,
        a.greenF() + (b.greenF() - a.greenF()) * u,
        a.blueF() + (b.blueF() - a.blueF()) * u,
        1.0);
}

QColor outlineBranchColor(const QPalette& palette) {
    QColor color = palette.color(QPalette::Mid);
    if (!color.isValid() || color.alpha() == 0) {
        const QColor base = palette.color(QPalette::Base);
        const QColor text = palette.color(QPalette::WindowText);
        const qreal mix = base.lightnessF() < 0.5 ? 0.34 : 0.18;
        color = blendWidgetColors(base, text, mix);
    }
    return color;
}

QColor sectionGridColor(const QPalette& palette) {
    QColor color = outlineBranchColor(palette);
    color.setAlpha(132);
    return color;
}

class HeaderBarWidget final : public QWidget {
public:
    explicit HeaderBarWidget(QWidget* parent = nullptr)
        : QWidget(parent) {
        setMouseTracking(true);
    }
};

class CompactModeComboBox final : public QComboBox {
public:
    static constexpr int CompactTextRole = Qt::UserRole + 1;

    explicit CompactModeComboBox(QWidget* parent = nullptr)
        : QComboBox(parent) {
    }

    QSize sizeHint() const override {
        return comboSizeHint();
    }

    QSize minimumSizeHint() const override {
        return comboSizeHint();
    }

protected:
    void paintEvent(QPaintEvent* event) override {
        Q_UNUSED(event)
        QStylePainter painter(this);
        QStyleOptionComboBox option;
        initStyleOption(&option);
        const QString compactText = currentData(CompactTextRole).toString();
        if (!compactText.isEmpty()) {
            option.currentText = compactText;
        }
        painter.drawComplexControl(QStyle::CC_ComboBox, option);
        painter.drawControl(QStyle::CE_ComboBoxLabel, option);
    }

private:
    QSize comboSizeHint() const {
        QStyleOptionComboBox option;
        initStyleOption(&option);

        const QFontMetrics metrics(font());
        int textWidth = 0;
        for (int index = 0; index < count(); ++index) {
            const QString compactText = itemData(index, CompactTextRole).toString();
            textWidth = std::max(textWidth, metrics.horizontalAdvance(compactText.isEmpty() ? itemText(index) : compactText));
        }

        const QSize contentSize(textWidth + 18, metrics.height() + 8);
        return style()->sizeFromContents(QStyle::CT_ComboBox, &option, contentSize, this);
    }
};

enum class ToolbarGlyph {
    InspectWind,
    InspectBuildings,
    RunWorkflow,
    Crop,
    Voxel,
    Batch,
    Validate,
    Solve,
    Stop,
    Console,
    SidePanel
};

enum class WindowControlGlyph {
    Minimize,
    Maximize,
    Restore,
    Close
};

class DotSeparatorWidget final : public QWidget {
public:
    explicit DotSeparatorWidget(QWidget* parent = nullptr)
        : QWidget(parent) {
        setFixedSize(10, 24);
        setAttribute(Qt::WA_StyledBackground, true);
        setStyleSheet("background: transparent;");
    }

protected:
    void paintEvent(QPaintEvent* event) override {
        Q_UNUSED(event)

        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(Qt::NoPen);
        QColor dotColor = palette().color(QPalette::ButtonText);
        dotColor.setAlpha(80);
        painter.setBrush(dotColor);

        const int centerX = width() / 2;
        for (int index = 0; index < 3; ++index) {
            painter.drawEllipse(QPointF(centerX, 4.5 + 7.0 * index), 1.0, 1.0);
        }
    }
};

bool isWidgetOrAncestor(const QWidget* candidate, const QWidget* target) {
    for (const QWidget* current = candidate; current; current = current->parentWidget()) {
        if (current == target) {
            return true;
        }
    }
    return false;
}

void resetWorkspaceSplitters(QSplitter* rootSplitter, QSplitter* workspaceSplitter) {
    if (rootSplitter) {
        rootSplitter->setSizes({380, 1420});
    }
    if (workspaceSplitter) {
        workspaceSplitter->setSizes({1420, 0});
    }
}

void allowHorizontalCompression(QWidget* widget) {
    if (!widget) {
        return;
    }
    widget->setMinimumWidth(0);
    QSizePolicy policy = widget->sizePolicy();
    policy.setHorizontalPolicy(QSizePolicy::Ignored);
    widget->setSizePolicy(policy);
}

Qt::CaseSensitivity projectPathCaseSensitivity() {
#ifdef Q_OS_WIN
    return Qt::CaseInsensitive;
#else
    return Qt::CaseSensitive;
#endif
}

QString normalizeProjectDeckPath(QString path) {
    path = path.trimmed();
    if (path.isEmpty()) {
        return {};
    }
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

bool sameProjectDeckPath(const QString& lhs, const QString& rhs) {
    return lhs.compare(rhs, projectPathCaseSensitivity()) == 0;
}

QString recentProjectActionText(const QString& filePath) {
    return QDir::toNativeSeparators(filePath);
}

const QString kGuiPropertiesDirName = QStringLiteral("gui_properties");
const QString kManagedRoleTool = QStringLiteral("tool");
const QString kManagedRoleDisplay = QStringLiteral("display");
const QString kManagedRoleDataProcessor = QStringLiteral("data_processor");
const QString kToolFileExtension = QStringLiteral(".luwtools");
const QString kVisFileExtension = QStringLiteral(".luwvis");
const QString kProcessorFileExtension = QStringLiteral(".luwpp");

struct ManagedFileNameInfo {
    QString baseName;
    bool trashed = false;
    int trashIndex = -1;
};

QString stripWrappingQuotes(QString text) {
    text = text.trimmed();
    if (text.size() >= 2) {
        const QChar first = text.front();
        const QChar last = text.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            text = text.mid(1, text.size() - 2).trimmed();
        }
    }
    return text;
}

QString managedFileExtension(const QString& role) {
    if (role == kManagedRoleTool) {
        return kToolFileExtension;
    }
    if (role == kManagedRoleDisplay) {
        return kVisFileExtension;
    }
    if (role == kManagedRoleDataProcessor) {
        return kProcessorFileExtension;
    }
    return {};
}

QString managedRoleDisplayName(const QString& role) {
    if (role == kManagedRoleTool) {
        return QStringLiteral("tool");
    }
    if (role == kManagedRoleDisplay) {
        return QStringLiteral("display");
    }
    if (role == kManagedRoleDataProcessor) {
        return QStringLiteral("data processor");
    }
    return QStringLiteral("node");
}

QString managedNodeId(const QString& role, const QString& filePath) {
    return role + "::" + normalizeProjectDeckPath(filePath);
}

QString managedNodeDefaultTitle(const QString& role, const QString& storageType) {
    if (role == kManagedRoleTool) {
        if (storageType == "crop") {
            return QStringLiteral("Crop region");
        }
        if (storageType == "multimoment") {
            return QStringLiteral("Multi-moment automation");
        }
    } else if (role == kManagedRoleDisplay) {
        if (storageType == "geometry") {
            return QStringLiteral("Geometry display");
        }
        if (storageType == "2d") {
            return QStringLiteral("2D visualization");
        }
        if (storageType == "3d") {
            return QStringLiteral("3D visualization");
        }
        if (storageType == "plot") {
            return QStringLiteral("Data plot");
        }
        if (storageType == "sheet") {
            return QStringLiteral("Spreadsheet view");
        }
    } else if (role == kManagedRoleDataProcessor) {
        return QStringLiteral("Data processor");
    }
    return QStringLiteral("Node");
}

QString managedNodeTypePropertyKey(const QString& role) {
    if (role == kManagedRoleTool) {
        return QStringLiteral("tool_type");
    }
    if (role == kManagedRoleDisplay) {
        return QStringLiteral("vis_type");
    }
    return {};
}

QString managedNodeFileContents(const QString& role, const QString& storageType) {
    const QString key = managedNodeTypePropertyKey(role);
    if (key.isEmpty()) {
        return {};
    }
    return QString("%1 = %2\n").arg(key, storageType);
}

QString managedNodeTrashSuffix(int trashIndex) {
    if (trashIndex <= 0) {
        return QStringLiteral(" (trashed)");
    }
    return QString(" (trashed %1)").arg(trashIndex);
}

QString managedNodeDisplayTitle(const QString& baseName, bool trashed, int trashIndex) {
    return trashed ? baseName + managedNodeTrashSuffix(trashIndex) : baseName;
}

QString managedNodeRecoverLabel(const QString& role) {
    if (role == kManagedRoleTool) {
        return QStringLiteral("Recover tool");
    }
    if (role == kManagedRoleDisplay) {
        return QStringLiteral("Recover display");
    }
    if (role == kManagedRoleDataProcessor) {
        return QStringLiteral("Recover data processor");
    }
    return QStringLiteral("Recover");
}

bool parseManagedFileName(const QString& fileName, const QString& extension, ManagedFileNameInfo* info) {
    const QRegularExpression expression(
        QString(R"(^(.+)%1(?:\.trash(\d*)?)?$)")
            .arg(QRegularExpression::escape(extension)),
        QRegularExpression::CaseInsensitiveOption);
    const QRegularExpressionMatch match = expression.match(fileName);
    if (!match.hasMatch()) {
        return false;
    }

    ManagedFileNameInfo parsed;
    parsed.baseName = match.captured(1);
    const bool hasTrashSuffix = !match.captured(2).isNull();
    parsed.trashed = hasTrashSuffix;
    parsed.trashIndex = hasTrashSuffix
        ? (match.captured(2).isEmpty() ? 0 : match.captured(2).toInt())
        : -1;

    if (info) {
        *info = parsed;
    }
    return true;
}

QHash<QString, QString> readLooseKeyValueFile(const QString& filePath) {
    QHash<QString, QString> values;
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return values;
    }

    const QString text = QString::fromUtf8(file.readAll());
    const QStringList lines = text.split(QRegularExpression(R"(\r\n|\r|\n)"));
    for (QString line : lines) {
        line = line.trimmed();
        if (line.isEmpty() || line.startsWith('#') || line.startsWith("//")) {
            continue;
        }
        const int equalsIndex = line.indexOf('=');
        if (equalsIndex <= 0) {
            continue;
        }
        const QString key = line.left(equalsIndex).trimmed().toLower();
        const QString value = stripWrappingQuotes(line.mid(equalsIndex + 1));
        if (!key.isEmpty()) {
            values.insert(key, value);
        }
    }

    return values;
}

bool writeManagedNodeFile(const QString& filePath, const QString& role, const QString& storageType, QString* errorMessage) {
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
        if (errorMessage) {
            *errorMessage = file.errorString();
        }
        return false;
    }

    const QByteArray data = managedNodeFileContents(role, storageType).toUtf8();
    if (!data.isEmpty() && file.write(data) != data.size()) {
        if (errorMessage) {
            *errorMessage = file.errorString();
        }
        return false;
    }

    return true;
}

QString nextAvailableTrashPath(const QString& activeFilePath) {
    const QString baseTrashPath = activeFilePath + ".trash";
    if (!QFileInfo::exists(baseTrashPath)) {
        return baseTrashPath;
    }
    for (int index = 2; index < std::numeric_limits<int>::max(); ++index) {
        const QString candidate = activeFilePath + ".trash" + QString::number(index);
        if (!QFileInfo::exists(candidate)) {
            return candidate;
        }
    }
    return baseTrashPath;
}

QString nextAvailableManagedPath(const QString& directoryPath, const QString& preferredBaseName, const QString& extension) {
    const QDir directory(directoryPath);
    const QString preferredPath = directory.filePath(preferredBaseName + extension);
    if (!QFileInfo::exists(preferredPath)) {
        return preferredPath;
    }

    for (int index = 2; index < std::numeric_limits<int>::max(); ++index) {
        const QString candidate = directory.filePath(
            QString("%1_%2%3").arg(preferredBaseName, QString::number(index), extension));
        if (!QFileInfo::exists(candidate)) {
            return candidate;
        }
    }

    return preferredPath;
}

bool isValidManagedNodeName(const QString& name, QString* errorMessage = nullptr) {
    const QString trimmed = name.trimmed();
    if (trimmed.isEmpty()) {
        if (errorMessage) {
            *errorMessage = "Name cannot be empty.";
        }
        return false;
    }
    if (trimmed == "." || trimmed == "..") {
        if (errorMessage) {
            *errorMessage = "Name is not valid.";
        }
        return false;
    }
    if (trimmed.endsWith('.') || trimmed.endsWith(' ')) {
        if (errorMessage) {
            *errorMessage = "Name cannot end with a space or period.";
        }
        return false;
    }
    if (trimmed.contains(QRegularExpression(R"([\\/:*?"<>|])"))) {
        if (errorMessage) {
            *errorMessage = "Name contains invalid path characters.";
        }
        return false;
    }
    return true;
}

QVariantList parseLooseList(const QString& text, bool integerMode) {
    QVariantList out;
    QString normalized = text;
    normalized.replace('[', ' ');
    normalized.replace(']', ' ');
    const QStringList parts = normalized.split(QRegularExpression(R"([,\s]+)"), Qt::SkipEmptyParts);
    for (const QString& part : parts) {
        bool ok = false;
        if (integerMode) {
            const int value = part.toInt(&ok);
            if (ok) {
                out.push_back(value);
            }
        } else {
            const double value = part.toDouble(&ok);
            if (ok) {
                out.push_back(value);
            }
        }
    }
    return out;
}

QString renderVariantList(const QVariantList& list, bool integerMode) {
    QStringList parts;
    for (const QVariant& value : list) {
        parts.push_back(integerMode ? QString::number(value.toInt()) : QString::number(value.toDouble(), 'f', 6));
    }
    return parts.join(",   ");
}

int modeMaskFor(RunMode mode) {
    switch (mode) {
    case RunMode::Luw:
        return ModeMaskLuw;
    case RunMode::Luwdg:
        return ModeMaskLuwdg;
    case RunMode::Luwpf:
        return ModeMaskLuwpf;
    }
    return ModeMaskAll;
}

bool specMatchesMode(const FieldSpec& spec, RunMode mode) {
    return (spec.modeMask & modeMaskFor(mode)) != 0;
}

bool hasNextSibling(const QModelIndex& index) {
    if (!index.isValid()) {
        return false;
    }
    return index.row() + 1 < index.model()->rowCount(index.parent());
}

bool rawValueTruthy(const QString& raw) {
    bool parsed = false;
    return tryParseDeckBool(raw, &parsed) && parsed;
}

QString cssColor(const QColor& color) {
    if (color.alpha() < 255) {
        return QStringLiteral("rgba(%1, %2, %3, %4)")
            .arg(color.red())
            .arg(color.green())
            .arg(color.blue())
            .arg(QString::number(color.alphaF(), 'f', 3));
    }
    return color.name(QColor::HexRgb);
}

struct PropertyEditorPalette {
    QColor background;
    QColor border;
    QColor text;
};

PropertyEditorPalette unlockedPropertyPalette(ThemeMode mode, const QPalette& palette) {
    switch (mode) {
    case ThemeMode::Frieren:
        return {QColor("#f1e3e4"), QColor("#bf9799"), QColor("#5d2628")};
    case ThemeMode::Himmel:
        return {QColor("#d7e7f8"), QColor("#2e5fa8"), QColor("#243b63")};
    case ThemeMode::Pink:
        return {QColor("#f7e1ea"), QColor("#c37b9c"), QColor("#5d233f")};
    case ThemeMode::Green:
        return {QColor("#e3f1e7"), QColor("#5d9a70"), QColor("#20422d")};
    case ThemeMode::Black:
        return {QColor("#1d1d1d"), QColor("#666666"), QColor("#f1f1f1")};
    case ThemeMode::White:
    case ThemeMode::LightDefault:
    case ThemeMode::DarkDefault:
    case ThemeMode::Siemens:
        break;
    }

    QColor background = palette.color(QPalette::Base);
    QColor border = palette.color(QPalette::Highlight);
    QColor text = palette.color(QPalette::Text);
    if (background.lightness() < 128) {
        background = background.lighter(118);
    } else {
        background = background.darker(102);
    }
    return {background, border, text};
}

void applyPropertyEditorVisual(QLineEdit* line, ThemeMode mode, bool unlocked) {
    if (!line) {
        return;
    }

    QPalette palette = line->palette();
    if (!unlocked) {
        line->setReadOnly(true);
        line->setFrame(false);
        line->setCursor(Qt::PointingHandCursor);
        QColor mutedText = palette.color(QPalette::PlaceholderText);
        if (!mutedText.isValid() || mutedText.alpha() == 0) {
            mutedText = palette.color(QPalette::Mid);
        }
        palette.setColor(QPalette::Text, mutedText);
        line->setPalette(palette);
        line->setStyleSheet("QLineEdit { background: transparent; border: none; padding: 1px 0px; }");
        line->deselect();
        line->setCursorPosition(0);
        return;
    }

    const PropertyEditorPalette unlockedColors = unlockedPropertyPalette(mode, palette);
    line->setReadOnly(false);
    line->setFrame(true);
    line->setCursor(Qt::IBeamCursor);
    palette.setColor(QPalette::Text, unlockedColors.text);
    line->setPalette(palette);
    line->setStyleSheet(QString(
        "QLineEdit { background-color: %1; border: 1px solid %2; padding: 2px 6px; color: %3; }")
        .arg(cssColor(unlockedColors.background),
             cssColor(unlockedColors.border),
             cssColor(unlockedColors.text)));
}

class OutlineTreeWidget final : public QTreeWidget {
public:
    explicit OutlineTreeWidget(QWidget* parent = nullptr)
        : QTreeWidget(parent) {
    }

protected:
    void drawBranches(QPainter* painter, const QRect& rect, const QModelIndex& index) const override {
        QTreeWidget::drawBranches(painter, rect, index);
        if (!index.isValid()) {
            return;
        }

        painter->save();
        painter->setRenderHint(QPainter::Antialiasing, false);
        QPen pen(outlineBranchColor(palette()), 1.0);
        painter->setPen(pen);

        QVector<bool> ancestorHasSibling;
        for (QModelIndex ancestor = index.parent(); ancestor.isValid(); ancestor = ancestor.parent()) {
            ancestorHasSibling.prepend(hasNextSibling(ancestor));
        }

        const int lineOffset = indentation() / 2;
        const int centerY = rect.top() + rect.height() / 2;
        for (int level = 0; level < ancestorHasSibling.size(); ++level) {
            if (!ancestorHasSibling[level]) {
                continue;
            }
            const int x = level * indentation() + lineOffset;
            painter->drawLine(x, rect.top(), x, rect.bottom());
        }

        if (index.parent().isValid()) {
            const int branchX = ancestorHasSibling.size() * indentation() + lineOffset;
            const int branchBottom = hasNextSibling(index) ? rect.bottom() : centerY;
            painter->drawLine(branchX, rect.top(), branchX, branchBottom);
            painter->drawLine(branchX, centerY, branchX + indentation() / 2, centerY);
        }

        painter->restore();
    }
};

class FormGridWidget final : public QWidget {
public:
    explicit FormGridWidget(bool drawGrid, QWidget* parent = nullptr)
        : QWidget(parent)
        , drawGrid_(drawGrid) {
    }

protected:
    void paintEvent(QPaintEvent* event) override {
        QWidget::paintEvent(event);
        if (!drawGrid_) {
            return;
        }

        std::set<int> rowBottoms;
        int top = std::numeric_limits<int>::max();
        int bottom = -1;
        const auto childWidgets = findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly);
        for (QWidget* child : childWidgets) {
            if (!child || !child->isVisible() || !child->property("formRow").isValid()) {
                continue;
            }
            const QRect geometry = child->geometry();
            top = std::min(top, geometry.top());
            bottom = std::max(bottom, geometry.bottom());
            rowBottoms.insert(geometry.bottom() + 2);
        }

        if (bottom < 0) {
            return;
        }

        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, false);
        painter.setPen(QPen(sectionGridColor(palette()), 1.0));

        const int leftInset = 6;
        const int rightInset = 6;
        const int contentLeft = leftInset;
        const int contentRight = width() - rightInset;
        const int centerX = contentLeft + std::max(0, (contentRight - contentLeft) / 2);
        const int topLine = std::max(6, top);
        const int bottomLine = bottom + 2;
        painter.drawLine(centerX, topLine, centerX, bottomLine);
        for (const int rowBottom : rowBottoms) {
            painter.drawLine(contentLeft, rowBottom, contentRight, rowBottom);
        }
    }

private:
    bool drawGrid_ = false;
};

QStringList splitLooseVectorTokens(QString text) {
    text.replace('[', ' ');
    text.replace(']', ' ');
    return text.split(QRegularExpression(R"([,\s]+)"), Qt::SkipEmptyParts);
}

QString canonicalizeLooseVectorText(const QString& text) {
    return splitLooseVectorTokens(text).join(", ");
}

QString normalizeTerrainHeightFieldValue(const QString& value) {
    const QString trimmed = value.trimmed();
    const QString normalized = trimmed.toLower();
    if (normalized.isEmpty() || normalized == "auto" || normalized == "inferred") {
        return QStringLiteral("auto");
    }
    return trimmed;
}

QString displayTerrainHeightFieldValue(const QString& value) {
    return normalizeTerrainHeightFieldValue(value) == "auto" ? QStringLiteral("Inferred") : value.trimmed();
}

QString formatFloatTokenForDisplay(const QString& token) {
    bool ok = false;
    const double value = token.trimmed().toDouble(&ok);
    return ok ? QString::number(value, 'f', 2) : token.trimmed();
}

QString formatFloatVectorForDisplay(const QString& text) {
    const QStringList tokens = splitLooseVectorTokens(text);
    if (tokens.isEmpty()) {
        return {};
    }

    QStringList displayTokens;
    displayTokens.reserve(tokens.size());
    for (const QString& token : tokens) {
        displayTokens.push_back(formatFloatTokenForDisplay(token));
    }
    return displayTokens.join(", ");
}

QString renderBracketedListText(const QString& text) {
    const QStringList tokens = splitLooseVectorTokens(text);
    if (tokens.isEmpty()) {
        return {};
    }
    return "[" + tokens.join(", ") + "]";
}

QString renderBracketedListText(const QStringList& parts) {
    QStringList tokens;
    tokens.reserve(parts.size());
    for (const QString& part : parts) {
        const QString trimmed = part.trimmed();
        if (!trimmed.isEmpty()) {
            tokens.push_back(trimmed);
        }
    }
    if (tokens.isEmpty()) {
        return {};
    }
    return "[" + tokens.join(", ") + "]";
}

QStringList tripletPartsFromRawText(const QString& text) {
    QStringList parts = splitLooseVectorTokens(text);
    while (parts.size() < 3) {
        parts.push_back({});
    }
    if (parts.size() > 3) {
        parts = parts.mid(0, 3);
    }
    return parts;
}

bool usesSplitTripletEditor(const FieldSpec& spec) {
    return spec.key == "n_gpu" || spec.key == "vk_inlet_anisotropy";
}

class PrecisionNumericLineEdit final : public QLineEdit {
public:
    enum class Mode {
        Scalar,
        Vector
    };

    explicit PrecisionNumericLineEdit(Mode mode, QWidget* parent = nullptr)
        : QLineEdit(parent)
        , mode_(mode) {
        if (mode_ == Mode::Scalar) {
            auto* validator = new QDoubleValidator(-1.0e100, 1.0e100, 15, this);
            validator->setNotation(QDoubleValidator::ScientificNotation);
            setValidator(validator);
        }
    }

    void setRawText(const QString& rawText) {
        fullText_ = normalizeText(rawText);
        applyPresentation();
    }

    QString canonicalText() const {
        if (hasFocus()) {
            return text().trimmed();
        }
        return fullText_.isEmpty() ? text().trimmed() : fullText_;
    }

    QString commitVisibleText() {
        fullText_ = normalizeText(text());
        return fullText_;
    }

protected:
    void focusInEvent(QFocusEvent* event) override {
        QLineEdit::focusInEvent(event);
        const QSignalBlocker blocker(this);
        setText(fullText_);
        selectAll();
    }

    void focusOutEvent(QFocusEvent* event) override {
        fullText_ = normalizeText(text());
        QLineEdit::focusOutEvent(event);
        applyPresentation();
    }

private:
    QString normalizeText(const QString& text) const {
        return mode_ == Mode::Scalar ? text.trimmed() : canonicalizeLooseVectorText(text);
    }

    QString displayText(const QString& text) const {
        if (text.trimmed().isEmpty()) {
            return {};
        }
        return mode_ == Mode::Scalar ? formatFloatTokenForDisplay(text) : formatFloatVectorForDisplay(text);
    }

    void applyPresentation() {
        const QSignalBlocker blocker(this);
        setText(hasFocus() ? fullText_ : displayText(fullText_));
        if (!hasFocus()) {
            deselect();
            setCursorPosition(0);
        }
    }

    Mode mode_;
    QString fullText_;
};

class SplitTripletEditor final : public QWidget {
public:
    enum class ValueMode {
        Integer,
        Float
    };

    explicit SplitTripletEditor(ValueMode mode, QWidget* parent = nullptr)
        : QWidget(parent)
        , mode_(mode) {
        auto* layout = new QHBoxLayout(this);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(6);

        for (int index = 0; index < 3; ++index) {
            QLineEdit* edit = nullptr;
            if (mode_ == ValueMode::Float) {
                edit = new PrecisionNumericLineEdit(PrecisionNumericLineEdit::Mode::Scalar, this);
            } else {
                edit = new QLineEdit(this);
                edit->setValidator(new QIntValidator(0, 999999999, edit));
            }
            edit->setMinimumWidth(0);
            edit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            layout->addWidget(edit, 1);
            edits_.push_back(edit);
        }
    }

    QList<QLineEdit*> lineEdits() const {
        return edits_;
    }

    QStringList canonicalParts() const {
        QStringList parts;
        parts.reserve(edits_.size());
        for (QLineEdit* edit : edits_) {
            if (auto* precision = dynamic_cast<PrecisionNumericLineEdit*>(edit)) {
                parts.push_back(precision->canonicalText());
            } else {
                parts.push_back(edit ? edit->text().trimmed() : QString());
            }
        }
        return parts;
    }

    QString rawTextForDeck() const {
        return renderBracketedListText(canonicalParts());
    }

    void setRawText(const QString& rawText) {
        const QStringList parts = tripletPartsFromRawText(rawText);
        for (int index = 0; index < edits_.size(); ++index) {
            QLineEdit* edit = edits_[index];
            if (auto* precision = dynamic_cast<PrecisionNumericLineEdit*>(edit)) {
                precision->setRawText(parts.value(index));
            } else {
                const QSignalBlocker blocker(edit);
                edit->setText(parts.value(index));
                edit->deselect();
                edit->setCursorPosition(0);
            }
        }
    }

private:
    ValueMode mode_;
    QList<QLineEdit*> edits_;
};

QString displayEnumLabel(const QString& key, const QString& value) {
    if (key == "geometry_mode") {
        if (value == "0") {
            return "building-only";
        }
        if (value == "1") {
            return "terrain-only";
        }
        if (value == "2") {
            return "building+terrain";
        }
    }
    if (key == "mesh_control") {
        if (value == "gpu_memory") {
            return "GPU memory";
        }
        if (value == "cell_size") {
            return "Cell size";
        }
    }
    if (key == "vk_inlet_uc_mode") {
        if (value == "NORMAL_COMPONENT") {
            return "Face-normal |U.n|";
        }
        if (value == "NORM_MEAN") {
            return "Mean speed |U|";
        }
    }
    if (key == "turb_inflow_approach") {
        if (value == "vonkarman") {
            return "Von Karman (harmonic)";
        }
        if (value == "smirnov") {
            return "Smirnov (Gaussian-like)";
        }
    }
    if (key == "terr_voxel_approach") {
        if (value == "idw") {
            return "IDW";
        }
        if (value == "kriging_gpu") {
            return "Kriging (GPU)";
        }
        if (value == "kriging") {
            return "Kriging";
        }
    }
    return value;
}

QString compactRunModeLabel(RunMode mode) {
    switch (mode) {
    case RunMode::Luw:
        return QStringLiteral("Normal");
    case RunMode::Luwdg:
        return QStringLiteral("Dataset");
    case RunMode::Luwpf:
        return QStringLiteral("Profile-based");
    }
    return QStringLiteral("Normal");
}

QString expandedRunModeLabel(RunMode mode) {
    switch (mode) {
    case RunMode::Luw:
        return QStringLiteral("Normal (LUW)");
    case RunMode::Luwdg:
        return QStringLiteral("Dataset generation (LUWDG)");
    case RunMode::Luwpf:
        return QStringLiteral("Profile-based dataset generation (LUWPF)");
    }
    return QStringLiteral("Normal (LUW)");
}

void populateModeComboBox(CompactModeComboBox* combo, RunMode selectedMode) {
    if (!combo) {
        return;
    }

    combo->clear();
    const QList<RunMode> modes = {RunMode::Luw, RunMode::Luwdg, RunMode::Luwpf};
    int popupWidth = 0;
    const QFontMetrics metrics(combo->font());
    for (RunMode mode : modes) {
        combo->addItem(expandedRunModeLabel(mode), static_cast<int>(mode));
        const int index = combo->count() - 1;
        combo->setItemData(index, compactRunModeLabel(mode), CompactModeComboBox::CompactTextRole);
        popupWidth = std::max(popupWidth, metrics.horizontalAdvance(expandedRunModeLabel(mode)));
    }
    combo->setCurrentIndex(static_cast<int>(selectedMode));
    if (combo->view()) {
        combo->view()->setMinimumWidth(popupWidth + 40);
    }
}

QColor iconStroke(const QPalette& palette) {
    return palette.color(QPalette::ButtonText);
}

QColor iconAccent(const QPalette& palette) {
    return palette.color(QPalette::Highlight);
}

QColor iconAccent(ToolbarGlyph glyph, ThemeMode mode, const QPalette& palette) {
    if (mode == ThemeMode::Himmel) {
        switch (glyph) {
        case ToolbarGlyph::RunWorkflow:
        case ToolbarGlyph::Solve:
        case ToolbarGlyph::Stop:
            return QColor("#8e2a2b");
        default:
            break;
        }
    }
    return palette.color(QPalette::LinkVisited);
}

bool isFrierenThemeFamily(ThemeMode mode) {
    return mode == ThemeMode::Frieren || mode == ThemeMode::Himmel;
}

QColor titleBarBaseColor(ThemeMode mode) {
    switch (mode) {
    case ThemeMode::LightDefault:
        return QColor("#356eab");
    case ThemeMode::DarkDefault:
        return QColor("#2d5f96");
    case ThemeMode::White:
        return QColor("#3c78b5");
    case ThemeMode::Black:
        return QColor("#000000");
    case ThemeMode::Pink:
        return QColor("#b75d84");
    case ThemeMode::Green:
        return QColor("#4a8f62");
    case ThemeMode::Siemens:
        return QColor(0, 100, 135);
    case ThemeMode::Frieren:
        return QColor("#c8a45a");
    case ThemeMode::Himmel:
        return QColor("#2e5fa8");
    }
    return QColor("#356eab");
}

QColor titleBarHoverColor(ThemeMode mode) {
    if (mode == ThemeMode::Black) {
        return QColor("#000000");
    }
    if (mode == ThemeMode::Frieren) {
        return QColor("#6b4f3a");
    }
    if (mode == ThemeMode::Himmel) {
        return QColor("#3d73bd");
    }
    return titleBarBaseColor(mode).lighter(116);
}

QColor titleBarPressedColor(ThemeMode mode) {
    if (mode == ThemeMode::Black) {
        return QColor("#000000");
    }
    if (mode == ThemeMode::Frieren) {
        return QColor("#5a412e");
    }
    if (mode == ThemeMode::Himmel) {
        return QColor("#244a83");
    }
    return titleBarBaseColor(mode).darker(112);
}

QIcon makeToolbarIcon(ToolbarGlyph glyph, ThemeMode mode, const QPalette& palette) {
    QPixmap pixmap(24, 24);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setBrush(Qt::NoBrush);

    QPen pen(iconStroke(palette), 1.7, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
    painter.setPen(pen);

    const QColor accent = iconAccent(glyph, mode, palette);
    QPen accentPen(accent, 1.7, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);

    switch (glyph) {
    case ToolbarGlyph::InspectWind: {
        QPainterPath gust;
        gust.moveTo(3.0, 7.0);
        gust.cubicTo(6.0, 4.0, 9.2, 4.0, 11.5, 7.0);
        gust.moveTo(4.0, 11.5);
        gust.cubicTo(7.0, 8.8, 10.4, 8.8, 13.0, 11.5);
        gust.moveTo(3.2, 16.0);
        gust.cubicTo(5.8, 13.5, 8.8, 13.5, 11.2, 16.0);
        painter.drawPath(gust);
        painter.drawEllipse(QRectF(13.0, 11.0, 6.0, 6.0));
        painter.drawLine(QPointF(17.4, 15.4), QPointF(20.0, 18.0));
        break;
    }
    case ToolbarGlyph::InspectBuildings: {
        painter.drawRect(QRectF(3.2, 10.0, 4.6, 9.0));
        painter.drawRect(QRectF(8.4, 6.6, 4.8, 12.4));
        painter.drawRect(QRectF(13.8, 11.8, 3.2, 7.2));
        painter.drawEllipse(QRectF(13.5, 3.0, 5.2, 5.2));
        painter.drawLine(QPointF(17.2, 6.8), QPointF(20.0, 9.4));
        break;
    }
    case ToolbarGlyph::RunWorkflow: {
        painter.drawEllipse(QRectF(3.0, 10.0, 3.6, 3.6));
        painter.drawEllipse(QRectF(9.0, 5.0, 3.6, 3.6));
        painter.drawEllipse(QRectF(15.2, 12.8, 3.6, 3.6));
        painter.drawLine(QPointF(6.4, 11.6), QPointF(10.2, 8.2));
        painter.drawLine(QPointF(12.6, 8.6), QPointF(16.2, 13.8));
        painter.setPen(accentPen);
        painter.drawLine(QPointF(8.0, 18.0), QPointF(17.2, 18.0));
        painter.drawLine(QPointF(14.8, 15.4), QPointF(17.2, 18.0));
        painter.drawLine(QPointF(14.8, 20.6), QPointF(17.2, 18.0));
        painter.setPen(pen);
        break;
    }
    case ToolbarGlyph::Crop: {
        painter.drawLine(QPointF(5.0, 3.2), QPointF(5.0, 10.2));
        painter.drawLine(QPointF(5.0, 3.2), QPointF(12.0, 3.2));
        painter.drawLine(QPointF(19.0, 9.4), QPointF(19.0, 16.4));
        painter.drawLine(QPointF(12.0, 16.4), QPointF(19.0, 16.4));
        painter.drawRect(QRectF(7.6, 6.2, 8.6, 8.6));
        break;
    }
    case ToolbarGlyph::Voxel: {
        for (int row = 0; row < 2; ++row) {
            for (int column = 0; column < 2; ++column) {
                painter.drawRect(QRectF(4.0 + 6.6 * column, 4.4 + 6.6 * row, 5.0, 5.0));
            }
        }
        painter.drawRect(QRectF(10.6, 11.0, 5.0, 5.0));
        break;
    }
    case ToolbarGlyph::Batch: {
        painter.drawRoundedRect(QRectF(4.0, 4.0, 11.8, 4.8), 1.4, 1.4);
        painter.drawRoundedRect(QRectF(6.0, 9.6, 11.8, 4.8), 1.4, 1.4);
        painter.drawRoundedRect(QRectF(8.0, 15.2, 10.8, 4.4), 1.4, 1.4);
        break;
    }
    case ToolbarGlyph::Validate: {
        painter.drawRoundedRect(QRectF(4.0, 4.0, 16.0, 16.0), 4.0, 4.0);
        painter.drawLine(QPointF(7.4, 12.0), QPointF(10.6, 15.4));
        painter.drawLine(QPointF(10.6, 15.4), QPointF(16.8, 8.2));
        break;
    }
    case ToolbarGlyph::Solve: {
        painter.setPen(accentPen);
        painter.drawArc(QRectF(4.2, 4.2, 15.6, 15.6), 30 * 16, 280 * 16);
        painter.drawLine(QPointF(15.8, 4.8), QPointF(18.8, 6.0));
        painter.drawLine(QPointF(15.8, 4.8), QPointF(17.0, 7.8));
        painter.setPen(pen);
        painter.drawLine(QPointF(6.0, 14.2), QPointF(10.2, 10.4));
        painter.drawLine(QPointF(10.2, 10.4), QPointF(13.2, 13.0));
        break;
    }
    case ToolbarGlyph::Stop: {
        painter.setPen(accentPen);
        painter.drawRoundedRect(QRectF(4.0, 4.0, 16.0, 16.0), 4.0, 4.0);
        painter.drawLine(QPointF(8.0, 8.0), QPointF(16.0, 16.0));
        painter.drawLine(QPointF(16.0, 8.0), QPointF(8.0, 16.0));
        painter.setPen(pen);
        break;
    }
    case ToolbarGlyph::Console: {
        painter.drawRoundedRect(QRectF(3.6, 5.0, 16.8, 13.2), 2.0, 2.0);
        painter.drawLine(QPointF(6.8, 9.0), QPointF(9.6, 12.0));
        painter.drawLine(QPointF(9.6, 12.0), QPointF(6.8, 15.0));
        painter.drawLine(QPointF(12.2, 15.0), QPointF(17.0, 15.0));
        break;
    }
    case ToolbarGlyph::SidePanel: {
        painter.drawRoundedRect(QRectF(3.4, 4.0, 17.2, 16.0), 2.0, 2.0);
        painter.drawLine(QPointF(14.2, 4.4), QPointF(14.2, 19.6));
        painter.drawLine(QPointF(6.4, 8.6), QPointF(11.0, 8.6));
        painter.drawLine(QPointF(6.4, 12.0), QPointF(11.0, 12.0));
        painter.drawLine(QPointF(6.4, 15.4), QPointF(11.0, 15.4));
        painter.drawLine(QPointF(16.8, 8.2), QPointF(18.8, 8.2));
        painter.drawLine(QPointF(16.8, 12.0), QPointF(18.8, 12.0));
        painter.drawLine(QPointF(16.8, 15.8), QPointF(18.8, 15.8));
        break;
    }
    }

    return QIcon(pixmap);
}

QIcon makeWindowControlIcon(WindowControlGlyph glyph, const QColor& color) {
    QPixmap pixmap(20, 20);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setBrush(Qt::NoBrush);
    QPen pen(color, glyph == WindowControlGlyph::Close ? 2.0 : 1.7, Qt::SolidLine, Qt::SquareCap, Qt::MiterJoin);
    painter.setPen(pen);

    switch (glyph) {
    case WindowControlGlyph::Minimize:
        painter.drawLine(QPointF(4.5, 10.5), QPointF(15.5, 10.5));
        break;
    case WindowControlGlyph::Maximize:
        painter.drawRect(QRectF(4.5, 4.5, 11.0, 11.0));
        break;
    case WindowControlGlyph::Restore:
        painter.drawRect(QRectF(6.5, 4.5, 9.0, 9.0));
        painter.drawLine(QPointF(6.5, 6.5), QPointF(4.5, 6.5));
        painter.drawLine(QPointF(4.5, 6.5), QPointF(4.5, 15.5));
        painter.drawLine(QPointF(4.5, 15.5), QPointF(13.5, 15.5));
        break;
    case WindowControlGlyph::Close:
        painter.drawLine(QPointF(5.2, 5.2), QPointF(14.8, 14.8));
        painter.drawLine(QPointF(14.8, 5.2), QPointF(5.2, 14.8));
        break;
    }

    return QIcon(pixmap);
}

QPixmap makeLogoPixmap(const QColor& stroke, const QColor& accent, const QString& text) {
    const bool hasText = !text.trimmed().isEmpty();
    QPixmap pixmap(hasText ? QSize(240, 30) : QSize(34, 30));
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);

    QPen pen(stroke, 1.7, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    painter.drawRect(QRectF(3.8, 9.4, 3.8, 8.6));
    painter.drawRect(QRectF(9.6, 6.6, 4.4, 11.4));
    painter.setPen(QPen(accent, 1.8, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    QPainterPath flowOne;
    flowOne.moveTo(1.5, 8.0);
    flowOne.cubicTo(7.0, 4.0, 11.0, 4.4, 16.0, 8.2);
    flowOne.cubicTo(20.4, 11.4, 24.2, 11.2, 28.2, 7.0);
    painter.drawPath(flowOne);
    QPainterPath flowTwo;
    flowTwo.moveTo(1.8, 14.0);
    flowTwo.cubicTo(7.4, 10.0, 11.8, 10.2, 16.2, 14.6);
    flowTwo.cubicTo(20.2, 18.2, 24.0, 18.0, 28.0, 13.6);
    painter.drawPath(flowTwo);
    QPainterPath flowThree;
    flowThree.moveTo(1.8, 20.0);
    flowThree.cubicTo(7.0, 16.2, 11.0, 16.6, 15.6, 20.2);
    flowThree.cubicTo(19.6, 23.2, 23.4, 23.0, 27.6, 18.8);
    painter.drawPath(flowThree);

    if (hasText) {
        QFont font("Segoe UI");
        font.setBold(true);
        font.setPointSize(11);
        painter.setFont(font);
        painter.setPen(stroke);
        painter.drawText(QRectF(40.0, 1.0, 194.0, 28.0), Qt::AlignVCenter | Qt::AlignLeft, text);
    }

    return pixmap;
}

bool copyFileReplacing(const QString& sourcePath, const QString& targetPath, QString* errorMessage = nullptr) {
    if (QFileInfo(sourcePath).absoluteFilePath() == QFileInfo(targetPath).absoluteFilePath()) {
        return true;
    }
    QFile::remove(targetPath);
    if (QFile::copy(sourcePath, targetPath)) {
        return true;
    }
    if (errorMessage) {
        *errorMessage = QString("Failed to copy %1 to %2.").arg(sourcePath, targetPath);
    }
    return false;
}

enum class ProjectImportKind {
    Unknown,
    Building,
    Terrain,
    Wind
};

ProjectImportKind importKindForSubdirectory(const QString& targetSubdirectory) {
    const QString normalized = targetSubdirectory.trimmed().toLower();
    if (normalized == "building_db") {
        return ProjectImportKind::Building;
    }
    if (normalized == "terrain_db") {
        return ProjectImportKind::Terrain;
    }
    if (normalized == "wind_bc") {
        return ProjectImportKind::Wind;
    }
    return ProjectImportKind::Unknown;
}

QString importKindDisplayName(ProjectImportKind kind) {
    switch (kind) {
    case ProjectImportKind::Building:
        return QStringLiteral("building database");
    case ProjectImportKind::Terrain:
        return QStringLiteral("terrain database");
    case ProjectImportKind::Wind:
        return QStringLiteral("wind data");
    case ProjectImportKind::Unknown:
        break;
    }
    return QStringLiteral("project asset");
}

QString importDialogFilter(ProjectImportKind kind) {
    switch (kind) {
    case ProjectImportKind::Building:
        return QStringLiteral("Building Shapefile (*.shp)");
    case ProjectImportKind::Terrain:
        return QStringLiteral("Terrain Data (*.shp *.tif *.tiff)");
    case ProjectImportKind::Wind:
        return QStringLiteral("Wind Data (*.nc *.out)");
    case ProjectImportKind::Unknown:
        break;
    }
    return QStringLiteral("All files (*.*)");
}

QStringList companionExtensionsFor(const QString& suffix) {
    const QString ext = suffix.toLower();
    if (ext == "shp") {
        return {"shp", "shx", "dbf", "prj", "cpg", "qix", "sbn", "sbx"};
    }
    if (ext == "tif" || ext == "tiff") {
        return {"tif", "tiff", "tfw", "prj", "aux.xml", "ovr"};
    }
    return {suffix};
}

bool isRelevantImportArtifact(ProjectImportKind kind, const QString& fileName) {
    const QString lower = fileName.trimmed().toLower();
    if (lower.isEmpty()) {
        return false;
    }

    switch (kind) {
    case ProjectImportKind::Building:
        return lower.endsWith(".shp")
            || lower.endsWith(".shx")
            || lower.endsWith(".dbf")
            || lower.endsWith(".prj")
            || lower.endsWith(".cpg")
            || lower.endsWith(".qix")
            || lower.endsWith(".sbn")
            || lower.endsWith(".sbx")
            || lower.endsWith(".stl");
    case ProjectImportKind::Terrain:
        return lower.endsWith(".shp")
            || lower.endsWith(".shx")
            || lower.endsWith(".dbf")
            || lower.endsWith(".prj")
            || lower.endsWith(".cpg")
            || lower.endsWith(".qix")
            || lower.endsWith(".sbn")
            || lower.endsWith(".sbx")
            || lower.endsWith(".tif")
            || lower.endsWith(".tiff")
            || lower.endsWith(".tfw")
            || lower.endsWith(".aux.xml")
            || lower.endsWith(".ovr");
    case ProjectImportKind::Wind:
        return lower.endsWith(".nc") || lower.endsWith(".out");
    case ProjectImportKind::Unknown:
        break;
    }
    return false;
}

QStringList existingImportArtifacts(const QString& directoryPath, ProjectImportKind kind) {
    QStringList files;
    const QDir directory(directoryPath);
    const QFileInfoList entries = directory.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);
    for (const QFileInfo& entry : entries) {
        if (isRelevantImportArtifact(kind, entry.fileName())) {
            files.push_back(entry.absoluteFilePath());
        }
    }
    return files;
}

bool removeFilesReplacing(const QStringList& filePaths, QString* errorMessage = nullptr) {
    for (const QString& filePath : filePaths) {
        const QFileInfo fileInfo(filePath);
        if (!fileInfo.exists()) {
            continue;
        }
        if (!QFile::remove(filePath)) {
            if (errorMessage) {
                *errorMessage = QString("Failed to remove existing file %1.").arg(QDir::toNativeSeparators(filePath));
            }
            return false;
        }
    }
    return true;
}

QString sanitizeImportedBaseName(QString text, const QString& fallback) {
    text = text.trimmed();
    text.replace(QRegularExpression(R"([\\/:*?"<>|\s]+)"), "_");
    text.replace(QRegularExpression(R"(_+)"), "_");
    while (text.startsWith('_')) {
        text.remove(0, 1);
    }
    while (text.endsWith('_')) {
        text.chop(1);
    }
    return text.isEmpty() ? fallback : text;
}

QString canonicalImportBaseName(ProjectImportKind kind, const QString& sourcePath, const luwgui::ConfigDocument* document) {
    switch (kind) {
    case ProjectImportKind::Building:
        return QStringLiteral("rawbuildings");
    case ProjectImportKind::Terrain: {
        const QString suffix = QFileInfo(sourcePath).suffix().toLower();
        if (suffix == "shp") {
            const QString caseName = document ? document->typedValue("casename").toString() : QString();
            return sanitizeImportedBaseName(caseName, QStringLiteral("terrain"));
        }
        return QStringLiteral("dem");
    }
    case ProjectImportKind::Wind:
    case ProjectImportKind::Unknown:
        break;
    }
    return sanitizeImportedBaseName(QFileInfo(sourcePath).completeBaseName(), QStringLiteral("imported"));
}

QString targetFileNameForBaseName(const QString& sourcePath, const QString& canonicalBaseName) {
    const QFileInfo sourceInfo(sourcePath);
    const QString fileName = sourceInfo.fileName();
    const QString lowerFileName = fileName.toLower();
    if (lowerFileName.endsWith(".aux.xml")) {
        const QString auxSuffix = QStringLiteral(".aux.xml");
        const QString prefix = fileName.left(fileName.size() - auxSuffix.size());
        const QString innerSuffix = QFileInfo(prefix).suffix();
        return innerSuffix.isEmpty()
            ? canonicalBaseName + auxSuffix
            : canonicalBaseName + "." + innerSuffix + auxSuffix;
    }

    const QString suffix = sourceInfo.suffix();
    return suffix.isEmpty() ? canonicalBaseName : canonicalBaseName + "." + suffix;
}

QString canonicalWindFileName(const QString& sourcePath, const luwgui::ConfigDocument* document) {
    const QString fallbackCaseName = sanitizeImportedBaseName(QFileInfo(sourcePath).completeBaseName(), QStringLiteral("wind"));
    const QString caseName = sanitizeImportedBaseName(
        document ? document->typedValue("casename").toString() : QString(),
        fallbackCaseName);

    QString datetimeToken = document ? document->typedValue("datetime").toString().trimmed() : QString();
    datetimeToken.remove(QRegularExpression(R"([\\/:*?"<>|\s]+)"));
    if (datetimeToken.isEmpty()) {
        return caseName + ".nc";
    }
    return caseName + "_" + datetimeToken + ".nc";
}

QStringList sourceFilesForImport(const QString& selectedFile) {
    std::set<QString> sourceFiles;
    const QFileInfo sourceInfo(selectedFile);
    const QString baseName = sourceInfo.completeBaseName();
    const QDir sourceDir = sourceInfo.dir();
    QStringList candidates;

    for (const QString& extension : companionExtensionsFor(sourceInfo.suffix())) {
        if (extension.contains('.')) {
            const QString sidecarName = sourceInfo.fileName() + "." + extension;
            if (sourceDir.exists(sidecarName)) {
                candidates.push_back(sourceDir.filePath(sidecarName));
            }
            continue;
        }
        const QStringList matches = sourceDir.entryList({baseName + "." + extension}, QDir::Files);
        for (const QString& match : matches) {
            candidates.push_back(sourceDir.filePath(match));
        }
    }

    if (candidates.isEmpty()) {
        candidates.push_back(selectedFile);
    }

    for (const QString& candidate : candidates) {
        sourceFiles.insert(QFileInfo(candidate).absoluteFilePath());
    }

    if (sourceInfo.fileName().toLower().endsWith(".aux.xml")) {
        sourceFiles.insert(sourceInfo.absoluteFilePath());
    }

    return QStringList(sourceFiles.begin(), sourceFiles.end());
}

QString chooseDeckFilePath(QWidget* parent,
                           const QString& title,
                           const QString& startPath,
                           QFileDialog::AcceptMode acceptMode,
                           QFileDialog::FileMode fileMode) {
    static const QString kDeckFilter = "LUW Deck (*.luw *.luwdg *.luwpf)";

    if (acceptMode == QFileDialog::AcceptOpen && fileMode == QFileDialog::ExistingFile) {
        return QFileDialog::getOpenFileName(parent, title, startPath, kDeckFilter);
    }

    if (acceptMode == QFileDialog::AcceptSave) {
        QString path = QFileDialog::getSaveFileName(parent, title, startPath, kDeckFilter);
        if (!path.isEmpty() && QFileInfo(path).suffix().isEmpty()) {
            path += ".luw";
        }
        return path;
    }

    QFileDialog dialog(parent, title, startPath, kDeckFilter);
    dialog.setAcceptMode(acceptMode);
    dialog.setFileMode(fileMode);
    if (dialog.exec() != QDialog::Accepted) {
        return {};
    }
    return dialog.selectedFiles().value(0);
}

} // namespace

MainWindow::MainWindow(const AppPreferences& preferences, QWidget* parent)
    : QMainWindow(parent)
    , document_(new ConfigDocument(this))
    , runner_(new CommandRunner(this))
    , preferences_(preferences) {
    setWindowTitle("LatticeUrbanWind Studio");
    setWindowFlags(Qt::Window | Qt::FramelessWindowHint);
    runner_->setDocument(document_);
    buildTitleBar();
    buildUi();
    buildMenus();

    connect(document_, &ConfigDocument::changed, this, &MainWindow::refreshEditors);
    connect(document_, &ConfigDocument::changed, this, &MainWindow::refreshRawDeck);
    connect(document_, &ConfigDocument::changed, this, &MainWindow::refreshFileTree);
    connect(document_, &ConfigDocument::changed, this, [this] {
        if (vtkView_) {
            vtkView_->setProjectDirectory(document_->projectDirectory());
        }
        if (boundaryCsvPanel_) {
            boundaryCsvPanel_->setProjectDirectory(document_->projectDirectory());
        }
    });
    connect(document_, &ConfigDocument::keyChanged, this, [this](const QString& key) {
        if (keyRequiresTreeRebuild(key)) {
            rebuildSectionPages(currentNodeId_);
        }
    });
    connect(document_, &ConfigDocument::documentReloaded, this, [this] {
        if (vtkView_) {
            vtkView_->setProjectDirectory(document_->projectDirectory());
        }
        if (boundaryCsvPanel_) {
            boundaryCsvPanel_->setProjectDirectory(document_->projectDirectory());
        }
        rebuildSectionPages(currentNodeId_);
    });
    connect(document_, &ConfigDocument::externalFileReloaded, this, [this](const QString& filePath) {
        const QString message = "Reloaded backend-updated deck: " + QDir::toNativeSeparators(filePath);
        statusBar()->showMessage(message, 5000);
        if (console_) {
            console_->appendText("[GUI] " + message + "\n");
        }
    });
    connect(document_, &ConfigDocument::externalFileReloadFailed, this, [this](const QString& filePath, const QString& error) {
        const QString message = QString("Failed to reload updated deck %1: %2")
            .arg(QDir::toNativeSeparators(filePath), error);
        statusBar()->showMessage(message, 5000);
        if (console_) {
            console_->appendText("[GUI] " + message + "\n");
        }
    });
    connect(document_, &ConfigDocument::modeChanged, this, [this](RunMode mode) {
        if (modeCombo_) {
            const QSignalBlocker blocker(modeCombo_);
            modeCombo_->setCurrentIndex(static_cast<int>(mode));
        }
        rebuildSectionPages(currentNodeId_);
        updateAuxiliaryPanelLayout();
        refreshEditors();
    });
    connect(runner_, &CommandRunner::outputReady, this, [this](const QString& text) {
        if (console_) {
            console_->appendText(text);
        }
    });
    connect(runner_, &CommandRunner::errorText, this, [this](const QString& text) {
        if (console_) {
            console_->appendText(text);
        }
    });
    connect(runner_, &CommandRunner::progressUpdated, this, [this](const QString& summary,
                                                                    const QString& detail,
                                                                    qint64 current,
                                                                    qint64 total,
                                                                    bool indeterminate) {
        if (progressPanel_) {
            progressPanel_->setProgress(summary, detail, current, total, indeterminate);
        }
    });
    connect(runner_, &CommandRunner::started, this, [this](const QString& title) {
        stopRequested_ = false;
        statusBar()->showMessage("Running: " + title);
        ensureProjectWorkspaceLoaded();
        if (console_) {
            console_->appendText("\n[GUI] Running " + title + "\n");
        }
        if (progressPanel_) {
            progressPanel_->setBusy(title, QString());
        }
    });
    connect(runner_, &CommandRunner::finished, this, [this](const QString& title, int exitCode, QProcess::ExitStatus exitStatus) {
        QString statusMessageText;
        if (stopRequested_) {
            statusMessageText = title + " stopped";
        } else if (exitStatus == QProcess::NormalExit && exitCode == 0) {
            statusMessageText = title + " completed";
        } else {
            statusMessageText = title + " finished with exit code " + QString::number(exitCode);
        }
        statusBar()->showMessage(statusMessageText, 5000);
        if (console_) {
            console_->appendText("[GUI] " + statusMessageText + "\n");
        }
        if (progressPanel_) {
            if (stopRequested_) {
                progressPanel_->showTerminalStatus("Stopped", title.toLower());
            } else if (exitStatus == QProcess::NormalExit && exitCode == 0) {
                progressPanel_->showTerminalStatus("Completed", title.toLower());
            } else {
                progressPanel_->showTerminalStatus("Failed", title.toLower() + ", exit code " + QString::number(exitCode));
            }
        }
        QString reloadError;
        if (!document_->reloadFromDisk(&reloadError) && !document_->filePath().isEmpty() && !reloadError.trimmed().isEmpty()) {
            if (console_) {
                console_->appendText("[GUI] Deck reload skipped: " + reloadError + "\n");
            }
        }
        if (title == "Solve" && exitCode == 0 && vtkView_) {
            vtkView_->handleSolverFinished();
        }
        stopRequested_ = false;
    });
    updateAuxiliaryPanelLayout();
    updateProjectAvailability();
    applyPreferences();
    qApp->installEventFilter(this);
    resize(1800, 1040);
}

void MainWindow::appendStartupReport(const StartupCheckResult& startupCheck) {
    ensureCenterWorkspaceCreated();
    if (!console_) {
        return;
    }

    console_->setCollapsed(false);
    if (centerSplitter_) {
        centerSplitter_->setVisible(true);
    }
    syncWorkflowChromeButtons();

    int packageFailures = 0;
    for (const PythonImportCheck& check : startupCheck.packageChecks) {
        if (!check.success) {
            ++packageFailures;
        }
    }

    QStringList lines;
    lines << "[Startup] ---- Runtime Summary ----";
    lines << "[Startup] Software: LatticeUrbanWind Studio";
    lines << QString("[Startup] Software version: %1").arg(buildinfo::kStudioVersion);
    lines << QString("[Startup] GUI version: %1").arg(buildinfo::kStudioVersion);
    lines << QString("[Startup] FluidX3D core: %1").arg(buildinfo::kCfdCoreVersion);
    lines << QString("[Startup] Build timestamp: %1").arg(buildinfo::kBuildTimestamp);
    lines << QString("[Startup] Repository root: %1").arg(QDir::toNativeSeparators(startupCheck.repoRoot));
    lines << QString("[Startup] Host platform: %1").arg(startupCheck.hostPlatform.isEmpty() ? QStringLiteral("unknown") : startupCheck.hostPlatform);
    lines << QString("[Startup] Python: %1 | %2")
                 .arg(startupCheck.pythonVersion.isEmpty() ? QStringLiteral("unknown") : startupCheck.pythonVersion,
                      startupCheck.pythonExecutable.isEmpty() ? QStringLiteral("unresolved") : QDir::toNativeSeparators(startupCheck.pythonExecutable));
    lines << QString("[Startup] Python environment: %1")
                 .arg(startupCheck.pythonSummary.isEmpty()
                          ? QStringLiteral("summary unavailable")
                          : startupCheck.pythonSummary);

    if (packageFailures > 0) {
        lines << QString("[Startup] Python import failures: %1").arg(packageFailures);
        for (const PythonImportCheck& check : startupCheck.packageChecks) {
            if (!check.success) {
                lines << QString("[Startup]   - %1 -> %2").arg(check.packageName, check.errorText);
            }
        }
    }

    QString cudaLine = startupCheck.cuda.summary;
    if (cudaLine.isEmpty()) {
        cudaLine = startupCheck.cuda.available
            ? QStringLiteral("CUDA available")
            : QStringLiteral("CUDA unavailable");
    }
    if (!startupCheck.cuda.runtimeVersion.isEmpty()) {
        cudaLine += QString(" | runtime %1").arg(startupCheck.cuda.runtimeVersion);
    }
    if (!startupCheck.cuda.driverVersion.isEmpty()) {
        cudaLine += QString(" | driver %1").arg(startupCheck.cuda.driverVersion);
    }
    if (!startupCheck.cuda.numbaVersion.isEmpty()) {
        cudaLine += QString(" | numba %1").arg(startupCheck.cuda.numbaVersion);
    }
    lines << (QStringLiteral("[Startup] CUDA: ") + cudaLine);
    if (!startupCheck.cuda.devices.isEmpty()) {
        const AcceleratorDeviceCheck& device = startupCheck.cuda.devices.front();
        QString deviceLine = QString("[Startup] CUDA primary device: %1").arg(device.name);
        if (!device.computeCapability.isEmpty()) {
            deviceLine += QString(" | cc %1").arg(device.computeCapability);
        }
        lines << deviceLine;
    }
    if (!startupCheck.cuda.errorText.isEmpty()) {
        lines << (QStringLiteral("[Startup] CUDA details: ") + startupCheck.cuda.errorText);
    }

    QString openclLine = startupCheck.opencl.summary;
    if (openclLine.isEmpty()) {
        openclLine = startupCheck.opencl.available
            ? QStringLiteral("OpenCL available")
            : QStringLiteral("OpenCL unavailable");
    }
    if (!startupCheck.opencl.version.isEmpty()) {
        openclLine += QString(" | %1").arg(startupCheck.opencl.version);
    }
    lines << (QStringLiteral("[Startup] OpenCL: ") + openclLine);
    if (!startupCheck.opencl.devices.isEmpty()) {
        const AcceleratorDeviceCheck& device = startupCheck.opencl.devices.front();
        QString deviceLine = QString("[Startup] OpenCL primary device: %1").arg(device.name);
        if (!device.driverVersion.isEmpty()) {
            deviceLine += QString(" | driver %1").arg(device.driverVersion);
        }
        lines << deviceLine;
    }
    if (!startupCheck.opencl.errorText.isEmpty()) {
        lines << (QStringLiteral("[Startup] OpenCL details: ") + startupCheck.opencl.errorText);
    }

    lines << "[Startup] -------------------------";
    console_->appendText(lines.join('\n') + '\n');
}

bool MainWindow::eventFilter(QObject* watched, QEvent* event) {
    if (event
        && event->type() == QEvent::MouseButtonDblClick
        && watched
        && watched->property("propertyField").toBool()) {
        const int bindingIndex = watched->property("bindingIndex").toInt();
        unlockPropertyEditor(bindingIndex);
        return true;
    }

    if (event && event->type() == QEvent::MouseButtonPress) {
        if (auto* widget = qobject_cast<QWidget*>(watched)) {
            if (widget->window() == this) {
                if (QWidget* panelShell = trackedPanelFor(widget)) {
                    setActiveTrackedPanel(panelShell);
                }
            }
        }
    }
    return QMainWindow::eventFilter(watched, event);
}

void MainWindow::changeEvent(QEvent* event) {
    QMainWindow::changeEvent(event);
    if (event && event->type() == QEvent::WindowStateChange) {
        refreshWindowButtons();
    }
}

#ifdef Q_OS_WIN
bool MainWindow::nativeEvent(const QByteArray& eventType, void* message, qintptr* result) {
    Q_UNUSED(eventType)

    MSG* msg = static_cast<MSG*>(message);
    if (!msg) {
        return false;
    }

    if (msg->message == WM_NCLBUTTONDOWN && msg->wParam == HTCAPTION) {
        recordTitleBarThemeClick();
    }

    if (msg->message == WM_NCHITTEST) {
        const QPoint globalPos(GET_X_LPARAM(msg->lParam), GET_Y_LPARAM(msg->lParam));
        const QPoint pos = mapFromGlobal(globalPos);
        const QRect bounds = rect();
        constexpr int kResizeMargin = 7;

        if (!isMaximized()) {
            const bool onLeft = pos.x() >= 0 && pos.x() < kResizeMargin;
            const bool onRight = pos.x() <= bounds.width() && pos.x() > bounds.width() - kResizeMargin;
            const bool onTop = pos.y() >= 0 && pos.y() < kResizeMargin;
            const bool onBottom = pos.y() <= bounds.height() && pos.y() > bounds.height() - kResizeMargin;

            if (onTop && onLeft) {
                *result = HTTOPLEFT;
                return true;
            }
            if (onTop && onRight) {
                *result = HTTOPRIGHT;
                return true;
            }
            if (onBottom && onLeft) {
                *result = HTBOTTOMLEFT;
                return true;
            }
            if (onBottom && onRight) {
                *result = HTBOTTOMRIGHT;
                return true;
            }
            if (onLeft) {
                *result = HTLEFT;
                return true;
            }
            if (onRight) {
                *result = HTRIGHT;
                return true;
            }
            if (onTop) {
                *result = HTTOP;
                return true;
            }
            if (onBottom) {
                *result = HTBOTTOM;
                return true;
            }
        }

        if (titleBarWidget_ && titleBarWidget_->geometry().contains(pos)) {
            const QPoint titlePos = titleBarWidget_->mapFrom(this, pos);
            QWidget* child = titleBarWidget_->childAt(titlePos);
            const bool overMenu = isWidgetOrAncestor(child, titleMenuStrip_);
            const bool overWindowButton = isWidgetOrAncestor(child, minimizeWindowButton_)
                || isWidgetOrAncestor(child, maximizeWindowButton_)
                || isWidgetOrAncestor(child, closeWindowButton_);
            if (!overMenu && !overWindowButton) {
                *result = HTCAPTION;
                return true;
            }
        }
    }

    return false;
}
#endif

void MainWindow::recordTitleBarThemeClick() {
    if (!isFrierenThemeFamily(preferences_.themeMode)) {
        titleBarThemeClickTimes_.clear();
        return;
    }

    constexpr qint64 kToggleWindowMs = 5000;
    constexpr qsizetype kToggleClickCount = 3;

    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    titleBarThemeClickTimes_.append(now);
    while (!titleBarThemeClickTimes_.isEmpty() && now - titleBarThemeClickTimes_.front() > kToggleWindowMs) {
        titleBarThemeClickTimes_.removeFirst();
    }

    if (titleBarThemeClickTimes_.size() >= kToggleClickCount) {
        titleBarThemeClickTimes_.clear();
        toggleHiddenThemeVariant();
    }
}

void MainWindow::toggleHiddenThemeVariant() {
    if (!isFrierenThemeFamily(preferences_.themeMode)) {
        return;
    }

    const ThemeMode previous = preferences_.themeMode;
    const ThemeMode next = preferences_.themeMode == ThemeMode::Himmel
        ? ThemeMode::Frieren
        : ThemeMode::Himmel;

    preferences_.themeMode = next;
    QString error;
    if (!persistPreferences(&error)) {
        preferences_.themeMode = previous;
        QMessageBox::critical(this, "Preferences", error);
        return;
    }

    applyPreferences();
    const QString consoleMessage = next == ThemeMode::Frieren
        ? QStringLiteral("(Theme) Frieren: Beyond Journey's End")
        : QStringLiteral("(Theme) Himmel: One Lifetime of Love.");
    const QString statusMessage = next == ThemeMode::Frieren
        ? QStringLiteral("Frieren: Beyond Journey's End")
        : QStringLiteral("Himmel: One Lifetime of Love.");
    statusBar()->showMessage(statusMessage, 3000);
    logGuiAction(consoleMessage);
}

void MainWindow::buildTitleBar() {
    auto* titleBar = new HeaderBarWidget(this);
    titleBar->setObjectName("appTitleBar");
    titleBar->setFixedHeight(42);

    auto* layout = new QHBoxLayout(titleBar);
    layout->setContentsMargins(10, 0, 0, 0);
    layout->setSpacing(6);

    titleBarLogo_ = new QLabel(titleBar);
    titleBarLogo_->setObjectName("titleBarLogo");
    titleBarLogo_->setFixedSize(34, 30);
    titleBarLogo_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    titleBarLogo_->setAttribute(Qt::WA_TransparentForMouseEvents, true);
    layout->addWidget(titleBarLogo_);

    titleMenuStrip_ = new QWidget(titleBar);
    titleMenuStrip_->setObjectName("titleMenuStrip");
    titleMenuStrip_->setFixedHeight(42);
    auto* menuStripLayout = new QHBoxLayout(titleMenuStrip_);
    menuStripLayout->setContentsMargins(0, 0, 0, 0);
    menuStripLayout->setSpacing(0);
    layout->addWidget(titleMenuStrip_, 0, Qt::AlignVCenter);
    layout->addStretch(1);

    titleBarTitle_ = new QLabel("LatticeUrbanWind", titleBar);
    titleBarTitle_->setObjectName("titleBarTitle");
    titleBarTitle_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    titleBarTitle_->setAttribute(Qt::WA_TransparentForMouseEvents, true);
    QFont titleFont("Segoe UI");
    titleFont.setBold(true);
    titleFont.setPixelSize(30);
    titleBarTitle_->setFont(titleFont);
    layout->addWidget(titleBarTitle_, 0, Qt::AlignVCenter);

    auto makeWindowButton = [titleBar](WindowControlGlyph glyph, const QString& toolTip, const QString& objectName) {
        auto* button = new QToolButton(titleBar);
        button->setObjectName(objectName);
        button->setToolTip(toolTip);
        button->setFixedSize(42, 42);
        button->setAutoRaise(true);
        button->setCursor(Qt::PointingHandCursor);
        button->setIcon(makeWindowControlIcon(glyph, QColor("#ffffff")));
        button->setIconSize(QSize(20, 20));
        return button;
    };

    minimizeWindowButton_ = makeWindowButton(WindowControlGlyph::Minimize, "minimize", "titleMinimize");
    maximizeWindowButton_ = makeWindowButton(WindowControlGlyph::Maximize, "maximize", "titleMaximize");
    closeWindowButton_ = makeWindowButton(WindowControlGlyph::Close, "close", "titleClose");
    layout->addWidget(minimizeWindowButton_);
    layout->addWidget(maximizeWindowButton_);
    layout->addWidget(closeWindowButton_);

    connect(minimizeWindowButton_, &QToolButton::clicked, this, &QWidget::showMinimized);
    connect(maximizeWindowButton_, &QToolButton::clicked, this, [this] {
        isMaximized() ? showNormal() : showMaximized();
        refreshWindowButtons();
    });
    connect(closeWindowButton_, &QToolButton::clicked, this, &QWidget::close);

    titleBarWidget_ = titleBar;
    setMenuWidget(titleBarWidget_);
    applyTitleBarStyle();
    refreshWindowButtons();
}

void MainWindow::buildUi() {
    auto* central = new QWidget(this);
    auto* centralLayout = new QVBoxLayout(central);
    centralLayout->setContentsMargins(0, 0, 0, 0);
    centralLayout->setSpacing(0);

    workflowBar_ = buildWorkflowBar();
    centralLayout->addWidget(workflowBar_);

    auto* root = new QSplitter(Qt::Horizontal, central);
    rootSplitter_ = root;
    centralLayout->addWidget(root, 1);
    setCentralWidget(central);
    root->setChildrenCollapsible(false);
    root->setCollapsible(0, false);

    leftSplitter_ = new QSplitter(Qt::Vertical, root);
    root->addWidget(leftSplitter_);
    leftSplitter_->setMinimumWidth(280);

    workspaceSplitter_ = new QSplitter(Qt::Horizontal, root);
    root->addWidget(workspaceSplitter_);
    workspaceSplitter_->setCollapsible(0, false);
    allowHorizontalCompression(workspaceSplitter_);

    progressPanel_ = new ProgressPanel();
    progressPanelShell_ = createPanelShell(leftSplitter_, progressPanel_, "progressPanelShell");
    leftSplitter_->addWidget(progressPanelShell_);

    sectionStack_ = new QStackedWidget();
    sectionPanelShell_ = createPanelShell(leftSplitter_, sectionStack_, "sectionPanelShell");
    leftSplitter_->addWidget(sectionPanelShell_);
    leftSplitter_->setStretchFactor(0, 0);
    leftSplitter_->setStretchFactor(1, 1);

    centerHost_ = new QWidget(workspaceSplitter_);
    allowHorizontalCompression(centerHost_);
    auto* centerHostLayout = new QVBoxLayout(centerHost_);
    centerHostLayout->setContentsMargins(0, 0, 0, 0);
    centerHostLayout->setSpacing(0);
    // Create the OpenGL-backed viewer before the top-level window is first shown.
    // On Windows, adding it lazily later can force the native window to be recreated.
    ensureCenterWorkspaceCreated();
    centerHostLayout->setStretch(0, 1);
    workspaceSplitter_->addWidget(centerHost_);

    rightHost_ = new QWidget(workspaceSplitter_);
    allowHorizontalCompression(rightHost_);
    auto* rightHostLayout = new QVBoxLayout(rightHost_);
    rightHostLayout->setContentsMargins(0, 0, 0, 0);
    rightHostLayout->setSpacing(0);
    rightHost_->hide();
    workspaceSplitter_->addWidget(rightHost_);
    root->setStretchFactor(0, 22);
    root->setStretchFactor(1, 78);
    workspaceSplitter_->setStretchFactor(0, 56);
    workspaceSplitter_->setStretchFactor(1, 22);

    configureStatusBarGeometry();
    statusBar()->showMessage("Ready");
    consoleExpandedSizes_ = {820, 160};
    if (centerSplitter_) {
        centerSplitter_->setSizes(consoleExpandedSizes_);
    }
    rightPanelExpandedSizes_ = {1040, 380};
    rebuildSectionPages();
    resetWorkspaceSplitters(rootSplitter_, workspaceSplitter_);
    syncWorkflowChromeButtons();
    setActiveTrackedPanel(viewerPanelShell_);
}

void MainWindow::configureStatusBarGeometry() {
    if (!statusBar()) {
        return;
    }

    statusBar()->setSizeGripEnabled(false);
    statusBar()->setContentsMargins(0, 0, 0, 0);
    statusBar()->setFixedHeight(statusBar()->fontMetrics().height() + 6);
    statusBar()->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);

    if (QLayout* statusLayout = statusBar()->layout()) {
        statusLayout->setContentsMargins(8, 1, 8, 0);
        statusLayout->setSpacing(0);
    }

}

void MainWindow::buildMenus() {
    projectScopedActions_.clear();
    if (!titleMenuStrip_) {
        return;
    }

    auto* menuStripLayout = qobject_cast<QHBoxLayout*>(titleMenuStrip_->layout());
    if (!menuStripLayout) {
        return;
    }

    while (QLayoutItem* item = menuStripLayout->takeAt(0)) {
        if (QWidget* widget = item->widget()) {
            widget->deleteLater();
        }
        delete item;
    }

    auto addTitleMenuButton = [this, menuStripLayout](const QString& text, QMenu* menu) {
        auto* button = new QToolButton(titleMenuStrip_);
        button->setObjectName("titleMenuButton");
        button->setText(text);
        button->setToolButtonStyle(Qt::ToolButtonTextOnly);
        button->setPopupMode(QToolButton::InstantPopup);
        button->setAutoRaise(true);
        button->setCursor(Qt::PointingHandCursor);
        button->setFixedHeight(30);
        button->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
        QFont font = button->font();
        const FontSizePreset topBarFontPreset = preferences_.fontSizePreset == FontSizePreset::Large
            ? FontSizePreset::Large
            : FontSizePreset::Normal;
        font.setPixelSize(interfaceFontPixelSize(topBarFontPreset));
        font.setWeight(QFont::Medium);
        button->setFont(font);
        button->setMenu(menu);
        button->setEnabled(menu->menuAction()->isEnabled());
        button->setVisible(menu->menuAction()->isVisible());
        connect(menu->menuAction(), &QAction::changed, button, [button, action = menu->menuAction()] {
            button->setEnabled(action->isEnabled());
            button->setVisible(action->isVisible());
        });
        menuStripLayout->addWidget(button, 0, Qt::AlignVCenter);
    };

    openRecentMenu_ = nullptr;

    auto* fileMenu = new QMenu("File", this);
    fileMenu->addAction("New Project", this, &MainWindow::createProjectWizard);
    fileMenu->addAction("Open Project", this, &MainWindow::openProject);
    openRecentMenu_ = fileMenu->addMenu("Open Recent");
    updateOpenRecentMenu();
    fileMenu->addSeparator();
    QAction* saveAction = fileMenu->addAction("Save", this, [this] { saveProject(); });
    QAction* saveAsAction = fileMenu->addAction("Save As", this, [this] { saveProjectAs(); });
    fileMenu->addSeparator();
    auto* importMenu = fileMenu->addMenu("Import");
    importMenu->addAction("Terrain Database", this, [this] {
        importProjectAsset("terrain_db", "Import Terrain Database");
    });
    importMenu->addAction("Building Database", this, [this] {
        importProjectAsset("building_db", "Import Building Database");
    });
    importMenu->addAction("Wind Data", this, [this] {
        importProjectAsset("wind_bc", "Import Wind Data");
    });
    auto* exportMenu = fileMenu->addMenu("Export");
    exportMenu->addAction("Generate visualization data", this, &MainWindow::runVisLuwExport);
    exportMenu->addAction("Export NetCDF", this, &MainWindow::runVtk2NcExport);
    exportMenu->addAction("Generate cut visuals", this, &MainWindow::runCutVis);

    auto* runMenu = new QMenu("Run", this);
    runMenu->addAction("Run preprocessing workflow", this, [this] { runPreset(CommandPreset::FullWorkflow); });
    runMenu->addAction("Build batch geometry", this, [this] { runPreset(CommandPreset::PrepareBatchGeometry); });
    runMenu->addAction("Solve", this, [this] { runPreset(CommandPreset::Solve); });
    runMenu->addAction("Stop", this, &MainWindow::stopActiveWork);

    auto* viewMenu = new QMenu("View", this);
    viewMenu->addAction("Load Latest Result", this, &MainWindow::loadLatestResult);
    viewMenu->addAction("Reset Camera", this, [this] {
        if (vtkView_) {
            vtkView_->resetCamera();
        }
    });
    viewMenu->addAction("Save View Image", this, [this] {
        if (vtkView_) {
            vtkView_->saveImage();
        }
    });

    auto* toolsMenu = new QMenu("Tools", this);
    toolsMenu->addAction("Fix deck file", this, &MainWindow::fixDeckFile);
    toolsMenu->addAction("Crop region (interactive)", this, [this] {
        if (progressPanel_) {
            progressPanel_->showTerminalStatus("Reserved", "crop region");
        }
        statusBar()->showMessage("Crop region (interactive) is reserved but not wired yet.", 4000);
    });
    toolsMenu->addAction("Inspect domain (interactive)", this, [this] {
        if (progressPanel_) {
            progressPanel_->showTerminalStatus("Reserved", "inspect domain");
        }
        statusBar()->showMessage("Inspect domain (interactive) is reserved but not wired yet.", 4000);
    });

    auto* helpMenu = new QMenu("About", this);
    helpMenu->addAction("Preferences", this, &MainWindow::showPreferencesDialog);
    helpMenu->addAction("About LatticeUrbanWind Studio", this, &MainWindow::showAboutDialog);

    addTitleMenuButton("File", fileMenu);
    addTitleMenuButton("Run", runMenu);
    addTitleMenuButton("View", viewMenu);
    addTitleMenuButton("Tools", toolsMenu);
    addTitleMenuButton("About", helpMenu);

    projectScopedActions_ << saveAction
                          << saveAsAction
                          << importMenu->menuAction()
                          << exportMenu->menuAction()
                          << runMenu->menuAction()
                          << viewMenu->menuAction()
                          << toolsMenu->menuAction();
}

void MainWindow::applyTitleBarStyle() {
    if (!titleBarWidget_ || !titleMenuStrip_ || !titleBarLogo_ || !titleBarTitle_) {
        return;
    }

    const QColor base = titleBarBaseColor(preferences_.themeMode);
    const QColor hover = titleBarHoverColor(preferences_.themeMode);
    const QColor pressed = titleBarPressedColor(preferences_.themeMode);
    const QString baseName = preferences_.themeMode == ThemeMode::Frieren
        ? QStringLiteral("#c8a45a")
        : base.name();
    const QString hoverName = hover.name();
    const QString pressedName = pressed.name();
    const QString titleBarBorderName = preferences_.themeMode == ThemeMode::Black
        ? QStringLiteral("transparent")
        : preferences_.themeMode == ThemeMode::Frieren
            ? QStringLiteral("rgba(244, 241, 232, 0.36)")
            : preferences_.themeMode == ThemeMode::Himmel
                ? QStringLiteral("rgba(244, 240, 230, 0.28)")
            : QStringLiteral("rgba(255, 255, 255, 0.16)");

    titleBarLogo_->setPixmap(makeLogoPixmap(QColor("#ffffff"), QColor("#ffffff"), QString()));

    titleBarWidget_->setStyleSheet(QString(
        "#appTitleBar {"
        " background: %1;"
        " border-bottom: 1px solid %4;"
        "}"
        "QLabel#titleBarLogo { background: transparent; }"
        "QWidget#titleMenuStrip {"
        " background: transparent;"
        "}"
        "QToolButton#titleMenuButton {"
        " color: white;"
        " background: transparent;"
        " border: none;"
        " border-radius: 6px;"
        " padding: 0 10px 1px 10px;"
        " margin: 0;"
        " min-height: 30px;"
        "}"
        "QToolButton#titleMenuButton::menu-indicator {"
        " image: none;"
        " width: 0px;"
        "}"
        "QToolButton#titleMenuButton:hover {"
        " background: %2;"
        "}"
        "QToolButton#titleMenuButton:pressed {"
        " background: %3;"
        "}"
        "QToolButton#titleMenuButton:disabled {"
        " color: rgba(255, 255, 255, 0.55);"
        "}"
        "QLabel#titleBarTitle {"
        " background: transparent;"
        " color: rgba(255, 255, 255, 0.4);"
        " font-size: 22px;"
        " font-weight: 700;"
        " padding-top: 0px;"
        " padding-right: 12px;"
        " padding-left: 18px;"
        "}"
        "QToolButton#titleMinimize, QToolButton#titleMaximize, QToolButton#titleClose {"
        " color: white;"
        " background: transparent;"
        " border: none;"
        "}"
        "QToolButton#titleMinimize:hover, QToolButton#titleMaximize:hover {"
        " background: %2;"
        "}"
        "QToolButton#titleMinimize:pressed, QToolButton#titleMaximize:pressed {"
        " background: %3;"
        "}"
        "QToolButton#titleClose:hover {"
        " background: %2;"
        "}"
        "QToolButton#titleClose:pressed {"
        " background: %3;"
        "}")
        .arg(baseName, hoverName, pressedName, titleBarBorderName));
}

void MainWindow::refreshWindowButtons() {
    if (maximizeWindowButton_) {
        maximizeWindowButton_->setIcon(makeWindowControlIcon(
            isMaximized() ? WindowControlGlyph::Restore : WindowControlGlyph::Maximize,
            QColor("#ffffff")));
        maximizeWindowButton_->setToolTip(isMaximized() ? "restore" : "maximize");
    }
}

QWidget* MainWindow::buildWorkflowBar() {
    auto* bar = new QWidget(this);
    bar->setObjectName("workflowBar");
    bar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    const int chromeSize = std::max(fontMetrics().height() + 10, 26);
    const int iconSize = std::max(chromeSize - 10, 16);

    auto* layout = new QHBoxLayout(bar);
    layout->setContentsMargins(8, 3, 8, 3);
    layout->setSpacing(5);

    auto* workflowModeCombo = new CompactModeComboBox(bar);
    modeCombo_ = workflowModeCombo;
    modeCombo_->setToolTip("workflow mode");
    populateModeComboBox(workflowModeCombo, document_->mode());
    modeCombo_->setFixedHeight(chromeSize);
    modeCombo_->setMinimumWidth(92);
    layout->addWidget(workflowModeCombo);

    auto makeButton = [this, bar, chromeSize, iconSize](ToolbarGlyph glyph,
                                                        const QString& tooltip,
                                                        bool checkable = false) {
        auto* button = new QToolButton(bar);
        button->setObjectName("workflowAction");
        button->setIcon(makeToolbarIcon(glyph, preferences_.themeMode, palette()));
        button->setIconSize(QSize(iconSize, iconSize));
        button->setFixedSize(chromeSize, chromeSize);
        button->setAutoRaise(true);
        button->setToolTip(tooltip);
        button->setCheckable(checkable);
        button->setCursor(Qt::PointingHandCursor);
        const auto cssColor = [](QColor color) {
            return QStringLiteral("rgba(%1, %2, %3, %4)")
                .arg(color.red())
                .arg(color.green())
                .arg(color.blue())
                .arg(QString::number(color.alphaF(), 'f', 3));
        };
        QColor borderColor = palette().color(QPalette::Highlight);
        QColor hoverFill = borderColor;
        hoverFill.setAlpha(28);
        QColor pressedFill = borderColor;
        pressedFill.setAlpha(46);
        QColor checkedFill = borderColor;
        checkedFill.setAlpha(38);
        QColor disabledBorder = palette().color(QPalette::Mid);
        disabledBorder.setAlpha(120);
        button->setStyleSheet(QString(
            "QToolButton {"
            " border: 1px solid %1;"
            " border-radius: 5px;"
            " background: transparent;"
            " background-color: rgba(0, 0, 0, 0);"
            " border-image: none;"
            " padding: 0px;"
            "}"
            "QToolButton:hover {"
            " background-color: %2;"
            "}"
            "QToolButton:pressed {"
            " background-color: %3;"
            " border-color: %1;"
            "}"
            "QToolButton:checked {"
            " background-color: %4;"
            " border-color: %1;"
            "}"
            "QToolButton:disabled {"
            " border-color: %5;"
            "}")
            .arg(cssColor(borderColor),
                 cssColor(hoverFill),
                 cssColor(pressedFill),
                 cssColor(checkedFill),
                 cssColor(disabledBorder)));
        return button;
    };

    auto* groupOne = new QWidget(bar);
    groupOne->setObjectName("workflowGroup");
    groupOne->setAttribute(Qt::WA_StyledBackground, true);
    groupOne->setStyleSheet("background: transparent; border: none;");
    auto* groupOneLayout = new QHBoxLayout(groupOne);
    groupOneLayout->setContentsMargins(0, 0, 0, 0);
    groupOneLayout->setSpacing(3);
    auto* windInspectButton = makeButton(ToolbarGlyph::InspectWind, "inspect wind climate inputs");
    auto* buildingInspectButton = makeButton(ToolbarGlyph::InspectBuildings, "inspect building footprints");
    auto* workflowButton = makeButton(ToolbarGlyph::RunWorkflow, "run preprocessing workflow");
    groupOneLayout->addWidget(windInspectButton);
    groupOneLayout->addWidget(buildingInspectButton);
    groupOneLayout->addWidget(workflowButton);
    layout->addWidget(groupOne);

    auto* separatorOne = new DotSeparatorWidget(bar);
    separatorOne->setObjectName("workflowSeparator");
    layout->addWidget(separatorOne);

    auto* groupTwo = new QWidget(bar);
    groupTwo->setObjectName("workflowGroup");
    groupTwo->setAttribute(Qt::WA_StyledBackground, true);
    groupTwo->setStyleSheet("background: transparent; border: none;");
    auto* groupTwoLayout = new QHBoxLayout(groupTwo);
    groupTwoLayout->setContentsMargins(0, 0, 0, 0);
    groupTwoLayout->setSpacing(3);
    auto* cropButton = makeButton(ToolbarGlyph::Crop, "crop geometry domain");
    auto* voxelButton = makeButton(ToolbarGlyph::Voxel, "generate voxel domain");
    auto* batchButton = makeButton(ToolbarGlyph::Batch, "build batch geometry");
    auto* validateButton = makeButton(ToolbarGlyph::Validate, "validate case setup");
    groupTwoLayout->addWidget(cropButton);
    groupTwoLayout->addWidget(voxelButton);
    groupTwoLayout->addWidget(batchButton);
    groupTwoLayout->addWidget(validateButton);
    layout->addWidget(groupTwo);

    auto* separatorTwo = new DotSeparatorWidget(bar);
    separatorTwo->setObjectName("workflowSeparator");
    layout->addWidget(separatorTwo);

    auto* groupThree = new QWidget(bar);
    groupThree->setObjectName("workflowGroup");
    groupThree->setAttribute(Qt::WA_StyledBackground, true);
    groupThree->setStyleSheet("background: transparent; border: none;");
    auto* groupThreeLayout = new QHBoxLayout(groupThree);
    groupThreeLayout->setContentsMargins(0, 0, 0, 0);
    groupThreeLayout->setSpacing(3);
    auto* solveButton = makeButton(ToolbarGlyph::Solve, "solve");
    auto* stopButton = makeButton(ToolbarGlyph::Stop, "stop");
    groupThreeLayout->addWidget(solveButton);
    groupThreeLayout->addWidget(stopButton);
    layout->addWidget(groupThree);

    layout->addStretch(1);

    consoleToggleButton_ = makeButton(ToolbarGlyph::Console, "toggle console", true);
    sidePanelToggleButton_ = makeButton(ToolbarGlyph::SidePanel, "toggle side panels", true);
    layout->addWidget(consoleToggleButton_);
    layout->addWidget(sidePanelToggleButton_);

    const bool frierenTheme = preferences_.themeMode == ThemeMode::Frieren;
    const bool himmelTheme = preferences_.themeMode == ThemeMode::Himmel;
    const QString workflowBarBorderColor = frierenTheme
        ? QStringLiteral("#c8a45a")
        : himmelTheme
            ? QStringLiteral("#2e5fa8")
            : palette().color(QPalette::Mid).name();
    const QString workflowBarBackgroundColor = frierenTheme
        ? QStringLiteral("#ead7ae")
        : himmelTheme
            ? QStringLiteral("#a9c8ec")
            : palette().color(QPalette::Window).name();

    bar->setStyleSheet(QString(
        "#workflowBar {"
        " border-bottom: 1px solid %1;"
        " background: %2;"
        "}")
        .arg(workflowBarBorderColor, workflowBarBackgroundColor));

    workflowModeCombo->setObjectName("workflowModeCombo");
    workflowModeCombo->setStyleSheet(QString(
        "QComboBox#workflowModeCombo {"
        " background-color: %1;"
        " border: 1px solid %2;"
        " border-radius: 5px;"
        " padding: 0px 24px 0px 8px;"
        " color: %3;"
        " }"
        "QComboBox#workflowModeCombo:on {"
        " background-color: %1;"
        " }"
        "QComboBox#workflowModeCombo::drop-down {"
        " subcontrol-origin: padding;"
        " subcontrol-position: top right;"
        " width: 20px;"
        " border: none;"
        " background: transparent;"
        " }"
        "QComboBox#workflowModeCombo QAbstractItemView {"
        " background-color: %4;"
        " color: %3;"
        " selection-background-color: %5;"
        " selection-color: %6;"
        " }")
        .arg(workflowBarBackgroundColor,
             workflowBarBorderColor,
             palette().color(QPalette::WindowText).name(),
             palette().color(QPalette::Base).name(),
             palette().color(QPalette::Highlight).name(),
             palette().color(QPalette::HighlightedText).name()));

    auto animateToggle = [iconSize](QToolButton* button) {
        if (!button) {
            return;
        }
        auto* animation = new QPropertyAnimation(button, "iconSize", button);
        animation->setDuration(120);
        animation->setKeyValueAt(0.0, button->iconSize());
        animation->setKeyValueAt(0.5, QSize(iconSize + 2, iconSize + 2));
        animation->setEndValue(QSize(iconSize, iconSize));
        animation->start(QAbstractAnimation::DeleteWhenStopped);
    };

    connect(modeCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
        const RunMode mode = static_cast<RunMode>(std::clamp(index, 0, 2));
        document_->setMode(mode);
    });

    connect(windInspectButton, &QToolButton::clicked, this, [this] { runPreset(CommandPreset::CdfInspect); });
    connect(buildingInspectButton, &QToolButton::clicked, this, [this] { runPreset(CommandPreset::ShpInspect); });
    connect(workflowButton, &QToolButton::clicked, this, [this] { runPreset(CommandPreset::FullWorkflow); });
    connect(cropButton, &QToolButton::clicked, this, [this] { runPreset(CommandPreset::CutGeometry); });
    connect(voxelButton, &QToolButton::clicked, this, [this] { runPreset(CommandPreset::Voxelize); });
    connect(batchButton, &QToolButton::clicked, this, [this] { runPreset(CommandPreset::PrepareBatchGeometry); });
    connect(validateButton, &QToolButton::clicked, this, [this] { runPreset(CommandPreset::Validate); });
    connect(solveButton, &QToolButton::clicked, this, [this] { runPreset(CommandPreset::Solve); });
    connect(stopButton, &QToolButton::clicked, this, &MainWindow::stopActiveWork);

    connect(consoleToggleButton_, &QToolButton::toggled, this, [this, animateToggle](bool checked) {
        animateToggle(consoleToggleButton_);
        if (console_) {
            console_->setCollapsed(!checked);
        }
    });
    connect(sidePanelToggleButton_, &QToolButton::toggled, this, [this, animateToggle](bool checked) {
        animateToggle(sidePanelToggleButton_);
        setRightPanelVisible(checked);
    });

    projectScopedWidgets_.clear();
    projectScopedWidgets_ << workflowModeCombo
                          << windInspectButton
                          << buildingInspectButton
                          << workflowButton
                          << cropButton
                          << voxelButton
                          << batchButton
                          << validateButton
                          << solveButton
                          << stopButton
                          << sidePanelToggleButton_;

    return bar;
}

bool MainWindow::isFieldCompatible(const FieldSpec& spec) const {
    return document_ && specMatchesMode(spec, document_->mode());
}

QString MainWindow::fieldDisplayLabel(const QString& nodeId, const FieldSpec& spec, bool propertyField) const {
    Q_UNUSED(nodeId)
    Q_UNUSED(propertyField)
    if (spec.key == "terr_voxel_grid_resolution") {
        return QStringLiteral("Grid resolution (m)");
    }
    if (spec.key == "terr_voxel_idw_neighbors") {
        return QStringLiteral("Neighboring points (N)");
    }
    if (spec.key == "terr_voxel_idw_sigma") {
        const QString approach = document_
            ? document_->typedValue("terr_voxel_approach").toString().trimmed().toLower()
            : QString();
        if (approach == "kriging" || approach == "kriging_gpu") {
            return QStringLiteral("Kriging sigma");
        }
        return QStringLiteral("IDW sigma");
    }
    static const QHash<QString, QString> abbreviations = {
        {"geometry_mode", "Geometry mode"},
        {"terr_voxel_approach", "Interpolation"},
        {"high_order", "High-order interpolation"},
        {"research_output", "Research stride"},
        {"unsteady_output", "Unsteady stride"},
        {"probes_output", "Probe stride"},
        {"purge_avg", "Avg purge stride"},
        {"purge_avg_stride", "Avg purge sub-stride"},
        {"output_tke_ti_tls", "Avg scalar outputs"},
        {"vk_inlet_ti", "Turbulence intensity"},
        {"vk_inlet_same_realization_all_faces", "Same realization (all)"},
        {"um_vol", "Volume mean vel."},
        {"um_bc", "Boundary mean vel."},
        {"origin_shift_applied", "Origin shift flag"}
    };
    return abbreviations.value(spec.key, spec.label);
}

QVector<MainWindow::TreeNodeInfo> MainWindow::buildTreeNodes() const {
    QVector<TreeNodeInfo> nodes;
    if (!document_) {
        return nodes;
    }

    const QString rawCaseName = document_->typedValue("casename").toString().trimmed();
    const QString caseName = QString("%1 (%2)")
        .arg(rawCaseName.isEmpty() ? QStringLiteral("Case name") : rawCaseName,
             runModeDisplayName(document_->mode()));
    const auto boolEnabled = [this](const QString& key) {
        return rawValueTruthy(document_->rawValue(key));
    };
    const auto intEnabled = [this](const QString& key) {
        return document_->typedValue(key).toInt() > 0;
    };
    const auto hasRawText = [this](const QString& key) {
        return !document_->rawValue(key).trimmed().isEmpty();
    };
    const QString geometryMode = document_->typedValue("geometry_mode").toString().trimmed();
    const bool showBuildingRepresentation = geometryMode == "0" || geometryMode == "2";
    const bool showTerrainRepresentation = geometryMode == "0" || geometryMode == "1" || geometryMode == "2";
    const auto pushNode = [&nodes](const QString& id,
                                   const QString& title,
                                   const QString& description,
                                   const QString& parentId,
                                   const QStringList& fieldKeys = {},
                                   const QStringList& propertyKeys = {},
                                   bool caseRoot = false,
                                   bool toolsRoot = false,
                                   bool resultsRoot = false,
                                   bool managedLeaf = false,
                                   const QString& managedRole = {},
                                   bool managedLeafTrashed = false) {
        TreeNodeInfo node;
        node.id = id;
        node.title = title;
        node.description = description;
        node.parentId = parentId;
        node.fieldKeys = fieldKeys;
        node.propertyKeys = propertyKeys;
        node.caseRoot = caseRoot;
        node.toolsRoot = toolsRoot;
        node.resultsRoot = resultsRoot;
        node.managedLeaf = managedLeaf;
        node.managedRole = managedRole;
        node.managedLeafTrashed = managedLeafTrashed;
        nodes.push_back(node);
    };
    const auto managedDescription = [](const ManagedNode& node) {
        if (node.trashed) {
            return QString("This %1 is mounted read-only from gui_properties because it is trashed.")
                .arg(managedRoleDisplayName(node.role));
        }
        if (node.role == kManagedRoleTool) {
            return QStringLiteral("Tool configuration will be added here in a later pass.");
        }
        if (node.role == kManagedRoleDisplay) {
            return QStringLiteral("Display configuration will be added here in a later pass.");
        }
        if (node.role == kManagedRoleDataProcessor) {
            return QStringLiteral("Data processor configuration will be added here in a later pass.");
        }
        return QStringLiteral("Node configuration will be added here in a later pass.");
    };

    pushNode("case_root", caseName, "Project-level case identity and timestamp.", {}, {"casename", "datetime"}, {}, true);

    pushNode("domain", "Domain", "Domain clipping, elevation limits, geometry representation, and derived coordinate metadata.", "case_root",
             {"cut_lon_manual", "cut_lat_manual", "base_height", "z_limit", "geometry_mode", "x_exp_rat", "y_exp_rat"},
             {"si_x_cfd", "si_y_cfd", "si_z_cfd", "utm_crs", "rotate_deg", "origin_shift_applied", "validation"});
    if (showBuildingRepresentation) {
        pushNode(
            "domain_building",
            "Building representation",
            "Building voxelization settings that control how building heights are read and filtered before extrusion.",
            "domain",
            {"terr_voxel_height_field", "terr_voxel_ignore_under"});
    }
    if (showTerrainRepresentation) {
        pushNode(
            "domain_terrain",
            "Terrain representation",
            "Terrain interpolation and smoothing settings used when generating terrain-aware voxel geometry and building base elevations.",
            "domain",
            {"terr_voxel_approach", "terr_voxel_grid_resolution", "terr_voxel_idw_sigma", "terr_voxel_idw_power", "terr_voxel_idw_neighbors"});
    }
    pushNode("domain_wind", "Wind", "Wind-domain inputs are grouped here when the selected mode supports them.", "domain",
             {"inflow", "angle"});

    pushNode("cfd_control", "CFD control", "Solver setup, numerical controls, boundary handling, and output configuration.", "case_root");
    pushNode("cfd_numerical", "Numerical methods", "Numerical-method groups and solver controls.", "cfd_control");
    pushNode("cfd_physics_model", "Physics model", "Physical source-term switches.", "cfd_numerical",
             {"buoyancy", "coriolis_term"});
    pushNode("cfd_boundary_conditions", "Boundary conditions",
             "Boundary-condition switches live here. Enabled features expose dedicated child nodes with their detailed settings.",
             "cfd_numerical",
             {"high_order", "flux_correction", "turb_inflow_enable", "downstream_open_face", "ibm_enabler", "enable_buffer_nudging", "enable_top_sponge"},
             {"um_vol", "um_bc", "downstream_bc", "downstream_bc_yaw"});
    if (boolEnabled("turb_inflow_enable")) {
        pushNode("bc_turbulence_inflow", "Turbulence inflow", "Synthetic turbulence inflow parameters.", "cfd_boundary_conditions",
                 {"turb_inflow_approach", "vk_inlet_ti", "vk_inlet_sigma", "vk_inlet_l", "vk_inlet_nmodes", "vk_inlet_seed", "vk_inlet_update_stride",
                  "vk_inlet_uc_mode", "vk_inlet_same_realization_all_faces", "vk_inlet_stride_interpolation",
                  "vk_inlet_inflow_only", "vk_inlet_anisotropy"});
    }
    if (boolEnabled("downstream_open_face")) {
        pushNode("bc_downstream_open", "Downstream open", "Downstream open is currently controlled by its parent switch.", "cfd_boundary_conditions");
    }
    if (boolEnabled("enable_buffer_nudging")) {
        pushNode("bc_buffer_nudging", "Buffer nudging", "Detailed buffer-nudging settings.", "cfd_boundary_conditions",
                 {"buffer_nudge_vertical", "buffer_thickness_m", "buffer_tau_s"});
    }
    if (boolEnabled("enable_top_sponge")) {
        pushNode("bc_top_sponge", "Top sponge layer", "Detailed top-sponge-layer settings.", "cfd_boundary_conditions",
                 {"sponge_thickness_m", "sponge_tau_s", "sponge_ref_mode"});
    }

    pushNode("cfd_compute", "Computational control", "GPU layout, mesh-control mode, and run-length controls.", "cfd_control",
             {"n_gpu", "mesh_control", "gpu_memory", "cell_size", "run_nstep"});

    pushNode("cfd_output", "Output", "Output groups are organized by mean, instantaneous, and probe-related controls.", "cfd_control");
    pushNode("output_mean_fields", "Mean fields", "Mean-field outputs and averaging windows.", "cfd_output",
             {"purge_avg"});
    if (intEnabled("purge_avg")) {
        pushNode("output_mean_settings", "Mean field settings", "Detailed mean-field output parameters.", "output_mean_fields",
                 {"purge_avg", "purge_avg_stride", "output_tke_ti_tls"});
    }
    pushNode("output_instant_fields", "Instantaneous fields", "Instantaneous and research-output controls.", "cfd_output",
             {"unsteady_output", "research_output"});
    if (intEnabled("unsteady_output")) {
        pushNode("output_unsteady_fields", "Unsteady fields", "Stride for unsteady field output.", "output_instant_fields",
                 {"unsteady_output"});
    }
    if (intEnabled("research_output")) {
        pushNode("output_research_output", "Research output", "Stride for research output snapshots.", "output_instant_fields",
                 {"research_output"});
    }
    pushNode("output_probes", "Probes", "Probe definitions and probe-output windows.", "cfd_output",
             {"probes", "probes_output"});
    if (intEnabled("probes_output")) {
        pushNode("output_probe_window", "Probe output", "Dedicated probe output window.", "output_probes",
                 {"probes_output"});
    }
    if (hasRawText("probes")) {
        pushNode("output_probe_definitions", "Probe definitions", "Detailed probe definitions.", "output_probes",
                 {"probes"});
    }

    pushNode("tools", "Tools", "Tool instances can be added here.", "case_root", {}, {}, false, true);
    for (const ManagedNode& tool : toolNodes_) {
        if (tool.trashed) {
            continue;
        }
        pushNode(tool.id, tool.title, managedDescription(tool), "tools", {}, {}, false, false, false, true, tool.role, false);
    }
    if (viewTrashedTools_) {
        for (const ManagedNode& tool : toolNodes_) {
            if (!tool.trashed) {
                continue;
            }
            pushNode(tool.id, tool.title, managedDescription(tool), "tools", {}, {}, false, false, false, true, tool.role, true);
        }
    }

    pushNode("results", "Results", "Display and data-processing nodes can be added here.", "case_root", {}, {}, false, false, true);
    for (const ManagedNode& result : resultNodes_) {
        if (result.trashed) {
            continue;
        }
        pushNode(result.id, result.title, managedDescription(result), "results", {}, {}, false, false, false, true, result.role, false);
    }
    if (viewTrashedResults_) {
        for (const ManagedNode& result : resultNodes_) {
            if (!result.trashed) {
                continue;
            }
            pushNode(result.id, result.title, managedDescription(result), "results", {}, {}, false, false, false, true, result.role, true);
        }
    }

    QHash<QString, TreeNodeInfo> nodeById;
    QHash<QString, QStringList> childIdsByParent;
    for (const TreeNodeInfo& node : nodes) {
        nodeById.insert(node.id, node);
        childIdsByParent[node.parentId].push_back(node.id);
    }

    const auto hasVisibleFields = [this](const TreeNodeInfo& node) {
        for (const QString& key : node.fieldKeys) {
            const FieldSpec* spec = findFieldSpec(key);
            if (spec && isFieldCompatible(*spec)) {
                return true;
            }
        }
        for (const QString& key : node.propertyKeys) {
            const FieldSpec* spec = findFieldSpec(key);
            if (spec && isFieldCompatible(*spec)) {
                return true;
            }
        }
        return false;
    };

    QHash<QString, bool> keepCache;
    const std::function<bool(const QString&)> shouldKeepNode = [&](const QString& nodeId) -> bool {
        if (keepCache.contains(nodeId)) {
            return keepCache.value(nodeId);
        }
        const TreeNodeInfo node = nodeById.value(nodeId);
        const bool explicitDomainRepresentationNode =
            node.id == "domain_building" || node.id == "domain_terrain";
        bool keep = node.caseRoot || node.toolsRoot || node.resultsRoot || node.managedLeaf
            || explicitDomainRepresentationNode || hasVisibleFields(node);
        const QStringList childIds = childIdsByParent.value(nodeId);
        for (const QString& childId : childIds) {
            if (shouldKeepNode(childId)) {
                keep = true;
                break;
            }
        }
        keepCache.insert(nodeId, keep);
        return keep;
    };

    QVector<TreeNodeInfo> visibleNodes;
    visibleNodes.reserve(nodes.size());
    for (const TreeNodeInfo& node : nodes) {
        if (shouldKeepNode(node.id)) {
            visibleNodes.push_back(node);
        }
    }
    return visibleNodes;
}

void MainWindow::syncManagedStateWithProject() {
    const QString projectKey = hasLoadedProject() ? QFileInfo(document_->filePath()).absoluteFilePath() : QString();
    if (managedStateProjectKey_ == projectKey) {
        return;
    }

    managedStateProjectKey_ = projectKey;
    toolNodes_.clear();
    resultNodes_.clear();
    viewTrashedTools_ = false;
    viewTrashedResults_ = false;

    if (!hasLoadedProject()) {
        return;
    }

    if (!ensureGuiPropertiesDirectory()) {
        return;
    }

    reloadToolNodesFromGuiProperties();
    reloadResultNodesFromGuiProperties();
}

void MainWindow::rebuildSectionPages(const QString& preferredNodeId) {
    syncManagedStateWithProject();

    QString selectedNodeId = preferredNodeId;
    if (selectedNodeId.isEmpty() && navTree_ && navTree_->currentItem()) {
        selectedNodeId = navTree_->currentItem()->data(0, Qt::UserRole).toString();
    }
    if (selectedNodeId.isEmpty()) {
        selectedNodeId = currentNodeId_;
    }

    QStringList expandedNodeIds;
    if (navTree_) {
        const std::function<void(QTreeWidgetItem*)> collectExpanded = [&](QTreeWidgetItem* item) {
            if (!item) {
                return;
            }
            if (item->isExpanded()) {
                expandedNodeIds.push_back(item->data(0, Qt::UserRole).toString());
            }
            for (int i = 0; i < item->childCount(); ++i) {
                collectExpanded(item->child(i));
            }
        };
        for (int i = 0; i < navTree_->topLevelItemCount(); ++i) {
            collectExpanded(navTree_->topLevelItem(i));
        }
    }

    bindings_.clear();
    pageById_.clear();
    nodeById_.clear();
    treeItemById_.clear();
    while (sectionStack_->count() > 0) {
        QWidget* widget = sectionStack_->widget(0);
        sectionStack_->removeWidget(widget);
        widget->deleteLater();
    }
    if (navTree_) {
        const QSignalBlocker blocker(navTree_);
        navTree_->clear();
    }

    if (!hasLoadedProject()) {
        currentNodeId_.clear();
        QWidget* page = buildEmptyPropertiesPage("No project");
        pageById_.insert("empty", sectionStack_->addWidget(page));
        sectionStack_->setCurrentWidget(page);
        return;
    }

    const QVector<TreeNodeInfo> nodes = buildTreeNodes();
    for (const TreeNodeInfo& node : nodes) {
        nodeById_.insert(node.id, node);
        pageById_.insert(node.id, sectionStack_->addWidget(buildNodePage(node)));

        if (!navTree_) {
            continue;
        }

        auto* item = new QTreeWidgetItem(QStringList{node.title});
        item->setData(0, Qt::UserRole, node.id);
        if (node.managedLeafTrashed) {
            item->setData(0, Qt::ForegroundRole, navTree_->palette().color(QPalette::Disabled, QPalette::Text));
        }
        if (node.managedLeaf
            && !node.managedLeafTrashed
            && (node.managedRole == kManagedRoleTool || node.managedRole == kManagedRoleDisplay)) {
            item->setFlags(item->flags() | Qt::ItemIsEditable);
        }
        treeItemById_.insert(node.id, item);
        if (!node.parentId.isEmpty() && treeItemById_.contains(node.parentId)) {
            treeItemById_.value(node.parentId)->addChild(item);
        } else {
            navTree_->addTopLevelItem(item);
        }
    }

    QSet<QString> pathNodeIds;
    QString cursorId = treeItemById_.contains(selectedNodeId) ? selectedNodeId : QString();
    while (!cursorId.isEmpty() && nodeById_.contains(cursorId)) {
        pathNodeIds.insert(cursorId);
        cursorId = nodeById_.value(cursorId).parentId;
    }

    for (auto it = treeItemById_.cbegin(); it != treeItemById_.cend(); ++it) {
        QTreeWidgetItem* item = it.value();
        const TreeNodeInfo treeNode = nodeById_.value(it.key());
        const bool keepExpanded = treeNode.caseRoot
            || expandedNodeIds.contains(it.key())
            || pathNodeIds.contains(it.key());
        item->setExpanded(keepExpanded);
    }

    const QString fallbackNodeId = treeItemById_.contains("case_root") ? QStringLiteral("case_root") : pageById_.isEmpty() ? QString() : pageById_.cbegin().key();
    const QString targetNodeId = treeItemById_.contains(selectedNodeId) ? selectedNodeId : fallbackNodeId;
    if (navTree_ && treeItemById_.contains(targetNodeId)) {
        navTree_->setCurrentItem(treeItemById_.value(targetNodeId));
    } else {
        setCurrentPage(targetNodeId);
    }
}

QWidget* MainWindow::buildWorkflowPage() {
    auto* page = new QWidget(sectionStack_);
    auto* root = new QVBoxLayout(page);
    root->setContentsMargins(8, 8, 8, 8);

    auto* summaryBox = new QGroupBox("Project Summary", page);
    auto* summaryLayout = new QVBoxLayout(summaryBox);
    auto* summaryControls = new QWidget(summaryBox);
    auto* summaryControlsLayout = new QHBoxLayout(summaryControls);
    summaryControlsLayout->setContentsMargins(0, 0, 0, 0);
    summaryControlsLayout->addWidget(new QLabel("Mode", summaryControls));
    auto* summaryModeCombo = new CompactModeComboBox(summaryControls);
    modeCombo_ = summaryModeCombo;
    populateModeComboBox(summaryModeCombo, document_->mode());
    summaryControlsLayout->addWidget(summaryModeCombo);
    summaryControlsLayout->addStretch(1);
    summaryLayout->addWidget(summaryControls);
    workflowSummary_ = new QLabel(summaryBox);
    workflowSummary_->setWordWrap(true);
    summaryLayout->addWidget(workflowSummary_);
    root->addWidget(summaryBox);

    auto* workflowBox = new QGroupBox("Workflow Actions", page);
    auto* workflowLayout = new QGridLayout(workflowBox);
    auto* runWorkflowButton = new QPushButton("Run Full Workflow", workflowBox);
    auto* cdfButton = new QPushButton("Inspect Wind Climate Inputs", workflowBox);
    auto* shpButton = new QPushButton("Inspect Building Footprints", workflowBox);
    auto* bcButton = new QPushButton("Generate Boundary Conditions", workflowBox);
    auto* cutButton = new QPushButton("Crop Geometry Domain", workflowBox);
    auto* voxButton = new QPushButton("Generate Voxel Domain", workflowBox);
    auto* valButton = new QPushButton("Validate Case Setup", workflowBox);
    auto* batchButton = new QPushButton("Build Batch Geometry", workflowBox);
    auto* solveButton = new QPushButton("Solve", workflowBox);
    auto* stopButton = new QPushButton("Stop", workflowBox);
    solveButton->setProperty("workflowDanger", true);
    stopButton->setProperty("workflowDanger", true);
    workflowLayout->addWidget(runWorkflowButton, 0, 0, 1, 2);
    workflowLayout->addWidget(cdfButton, 1, 0);
    workflowLayout->addWidget(shpButton, 1, 1);
    workflowLayout->addWidget(bcButton, 2, 0);
    workflowLayout->addWidget(cutButton, 2, 1);
    workflowLayout->addWidget(voxButton, 3, 0);
    workflowLayout->addWidget(valButton, 3, 1);
    workflowLayout->addWidget(batchButton, 4, 0);
    workflowLayout->addWidget(solveButton, 4, 1);
    workflowLayout->addWidget(stopButton, 5, 0, 1, 2);
    root->addWidget(workflowBox);

    auto* postBox = new QGroupBox("Post Processing", page);
    auto* postLayout = new QGridLayout(postBox);
    auto* visButton = new QPushButton("Generate Visualization Data", postBox);
    auto* ncButton = new QPushButton("Export NetCDF", postBox);
    auto* latestButton = new QPushButton("Load Latest Result", postBox);
    auto* cutVisButton = new QPushButton("Generate Cut Visuals", postBox);
    cutVisExtraArgsEdit_ = new QLineEdit(postBox);
    cutVisExtraArgsEdit_->setPlaceholderText("--export-cropped-vtk --cell-size 10");
    postLayout->addWidget(visButton, 0, 0);
    postLayout->addWidget(ncButton, 0, 1);
    postLayout->addWidget(latestButton, 1, 0);
    postLayout->addWidget(cutVisButton, 1, 1);
    postLayout->addWidget(new QLabel("Cut visual extra arguments"), 2, 0, 1, 2);
    postLayout->addWidget(cutVisExtraArgsEdit_, 3, 0, 1, 2);
    root->addWidget(postBox);
    root->addStretch(1);

    connect(runWorkflowButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::FullWorkflow); });
    connect(cdfButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::CdfInspect); });
    connect(shpButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::ShpInspect); });
    connect(bcButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::BuildBoundaryConditions); });
    connect(cutButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::CutGeometry); });
    connect(voxButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::Voxelize); });
    connect(valButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::Validate); });
    connect(batchButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::PrepareBatchGeometry); });
    connect(solveButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::Solve); });
    connect(stopButton, &QPushButton::clicked, this, &MainWindow::stopActiveWork);
    connect(visButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::VisLuw); });
    connect(ncButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::Vtk2Nc); });
    connect(latestButton, &QPushButton::clicked, this, &MainWindow::loadLatestResult);
    connect(cutVisButton, &QPushButton::clicked, this, &MainWindow::runCutVis);
    connect(summaryModeCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
        const RunMode mode = static_cast<RunMode>(std::clamp(index, 0, 2));
        document_->setMode(mode);
    });

    return page;
}

QWidget* MainWindow::createPropertyEditor(const FieldSpec& spec) {
    auto* line = new QLineEdit();
    line->setToolTip(spec.help + "\n\nDouble-click to force-edit this property.");
    applyPropertyEditorVisual(line, preferences_.themeMode, false);
    return line;
}

QWidget* MainWindow::buildNodePage(const TreeNodeInfo& node) {
    if (!hasLoadedProject()) {
        return buildEmptyPropertiesPage(node.title);
    }

    QStringList compatibleFieldKeys;
    for (const QString& key : node.fieldKeys) {
        const FieldSpec* spec = findFieldSpec(key);
        if (spec && isFieldCompatible(*spec)) {
            compatibleFieldKeys.push_back(key);
        }
    }

    QStringList compatiblePropertyKeys;
    for (const QString& key : node.propertyKeys) {
        const FieldSpec* spec = findFieldSpec(key);
        if (spec && isFieldCompatible(*spec)) {
            compatiblePropertyKeys.push_back(key);
        }
    }

    auto* page = new QWidget(sectionStack_);
    auto* pageLayout = new QVBoxLayout(page);
    pageLayout->setContentsMargins(0, 0, 0, 0);
    pageLayout->setSpacing(4);

    const bool hasManagedRootControls = node.toolsRoot || node.resultsRoot;
    auto buildManagedRootControlsSection = [this, &node, page]() -> QWidget* {
        auto* section = new QWidget(page);
        auto* sectionLayout = new QVBoxLayout(section);
        sectionLayout->setContentsMargins(0, 0, 0, 0);
        sectionLayout->setSpacing(0);

        auto* header = new QLabel("Configurations", section);
        header->setContentsMargins(6, 0, 0, 0);
        sectionLayout->addWidget(header);

        const bool toolsRoot = node.toolsRoot;
        const QString targetNodeId = node.id;
        auto* scroll = new QScrollArea(section);
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        sectionLayout->addWidget(scroll, 1);

        auto* container = new FormGridWidget(false, scroll);
        auto* containerLayout = new QVBoxLayout(container);
        containerLayout->setContentsMargins(6, 6, 6, 0);
        containerLayout->setSpacing(0);
        const QString dividerColor = cssColor(sectionGridColor(container->palette()));

        auto* rowWidget = new QWidget(container);
        rowWidget->setProperty("formRow", 0);
        rowWidget->setMinimumWidth(0);
        rowWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        auto* rowLayout = new QHBoxLayout(rowWidget);
        rowLayout->setContentsMargins(0, 0, 0, 0);
        rowLayout->setSpacing(0);

        auto* label = new QLabel(
            toolsRoot ? "View trashed tools" : "View trashed displays",
            rowWidget);
        label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        label->setMinimumWidth(0);
        label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

        auto* labelHost = new QWidget(rowWidget);
        labelHost->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        labelHost->setMinimumWidth(0);
        auto* labelLayout = new QHBoxLayout(labelHost);
        labelLayout->setContentsMargins(0, 0, 5, 0);
        labelLayout->setSpacing(0);
        labelLayout->addWidget(label, 1, Qt::AlignLeft | Qt::AlignVCenter);

        auto* dividerFrame = new QFrame(rowWidget);
        dividerFrame->setFrameShape(QFrame::NoFrame);
        dividerFrame->setFixedWidth(1);
        dividerFrame->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
        dividerFrame->setStyleSheet(
            QString("QFrame { background-color: %1; border: none; }").arg(dividerColor));

        auto* host = new QWidget(rowWidget);
        host->setMinimumWidth(0);
        host->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        auto* hostLayout = new QHBoxLayout(host);
        hostLayout->setContentsMargins(5, 0, 0, 0);
        hostLayout->setSpacing(0);

        auto* trashedToggle = new QCheckBox(host);
        trashedToggle->setText({});
        trashedToggle->setChecked(toolsRoot ? viewTrashedTools_ : viewTrashedResults_);
        connect(trashedToggle, &QCheckBox::toggled, this, [this, toolsRoot, targetNodeId](bool checked) {
            if (toolsRoot) {
                viewTrashedTools_ = checked;
                reloadToolNodesFromGuiProperties();
            } else {
                viewTrashedResults_ = checked;
                reloadResultNodesFromGuiProperties();
            }
            rebuildSectionPages(targetNodeId);
            refreshEditors();
        });
        trashedToggle->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
        hostLayout->addWidget(trashedToggle, 0, Qt::AlignLeft | Qt::AlignVCenter);
        hostLayout->addStretch(1);

        rowLayout->addWidget(labelHost, 1);
        rowLayout->addWidget(dividerFrame, 0);
        rowLayout->addWidget(host, 1);
        containerLayout->addWidget(rowWidget);

        auto* rowDivider = new QFrame(container);
        rowDivider->setFrameShape(QFrame::NoFrame);
        rowDivider->setFixedHeight(1);
        rowDivider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        rowDivider->setStyleSheet(
            QString("QFrame { background-color: %1; border: none; }").arg(dividerColor));
        containerLayout->addWidget(rowDivider);
        containerLayout->addStretch(1);
        scroll->setWidget(container);

        return section;
    };

    if (compatibleFieldKeys.isEmpty() && compatiblePropertyKeys.isEmpty() && !hasManagedRootControls) {
        auto* emptyPage = buildEmptyPropertiesPage(node.title);
        pageLayout->addWidget(emptyPage, 1);
        return page;
    }

    auto buildFormSection = [this, &node](const QString& headerTitle, const QStringList& keys, bool propertyField) -> QWidget* {
        auto* section = new QWidget();
        auto* sectionLayout = new QVBoxLayout(section);
        sectionLayout->setContentsMargins(0, 0, 0, 0);
        sectionLayout->setSpacing(propertyField ? 2 : 4);

        if (!headerTitle.isEmpty()) {
            auto* header = new QLabel(headerTitle, section);
            header->setContentsMargins(6, 0, 0, 0);
            sectionLayout->addWidget(header);
        }

        auto* scroll = new QScrollArea(section);
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        sectionLayout->addWidget(scroll, 1);

        auto* container = new FormGridWidget(false, scroll);
        auto* containerLayout = new QVBoxLayout(container);
        containerLayout->setContentsMargins(6, 6, 6, 0);
        containerLayout->setSpacing(0);
        const QString dividerColor = cssColor(sectionGridColor(container->palette()));

        if (keys.isEmpty()) {
            auto* empty = new QLabel(propertyField ? "<No Properties>" : "<No Fields>", container);
            empty->setProperty("emptyState", true);
            containerLayout->addWidget(empty);
        } else {
            int row = 0;
            for (const QString& key : keys) {
                const FieldSpec* spec = findFieldSpec(key);
                if (!spec) {
                    continue;
                }

                QWidget* editor = propertyField ? createPropertyEditor(*spec) : createEditor(*spec);
                if (!propertyField) {
                    editor->setToolTip(spec->help);
                }
                if (!propertyField) {
                    editor->setEnabled(!spec->readOnly);
                }

                auto* label = new QLabel(fieldDisplayLabel(node.id, *spec, propertyField), container);
                label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
                label->setToolTip(spec->label);
                label->setMinimumWidth(0);
                label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

                auto* rowWidget = new QWidget(container);
                rowWidget->setProperty("formRow", row);
                rowWidget->setMinimumWidth(0);
                rowWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
                auto* rowLayout = new QHBoxLayout(rowWidget);
                rowLayout->setContentsMargins(0, 0, 0, 0);
                rowLayout->setSpacing(0);

                auto* labelHost = new QWidget(rowWidget);
                labelHost->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
                labelHost->setMinimumWidth(0);
                auto* labelLayout = new QHBoxLayout(labelHost);
                labelLayout->setContentsMargins(0, 0, 5, 0);
                labelLayout->setSpacing(0);
                labelLayout->addWidget(label, 1, Qt::AlignLeft | Qt::AlignVCenter);

                QWidget* divider = nullptr;
                if (!propertyField) {
                    auto* dividerFrame = new QFrame(rowWidget);
                    dividerFrame->setFrameShape(QFrame::NoFrame);
                    dividerFrame->setFixedWidth(1);
                    dividerFrame->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
                    dividerFrame->setStyleSheet(
                        QString("QFrame { background-color: %1; border: none; }").arg(dividerColor));
                    divider = dividerFrame;
                }

                auto* host = new QWidget(rowWidget);
                host->setMinimumWidth(0);
                host->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
                auto* hostLayout = new QHBoxLayout(host);
                hostLayout->setContentsMargins(5, 0, 0, 0);
                hostLayout->setSpacing(0);

                if (propertyField || spec->kind != FieldKind::Boolean) {
                    editor->setMinimumWidth(0);
                    const QSizePolicy::Policy verticalPolicy =
                        spec->kind == FieldKind::Multiline ? QSizePolicy::Expanding : QSizePolicy::Fixed;
                    editor->setSizePolicy(QSizePolicy::Expanding, verticalPolicy);
                    if (auto* combo = qobject_cast<QComboBox*>(editor)) {
                        if (spec->key == "terr_voxel_approach") {
                            combo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
                            combo->setMinimumContentsLength(8);
                        } else {
                            combo->setSizeAdjustPolicy(QComboBox::AdjustToContentsOnFirstShow);
                        }
                    }
                    hostLayout->addWidget(editor, 1);
                } else {
                    editor->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
                    hostLayout->addWidget(editor, 0, Qt::AlignLeft | Qt::AlignVCenter);
                    hostLayout->addStretch(1);
                }

                rowLayout->addWidget(labelHost, 1);
                if (divider) {
                    rowLayout->addWidget(divider, 0);
                }
                rowLayout->addWidget(host, 1);
                containerLayout->addWidget(rowWidget);
                if (!propertyField) {
                    auto* rowDivider = new QFrame(container);
                    rowDivider->setFrameShape(QFrame::NoFrame);
                    rowDivider->setFixedHeight(1);
                    rowDivider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
                    rowDivider->setStyleSheet(
                        QString("QFrame { background-color: %1; border: none; }").arg(dividerColor));
                    containerLayout->addWidget(rowDivider);
                }
                const int bindingIndex = bindings_.size();
                bindings_.push_back({*spec, editor, label, node.id, propertyField});

                if (propertyField) {
                    editor->setProperty("bindingIndex", bindingIndex);
                    editor->setProperty("propertyField", true);
                    editor->installEventFilter(this);
                    auto* line = qobject_cast<QLineEdit*>(editor);
                    connect(line, &QLineEdit::editingFinished, this, [this, spec, line] {
                        if (line->isReadOnly()) {
                            return;
                        }
                        document_->setRawValue(spec->key, line->text());
                        applyPropertyEditorVisual(line, preferences_.themeMode, false);
                        line->clearFocus();
                    });
                }
                ++row;
            }
        }

        containerLayout->addStretch(1);
        scroll->setWidget(container);
        return section;
    };

    if (hasManagedRootControls && compatibleFieldKeys.isEmpty()) {
        pageLayout->addWidget(buildManagedRootControlsSection(), 1);
    } else {
        pageLayout->addWidget(buildFormSection("Configurations", compatibleFieldKeys, false), 1);
    }
    if (!compatiblePropertyKeys.isEmpty()) {
        auto* propertyToggle = new QToolButton(page);
        propertyToggle->setText("Properties");
        propertyToggle->setCheckable(true);
        propertyToggle->setChecked(false);
        propertyToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        propertyToggle->setArrowType(Qt::RightArrow);
        propertyToggle->setAutoRaise(true);
        propertyToggle->setStyleSheet(
            "QToolButton { font-weight: normal; border: none; background: transparent; padding: 0px; margin: 0px; }");
        pageLayout->addWidget(propertyToggle, 0, Qt::AlignLeft);

        QWidget* propertySection = buildFormSection({}, compatiblePropertyKeys, true);
        propertySection->setVisible(false);
        pageLayout->addWidget(propertySection, 0);

        connect(propertyToggle, &QToolButton::toggled, this, [propertyToggle, propertySection](bool expanded) {
            propertyToggle->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);
            propertySection->setVisible(expanded);
        });
    }
    return page;
}

QWidget* MainWindow::buildSectionPage(const SectionSpec& section) {
    if (!hasLoadedProject()) {
        return buildEmptyPropertiesPage(section.title);
    }

    auto* page = new QWidget(sectionStack_);
    auto* pageLayout = new QVBoxLayout(page);
    pageLayout->setContentsMargins(0, 0, 0, 0);

    auto* scroll = new QScrollArea(page);
    scroll->setWidgetResizable(true);
    pageLayout->addWidget(scroll);

    auto* container = new QWidget(scroll);
    auto* layout = new QVBoxLayout(container);
    layout->setContentsMargins(8, 8, 8, 8);
    auto* description = new QLabel(section.description, container);
    description->setWordWrap(true);
    description->setProperty("muted", true);
    layout->addWidget(description);

    auto* formContainer = new QWidget(container);
    auto* form = new QFormLayout(formContainer);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignTop);
    form->setFormAlignment(Qt::AlignTop);
    form->setContentsMargins(0, 0, 0, 0);

    for (const FieldSpec& spec : fieldsForSection(section.id, document_->mode())) {
        QWidget* editor = createEditor(spec);
        editor->setEnabled(!spec.readOnly);
        editor->setToolTip(spec.help);
        form->addRow(spec.label, editor);
        bindings_.push_back({spec, editor, form->labelForField(editor)});
    }

    layout->addWidget(formContainer);
    layout->addStretch(1);
    scroll->setWidget(container);
    return page;
}

void MainWindow::expandNodeRecursive(QTreeWidgetItem* item, bool expand) {
    if (!item) {
        return;
    }
    item->setExpanded(expand);
    for (int i = 0; i < item->childCount(); ++i) {
        expandNodeRecursive(item->child(i), expand);
    }
}

void MainWindow::renameCase() {
    if (!document_) {
        return;
    }

    bool accepted = false;
    const QString currentName = document_->typedValue("casename").toString().trimmed();
    const QString nextName = QInputDialog::getText(this, "Rename case", "Case name", QLineEdit::Normal, currentName, &accepted).trimmed();
    if (!accepted || nextName.isEmpty() || nextName == currentName) {
        return;
    }
    if (nextName.contains(QRegularExpression(R"([\\/:*?"<>|])"))) {
        QMessageBox::warning(this, "Rename case", "Case name contains invalid path characters.");
        return;
    }

    document_->setTypedValue("casename", nextName);
}

void MainWindow::revealProjectInExplorer() {
    if (!hasLoadedProject()) {
        return;
    }

    const QUrl projectUrl = QUrl::fromLocalFile(document_->projectDirectory());
    if (!QDesktopServices::openUrl(projectUrl)) {
        QMessageBox::warning(
            this,
            "Reveal in explorer",
            "Failed to open the project folder in the system file browser.");
    }
}

QString MainWindow::guiPropertiesDirectory() const {
    if (!hasLoadedProject()) {
        return {};
    }
    return QDir(document_->projectDirectory()).filePath(kGuiPropertiesDirName);
}

bool MainWindow::ensureGuiPropertiesDirectory() {
    const QString directoryPath = guiPropertiesDirectory();
    if (directoryPath.isEmpty()) {
        return false;
    }

    QDir directory(directoryPath);
    if (directory.exists()) {
        return true;
    }

    if (QDir().mkpath(directoryPath)) {
        return true;
    }

    const QString message = "Failed to create gui_properties under the project folder.";
    statusBar()->showMessage(message, 5000);
    if (console_) {
        console_->appendText("[GUI] " + message + "\n");
    }
    return false;
}

void MainWindow::reloadToolNodesFromGuiProperties() {
    toolNodes_.clear();
    if (!hasLoadedProject() || !ensureGuiPropertiesDirectory()) {
        return;
    }

    const QDir guiPropertiesDir(guiPropertiesDirectory());
    const QFileInfoList entries = guiPropertiesDir.entryInfoList(QDir::Files, QDir::Name | QDir::IgnoreCase);
    for (const QFileInfo& entry : entries) {
        ManagedFileNameInfo fileInfo;
        if (!parseManagedFileName(entry.fileName(), kToolFileExtension, &fileInfo)) {
            continue;
        }

        const QHash<QString, QString> values = readLooseKeyValueFile(entry.absoluteFilePath());
        ManagedNode node;
        node.name = fileInfo.baseName;
        node.title = managedNodeDisplayTitle(node.name, fileInfo.trashed, fileInfo.trashIndex);
        node.filePath = entry.absoluteFilePath();
        node.role = kManagedRoleTool;
        node.storageType = values.value(QStringLiteral("tool_type")).trimmed().toLower();
        node.trashed = fileInfo.trashed;
        node.trashIndex = fileInfo.trashIndex;
        node.id = managedNodeId(node.role, node.filePath);
        toolNodes_.push_back(node);
    }

    std::sort(toolNodes_.begin(), toolNodes_.end(), [](const ManagedNode& lhs, const ManagedNode& rhs) {
        if (lhs.trashed != rhs.trashed) {
            return !lhs.trashed;
        }
        const int nameCompare = lhs.name.compare(rhs.name, Qt::CaseInsensitive);
        if (nameCompare != 0) {
            return nameCompare < 0;
        }
        return lhs.trashIndex < rhs.trashIndex;
    });
}

void MainWindow::reloadResultNodesFromGuiProperties() {
    resultNodes_.clear();
    if (!hasLoadedProject() || !ensureGuiPropertiesDirectory()) {
        return;
    }

    bool activeDataProcessorLoaded = false;
    QStringList skippedDataProcessorNames;
    const QDir guiPropertiesDir(guiPropertiesDirectory());
    const QFileInfoList entries = guiPropertiesDir.entryInfoList(QDir::Files, QDir::Name | QDir::IgnoreCase);
    for (const QFileInfo& entry : entries) {
        ManagedFileNameInfo fileInfo;
        QString role;
        QString storageType;

        if (parseManagedFileName(entry.fileName(), kVisFileExtension, &fileInfo)) {
            role = kManagedRoleDisplay;
            const QHash<QString, QString> values = readLooseKeyValueFile(entry.absoluteFilePath());
            storageType = values.value(QStringLiteral("vis_type")).trimmed().toLower();
        } else if (parseManagedFileName(entry.fileName(), kProcessorFileExtension, &fileInfo)) {
            role = kManagedRoleDataProcessor;
            if (!fileInfo.trashed) {
                if (activeDataProcessorLoaded) {
                    skippedDataProcessorNames.push_back(entry.fileName());
                    continue;
                }
                activeDataProcessorLoaded = true;
            }
        } else {
            continue;
        }

        ManagedNode node;
        node.name = fileInfo.baseName;
        node.title = managedNodeDisplayTitle(node.name, fileInfo.trashed, fileInfo.trashIndex);
        node.filePath = entry.absoluteFilePath();
        node.role = role;
        node.storageType = storageType;
        node.trashed = fileInfo.trashed;
        node.trashIndex = fileInfo.trashIndex;
        node.id = managedNodeId(node.role, node.filePath);
        resultNodes_.push_back(node);
    }

    std::sort(resultNodes_.begin(), resultNodes_.end(), [](const ManagedNode& lhs, const ManagedNode& rhs) {
        const auto rank = [](const ManagedNode& node) {
            if (!node.trashed && node.role == kManagedRoleDisplay) {
                return 0;
            }
            if (!node.trashed && node.role == kManagedRoleDataProcessor) {
                return 1;
            }
            if (node.trashed && node.role == kManagedRoleDisplay) {
                return 2;
            }
            return 3;
        };

        const int leftRank = rank(lhs);
        const int rightRank = rank(rhs);
        if (leftRank != rightRank) {
            return leftRank < rightRank;
        }
        const int nameCompare = lhs.name.compare(rhs.name, Qt::CaseInsensitive);
        if (nameCompare != 0) {
            return nameCompare < 0;
        }
        return lhs.trashIndex < rhs.trashIndex;
    });

    if (!skippedDataProcessorNames.isEmpty()) {
        statusBar()->showMessage(
            "Multiple active data processors were detected; only the first was loaded.",
            5000);
    }
}

const MainWindow::ManagedNode* MainWindow::findToolNode(const QString& nodeId) const {
    for (const ManagedNode& node : toolNodes_) {
        if (node.id == nodeId) {
            return &node;
        }
    }
    return nullptr;
}

const MainWindow::ManagedNode* MainWindow::findResultNode(const QString& nodeId) const {
    for (const ManagedNode& node : resultNodes_) {
        if (node.id == nodeId) {
            return &node;
        }
    }
    return nullptr;
}

const MainWindow::ManagedNode* MainWindow::findManagedNode(const QString& nodeId) const {
    if (const ManagedNode* toolNode = findToolNode(nodeId)) {
        return toolNode;
    }
    return findResultNode(nodeId);
}

bool MainWindow::hasActiveDataProcessor() const {
    return std::any_of(resultNodes_.cbegin(), resultNodes_.cend(), [](const ManagedNode& node) {
        return node.role == kManagedRoleDataProcessor && !node.trashed;
    });
}

void MainWindow::addManagedNode(const QString& role, const QString& type) {
    if (!hasLoadedProject() || !ensureGuiPropertiesDirectory()) {
        return;
    }

    if (role == kManagedRoleDataProcessor && hasActiveDataProcessor()) {
        QMessageBox::warning(
            this,
            "Add data processor",
            "Active data processor detected: only one data processor is allowed.");
        return;
    }

    const QString baseTitle = managedNodeDefaultTitle(role, type);
    QString preferredName = baseTitle;
    if (role != kManagedRoleDataProcessor) {
        int sameTypeCount = 0;
        const QVector<ManagedNode>& nodes = role == kManagedRoleTool ? toolNodes_ : resultNodes_;
        for (const ManagedNode& existing : nodes) {
            if (!existing.trashed && existing.role == role && existing.storageType == type) {
                ++sameTypeCount;
            }
        }
        if (sameTypeCount > 0) {
            preferredName = QString("%1 %2").arg(baseTitle).arg(sameTypeCount + 1);
        }
    }

    const QString filePath = nextAvailableManagedPath(
        guiPropertiesDirectory(),
        preferredName,
        managedFileExtension(role));
    QString error;
    if (!writeManagedNodeFile(filePath, role, type, &error)) {
        const QString roleName = managedRoleDisplayName(role);
        QMessageBox::critical(
            this,
            QString("Add %1").arg(roleName),
            QString("Failed to create %1 file:\n%2").arg(roleName, error));
        return;
    }

    if (role == kManagedRoleTool) {
        reloadToolNodesFromGuiProperties();
        rebuildSectionPages(managedNodeId(role, filePath));
    } else {
        reloadResultNodesFromGuiProperties();
        rebuildSectionPages(managedNodeId(role, filePath));
    }
    refreshEditors();
    refreshFileTree();
}

void MainWindow::renameManagedNode(const QString& nodeId) {
    const ManagedNode* node = findManagedNode(nodeId);
    if (!node || node->trashed
        || (node->role != kManagedRoleTool && node->role != kManagedRoleDisplay)) {
        return;
    }

    if (!navTree_ || !treeItemById_.contains(nodeId)) {
        return;
    }

    QTreeWidgetItem* item = treeItemById_.value(nodeId);
    navTree_->setCurrentItem(item);
    navTree_->editItem(item, 0);
}

bool MainWindow::commitManagedNodeRename(const QString& nodeId, const QString& nextName, QString* errorMessage) {
    const ManagedNode* node = findManagedNode(nodeId);
    if (!node || node->trashed
        || (node->role != kManagedRoleTool && node->role != kManagedRoleDisplay)) {
        return false;
    }

    if (nextName == node->name) {
        return true;
    }

    QString validationError;
    if (!isValidManagedNodeName(nextName, &validationError)) {
        if (errorMessage) {
            *errorMessage = validationError;
        }
        return false;
    }

    const QString targetPath = QDir(QFileInfo(node->filePath).absolutePath())
        .filePath(nextName + managedFileExtension(node->role));
    if (sameProjectDeckPath(targetPath, node->filePath)) {
        return true;
    }
    if (QFileInfo::exists(targetPath)) {
        if (errorMessage) {
            *errorMessage = "A node with this name already exists.";
        }
        return false;
    }

    QFile file(node->filePath);
    if (!file.rename(targetPath)) {
        if (errorMessage) {
            *errorMessage = "Failed to rename file:\n" + file.errorString();
        }
        return false;
    }

    if (node->role == kManagedRoleTool) {
        reloadToolNodesFromGuiProperties();
        rebuildSectionPages(managedNodeId(node->role, targetPath));
    } else {
        reloadResultNodesFromGuiProperties();
        rebuildSectionPages(managedNodeId(node->role, targetPath));
    }
    refreshEditors();
    refreshFileTree();

    return true;
}

void MainWindow::removeManagedNode(const QString& nodeId) {
    const ManagedNode* node = findManagedNode(nodeId);
    if (!node || node->trashed) {
        return;
    }

    const QString trashPath = nextAvailableTrashPath(node->filePath);
    const QString roleName = managedRoleDisplayName(node->role);
    QFile file(node->filePath);
    if (!file.rename(trashPath)) {
        QMessageBox::critical(
            this,
            QString("Remove %1").arg(roleName),
            "Failed to trash file:\n" + file.errorString());
        return;
    }

    if (node->role == kManagedRoleTool) {
        reloadToolNodesFromGuiProperties();
        rebuildSectionPages("tools");
    } else {
        reloadResultNodesFromGuiProperties();
        rebuildSectionPages("results");
    }
    refreshEditors();
    refreshFileTree();
}

void MainWindow::removeAllToolNodes() {
    const bool hasActiveTools = std::any_of(toolNodes_.cbegin(), toolNodes_.cend(), [](const ManagedNode& node) {
        return !node.trashed;
    });
    if (!hasActiveTools) {
        return;
    }

    const auto answer = QMessageBox::question(
        this,
        "Remove all tools",
        "Remove all tool nodes from the project tree?",
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);
    if (answer != QMessageBox::Yes) {
        return;
    }

    QStringList errors;
    const QVector<ManagedNode> activeTools = toolNodes_;
    for (const ManagedNode& node : activeTools) {
        if (node.trashed) {
            continue;
        }
        QFile file(node.filePath);
        if (!file.rename(nextAvailableTrashPath(node.filePath))) {
            errors.push_back(node.name + ": " + file.errorString());
        }
    }

    reloadToolNodesFromGuiProperties();
    rebuildSectionPages("tools");
    refreshEditors();
    refreshFileTree();

    if (!errors.isEmpty()) {
        QMessageBox::warning(
            this,
            "Remove all tools",
            "Some tools could not be removed:\n" + errors.join('\n'));
    }
}

void MainWindow::removeAllResultNodes() {
    const bool hasActiveResults = std::any_of(resultNodes_.cbegin(), resultNodes_.cend(), [](const ManagedNode& node) {
        return !node.trashed && node.role == kManagedRoleDisplay;
    });
    if (!hasActiveResults) {
        return;
    }

    const auto answer = QMessageBox::question(
        this,
        "Remove all displays",
        "Remove all display nodes from the project tree?",
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);
    if (answer != QMessageBox::Yes) {
        return;
    }

    QStringList errors;
    const QVector<ManagedNode> activeResults = resultNodes_;
    for (const ManagedNode& node : activeResults) {
        if (node.trashed || node.role != kManagedRoleDisplay) {
            continue;
        }
        QFile file(node.filePath);
        if (!file.rename(nextAvailableTrashPath(node.filePath))) {
            errors.push_back(node.name + ": " + file.errorString());
        }
    }

    reloadResultNodesFromGuiProperties();
    rebuildSectionPages("results");
    refreshEditors();
    refreshFileTree();

    if (!errors.isEmpty()) {
        QMessageBox::warning(
            this,
            "Remove all displays",
            "Some display nodes could not be removed:\n" + errors.join('\n'));
    }
}

void MainWindow::recoverManagedNode(const QString& nodeId) {
    const ManagedNode* node = findManagedNode(nodeId);
    if (!node || !node->trashed) {
        return;
    }
    if (node->role == kManagedRoleDataProcessor && hasActiveDataProcessor()) {
        QMessageBox::warning(
            this,
            "Recover data processor",
            "Active data processor detected: only one data processor is allowed.");
        return;
    }

    QString preferredBaseName = node->name + "_recovery";
    if (node->trashIndex > 0) {
        preferredBaseName += QString::number(node->trashIndex);
    }
    const QString targetPath = nextAvailableManagedPath(
        guiPropertiesDirectory(),
        preferredBaseName,
        managedFileExtension(node->role));

    QFile file(node->filePath);
    if (!file.rename(targetPath)) {
        QMessageBox::critical(
            this,
            QString("Recover %1").arg(managedRoleDisplayName(node->role)),
            "Failed to recover file:\n" + file.errorString());
        return;
    }

    if (node->role == kManagedRoleTool) {
        reloadToolNodesFromGuiProperties();
        rebuildSectionPages(managedNodeId(node->role, targetPath));
    } else {
        reloadResultNodesFromGuiProperties();
        rebuildSectionPages(managedNodeId(node->role, targetPath));
    }
    refreshEditors();
    refreshFileTree();
}

void MainWindow::permanentlyDeleteManagedNode(const QString& nodeId) {
    const ManagedNode* node = findManagedNode(nodeId);
    if (!node || !node->trashed) {
        return;
    }

    const QString roleName = managedRoleDisplayName(node->role);
    const auto answer = QMessageBox::warning(
        this,
        QString("Delete %1 permanently").arg(roleName),
        QString("Delete this trashed %1 permanently?").arg(roleName),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);
    if (answer != QMessageBox::Yes) {
        return;
    }

    QFile file(node->filePath);
    if (!file.remove()) {
        QMessageBox::critical(
            this,
            QString("Delete %1 permanently").arg(roleName),
            "Failed to delete file:\n" + file.errorString());
        return;
    }

    if (node->role == kManagedRoleTool) {
        reloadToolNodesFromGuiProperties();
        rebuildSectionPages("tools");
    } else {
        reloadResultNodesFromGuiProperties();
        rebuildSectionPages("results");
    }
    refreshEditors();
    refreshFileTree();
}

bool MainWindow::keyRequiresTreeRebuild(const QString& key) const {
    static const QStringList keys{
        "casename",
        "geometry_mode",
        "enable_buffer_nudging",
        "buffer_nudge_vertical",
        "enable_top_sponge",
        "turb_inflow_enable",
        "downstream_open_face",
        "purge_avg",
        "unsteady_output",
        "research_output",
        "probes_output",
        "probes",
    };
    return keys.contains(key.trimmed().toLower());
}

void MainWindow::unlockPropertyEditor(int bindingIndex) {
    if (bindingIndex < 0 || bindingIndex >= bindings_.size()) {
        return;
    }

    auto* editor = qobject_cast<QLineEdit*>(bindings_[bindingIndex].editor);
    if (!editor || !editor->isReadOnly()) {
        return;
    }

    const auto answer = QMessageBox::warning(
        this,
        "Force edit property",
        "This value is treated as a property or derived field and may be overwritten by preprocessing or validation.\n\nDo you want to force-edit the raw deck value anyway?",
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);
    if (answer != QMessageBox::Yes) {
        return;
    }

    editor->setReadOnly(false);
    applyPropertyEditorVisual(editor, preferences_.themeMode, true);
    editor->setFocus();
    editor->selectAll();
    statusBar()->showMessage("Property field unlocked for raw editing.", 4000);
}

void MainWindow::showNavTreeContextMenu(const QPoint& pos) {
    if (!navTree_) {
        return;
    }

    QTreeWidgetItem* item = navTree_->itemAt(pos);
    if (item) {
        navTree_->setCurrentItem(item);
    }

    const QString nodeId = item ? item->data(0, Qt::UserRole).toString() : QString();
    const TreeNodeInfo node = item ? nodeById_.value(nodeId) : TreeNodeInfo{};

    QMenu menu(this);
    QAction* expandAction = nullptr;
    QAction* collapseAction = nullptr;
    QAction* expandAllAction = nullptr;
    QAction* collapseAllAction = nullptr;

    QAction* renameAction = nullptr;
    QAction* revealInExplorerAction = nullptr;
    QAction* runAllToolsAction = nullptr;
    QAction* addCropRegionAction = nullptr;
    QAction* addMultiMomentAction = nullptr;
    QAction* removeToolAction = nullptr;
    QAction* removeAllToolsAction = nullptr;
    QAction* refreshToolsAction = nullptr;
    QAction* addGeometryDisplayAction = nullptr;
    QAction* add2dVisualizationAction = nullptr;
    QAction* add3dVisualizationAction = nullptr;
    QAction* addDataPlotAction = nullptr;
    QAction* addSpreadsheetAction = nullptr;
    QAction* addDataProcessorAction = nullptr;
    QAction* removeAllResultsAction = nullptr;
    QAction* refreshResultsAction = nullptr;
    QAction* renameManagedAction = nullptr;
    QAction* recoverManagedAction = nullptr;
    QAction* permanentlyDeleteManagedAction = nullptr;

    if (item && (node.caseRoot || node.toolsRoot || node.resultsRoot || node.managedLeaf)) {
        menu.addSeparator();
    }
    if (item && node.caseRoot) {
        renameAction = menu.addAction("Rename case...");
        revealInExplorerAction = menu.addAction("Reveal in explorer");
    }
    if (item && node.toolsRoot) {
        auto* addToolMenu = menu.addMenu("Add a tool");
        addCropRegionAction = addToolMenu->addAction("Crop region");
        addMultiMomentAction = addToolMenu->addAction("Multi-moment automation");
        runAllToolsAction = menu.addAction("Run all tools");
        removeAllToolsAction = menu.addAction("Remove all tools");
        removeAllToolsAction->setEnabled(std::any_of(toolNodes_.cbegin(), toolNodes_.cend(), [](const ManagedNode& managedNode) {
            return !managedNode.trashed;
        }));
        refreshToolsAction = menu.addAction("Refresh changes");
    }
    if (item && node.resultsRoot) {
        auto* addDisplayMenu = menu.addMenu("Add a display");
        addGeometryDisplayAction = addDisplayMenu->addAction("Geometry display");
        add2dVisualizationAction = addDisplayMenu->addAction("2D visualization");
        add3dVisualizationAction = addDisplayMenu->addAction("3D visualization");
        addDataPlotAction = addDisplayMenu->addAction("Data plot");
        addSpreadsheetAction = addDisplayMenu->addAction("Spreadsheet view");
        addDataProcessorAction = menu.addAction("Add data processor");
        addDataProcessorAction->setEnabled(!hasActiveDataProcessor());
        removeAllResultsAction = menu.addAction("Remove all displays");
        removeAllResultsAction->setEnabled(std::any_of(resultNodes_.cbegin(), resultNodes_.cend(), [](const ManagedNode& managedNode) {
            return !managedNode.trashed && managedNode.role == kManagedRoleDisplay;
        }));
        refreshResultsAction = menu.addAction("Refresh changes");
    }
    if (item && node.managedLeaf) {
        const ManagedNode* managedNode = findManagedNode(nodeId);
        if (managedNode && managedNode->trashed) {
            recoverManagedAction = menu.addAction(managedNodeRecoverLabel(managedNode->role));
            permanentlyDeleteManagedAction = menu.addAction("Delete (permanent)");
        } else if (managedNode) {
            if (managedNode->role == kManagedRoleTool || managedNode->role == kManagedRoleDisplay) {
                renameManagedAction = menu.addAction("Rename");
            }
            removeToolAction = menu.addAction("Remove");
        }
    }

    if (!menu.actions().isEmpty()) {
        menu.addSeparator();
    }
    if (item && !node.caseRoot) {
        expandAction = menu.addAction("Expand");
        collapseAction = menu.addAction("Collapse");
        expandAction->setEnabled(item->childCount() > 0);
        collapseAction->setEnabled(item->childCount() > 0);
    }
    expandAllAction = menu.addAction("Expand all");
    collapseAllAction = menu.addAction("Collapse all");

    QAction* chosen = menu.exec(navTree_->viewport()->mapToGlobal(pos));
    if (!chosen) {
        return;
    }

    if (chosen == expandAction) {
        item->setExpanded(true);
        return;
    }
    if (chosen == collapseAction) {
        item->setExpanded(false);
        return;
    }
    if (chosen == expandAllAction) {
        navTree_->expandAll();
        return;
    }
    if (chosen == collapseAllAction) {
        navTree_->collapseAll();
        if (treeItemById_.contains("case_root")) {
            treeItemById_.value("case_root")->setExpanded(true);
        }
        return;
    }
    if (chosen == renameAction) {
        renameCase();
        return;
    }
    if (chosen == revealInExplorerAction) {
        revealProjectInExplorer();
        return;
    }
    if (chosen == addCropRegionAction) {
        addManagedNode(kManagedRoleTool, "crop");
        return;
    }
    if (chosen == addMultiMomentAction) {
        addManagedNode(kManagedRoleTool, "multimoment");
        return;
    }
    if (chosen == removeToolAction) {
        removeManagedNode(nodeId);
        return;
    }
    if (chosen == removeAllToolsAction) {
        removeAllToolNodes();
        return;
    }
    if (chosen == refreshToolsAction) {
        reloadToolNodesFromGuiProperties();
        rebuildSectionPages("tools");
        refreshEditors();
        refreshFileTree();
        return;
    }
    if (chosen == runAllToolsAction) {
        statusBar()->showMessage("Run all tools is reserved but not wired yet.", 4000);
        return;
    }
    if (chosen == addGeometryDisplayAction) {
        addManagedNode(kManagedRoleDisplay, "geometry");
        return;
    }
    if (chosen == add2dVisualizationAction) {
        addManagedNode(kManagedRoleDisplay, "2d");
        return;
    }
    if (chosen == add3dVisualizationAction) {
        addManagedNode(kManagedRoleDisplay, "3d");
        return;
    }
    if (chosen == addDataPlotAction) {
        addManagedNode(kManagedRoleDisplay, "plot");
        return;
    }
    if (chosen == addSpreadsheetAction) {
        addManagedNode(kManagedRoleDisplay, "sheet");
        return;
    }
    if (chosen == addDataProcessorAction) {
        addManagedNode(kManagedRoleDataProcessor);
        return;
    }
    if (chosen == removeAllResultsAction) {
        removeAllResultNodes();
        return;
    }
    if (chosen == refreshResultsAction) {
        reloadResultNodesFromGuiProperties();
        rebuildSectionPages("results");
        refreshEditors();
        refreshFileTree();
        return;
    }
    if (chosen == renameManagedAction) {
        renameManagedNode(nodeId);
        return;
    }
    if (chosen == recoverManagedAction) {
        recoverManagedNode(nodeId);
        return;
    }
    if (chosen == permanentlyDeleteManagedAction) {
        permanentlyDeleteManagedNode(nodeId);
        return;
    }
}

QWidget* MainWindow::buildEmptyPropertiesPage(const QString& title) {
    auto* page = new QWidget(sectionStack_);
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(0);

    auto* frame = new QFrame(page);
    frame->setObjectName("emptyPropertiesFrame");
    frame->setFrameShape(QFrame::StyledPanel);
    frame->setFrameShadow(QFrame::Plain);
    frame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    auto* frameLayout = new QVBoxLayout(frame);
    frameLayout->setContentsMargins(0, 0, 0, 0);
    frameLayout->addStretch(1);

    auto* label = new QLabel("<No Properties>", frame);
    label->setAlignment(Qt::AlignCenter);
    label->setProperty("emptyState", true);
    label->setToolTip(title);
    frameLayout->addWidget(label, 0, Qt::AlignCenter);
    frameLayout->addStretch(1);

    layout->addWidget(frame, 1);
    return page;
}

QWidget* MainWindow::buildShellPlaceholderPage() {
    auto* page = new QWidget(this);
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(0);

    auto* frame = new QFrame(page);
    frame->setObjectName("emptyPropertiesFrame");
    frame->setFrameShape(QFrame::StyledPanel);
    frame->setFrameShadow(QFrame::Plain);
    frame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    layout->addWidget(frame, 1);
    return page;
}

QWidget* MainWindow::createPanelShell(QWidget* parent, QWidget* child, const QString& objectName) {
    auto* shell = new QWidget(parent);
    shell->setObjectName(objectName);
    shell->setAttribute(Qt::WA_StyledBackground, true);
    auto* layout = new QVBoxLayout(shell);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    child->setParent(shell);
    layout->addWidget(child);
    shell->setSizePolicy(child->sizePolicy());
    registerTrackedPanel(shell);
    return shell;
}

void MainWindow::registerTrackedPanel(QWidget* panelShell) {
    if (!panelShell || trackedPanels_.contains(panelShell)) {
        return;
    }

    panelShell->setProperty("panelShell", true);
    panelShell->setProperty("panelActive", false);
    trackedPanels_.append(panelShell);
    repolishTrackedPanel(panelShell);
}

QWidget* MainWindow::trackedPanelFor(QWidget* widget) const {
    for (QWidget* current = widget; current; current = current->parentWidget()) {
        if (trackedPanels_.contains(current)) {
            return current;
        }
        if (current == this) {
            break;
        }
    }
    return nullptr;
}

void MainWindow::repolishTrackedPanel(QWidget* panelShell) {
    if (!panelShell) {
        return;
    }
    panelShell->style()->unpolish(panelShell);
    panelShell->style()->polish(panelShell);
    panelShell->update();
}

void MainWindow::setActiveTrackedPanel(QWidget* panelShell) {
    if (activeTrackedPanel_ == panelShell) {
        return;
    }

    if (activeTrackedPanel_) {
        activeTrackedPanel_->setProperty("panelActive", false);
        repolishTrackedPanel(activeTrackedPanel_);
    }

    activeTrackedPanel_ = panelShell;

    if (activeTrackedPanel_) {
        activeTrackedPanel_->setProperty("panelActive", true);
        repolishTrackedPanel(activeTrackedPanel_);
    }
}

QWidget* MainWindow::createEditor(const FieldSpec& spec) {
    QWidget* editor = nullptr;

    switch (spec.kind) {
    case FieldKind::Boolean: {
        auto* box = new QCheckBox();
        connect(box, &QCheckBox::toggled, this, [this, spec](bool value) {
            document_->setTypedValue(spec.key, value);
        });
        editor = box;
        break;
    }
    case FieldKind::Enum: {
        auto* combo = new QComboBox();
        for (const QString& value : spec.enumValues) {
            combo->addItem(displayEnumLabel(spec.key, value), value);
        }
        if (spec.key == "terr_voxel_approach" && combo->view()) {
            const QFontMetrics metrics(combo->font());
            int popupWidth = 0;
            for (int index = 0; index < combo->count(); ++index) {
                popupWidth = std::max(popupWidth, metrics.horizontalAdvance(combo->itemText(index)));
            }
            combo->view()->setMinimumWidth(popupWidth + 36);
        }
        connect(combo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this, spec, combo](int index) {
            if (index < 0) {
                return;
            }
            document_->setTypedValue(spec.key, combo->itemData(index).toString());
        });
        editor = combo;
        break;
    }
    case FieldKind::Integer: {
        auto* spin = new QSpinBox();
        spin->setRange(-999999999, 999999999);
        connect(spin, qOverload<int>(&QSpinBox::valueChanged), this, [this, spec](int value) {
            document_->setTypedValue(spec.key, value);
        });
        editor = spin;
        break;
    }
    case FieldKind::Float: {
        auto* line = new PrecisionNumericLineEdit(PrecisionNumericLineEdit::Mode::Scalar);
        connect(line, &QLineEdit::editingFinished, this, [this, spec, line] {
            document_->setRawValue(spec.key, line->commitVisibleText());
        });
        editor = line;
        break;
    }
    case FieldKind::Multiline: {
        auto* text = new QPlainTextEdit();
        text->setMinimumHeight(110);
        connect(text, &QPlainTextEdit::textChanged, this, [this, spec, text] {
            document_->setTypedValue(spec.key, text->toPlainText());
        });
        editor = text;
        break;
    }
    case FieldKind::UIntTriplet:
        if (usesSplitTripletEditor(spec)) {
            auto* triplet = new SplitTripletEditor(SplitTripletEditor::ValueMode::Integer);
            for (QLineEdit* line : triplet->lineEdits()) {
                connect(line, &QLineEdit::editingFinished, this, [this, spec, triplet] {
                    document_->setRawValue(spec.key, triplet->rawTextForDeck());
                });
            }
            editor = triplet;
            break;
        }
        [[fallthrough]];
    case FieldKind::FloatPair:
    case FieldKind::FloatTriplet:
        if (usesSplitTripletEditor(spec)) {
            auto* triplet = new SplitTripletEditor(SplitTripletEditor::ValueMode::Float);
            for (QLineEdit* line : triplet->lineEdits()) {
                connect(line, &QLineEdit::editingFinished, this, [this, spec, triplet] {
                    document_->setRawValue(spec.key, triplet->rawTextForDeck());
                });
            }
            editor = triplet;
            break;
        }
        [[fallthrough]];
    case FieldKind::FloatList:
    case FieldKind::TokenList:
    case FieldKind::String: {
        auto* line = new QLineEdit();
        if (spec.kind == FieldKind::FloatPair
            || spec.kind == FieldKind::FloatTriplet
            || spec.kind == FieldKind::FloatList) {
            delete line;
            auto* precisionLine = new PrecisionNumericLineEdit(PrecisionNumericLineEdit::Mode::Vector);
            connect(precisionLine, &QLineEdit::editingFinished, this, [this, spec, precisionLine] {
                document_->setRawValue(spec.key, renderBracketedListText(precisionLine->commitVisibleText()));
            });
            editor = precisionLine;
        } else if (spec.kind == FieldKind::UIntTriplet) {
            connect(line, &QLineEdit::editingFinished, this, [this, spec, line] {
                document_->setRawValue(spec.key, renderBracketedListText(line->text()));
            });
            editor = line;
        } else {
            connect(line, &QLineEdit::editingFinished, this, [this, spec, line] {
                document_->setTypedValue(spec.key, readEditorValue({spec, line}));
            });
            editor = line;
        }
        break;
    }
    }

    return editor;
}

void MainWindow::ensureCenterWorkspaceCreated() {
    if (centerSplitter_ || !centerHost_) {
        return;
    }

    auto* centerHostLayout = qobject_cast<QVBoxLayout*>(centerHost_->layout());
    if (!centerHostLayout) {
        return;
    }

    centerSplitter_ = new QSplitter(Qt::Vertical, centerHost_);
    viewerStack_ = new QStackedWidget(centerSplitter_);
    centerPlaceholder_ = buildShellPlaceholderPage();
    centerPlaceholder_->setObjectName("centerPlaceholder");
    vtkView_ = new VtkViewWidget();
    viewerStack_->addWidget(centerPlaceholder_);
    viewerStack_->addWidget(vtkView_);
    viewerStack_->setCurrentWidget(centerPlaceholder_);
    viewerPanelShell_ = createPanelShell(centerSplitter_, viewerStack_, "viewerPanelShell");
    console_ = new ConsolePanel();
    consolePanelShell_ = createPanelShell(centerSplitter_, console_, "consolePanelShell");
    centerSplitter_->addWidget(viewerPanelShell_);
    centerSplitter_->addWidget(consolePanelShell_);
    centerSplitter_->setStretchFactor(0, 3);
    centerSplitter_->setStretchFactor(1, 1);
    allowHorizontalCompression(centerSplitter_);
    allowHorizontalCompression(viewerStack_);
    allowHorizontalCompression(vtkView_);
    allowHorizontalCompression(console_);
    centerSplitter_->setSizes(consoleExpandedSizes_.size() == 2 ? consoleExpandedSizes_ : QList<int>{820, 160});
    centerHostLayout->addWidget(centerSplitter_, 1);

    connect(vtkView_, &VtkViewWidget::statusMessage, this, [this](const QString& message) {
        statusBar()->showMessage(message, 5000);
    });
    connect(vtkView_, &VtkViewWidget::guiActionRequested, this, &MainWindow::logGuiAction);
    connect(vtkView_, &VtkViewWidget::fileLoaded, this, [this](const QString& filePath) {
        if (filePath.endsWith(".vtk", Qt::CaseInsensitive) && wavenumberPanel_) {
            const bool autoAnalyze = QRegularExpression(R"(_raw_u-\d+\.vtk$)", QRegularExpression::CaseInsensitiveOption)
                .match(QFileInfo(filePath).fileName())
                .hasMatch();
            wavenumberPanel_->setSuggestedFilePath(filePath, autoAnalyze);
        }
    });
    connect(console_, &ConsolePanel::collapseToggled, this, [this](bool collapsed) {
        if (!centerSplitter_ || !console_) {
            return;
        }
        if (collapsed) {
            consoleExpandedSizes_ = centerSplitter_->sizes();
            const QList<int> sizes = consoleExpandedSizes_.isEmpty() ? QList<int>{1, 1} : consoleExpandedSizes_;
            const int total = sizes.value(0, 1) + sizes.value(1, 1);
            centerSplitter_->setSizes({std::max(total - console_->collapsedHeight(), 1), console_->collapsedHeight()});
        } else if (consoleExpandedSizes_.size() == 2) {
            centerSplitter_->setSizes(consoleExpandedSizes_);
        } else {
            centerSplitter_->setSizes({820, 160});
        }
        syncWorkflowChromeButtons();
    });
}

void MainWindow::ensureProjectWorkspaceLoaded() {
    if (!hasLoadedProject()) {
        return;
    }

    const bool firstWorkspaceLoad = !navTree_ || !centerSplitter_ || !rightSplitter_;
    QWidget* const central = centralWidget();
    if (firstWorkspaceLoad && central) {
        central->setUpdatesEnabled(false);
    }

    if (!navTree_ && leftSplitter_) {
        navTree_ = new OutlineTreeWidget();
        navTree_->setObjectName("projectNavTree");
        navTree_->setHeaderHidden(true);
        navTree_->setRootIsDecorated(true);
        navTree_->setItemsExpandable(true);
        navTree_->setExpandsOnDoubleClick(true);
        navTree_->setEditTriggers(QAbstractItemView::SelectedClicked | QAbstractItemView::EditKeyPressed);
        navTree_->setIndentation(22);
        navTree_->setUniformRowHeights(true);
        navTree_->setContextMenuPolicy(Qt::CustomContextMenu);
        navPanelShell_ = createPanelShell(leftSplitter_, navTree_, "navPanelShell");
        leftSplitter_->insertWidget(0, navPanelShell_);
        leftSplitter_->setStretchFactor(0, 1);
        leftSplitter_->setStretchFactor(1, 0);
        leftSplitter_->setStretchFactor(2, 1);
        leftSplitter_->setSizes({440, 18, 440});
        connect(navTree_, &QTreeWidget::currentItemChanged, this, [this](QTreeWidgetItem* current) {
            if (!current) {
                return;
            }
            setCurrentPage(current->data(0, Qt::UserRole).toString());
        });
        connect(navTree_, &QTreeWidget::itemChanged, this, [this](QTreeWidgetItem* item, int column) {
            if (!item || column != 0) {
                return;
            }

            const QString nodeId = item->data(0, Qt::UserRole).toString();
            const TreeNodeInfo node = nodeById_.value(nodeId);
            if (!node.managedLeaf
                || node.managedLeafTrashed
                || (node.managedRole != kManagedRoleTool && node.managedRole != kManagedRoleDisplay)) {
                return;
            }

            const ManagedNode* managedNode = findManagedNode(nodeId);
            if (!managedNode) {
                return;
            }

            const QString nextName = item->text(0).trimmed();
            if (nextName == managedNode->name) {
                return;
            }

            QString error;
            if (commitManagedNodeRename(nodeId, nextName, &error)) {
                return;
            }

            const QSignalBlocker blocker(navTree_);
            item->setText(0, managedNode->title);
            if (!error.trimmed().isEmpty()) {
                QMessageBox::warning(this,
                                     QString("Rename %1").arg(managedRoleDisplayName(managedNode->role)),
                                     error);
            }
        });
        connect(navTree_, &QTreeWidget::itemCollapsed, this, [this](QTreeWidgetItem* item) {
            if (!item) {
                return;
            }
            const QString nodeId = item->data(0, Qt::UserRole).toString();
            if (!nodeById_.value(nodeId).caseRoot) {
                return;
            }
            const QSignalBlocker blocker(navTree_);
            item->setExpanded(true);
        });
        connect(navTree_, &QWidget::customContextMenuRequested, this, &MainWindow::showNavTreeContextMenu);
    }

    ensureCenterWorkspaceCreated();

    if (!rightSplitter_ && rightHost_) {
        auto* rightHostLayout = qobject_cast<QVBoxLayout*>(rightHost_->layout());
        rightSplitter_ = new QSplitter(Qt::Vertical, rightHost_);
        rightTopTabs_ = new QTabWidget();
        batchPanel_ = new BatchBoundaryPanel(rightTopTabs_);
        boundaryCsvPanel_ = new BoundaryCsvPanel(rightTopTabs_);
        rightTopTabs_->addTab(batchPanel_, "Batch Boundary");
        rightTopTabs_->addTab(boundaryCsvPanel_, "Boundary CSV");
        allowHorizontalCompression(rightSplitter_);
        allowHorizontalCompression(rightTopTabs_);
        allowHorizontalCompression(batchPanel_);
        allowHorizontalCompression(boundaryCsvPanel_);
        rightTopPanelShell_ = createPanelShell(rightSplitter_, rightTopTabs_, "rightTopPanelShell");
        rightSplitter_->addWidget(rightTopPanelShell_);
        rightBottomPanelShell_ = createPanelShell(rightSplitter_, buildAuxiliaryTabs(), "rightBottomPanelShell");
        rightSplitter_->addWidget(rightBottomPanelShell_);
        rightSplitter_->setStretchFactor(0, 3);
        rightSplitter_->setStretchFactor(1, 2);
        rightHostLayout->addWidget(rightSplitter_, 1);

        batchPanel_->setDocument(document_);
        connect(wavenumberPanel_, &WavenumberPanel::statusMessage, this, [this](const QString& message) {
            statusBar()->showMessage(message, 5000);
        });
        connect(wavenumberPanel_, &WavenumberPanel::guiActionRequested, this, &MainWindow::logGuiAction);
        connect(buildingScalePanel_, &BuildingScalePanel::statusMessage, this, [this](const QString& message) {
            statusBar()->showMessage(message, 5000);
        });
        connect(boundaryCsvPanel_, &BoundaryCsvPanel::statusMessage, this, [this](const QString& message) {
            statusBar()->showMessage(message, 5000);
        });
    }

    if (centerSplitter_) {
        centerSplitter_->show();
    }
    if (viewerStack_ && vtkView_) {
        viewerStack_->setCurrentWidget(vtkView_);
    }
    if (navTree_) {
        navPanelShell_->show();
    }
    if (vtkView_) {
        vtkView_->setProjectDirectory(document_->projectDirectory());
    }
    if (boundaryCsvPanel_) {
        boundaryCsvPanel_->setProjectDirectory(document_->projectDirectory());
    }
    updateAuxiliaryPanelLayout();

    if (!activeTrackedPanel_ || activeTrackedPanel_ == centerPlaceholderShell_) {
        setActiveTrackedPanel(navPanelShell_ ? navPanelShell_ : viewerPanelShell_);
    }

    if (firstWorkspaceLoad && central) {
        resetWorkspaceSplitters(rootSplitter_, workspaceSplitter_);
        central->layout()->activate();
        central->setUpdatesEnabled(true);
        central->update();
    }
}

QWidget* MainWindow::buildAuxiliaryTabs() {
    auto* tabs = new QTabWidget(rightSplitter_ ? static_cast<QWidget*>(rightSplitter_) : this);
    auxiliaryTabs_ = tabs;
    allowHorizontalCompression(tabs);

    wavenumberPanel_ = new WavenumberPanel(tabs);
    allowHorizontalCompression(wavenumberPanel_);
    tabs->addTab(wavenumberPanel_, "Wavenumber");

    buildingScalePanel_ = new BuildingScalePanel(tabs);
    allowHorizontalCompression(buildingScalePanel_);
    tabs->addTab(buildingScalePanel_, "Building Scale");

    auto* filesPage = new QWidget(tabs);
    auto* filesLayout = new QVBoxLayout(filesPage);
    fileModel_ = new QFileSystemModel(filesPage);
    fileModel_->setRootPath(document_->projectDirectory());
    fileTree_ = new QTreeView(filesPage);
    fileTree_->setObjectName("projectFileTree");
    fileTree_->setModel(fileModel_);
    fileTree_->setRootIndex(fileModel_->index(document_->projectDirectory()));
    fileTree_->setAlternatingRowColors(true);
    allowHorizontalCompression(fileTree_);
    fileTree_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    filesLayout->addWidget(fileTree_);
    tabs->addTab(filesPage, "Files");

    auto* rawPage = new QWidget(tabs);
    auto* rawLayout = new QVBoxLayout(rawPage);
    rawDeckEdit_ = new QPlainTextEdit(rawPage);
    allowHorizontalCompression(rawDeckEdit_);
    rawLayout->addWidget(rawDeckEdit_, 1);
    auto* rawButtons = new QWidget(rawPage);
    auto* rawButtonsLayout = new QHBoxLayout(rawButtons);
    rawButtonsLayout->setContentsMargins(0, 0, 0, 0);
    auto* applyButton = new QPushButton("Apply changes", rawButtons);
    auto* saveButton = new QPushButton("Save file", rawButtons);
    applyButton->setToolTip("apply the edited raw deck text to the current project");
    saveButton->setToolTip("save the current deck file to disk");
    rawButtonsLayout->addWidget(applyButton);
    rawButtonsLayout->addWidget(saveButton);
    rawButtonsLayout->addStretch(1);
    rawLayout->addWidget(rawButtons);
    tabs->addTab(rawPage, "Raw Deck");

    connect(applyButton, &QPushButton::clicked, this, [this] {
        QString error;
        if (!document_->applyRawText(rawDeckEdit_->toPlainText(), &error)) {
            QMessageBox::critical(this, "Apply raw deck", error);
            return;
        }
        logGuiAction("Applied raw deck edits");
    });
    connect(saveButton, &QPushButton::clicked, this, &MainWindow::saveProject);
    connect(fileTree_, &QTreeView::doubleClicked, this, [this](const QModelIndex& index) {
        const QString path = fileModel_->filePath(index);
        if (path.endsWith(".vtk", Qt::CaseInsensitive) || path.endsWith(".stl", Qt::CaseInsensitive)) {
            loadViewerFile(path);
            if (path.endsWith(".vtk", Qt::CaseInsensitive)) {
                wavenumberPanel_->setSuggestedFilePath(path);
            }
        } else if (path.endsWith(".shp", Qt::CaseInsensitive)) {
            buildingScalePanel_->setSuggestedFilePath(path);
        }
    });

    return tabs;
}

void MainWindow::showPreferencesDialog() {
    QDialog dialog(this);
    dialog.setWindowTitle("Preferences");
    dialog.resize(520, 320);

    auto* layout = new QVBoxLayout(&dialog);
    auto* form = new QFormLayout();
    auto* themeCombo = new QComboBox(&dialog);
    for (ThemeMode mode : availableThemeModes()) {
        themeCombo->addItem(themeModeDisplayName(mode), themeModeStorageKey(mode));
    }
    const ThemeMode visibleThemeMode = preferences_.themeMode == ThemeMode::Himmel
        ? ThemeMode::Frieren
        : preferences_.themeMode;
    themeCombo->setCurrentIndex(themeCombo->findData(themeModeStorageKey(visibleThemeMode)));
    form->addRow("Appearance", themeCombo);

    auto* fontSizeCombo = new QComboBox(&dialog);
    for (FontSizePreset preset : {FontSizePreset::PerfectForGenZ, FontSizePreset::Small, FontSizePreset::Normal, FontSizePreset::Large}) {
        fontSizeCombo->addItem(fontSizePresetDisplayName(preset), fontSizePresetStorageKey(preset));
    }
    fontSizeCombo->setCurrentIndex(fontSizeCombo->findData(fontSizePresetStorageKey(preferences_.fontSizePreset)));
    form->addRow("Font size", fontSizeCombo);

    auto* recentProjectsSpin = new QSpinBox(&dialog);
    recentProjectsSpin->setRange(1, kMaxRecentProjectsLimit);
    recentProjectsSpin->setValue(std::clamp(preferences_.recentProjectsLimit, 1, kMaxRecentProjectsLimit));
    form->addRow("Open recent count", recentProjectsSpin);

    auto* locationRow = new QWidget(&dialog);
    auto* locationLayout = new QHBoxLayout(locationRow);
    locationLayout->setContentsMargins(0, 0, 0, 0);
    locationLayout->setSpacing(6);
    auto* defaultLocationEdit = new QLineEdit(locationRow);
    defaultLocationEdit->setText(preferences_.defaultProjectLocation);
    defaultLocationEdit->setPlaceholderText(QDir::toNativeSeparators(preferredProjectLocation()));
    auto* browseButton = new QPushButton("Browse", locationRow);
    locationLayout->addWidget(defaultLocationEdit, 1);
    locationLayout->addWidget(browseButton);
    form->addRow("Default project location", locationRow);

    layout->addLayout(form);
    layout->addStretch(1);

    auto* footer = new QLabel("Preference file: " + QDir::toNativeSeparators(preferencesFilePath()), &dialog);
    footer->setStyleSheet("color: rgba(120, 120, 120, 0.60); font-size: 8px;");
    footer->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(footer, 0, Qt::AlignLeft);

    bool themeSelectionTouched = false;
    auto applyPreferenceChanges = [this, &dialog, themeCombo, fontSizeCombo, recentProjectsSpin, defaultLocationEdit, &themeSelectionTouched]() {
        AppPreferences updated = preferences_;
        const QString selectedThemeKey = themeCombo->currentData().toString();
        if (preferences_.themeMode == ThemeMode::Himmel
            && !themeSelectionTouched
            && selectedThemeKey == themeModeStorageKey(ThemeMode::Frieren)) {
            updated.themeMode = ThemeMode::Himmel;
        } else {
            updated.themeMode = themeModeFromString(selectedThemeKey);
        }
        updated.fontSizePreset = fontSizePresetFromString(fontSizeCombo->currentData().toString());
        updated.recentProjectsLimit = std::clamp(recentProjectsSpin->value(), 1, kMaxRecentProjectsLimit);
        updated.defaultProjectLocation = defaultLocationEdit->text().trimmed();
        if (!updated.defaultProjectLocation.isEmpty() && !QDir(updated.defaultProjectLocation).exists()) {
            QMessageBox::warning(&dialog, "Preferences", "Default project location must be an existing folder.");
            defaultLocationEdit->setFocus();
            return false;
        }
        preferences_ = updated;
        trimRecentProjectFiles();
        QString error;
        if (!persistPreferences(&error)) {
            QMessageBox::critical(&dialog, "Preferences", error);
            return false;
        }
        applyPreferences();
        statusBar()->showMessage("Preferences applied.", 3000);
        return true;
    };

    connect(themeCombo, qOverload<int>(&QComboBox::currentIndexChanged), &dialog, [&themeSelectionTouched, applyPreferenceChanges](int) {
        themeSelectionTouched = true;
        applyPreferenceChanges();
    });
    connect(fontSizeCombo, qOverload<int>(&QComboBox::currentIndexChanged), &dialog, [applyPreferenceChanges](int) {
        applyPreferenceChanges();
    });
    connect(recentProjectsSpin, qOverload<int>(&QSpinBox::valueChanged), &dialog, [applyPreferenceChanges](int) {
        applyPreferenceChanges();
    });
    connect(defaultLocationEdit, &QLineEdit::editingFinished, &dialog, [applyPreferenceChanges] {
        applyPreferenceChanges();
    });
    connect(browseButton, &QPushButton::clicked, &dialog, [&, this] {
        const QString basePath = defaultLocationEdit->text().trimmed().isEmpty()
            ? preferredProjectLocation()
            : defaultLocationEdit->text().trimmed();
        const QString path = QFileDialog::getExistingDirectory(&dialog, "Select default project location", basePath);
        if (path.isEmpty()) {
            return;
        }
        defaultLocationEdit->setText(path);
        applyPreferenceChanges();
    });

    dialog.exec();
}

void MainWindow::showAboutDialog() {
    QDialog dialog(this);
    dialog.setWindowTitle("About LatticeUrbanWind Studio");
    dialog.resize(580, 340);

    auto* layout = new QVBoxLayout(&dialog);
    auto* title = new QLabel("LatticeUrbanWind Studio", &dialog);
    QFont titleFont = title->font();
    titleFont.setBold(true);
    titleFont.setPixelSize(std::max(interfaceFontPixelSize(preferences_.fontSizePreset) + 4, 16));
    title->setFont(titleFont);
    layout->addWidget(title);

    auto* intro = new QLabel(
        "LatticeUrbanWind (LUW) is an urban wind workflow toolkit that links preprocessing, "
        "CFD solving, and result inspection inside a single desktop workspace.",
        &dialog);
    intro->setWordWrap(true);
    layout->addWidget(intro);

    auto* facts = new QFormLayout();
    facts->setLabelAlignment(Qt::AlignLeft | Qt::AlignTop);
    facts->setFormAlignment(Qt::AlignLeft | Qt::AlignTop);
    facts->setVerticalSpacing(8);
    auto makeRowTitle = [&dialog](const QString& text) {
        auto* label = new QLabel(text, &dialog);
        label->setAlignment(Qt::AlignLeft | Qt::AlignTop);
        return label;
    };
    auto makeFactLabel = [&dialog](const QString& text) {
        auto* label = new QLabel(text, &dialog);
        label->setWordWrap(true);
        label->setAlignment(Qt::AlignLeft | Qt::AlignTop);
        label->setTextInteractionFlags(Qt::TextSelectableByMouse);
        return label;
    };
    facts->addRow(makeRowTitle("Version"), makeFactLabel(
        QString("%1  |  built %2")
            .arg(buildinfo::kStudioVersion, buildinfo::kBuildTimestamp)));
    facts->addRow(makeRowTitle("CFD core"), makeFactLabel(buildinfo::kCfdCoreVersion));
    facts->addRow(makeRowTitle("Copyright"), makeFactLabel(buildinfo::kCopyrightNotice));
    facts->addRow(makeRowTitle("Author"), makeFactLabel(buildinfo::kAuthorName));

    auto* contactWidget = new QWidget(&dialog);
    auto* contactLayout = new QVBoxLayout(contactWidget);
    contactLayout->setContentsMargins(0, 0, 0, 0);
    contactLayout->setSpacing(3);

    auto* emailLabel = new QLabel(
        QString("<a href=\"mailto:%1\">%1</a>").arg(buildinfo::kAuthorEmail),
        contactWidget);
    emailLabel->setTextFormat(Qt::RichText);
    emailLabel->setTextInteractionFlags(Qt::TextBrowserInteraction);
    emailLabel->setOpenExternalLinks(true);
    emailLabel->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    contactLayout->addWidget(emailLabel, 0, Qt::AlignLeft | Qt::AlignTop);

    QString wechatId = QString::fromUtf8(buildinfo::kAuthorContact);
    const int separatorIndex = wechatId.indexOf(':');
    if (separatorIndex >= 0) {
        wechatId = wechatId.mid(separatorIndex + 1).trimmed();
    }
    auto* wechatButton = new QToolButton(contactWidget);
    wechatButton->setAutoRaise(true);
    wechatButton->setCursor(Qt::PointingHandCursor);
    wechatButton->setToolButtonStyle(Qt::ToolButtonTextOnly);
    wechatButton->setText("WeChat: " + wechatId);
    wechatButton->setStyleSheet("QToolButton { border: none; padding: 0; color: palette(highlight); }");
    connect(wechatButton, &QToolButton::clicked, &dialog, [wechatButton, wechatId] {
        if (QClipboard* clipboard = QGuiApplication::clipboard()) {
            clipboard->setText(wechatId);
        }
        QToolTip::showText(wechatButton->mapToGlobal(QPoint(wechatButton->width() / 2, wechatButton->height())),
                           "WeChat ID copied");
    });
    contactLayout->addWidget(wechatButton, 0, Qt::AlignLeft | Qt::AlignTop);

    facts->addRow(makeRowTitle("Contact"), contactWidget);
    layout->addLayout(facts, 1);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok, &dialog);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    layout->addWidget(buttons);

    dialog.exec();
}

void MainWindow::applyPreferences() {
    titleBarThemeClickTimes_.clear();
    applyTheme(*qApp, preferences_.themeMode, preferences_.fontSizePreset);
    applyTitleBarStyle();
    buildMenus();
    configureStatusBarGeometry();
    refreshWindowButtons();
    if (workflowBar_ && centralWidget() && centralWidget()->layout()) {
        auto* layout = qobject_cast<QVBoxLayout*>(centralWidget()->layout());
        if (layout) {
            layout->removeWidget(workflowBar_);
            workflowBar_->deleteLater();
            workflowBar_ = buildWorkflowBar();
            layout->insertWidget(0, workflowBar_);
            updateProjectAvailability();
        }
    }
    for (QWidget* panelShell : trackedPanels_) {
        repolishTrackedPanel(panelShell);
    }
}

bool MainWindow::persistPreferences(QString* errorMessage) {
    return savePreferences(preferences_, errorMessage);
}

QString MainWindow::preferredProjectLocation() const {
    const QString preferred = preferences_.defaultProjectLocation.trimmed();
    if (!preferred.isEmpty() && QDir(preferred).exists()) {
        return preferred;
    }
    if (hasLoadedProject()) {
        return document_->projectDirectory();
    }
    return QDir::homePath();
}

QString MainWindow::preferredProjectDeckPath(RunMode mode) const {
    return QDir(preferredProjectLocation()).filePath(defaultDeckName(mode));
}

bool MainWindow::openProjectFile(const QString& path) {
    const QString normalizedPath = normalizeProjectDeckPath(path);
    if (normalizedPath.isEmpty()) {
        return false;
    }
    if (!QFileInfo::exists(normalizedPath)) {
        QMessageBox::warning(
            this,
            "Open project",
            "Project file does not exist:\n" + QDir::toNativeSeparators(normalizedPath));

        bool removed = false;
        for (int index = static_cast<int>(preferences_.recentProjectFiles.size()) - 1; index >= 0; --index) {
            if (sameProjectDeckPath(
                    normalizeProjectDeckPath(preferences_.recentProjectFiles.at(index)),
                    normalizedPath)) {
                preferences_.recentProjectFiles.removeAt(index);
                removed = true;
            }
        }
        if (removed) {
            QString error;
            if (!persistPreferences(&error)) {
                statusBar()->showMessage("Failed to update recent projects: " + error, 5000);
            }
            updateOpenRecentMenu();
        }
        return false;
    }

    QString error;
    {
        const QSignalBlocker blocker(document_);
        if (!document_->loadFromFile(normalizedPath, &error)) {
            QMessageBox::critical(this, "Open project", error);
            return false;
        }
    }
    syncProjectUiFromDocument();
    rememberRecentProjectFile(normalizedPath);
    logGuiAction("Opened project " + QFileInfo(normalizedPath).fileName());
    return true;
}

void MainWindow::rememberRecentProjectFile(const QString& filePath) {
    const QString normalizedPath = normalizeProjectDeckPath(filePath);
    if (normalizedPath.isEmpty()) {
        return;
    }

    QStringList updatedFiles;
    updatedFiles.reserve(static_cast<int>(preferences_.recentProjectFiles.size()) + 1);
    updatedFiles.push_back(normalizedPath);
    for (const QString& existingPath : preferences_.recentProjectFiles) {
        const QString normalizedExistingPath = normalizeProjectDeckPath(existingPath);
        if (!normalizedExistingPath.isEmpty()
            && !sameProjectDeckPath(normalizedExistingPath, normalizedPath)) {
            updatedFiles.push_back(normalizedExistingPath);
        }
    }
    preferences_.recentProjectFiles = updatedFiles;
    trimRecentProjectFiles();

    QString error;
    if (!persistPreferences(&error)) {
        statusBar()->showMessage("Failed to update recent projects: " + error, 5000);
    }
    updateOpenRecentMenu();
}

void MainWindow::trimRecentProjectFiles() {
    preferences_.recentProjectsLimit = std::clamp(
        preferences_.recentProjectsLimit,
        1,
        kMaxRecentProjectsLimit);

    QStringList trimmedFiles;
    trimmedFiles.reserve(std::min(static_cast<int>(preferences_.recentProjectFiles.size()), preferences_.recentProjectsLimit));
    for (const QString& filePath : preferences_.recentProjectFiles) {
        const QString normalizedPath = normalizeProjectDeckPath(filePath);
        if (normalizedPath.isEmpty()) {
            continue;
        }

        bool duplicate = false;
        for (const QString& existingPath : trimmedFiles) {
            if (sameProjectDeckPath(existingPath, normalizedPath)) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }

        trimmedFiles.push_back(normalizedPath);
        if (trimmedFiles.size() >= preferences_.recentProjectsLimit) {
            break;
        }
    }
    preferences_.recentProjectFiles = trimmedFiles;
}

void MainWindow::updateOpenRecentMenu() {
    if (!openRecentMenu_) {
        return;
    }

    trimRecentProjectFiles();
    openRecentMenu_->clear();
    if (preferences_.recentProjectFiles.isEmpty()) {
        QAction* emptyAction = openRecentMenu_->addAction("No recent projects");
        emptyAction->setEnabled(false);
        openRecentMenu_->setEnabled(false);
        return;
    }

    openRecentMenu_->setEnabled(true);
    for (const QString& filePath : preferences_.recentProjectFiles) {
        const QString normalizedPath = normalizeProjectDeckPath(filePath);
        if (normalizedPath.isEmpty()) {
            continue;
        }

        QAction* action = openRecentMenu_->addAction(recentProjectActionText(normalizedPath));
        action->setToolTip(QDir::toNativeSeparators(normalizedPath));
        connect(action, &QAction::triggered, this, [this, normalizedPath] {
            openProjectFile(normalizedPath);
        });
    }
}

void MainWindow::syncProjectUiFromDocument() {
    if (modeCombo_) {
        const QSignalBlocker blocker(modeCombo_);
        modeCombo_->setCurrentIndex(static_cast<int>(document_->mode()));
    }
    ensureProjectWorkspaceLoaded();
    if (batchPanel_) {
        batchPanel_->refresh();
    }
    rebuildSectionPages();
    refreshEditors();
    refreshRawDeck();
    refreshFileTree();
    updateProjectAvailability();
}

void MainWindow::openProject() {
    const QString path = chooseDeckFilePath(
        this,
        "Open Project",
        preferredProjectLocation(),
        QFileDialog::AcceptOpen,
        QFileDialog::ExistingFile);
    if (path.isEmpty()) {
        return;
    }
    openProjectFile(path);
}

bool MainWindow::saveProject() {
    QString error;
    if (document_->filePath().isEmpty()) {
        return saveProjectAs();
    }
    if (!document_->save(&error)) {
        QMessageBox::critical(this, "Save project", error);
        return false;
    }
    rememberRecentProjectFile(document_->filePath());
    logGuiAction("Saved project " + QFileInfo(document_->filePath()).fileName());
    return true;
}

bool MainWindow::saveProjectAs() {
    const QString path = chooseDeckFilePath(
        this,
        "Save Project As",
        document_->filePath().isEmpty()
            ? preferredProjectDeckPath(document_->mode())
            : QDir(document_->projectDirectory()).filePath(defaultDeckName(document_->mode())),
        QFileDialog::AcceptSave,
        QFileDialog::AnyFile);
    if (path.isEmpty()) {
        return false;
    }
    QString error;
    {
        const QSignalBlocker blocker(document_);
        if (!document_->saveAs(path, &error)) {
            QMessageBox::critical(this, "Save project", error);
            return false;
        }
    }
    syncProjectUiFromDocument();
    rememberRecentProjectFile(path);
    logGuiAction("Saved project as " + QFileInfo(path).fileName());
    return true;
}

void MainWindow::createProjectWizard() {
    QDialog dialog(this);
    dialog.setWindowTitle("New Project");
    dialog.resize(520, 220);

    auto* layout = new QVBoxLayout(&dialog);
    auto* form = new QFormLayout();
    auto* folderRow = new QWidget(&dialog);
    auto* folderLayout = new QHBoxLayout(folderRow);
    folderLayout->setContentsMargins(0, 0, 0, 0);
    auto* folderEdit = new QLineEdit(folderRow);
    folderEdit->setText(preferredProjectLocation());
    auto* browseButton = new QPushButton("Browse", folderRow);
    folderLayout->addWidget(folderEdit, 1);
    folderLayout->addWidget(browseButton);
    form->addRow("Parent folder", folderRow);

    auto* caseEdit = new QLineEdit(&dialog);
    caseEdit->setText(document_->hasKey("casename") ? document_->typedValue("casename").toString() : "case_luw");
    form->addRow("Case name", caseEdit);

    auto* modeCombo = new QComboBox(&dialog);
    modeCombo->addItems({"LUW", "LUWDG", "LUWPF"});
    modeCombo->setCurrentIndex(static_cast<int>(document_->mode()));
    form->addRow("Run mode", modeCombo);
    layout->addLayout(form);
    layout->addStretch(1);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    layout->addWidget(buttons);

    connect(browseButton, &QPushButton::clicked, &dialog, [&dialog, folderEdit] {
        const QString path = QFileDialog::getExistingDirectory(&dialog, "Select Parent Folder", folderEdit->text());
        if (!path.isEmpty()) {
            folderEdit->setText(path);
        }
    });
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    const QString parentFolder = folderEdit->text().trimmed();
    const QString caseName = caseEdit->text().trimmed();
    if (parentFolder.isEmpty()) {
        QMessageBox::warning(this, "New project", "Select a parent folder.");
        return;
    }
    if (caseName.isEmpty()) {
        QMessageBox::warning(this, "New project", "Enter a case name.");
        return;
    }
    if (caseName.contains(QRegularExpression(R"([\\/:*?"<>|])"))) {
        QMessageBox::warning(this, "New project", "Case name contains invalid path characters.");
        return;
    }

    const RunMode mode = static_cast<RunMode>(std::clamp(modeCombo->currentIndex(), 0, 2));
    const QString caseDirectory = QDir(parentFolder).filePath(caseName);
    const QDir caseDir(caseDirectory);
    if (caseDir.exists()) {
        const QFileInfoList existingEntries = caseDir.entryInfoList(QDir::NoDotAndDotDot | QDir::AllEntries);
        if (!existingEntries.isEmpty()) {
            const auto answer = QMessageBox::question(
                this,
                "New project",
                "The case folder already exists and is not empty. Continue and overwrite the project file?");
            if (answer != QMessageBox::Yes) {
                return;
            }
        }
    }

    for (const QString& subdirectory : {"building_db", "gui_properties", "proj_temp", "RESULTS", "terrain_db", "wind_bc"}) {
        if (!QDir().mkpath(QDir(caseDirectory).filePath(subdirectory))) {
            QMessageBox::critical(this, "New project", "Failed to create " + subdirectory + ".");
            return;
        }
    }

    QString error;
    const QString deckPath = QDir(caseDirectory).filePath(defaultDeckName(mode));
    {
        const QSignalBlocker blocker(document_);
        newDeck(mode, caseName);
        if (!document_->saveAs(deckPath, &error)) {
            QMessageBox::critical(this, "New project", error);
            return;
        }
    }

    syncProjectUiFromDocument();
    rememberRecentProjectFile(deckPath);
    logGuiAction("Created project " + caseName + " in " + QDir::toNativeSeparators(caseDirectory));
}

void MainWindow::importProjectAsset(const QString& targetSubdirectory, const QString& dialogTitle) {
    if (document_->filePath().isEmpty()) {
        QMessageBox::warning(this, "Import", "Create or open a project before importing files.");
        return;
    }

    const ProjectImportKind importKind = importKindForSubdirectory(targetSubdirectory);
    if (importKind == ProjectImportKind::Unknown) {
        QMessageBox::critical(this, "Import", "Unsupported import target: " + targetSubdirectory + ".");
        return;
    }

    const QString selectedFile = QFileDialog::getOpenFileName(
        this,
        dialogTitle,
        document_->projectDirectory(),
        importDialogFilter(importKind));
    if (selectedFile.isEmpty()) {
        return;
    }

    const QString suffix = QFileInfo(selectedFile).suffix().toLower();
    if (importKind == ProjectImportKind::Building && suffix != "shp") {
        QMessageBox::warning(this, "Import", "Building Database import requires a .shp file.");
        return;
    }
    if (importKind == ProjectImportKind::Terrain
        && suffix != "shp"
        && suffix != "tif"
        && suffix != "tiff") {
        QMessageBox::warning(this, "Import", "Terrain Database import supports .shp, .tif, or .tiff.");
        return;
    }
    if (importKind == ProjectImportKind::Wind
        && suffix != "nc"
        && suffix != "out") {
        QMessageBox::warning(this, "Import", "Wind Data import supports .nc or .out.");
        return;
    }

    const QString targetDirectory = QDir(document_->projectDirectory()).filePath(targetSubdirectory);
    if (!QDir().mkpath(targetDirectory)) {
        QMessageBox::critical(this, "Import", "Failed to create " + targetSubdirectory + ".");
        return;
    }

    const QStringList existingFiles = existingImportArtifacts(targetDirectory, importKind);
    if (!existingFiles.isEmpty()) {
        const auto answer = QMessageBox::question(
            this,
            dialogTitle,
            QString("The project already contains %1 files under %2.\n\nReplace the existing files?")
                .arg(importKindDisplayName(importKind), QDir::toNativeSeparators(targetDirectory)),
            QMessageBox::Yes | QMessageBox::No,
            QMessageBox::No);
        if (answer != QMessageBox::Yes) {
            return;
        }
        QString removeError;
        if (!removeFilesReplacing(existingFiles, &removeError)) {
            QMessageBox::critical(this, "Import", removeError);
            return;
        }
    }

    const QStringList sourceFiles = sourceFilesForImport(selectedFile);
    QString canonicalBaseName;
    QString canonicalWindName;
    if (importKind == ProjectImportKind::Wind) {
        canonicalWindName = canonicalWindFileName(selectedFile, document_);
    } else {
        canonicalBaseName = canonicalImportBaseName(importKind, selectedFile, document_);
    }

    QString error;
    int copiedCount = 0;
    for (const QString& sourcePath : sourceFiles) {
        QString targetFileName;
        if (importKind == ProjectImportKind::Wind) {
            targetFileName = canonicalWindName;
        } else {
            targetFileName = targetFileNameForBaseName(sourcePath, canonicalBaseName);
        }
        const QString targetPath = QDir(targetDirectory).filePath(targetFileName);
        if (!copyFileReplacing(sourcePath, targetPath, &error)) {
            QMessageBox::critical(this, "Import", error);
            return;
        }
        ++copiedCount;
    }

    const QString successMessage = QString("Imported %1 file(s) into %2.")
        .arg(copiedCount)
        .arg(QDir::toNativeSeparators(targetDirectory));
    statusBar()->showMessage(successMessage, 5000);
    logGuiAction(successMessage);
    refreshFileTree();
}

void MainWindow::stopActiveWork() {
    if (!runner_->isRunning()) {
        statusBar()->showMessage("No backend process is running.", 3000);
        return;
    }
    stopRequested_ = true;
    logGuiAction("Stop requested for the active backend process");
    if (progressPanel_) {
        progressPanel_->setBusy("Stopping", runner_->activeTitle());
    }
    runner_->stop();
}

bool MainWindow::hasLoadedProject() const {
    return document_ && !document_->filePath().trimmed().isEmpty();
}

void MainWindow::updateAuxiliaryPanelLayout() {
    if (!batchPanel_ || !boundaryCsvPanel_ || !rightSplitter_ || !rightTopTabs_ || !document_) {
        return;
    }
    const bool luwMode = document_->mode() == RunMode::Luw;
    rightTopTabs_->setTabVisible(0, !luwMode);
    rightTopTabs_->setTabVisible(1, luwMode);
    if (luwMode) {
        rightTopTabs_->setCurrentWidget(boundaryCsvPanel_);
    } else {
        rightTopTabs_->setCurrentWidget(batchPanel_);
    }
    rightSplitter_->setSizes({420, 280});
}

void MainWindow::setRightPanelVisible(bool visible) {
    if (!workspaceSplitter_ || !rightHost_) {
        return;
    }

    const QList<int> rootSizes = rootSplitter_ ? rootSplitter_->sizes() : QList<int>{380, 1420};
    const int currentLeft = rootSizes.value(0, 380);
    const int currentWorkspace = rootSizes.value(1, 1420);
    const QList<int> currentSizes = workspaceSplitter_->sizes();
    const int currentCenter = currentSizes.value(0, 1040);
    const int currentRight = currentSizes.value(1, 0);
    const int totalWidth = std::max(currentCenter + currentRight, 0);

    if (visible) {
        if (hasLoadedProject()) {
            ensureProjectWorkspaceLoaded();
        }
        if (leftSplitter_) {
            leftSplitter_->setMinimumWidth(std::max(currentLeft, 280));
        }
        rightHost_->setVisible(hasLoadedProject());
        if (rootSplitter_) {
            rootSplitter_->setSizes({currentLeft, currentWorkspace});
        }
        if (hasLoadedProject()) {
            constexpr int kMinimumCenterWidth = 320;
            const int targetRight = rightPanelExpandedSizes_.size() == 2 && rightPanelExpandedSizes_.value(1) > 0
                ? rightPanelExpandedSizes_.value(1)
                : 380;
            const int maxRightFromCenter = std::max(totalWidth - kMinimumCenterWidth, 0);
            const int clampedRight = std::max(0, std::min(targetRight, maxRightFromCenter));
            const int targetCenter = std::max(totalWidth - clampedRight, 0);
            workspaceSplitter_->setSizes({targetCenter, clampedRight});
        }
    } else {
        if (currentSizes.size() == 2 && currentRight > 0) {
            rightPanelExpandedSizes_ = currentSizes;
        }
        rightHost_->setVisible(false);
        if (leftSplitter_) {
            leftSplitter_->setMinimumWidth(280);
        }
        if (rootSplitter_) {
            rootSplitter_->setSizes({currentLeft, currentWorkspace});
        }
        workspaceSplitter_->setSizes({currentCenter + currentRight, 0});
        if (activeTrackedPanel_ && isWidgetOrAncestor(activeTrackedPanel_, rightHost_)) {
            setActiveTrackedPanel(viewerPanelShell_ ? viewerPanelShell_ : sectionPanelShell_);
        }
    }
    syncWorkflowChromeButtons();
}

void MainWindow::syncWorkflowChromeButtons() {
    if (consoleToggleButton_) {
        const QSignalBlocker blocker(consoleToggleButton_);
        consoleToggleButton_->setChecked(console_ && !console_->isCollapsed());
    }
    if (sidePanelToggleButton_) {
        const QSignalBlocker blocker(sidePanelToggleButton_);
        sidePanelToggleButton_->setChecked(rightHost_ && rightHost_->isVisible());
    }
}

void MainWindow::logGuiAction(const QString& message) {
    if (!console_ || runner_->isRunning() || message.trimmed().isEmpty()) {
        return;
    }
    console_->appendText("[GUI] " + message.trimmed() + "\n");
}

void MainWindow::newDeck(RunMode mode, const QString& caseName) {
    document_->clearFilePath();
    document_->setMode(mode);
    document_->applyRawText(buildSkeletonText(mode, caseName));
    refreshEditors();
    refreshRawDeck();
}

QString MainWindow::buildSkeletonText(RunMode mode, const QString& caseName) const {
    QStringList lines;
    lines << "// LUW Studio generated starter deck";
    const QString resolvedCaseName = caseName.trimmed().isEmpty()
        ? QString("case_%1").arg(runModeDisplayName(mode).toLower())
        : caseName.trimmed();
    lines << QString("casename = %1").arg(resolvedCaseName);
    lines << QString("datetime = %1").arg(QDateTime::currentDateTimeUtc().toString("yyyyMMddhhmmss"));
    lines << "cut_lon_manual = [121.300000, 121.700000]";
    lines << "cut_lat_manual = [31.100000, 31.400000]";
    lines << "base_height = 50.000000";
    lines << "z_limit = 500.000000";
    if (mode == RunMode::Luw || mode == RunMode::Luwpf) {
        lines << "geometry_mode = 2";
        lines << "terr_voxel_height_field = auto";
        lines << "terr_voxel_ignore_under = 0.000000";
        lines << "terr_voxel_approach = idw";
        lines << "terr_voxel_grid_resolution = 50.000000";
        lines << "terr_voxel_idw_sigma = 1.000000";
        lines << "terr_voxel_idw_power = 2.000000";
        lines << "terr_voxel_idw_neighbors = 12";
    }
    lines << "si_x_cfd = [0.000000, 1000.000000]";
    lines << "si_y_cfd = [0.000000, 1000.000000]";
    lines << "si_z_cfd = [0.000000, 300.000000]";
    lines << "";
    lines << "// CFD Controls";
    lines << "n_gpu = [1, 1, 1]";
    lines << "mesh_control = \"gpu_memory\"";
    lines << "gpu_memory = 20000";
    lines << "cell_size = ";
    lines << "validation = pass";
    lines << "high_order = true";
    lines << "flux_correction = true";
    lines << "downstream_open_face = false";
    lines << "coriolis_term = true";
    lines << "buoyancy = false";
    lines << "";
    lines << "// Output & Probes";
    lines << "unsteady_output = 0";
    lines << "probes_output = 0";
    lines << "purge_avg = 0";
    lines << "purge_avg_stride = 1";
    lines << "output_tke_ti_tls = [tke, ti, tls]";
    lines << "probes = ";
    lines << "";
    lines << "// Physics";
    lines << "ibm_enabler = false";
    lines << "enable_buffer_nudging = 1";
    lines << "buffer_thickness_m = 160.000000";
    lines << "buffer_tau_s = 300.000000";
    lines << "buffer_nudge_vertical = 0";
    lines << "enable_top_sponge = 1";
    lines << "sponge_thickness_m = 200.000000";
    lines << "sponge_tau_s = 120.000000";
    lines << "sponge_ref_mode = 0";
    lines << "";
    lines << "// Turbulence inflow";
    lines << "turb_inflow_enable = true";
    lines << "turb_inflow_approach = vonkarman";
    lines << "vk_inlet_ti = 0.050000";
    lines << "vk_inlet_sigma = 0.0";
    lines << "vk_inlet_l = 100.000000";
    lines << "vk_inlet_nmodes = 256";
    lines << "vk_inlet_seed = 100";
    lines << "vk_inlet_update_stride = 1";
    lines << "vk_inlet_uc_mode = NORM_MEAN";
    lines << "vk_inlet_same_realization_all_faces = true";
    lines << "vk_inlet_stride_interpolation = false";
    lines << "vk_inlet_inflow_only = false";
    lines << "vk_inlet_anisotropy = [1.000000, 1.000000, 1.000000]";
    if (mode == RunMode::Luwdg || mode == RunMode::Luwpf) {
        lines << "";
        lines << "// Batch";
        lines << "x_exp_rat = 5.000000";
        lines << "y_exp_rat = 5.000000";
        lines << "angle = [0.000000, 90.000000, 180.000000, 270.000000]";
    }
    if (mode == RunMode::Luwdg) {
        lines << "inflow = [5.000000, 10.000000]";
    }
    return lines.join('\n') + '\n';
}

void MainWindow::refreshEditors() {
    if (workflowSummary_) {
        workflowSummary_->setText(
            QString("Mode: %1\nDeck: %2\nProject: %3\nRESULTS: %4")
                .arg(runModeDisplayName(document_->mode()))
                .arg(document_->filePath().isEmpty() ? "<unsaved>" : document_->filePath())
                .arg(document_->projectDirectory())
                .arg(document_->resultsDirectory()));
    }

    for (const EditorBinding& binding : bindings_) {
        if (auto* label = qobject_cast<QLabel*>(binding.label)) {
            const QString displayLabel = fieldDisplayLabel(binding.nodeId, binding.spec, binding.propertyField);
            label->setText(displayLabel);
            label->setToolTip(displayLabel);
        }
        writeEditorValue(binding, binding.propertyField ? QVariant(document_->rawValue(binding.spec.key))
                                                       : document_->typedValue(binding.spec.key));
    }
    refreshEditorStates();
}

void MainWindow::refreshRawDeck() {
    if (!rawDeckEdit_) {
        return;
    }
    if (!hasLoadedProject()) {
        rawDeckEdit_->blockSignals(true);
        rawDeckEdit_->clear();
        rawDeckEdit_->blockSignals(false);
        return;
    }
    const QString rendered = document_->renderedText();
    if (rawDeckEdit_->toPlainText() == rendered) {
        return;
    }
    rawDeckEdit_->blockSignals(true);
    rawDeckEdit_->setPlainText(rendered);
    rawDeckEdit_->blockSignals(false);
}

void MainWindow::refreshEditorStates() {
    for (const EditorBinding& binding : bindings_) {
        if (binding.propertyField) {
            if (binding.label) {
                binding.label->setVisible(true);
                binding.label->setEnabled(true);
            }
            binding.editor->setVisible(true);
            binding.editor->setEnabled(true);
            continue;
        }

        const bool active = isFieldActive(binding.spec.key);
        if (binding.label) {
            binding.label->setVisible(active || binding.spec.readOnly);
            binding.label->setEnabled(active || binding.spec.readOnly);
        }
        if (!active) {
            clearEditorDisplay(binding);
        }
        binding.editor->setVisible(active);
        binding.editor->setEnabled(active && !binding.spec.readOnly);
    }
}

void MainWindow::refreshFileTree() {
    if (!fileModel_ || !fileTree_) {
        return;
    }
    if (!hasLoadedProject()) {
        fileTree_->setRootIndex(QModelIndex());
        return;
    }
    fileModel_->setRootPath(document_->projectDirectory());
    fileTree_->setRootIndex(fileModel_->index(document_->projectDirectory()));
}

void MainWindow::updateProjectAvailability() {
    const bool active = hasLoadedProject();

    if (active) {
        ensureProjectWorkspaceLoaded();
    }

    for (QWidget* widget : projectScopedWidgets_) {
        if (widget) {
            widget->setEnabled(active);
        }
    }
    for (QAction* action : projectScopedActions_) {
        if (action) {
            action->setEnabled(active);
        }
    }

    if (navPanelShell_) {
        navPanelShell_->setVisible(active);
    }
    if (centerPlaceholderShell_) {
        centerPlaceholderShell_->setVisible(!active);
    }
    if (centerSplitter_) {
        centerSplitter_->setVisible(true);
        centerSplitter_->setEnabled(true);
    }
    if (viewerStack_ && centerPlaceholder_ && vtkView_) {
        viewerStack_->setCurrentWidget(active ? static_cast<QWidget*>(vtkView_) : centerPlaceholder_);
    }
    if (viewerPanelShell_) {
        viewerPanelShell_->setVisible(true);
    }
    if (vtkView_) {
        vtkView_->setEnabled(active);
    }
    if (console_) {
        console_->setEnabled(true);
    }
    if (rightTopTabs_) {
        rightTopTabs_->setEnabled(active);
    }
    if (auxiliaryTabs_) {
        auxiliaryTabs_->setEnabled(active);
    }
    setRightPanelVisible(active && sidePanelToggleButton_ && sidePanelToggleButton_->isChecked());
    if (progressPanel_ && !active) {
        progressPanel_->setIdle();
    }
    if (!active) {
        setActiveTrackedPanel(viewerPanelShell_ ? viewerPanelShell_ : centerPlaceholderShell_);
        statusBar()->showMessage("Open or create a project to enable editing.", 5000);
        resetWorkspaceSplitters(rootSplitter_, workspaceSplitter_);
    }
    syncWorkflowChromeButtons();
}

void MainWindow::setCurrentPage(const QString& pageId) {
    const auto it = pageById_.constFind(pageId);
    if (it == pageById_.cend()) {
        return;
    }
    currentNodeId_ = pageId;
    sectionStack_->setCurrentIndex(it.value());
}

void MainWindow::runPreset(CommandPreset preset) {
    if (!ensureProjectSaved("Save project before running")) {
        return;
    }
    stopRequested_ = false;
    runner_->startPreset(preset);
}

bool MainWindow::ensureProjectSaved(const QString& dialogTitle) {
    if (!hasLoadedProject()) {
        statusBar()->showMessage("Create or open a project first.", 4000);
        return false;
    }
    QString error;
    if (document_->filePath().isEmpty()) {
        const QString path = chooseDeckFilePath(
            this,
            dialogTitle,
            QDir(document_->projectDirectory()).filePath(defaultDeckName(document_->mode())),
            QFileDialog::AcceptSave,
            QFileDialog::AnyFile);
        if (path.isEmpty()) {
            return false;
        }
        if (!document_->saveAs(path, &error)) {
            QMessageBox::critical(this, "Save project", error);
            return false;
        }
    } else if (!document_->save(&error)) {
        QMessageBox::critical(this, "Save project", error);
        return false;
    }
    return true;
}

QString MainWindow::latestResultBaseName() const {
    const QString vtkPath = latestResultVtk();
    if (!vtkPath.isEmpty()) {
        return QFileInfo(vtkPath).completeBaseName();
    }
    const QString caseName = document_->typedValue("casename").toString().trimmed();
    const QString timestamp = document_->typedValue("datetime").toString().trimmed();
    if (!caseName.isEmpty() && !timestamp.isEmpty()) {
        return QString("uvw-%1_%2").arg(caseName, timestamp);
    }
    return "luw_export";
}

QString MainWindow::defaultNetCdfPath() const {
    QString path = QDir(document_->resultsDirectory()).filePath(latestResultBaseName() + ".nc");
    if (!path.endsWith(".nc", Qt::CaseInsensitive)) {
        path += ".nc";
    }
    return path;
}

QString MainWindow::resolveTargetCrs() const {
    auto normalize = [](QString value) {
        value = value.trimmed();
        if (value.isEmpty()) {
            return value;
        }
        if (QRegularExpression(R"(^\d+$)").match(value).hasMatch()) {
            return "EPSG:" + value;
        }
        return value;
    };

    const QString utmCrs = normalize(document_->typedValue("utm_crs").toString());
    if (!utmCrs.isEmpty()) {
        return utmCrs;
    }
    for (const QString& key : {QStringLiteral("utm_epsg"), QStringLiteral("utm")}) {
        const int code = document_->typedValue(key).toInt();
        if (code > 0) {
            return QString("EPSG:%1").arg(code);
        }
    }
    const int zone = document_->typedValue("utm_zone").toInt();
    if (zone > 0) {
        const QString hemisphere = document_->typedValue("utm_hemisphere").toString().trimmed().toUpper();
        const int base = hemisphere == "S" ? 32700 : 32600;
        return QString("EPSG:%1").arg(base + zone);
    }
    return {};
}

void MainWindow::runVtk2NcExport() {
    if (!ensureProjectSaved("Save project before exporting NetCDF")) {
        return;
    }

    QString path = QFileDialog::getSaveFileName(
        this,
        "Export NetCDF",
        defaultNetCdfPath(),
        "NetCDF (*.nc)");
    if (path.isEmpty()) {
        return;
    }
    if (QFileInfo(path).suffix().isEmpty()) {
        path += ".nc";
    }

    if (console_) {
        console_->appendText("[GUI] NetCDF output: " + QDir::toNativeSeparators(path) + "\n");
    }
    stopRequested_ = false;
    runner_->startPreset(CommandPreset::Vtk2Nc, {"--output", path});
}

void MainWindow::runVisLuwExport() {
    if (!ensureProjectSaved("Save project before generating visualization data")) {
        return;
    }

    const QVariantList lon = document_->typedValue("cut_lon_manual").toList();
    const QVariantList lat = document_->typedValue("cut_lat_manual").toList();
    const QString defaultOutputDir = QDir(document_->resultsDirectory()).filePath("visualization_data");
    const QString defaultNcPath = QDir(defaultOutputDir).filePath(latestResultBaseName() + "_visluw.nc");

    QDialog dialog(this);
    dialog.setWindowTitle("Generate visualization data");
    dialog.resize(560, 260);

    auto* layout = new QVBoxLayout(&dialog);
    auto* form = new QFormLayout();
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);

    auto* outputRow = new QWidget(&dialog);
    auto* outputLayout = new QHBoxLayout(outputRow);
    outputLayout->setContentsMargins(0, 0, 0, 0);
    auto* outputDirEdit = new QLineEdit(defaultOutputDir, outputRow);
    auto* outputBrowseButton = new QPushButton("Browse", outputRow);
    outputLayout->addWidget(outputDirEdit, 1);
    outputLayout->addWidget(outputBrowseButton);
    form->addRow("Export folder", outputRow);

    auto* lonMinSpin = new QDoubleSpinBox(&dialog);
    auto* lonMaxSpin = new QDoubleSpinBox(&dialog);
    auto* latMinSpin = new QDoubleSpinBox(&dialog);
    auto* latMaxSpin = new QDoubleSpinBox(&dialog);
    for (QDoubleSpinBox* spin : {lonMinSpin, lonMaxSpin, latMinSpin, latMaxSpin}) {
        spin->setDecimals(6);
        spin->setRange(-180.0, 180.0);
        spin->setSingleStep(0.01);
    }
    lonMinSpin->setValue(lon.size() >= 1 ? lon[0].toDouble() : 0.0);
    lonMaxSpin->setValue(lon.size() >= 2 ? lon[1].toDouble() : lonMinSpin->value());
    latMinSpin->setRange(-90.0, 90.0);
    latMaxSpin->setRange(-90.0, 90.0);
    latMinSpin->setValue(lat.size() >= 1 ? lat[0].toDouble() : 0.0);
    latMaxSpin->setValue(lat.size() >= 2 ? lat[1].toDouble() : latMinSpin->value());
    form->addRow("Longitude min", lonMinSpin);
    form->addRow("Longitude max", lonMaxSpin);
    form->addRow("Latitude min", latMinSpin);
    form->addRow("Latitude max", latMaxSpin);

    auto* layerSpin = new QSpinBox(&dialog);
    layerSpin->setRange(1, 30);
    layerSpin->setValue(6);
    form->addRow("Section layers", layerSpin);

    auto* exportNcBox = new QCheckBox("Export NetCDF", &dialog);
    auto* ncRow = new QWidget(&dialog);
    auto* ncLayout = new QHBoxLayout(ncRow);
    ncLayout->setContentsMargins(0, 0, 0, 0);
    auto* ncPathEdit = new QLineEdit(defaultNcPath, ncRow);
    auto* ncBrowseButton = new QPushButton("Browse", ncRow);
    ncLayout->addWidget(ncPathEdit, 1);
    ncLayout->addWidget(ncBrowseButton);
    form->addRow(exportNcBox, ncRow);

    layout->addLayout(form);
    layout->addStretch(1);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    layout->addWidget(buttons);

    connect(outputBrowseButton, &QPushButton::clicked, &dialog, [&dialog, outputDirEdit] {
        const QString path = QFileDialog::getExistingDirectory(&dialog, "Select export folder", outputDirEdit->text());
        if (!path.isEmpty()) {
            outputDirEdit->setText(path);
        }
    });
    connect(ncBrowseButton, &QPushButton::clicked, &dialog, [&dialog, ncPathEdit] {
        QString path = QFileDialog::getSaveFileName(&dialog, "Select NetCDF path", ncPathEdit->text(), "NetCDF (*.nc)");
        if (!path.isEmpty() && QFileInfo(path).suffix().isEmpty()) {
            path += ".nc";
        }
        if (!path.isEmpty()) {
            ncPathEdit->setText(path);
        }
    });
    connect(exportNcBox, &QCheckBox::toggled, ncRow, &QWidget::setEnabled);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    ncRow->setEnabled(false);

    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    const QString outputDir = outputDirEdit->text().trimmed();
    if (outputDir.isEmpty()) {
        QMessageBox::warning(this, "Generate visualization data", "Select an export folder.");
        return;
    }

    QStringList args = {
        "--lon-min", QString::number(std::min(lonMinSpin->value(), lonMaxSpin->value()), 'f', 6),
        "--lon-max", QString::number(std::max(lonMinSpin->value(), lonMaxSpin->value()), 'f', 6),
        "--lat-min", QString::number(std::min(latMinSpin->value(), latMaxSpin->value()), 'f', 6),
        "--lat-max", QString::number(std::max(latMinSpin->value(), latMaxSpin->value()), 'f', 6),
        "--layers", QString::number(layerSpin->value()),
        "--output-dir", outputDir
    };

    if (exportNcBox->isChecked()) {
        QString ncPath = ncPathEdit->text().trimmed();
        if (ncPath.isEmpty()) {
            QMessageBox::warning(this, "Generate visualization data", "Select a NetCDF output path or disable NetCDF export.");
            return;
        }
        if (QFileInfo(ncPath).suffix().isEmpty()) {
            ncPath += ".nc";
        }
        args << "--export-nc" << "--nc-output" << ncPath;
        if (console_) {
            console_->appendText("[GUI] Visualization NetCDF output: " + QDir::toNativeSeparators(ncPath) + "\n");
        }
    }

    if (console_) {
        console_->appendText("[GUI] Visualization export folder: " + QDir::toNativeSeparators(outputDir) + "\n");
    }
    stopRequested_ = false;
    runner_->startPreset(CommandPreset::VisLuw, args);
}

void MainWindow::runCutVis() {
    if (!ensureProjectSaved("Save project before generating cut visuals")) {
        return;
    }

    const QString vtkPath = latestResultVtk();
    if (vtkPath.isEmpty()) {
        QMessageBox::warning(this, "Generate cut visuals", "No VTK result was found in the RESULTS folder.");
        return;
    }

    const QVariantList lon = document_->typedValue("cut_lon_manual").toList();
    const QVariantList lat = document_->typedValue("cut_lat_manual").toList();
    const QString targetCrs = resolveTargetCrs();
    if (targetCrs.isEmpty()) {
        QMessageBox::warning(this, "Generate cut visuals", "The deck does not contain a usable projected CRS.");
        return;
    }

    double defaultCellSize = 10.0;
    if (document_->typedValue("mesh_control").toString().trimmed().compare("cell_size", Qt::CaseInsensitive) == 0) {
        bool ok = false;
        const double cellSize = document_->rawValue("cell_size").trimmed().toDouble(&ok);
        if (ok && cellSize > 0.0) {
            defaultCellSize = cellSize;
        }
    }

    const QString defaultOutputDir = QDir(document_->resultsDirectory()).filePath("cut_visuals");
    const QString defaultCroppedPath = QDir(defaultOutputDir).filePath(latestResultBaseName() + "_cropped.vtk");

    QDialog dialog(this);
    dialog.setWindowTitle("Generate cut visuals");
    dialog.resize(600, 360);

    auto* layout = new QVBoxLayout(&dialog);
    auto* form = new QFormLayout();
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);

    auto* vtkRow = new QWidget(&dialog);
    auto* vtkLayout = new QHBoxLayout(vtkRow);
    vtkLayout->setContentsMargins(0, 0, 0, 0);
    auto* vtkEdit = new QLineEdit(vtkPath, vtkRow);
    auto* vtkBrowseButton = new QPushButton("Browse", vtkRow);
    vtkLayout->addWidget(vtkEdit, 1);
    vtkLayout->addWidget(vtkBrowseButton);
    form->addRow("VTK field", vtkRow);

    auto* outputRow = new QWidget(&dialog);
    auto* outputLayout = new QHBoxLayout(outputRow);
    outputLayout->setContentsMargins(0, 0, 0, 0);
    auto* outputDirEdit = new QLineEdit(defaultOutputDir, outputRow);
    auto* outputBrowseButton = new QPushButton("Browse", outputRow);
    outputLayout->addWidget(outputDirEdit, 1);
    outputLayout->addWidget(outputBrowseButton);
    form->addRow("Export folder", outputRow);

    auto* lonMinSpin = new QDoubleSpinBox(&dialog);
    auto* lonMaxSpin = new QDoubleSpinBox(&dialog);
    auto* latMinSpin = new QDoubleSpinBox(&dialog);
    auto* latMaxSpin = new QDoubleSpinBox(&dialog);
    for (QDoubleSpinBox* spin : {lonMinSpin, lonMaxSpin, latMinSpin, latMaxSpin}) {
        spin->setDecimals(6);
        spin->setRange(-180.0, 180.0);
        spin->setSingleStep(0.01);
    }
    latMinSpin->setRange(-90.0, 90.0);
    latMaxSpin->setRange(-90.0, 90.0);
    lonMinSpin->setValue(lon.size() >= 1 ? lon[0].toDouble() : 0.0);
    lonMaxSpin->setValue(lon.size() >= 2 ? lon[1].toDouble() : lonMinSpin->value());
    latMinSpin->setValue(lat.size() >= 1 ? lat[0].toDouble() : 0.0);
    latMaxSpin->setValue(lat.size() >= 2 ? lat[1].toDouble() : latMinSpin->value());
    form->addRow("Longitude min", lonMinSpin);
    form->addRow("Longitude max", lonMaxSpin);
    form->addRow("Latitude min", latMinSpin);
    form->addRow("Latitude max", latMaxSpin);

    auto* cellSizeSpin = new QDoubleSpinBox(&dialog);
    cellSizeSpin->setRange(0.1, 1000.0);
    cellSizeSpin->setDecimals(3);
    cellSizeSpin->setValue(defaultCellSize);
    form->addRow("Grid cell size", cellSizeSpin);

    auto* quiverStepSpin = new QSpinBox(&dialog);
    quiverStepSpin->setRange(1, 500);
    quiverStepSpin->setValue(50);
    form->addRow("Arrow stride", quiverStepSpin);

    auto* dpiSpin = new QSpinBox(&dialog);
    dpiSpin->setRange(72, 600);
    dpiSpin->setValue(220);
    form->addRow("Figure DPI", dpiSpin);

    auto* croppedBox = new QCheckBox("Export cropped VTK", &dialog);
    auto* croppedRow = new QWidget(&dialog);
    auto* croppedLayout = new QHBoxLayout(croppedRow);
    croppedLayout->setContentsMargins(0, 0, 0, 0);
    auto* croppedEdit = new QLineEdit(defaultCroppedPath, croppedRow);
    auto* croppedBrowseButton = new QPushButton("Browse", croppedRow);
    croppedLayout->addWidget(croppedEdit, 1);
    croppedLayout->addWidget(croppedBrowseButton);
    form->addRow(croppedBox, croppedRow);

    layout->addLayout(form);
    layout->addStretch(1);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    layout->addWidget(buttons);

    connect(vtkBrowseButton, &QPushButton::clicked, &dialog, [&dialog, vtkEdit] {
        const QString path = QFileDialog::getOpenFileName(&dialog, "Select VTK field", vtkEdit->text(), "VTK (*.vtk)");
        if (!path.isEmpty()) {
            vtkEdit->setText(path);
        }
    });
    connect(outputBrowseButton, &QPushButton::clicked, &dialog, [&dialog, outputDirEdit] {
        const QString path = QFileDialog::getExistingDirectory(&dialog, "Select export folder", outputDirEdit->text());
        if (!path.isEmpty()) {
            outputDirEdit->setText(path);
        }
    });
    connect(croppedBrowseButton, &QPushButton::clicked, &dialog, [&dialog, croppedEdit] {
        QString path = QFileDialog::getSaveFileName(&dialog, "Select cropped VTK path", croppedEdit->text(), "VTK (*.vtk)");
        if (!path.isEmpty() && QFileInfo(path).suffix().isEmpty()) {
            path += ".vtk";
        }
        if (!path.isEmpty()) {
            croppedEdit->setText(path);
        }
    });
    connect(croppedBox, &QCheckBox::toggled, croppedRow, &QWidget::setEnabled);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    croppedRow->setEnabled(false);

    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    if (vtkEdit->text().trimmed().isEmpty() || outputDirEdit->text().trimmed().isEmpty()) {
        QMessageBox::warning(this, "Generate cut visuals", "Select a VTK field and an export folder.");
        return;
    }

    QStringList args = {
        "--skip-config",
        vtkEdit->text().trimmed(),
        "--data-dir", document_->resultsDirectory(),
        "--output-dir", outputDirEdit->text().trimmed(),
        "--target-crs", targetCrs,
        "--utm-crs", targetCrs,
        "--min-lon", QString::number(std::min(lonMinSpin->value(), lonMaxSpin->value()), 'f', 6),
        "--max-lon", QString::number(std::max(lonMinSpin->value(), lonMaxSpin->value()), 'f', 6),
        "--min-lat", QString::number(std::min(latMinSpin->value(), latMaxSpin->value()), 'f', 6),
        "--max-lat", QString::number(std::max(latMinSpin->value(), latMaxSpin->value()), 'f', 6),
        "--cell-size", QString::number(cellSizeSpin->value(), 'f', 3),
        "--quiver-step", QString::number(quiverStepSpin->value()),
        "--dpi", QString::number(dpiSpin->value())
    };

    if (lon.size() >= 2) {
        args << "--domain-lon-min" << QString::number(std::min(lon[0].toDouble(), lon[1].toDouble()), 'f', 6);
        args << "--domain-lon-max" << QString::number(std::max(lon[0].toDouble(), lon[1].toDouble()), 'f', 6);
    }
    if (lat.size() >= 2) {
        args << "--domain-lat-min" << QString::number(std::min(lat[0].toDouble(), lat[1].toDouble()), 'f', 6);
        args << "--domain-lat-max" << QString::number(std::max(lat[0].toDouble(), lat[1].toDouble()), 'f', 6);
    }
    if (document_->hasKey("rotate_deg")) {
        args << "--rotate-deg" << QString::number(document_->typedValue("rotate_deg").toDouble(), 'f', 6);
    }
    if (croppedBox->isChecked()) {
        QString croppedPath = croppedEdit->text().trimmed();
        if (croppedPath.isEmpty()) {
            QMessageBox::warning(this, "Generate cut visuals", "Select a path for the cropped VTK export.");
            return;
        }
        if (QFileInfo(croppedPath).suffix().isEmpty()) {
            croppedPath += ".vtk";
        }
        args << "--export-cropped-vtk" << "--cropped-vtk-path" << croppedPath;
        if (console_) {
            console_->appendText("[GUI] Cropped VTK output: " + QDir::toNativeSeparators(croppedPath) + "\n");
        }
    }

    if (console_) {
        console_->appendText("[GUI] Cut visuals export folder: " + QDir::toNativeSeparators(outputDirEdit->text().trimmed()) + "\n");
    }
    stopRequested_ = false;
    runner_->startPreset(CommandPreset::CutVis, args);
}

QStringList MainWindow::buildCutVisArguments() const {
    QStringList args;
    const QString vtkPath = latestResultVtk();
    if (!vtkPath.isEmpty()) {
        args << vtkPath;
    }
    args << "--data-dir" << document_->resultsDirectory();
    args << "--output-dir" << QDir(document_->resultsDirectory()).filePath("cut_visuals");

    const QString targetCrs = resolveTargetCrs();
    if (!targetCrs.isEmpty()) {
        args << "--target-crs" << targetCrs;
        args << "--utm-crs" << targetCrs;
    }

    const QVariantList lon = document_->typedValue("cut_lon_manual").toList();
    if (lon.size() >= 2) {
        args << "--min-lon" << QString::number(std::min(lon[0].toDouble(), lon[1].toDouble()), 'f', 6);
        args << "--max-lon" << QString::number(std::max(lon[0].toDouble(), lon[1].toDouble()), 'f', 6);
        args << "--domain-lon-min" << QString::number(std::min(lon[0].toDouble(), lon[1].toDouble()), 'f', 6);
        args << "--domain-lon-max" << QString::number(std::max(lon[0].toDouble(), lon[1].toDouble()), 'f', 6);
    }
    const QVariantList lat = document_->typedValue("cut_lat_manual").toList();
    if (lat.size() >= 2) {
        args << "--min-lat" << QString::number(std::min(lat[0].toDouble(), lat[1].toDouble()), 'f', 6);
        args << "--max-lat" << QString::number(std::max(lat[0].toDouble(), lat[1].toDouble()), 'f', 6);
        args << "--domain-lat-min" << QString::number(std::min(lat[0].toDouble(), lat[1].toDouble()), 'f', 6);
        args << "--domain-lat-max" << QString::number(std::max(lat[0].toDouble(), lat[1].toDouble()), 'f', 6);
    }

    return args;
}

QString MainWindow::latestResultVtk() const {
    QDir solverVtkDir(QDir(document_->resultsDirectory()).filePath("vtk"));
    if (solverVtkDir.exists()) {
        QFileInfoList avgFiles = solverVtkDir.entryInfoList({"*_avg-*.vtk"}, QDir::Files, QDir::Time);
        if (!avgFiles.isEmpty()) {
            return avgFiles.front().absoluteFilePath();
        }
        QFileInfoList allSolverFiles = solverVtkDir.entryInfoList({"*.vtk"}, QDir::Files, QDir::Time);
        if (!allSolverFiles.isEmpty()) {
            return allSolverFiles.front().absoluteFilePath();
        }
    }

    QDir resultsDir(document_->resultsDirectory());
    QFileInfoList files = resultsDir.entryInfoList({"*.vtk"}, QDir::Files, QDir::Time);
    if (files.isEmpty()) {
        return {};
    }
    return files.front().absoluteFilePath();
}

void MainWindow::loadLatestResult() {
    QString error;
    if (!vtkView_ || !vtkView_->loadLatestResult(&error)) {
        statusBar()->showMessage(error.isEmpty() ? "No VTK file found in RESULTS/vtk." : error, 5000);
        return;
    }
}

void MainWindow::loadViewerFile(const QString& filePath) {
    if (!vtkView_) {
        return;
    }
    QString error;
    if (!vtkView_->loadFile(filePath, &error)) {
        QMessageBox::critical(this, "Load visualization file", error);
        return;
    }
    if (filePath.endsWith(".vtk", Qt::CaseInsensitive) && wavenumberPanel_) {
        wavenumberPanel_->setSuggestedFilePath(filePath);
    }
}

void MainWindow::fixDeckFile() {
    if (!hasLoadedProject()) {
        statusBar()->showMessage("Create or open a project first.", 4000);
        return;
    }
    if (runner_->isRunning()) {
        QMessageBox::information(this,
                                 "Fix deck file",
                                 "Stop the active backend job before fixing the deck file.");
        return;
    }

    const QString sourceText = rawDeckEdit_ ? rawDeckEdit_->toPlainText() : document_->rawText();
    if (sourceText.trimmed().isEmpty()) {
        statusBar()->showMessage("The current raw deck is empty.", 4000);
        return;
    }

    const QMessageBox::StandardButton confirmed = QMessageBox::question(
        this,
        "Fix deck file",
        "This will reorganize the current raw deck using canonical ordering, fill any missing required "
        "fields with defaults, print the before/after deck to the console, and save the fixed deck back "
        "to disk.\n\nContinue?",
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);
    if (confirmed != QMessageBox::Yes) {
        return;
    }

    auto appendConsoleBlock = [this](const QString& title, const QString& text) {
        if (!console_) {
            return;
        }
        QString block = text;
        if (!block.endsWith('\n')) {
            block += '\n';
        }
        console_->appendText("[GUI] ---- " + title + " ----\n");
        console_->appendText(block);
        console_->appendText("[GUI] ---- end " + title + " ----\n");
    };

    if (console_) {
        console_->appendText("[GUI] Fix deck file requested for "
                             + QDir::toNativeSeparators(document_->filePath()) + "\n");
    }
    appendConsoleBlock("original deck", sourceText);

    if (progressPanel_) {
        progressPanel_->setBusy("Fix deck file", "Scanning raw deck");
    }
    statusBar()->showMessage("Fixing deck file...");
    QCoreApplication::processEvents();

    QString fixedText;
    QStringList operations;
    QString error;
    if (!document_->repairRawText(sourceText, &fixedText, &operations, &error)) {
        if (progressPanel_) {
            progressPanel_->showTerminalStatus("Failed", "deck fix");
        }
        const QString detail = error.trimmed().isEmpty() ? "Failed to reorganize the current deck." : error;
        if (console_) {
            console_->appendText("[GUI] Fix deck file failed: " + detail + "\n");
        }
        QMessageBox::critical(this, "Fix deck file", detail);
        statusBar()->showMessage("Fix deck file failed.", 5000);
        return;
    }

    if (console_) {
        if (operations.isEmpty()) {
            console_->appendText("[GUI] Deck fix actions: none. Deck is already canonical.\n");
        } else {
            console_->appendText("[GUI] Deck fix actions:\n");
            for (const QString& operation : operations) {
                console_->appendText("  - " + operation + "\n");
            }
        }
    }

    if (operations.isEmpty()) {
        if (progressPanel_) {
            progressPanel_->showTerminalStatus("Completed", "deck already clean");
        }
        statusBar()->showMessage("Deck is already canonical; no fix actions were needed.", 5000);
        return;
    }

    if (!document_->applyRawText(fixedText, &error)) {
        if (progressPanel_) {
            progressPanel_->showTerminalStatus("Failed", "deck fix");
        }
        const QString detail = error.trimmed().isEmpty() ? "Failed to apply the fixed deck text." : error;
        if (console_) {
            console_->appendText("[GUI] Fix deck file failed while applying the repaired deck: "
                                 + detail + "\n");
        }
        QMessageBox::critical(this, "Fix deck file", detail);
        statusBar()->showMessage("Fix deck file failed.", 5000);
        return;
    }

    appendConsoleBlock("fixed deck", fixedText);

    QString saveError;
    if (!document_->save(&saveError)) {
        if (progressPanel_) {
            progressPanel_->showTerminalStatus("Completed", "save pending");
        }
        const QString detail = saveError.trimmed().isEmpty() ? "Unknown save error." : saveError;
        if (console_) {
            console_->appendText("[GUI] Deck fix completed, but saving failed: " + detail + "\n");
        }
        QMessageBox::warning(this,
                             "Fix deck file",
                             "The deck was fixed in the editor, but saving to disk failed:\n\n" + detail);
        statusBar()->showMessage("Deck fixed in the editor, but saving failed.", 7000);
        return;
    }

    if (progressPanel_) {
        progressPanel_->showTerminalStatus("Completed", "deck file fixed");
    }
    statusBar()->showMessage("Deck file fixed and saved.", 5000);
    logGuiAction("Fixed deck file " + QFileInfo(document_->filePath()).fileName());
}

QVariant MainWindow::readEditorValue(const EditorBinding& binding) const {
    switch (binding.spec.kind) {
    case FieldKind::Boolean:
        return qobject_cast<QCheckBox*>(binding.editor)->isChecked();
    case FieldKind::Enum: {
        auto* combo = qobject_cast<QComboBox*>(binding.editor);
        const QVariant value = combo->currentData();
        return value.isValid() ? value : QVariant(combo->currentText());
    }
    case FieldKind::Integer:
        return qobject_cast<QSpinBox*>(binding.editor)->value();
    case FieldKind::Float: {
        if (auto* line = dynamic_cast<PrecisionNumericLineEdit*>(binding.editor)) {
            bool ok = false;
            const QString text = line->canonicalText();
            const double value = text.toDouble(&ok);
            return ok ? QVariant(value) : QVariant(text);
        }
        return qobject_cast<QDoubleSpinBox*>(binding.editor)->value();
    }
    case FieldKind::FloatPair:
    case FieldKind::FloatTriplet:
    case FieldKind::FloatList:
        if (auto* triplet = dynamic_cast<SplitTripletEditor*>(binding.editor)) {
            return parseLooseList(triplet->rawTextForDeck(), false);
        }
        if (auto* line = dynamic_cast<PrecisionNumericLineEdit*>(binding.editor)) {
            return parseLooseList(line->canonicalText(), false);
        }
        return parseLooseList(qobject_cast<QLineEdit*>(binding.editor)->text(), false);
    case FieldKind::UIntTriplet:
        if (auto* triplet = dynamic_cast<SplitTripletEditor*>(binding.editor)) {
            return parseLooseList(triplet->rawTextForDeck(), true);
        }
        return parseLooseList(qobject_cast<QLineEdit*>(binding.editor)->text(), true);
    case FieldKind::TokenList: {
        const QStringList tokens = qobject_cast<QLineEdit*>(binding.editor)->text().split(QRegularExpression(R"([,\s]+)"), Qt::SkipEmptyParts);
        return tokens;
    }
    case FieldKind::Multiline:
        return qobject_cast<QPlainTextEdit*>(binding.editor)->toPlainText();
    case FieldKind::String:
        if (binding.spec.key == "terr_voxel_height_field") {
            return normalizeTerrainHeightFieldValue(qobject_cast<QLineEdit*>(binding.editor)->text());
        }
        return qobject_cast<QLineEdit*>(binding.editor)->text();
    }
    return {};
}

bool MainWindow::isFieldActive(const QString& key) const {
    if (!document_) {
        return true;
    }

    const QString normalized = key.trimmed().toLower();
    const QString meshControl = document_->typedValue("mesh_control").toString().trimmed().toLower();
    if (normalized == "gpu_memory") {
        return meshControl != "cell_size";
    }
    if (normalized == "cell_size") {
        return meshControl == "cell_size";
    }
    if (normalized == "buffer_thickness_m" ||
        normalized == "buffer_tau_s") {
        return document_->typedValue("enable_buffer_nudging").toBool();
    }
    if (normalized == "sponge_thickness_m" ||
        normalized == "sponge_tau_s" ||
        normalized == "sponge_ref_mode") {
        return document_->typedValue("enable_top_sponge").toBool();
    }
    if (normalized == "purge_avg_stride" ||
        normalized == "output_tke_ti_tls") {
        return document_->typedValue("purge_avg").toInt() > 0;
    }
    if (normalized == "turb_inflow_approach" ||
        normalized.startsWith("vk_inlet_")) {
        return document_->typedValue("turb_inflow_enable").toBool();
    }
    if (normalized == "terr_voxel_idw_power") {
        const QString approach = document_->typedValue("terr_voxel_approach").toString().trimmed().toLower();
        return approach != "kriging" && approach != "kriging_gpu";
    }
    return true;
}

void MainWindow::clearEditorDisplay(const EditorBinding& binding) {
    if (binding.propertyField) {
        auto* widget = qobject_cast<QLineEdit*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->clear();
        return;
    }

    switch (binding.spec.kind) {
    case FieldKind::Boolean: {
        auto* widget = qobject_cast<QCheckBox*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->setChecked(false);
        break;
    }
    case FieldKind::Enum: {
        auto* widget = qobject_cast<QComboBox*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->setCurrentIndex(-1);
        break;
    }
    case FieldKind::Integer: {
        auto* widget = qobject_cast<QSpinBox*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->clear();
        break;
    }
    case FieldKind::Float: {
        if (auto* precisionWidget = dynamic_cast<PrecisionNumericLineEdit*>(binding.editor)) {
            precisionWidget->setRawText({});
        } else {
            auto* spinWidget = qobject_cast<QDoubleSpinBox*>(binding.editor);
            const QSignalBlocker blocker(spinWidget);
            spinWidget->clear();
        }
        break;
    }
    case FieldKind::FloatPair:
    case FieldKind::FloatTriplet:
    case FieldKind::UIntTriplet:
    case FieldKind::FloatList:
    case FieldKind::TokenList:
    case FieldKind::String: {
        if (auto* triplet = dynamic_cast<SplitTripletEditor*>(binding.editor)) {
            triplet->setRawText({});
        } else if (auto* precision = dynamic_cast<PrecisionNumericLineEdit*>(binding.editor)) {
            precision->setRawText({});
        } else {
            auto* widget = qobject_cast<QLineEdit*>(binding.editor);
            const QSignalBlocker blocker(widget);
            widget->clear();
        }
        break;
    }
    case FieldKind::Multiline: {
        auto* widget = qobject_cast<QPlainTextEdit*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->clear();
        break;
    }
    }
}

void MainWindow::writeEditorValue(const EditorBinding& binding, const QVariant& value) {
    if (binding.propertyField) {
        auto* widget = qobject_cast<QLineEdit*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->setText(value.toString());
        widget->deselect();
        widget->setCursorPosition(0);
        return;
    }

    switch (binding.spec.kind) {
    case FieldKind::Boolean: {
        auto* widget = qobject_cast<QCheckBox*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->setChecked(value.toBool());
        break;
    }
    case FieldKind::Enum: {
        auto* widget = qobject_cast<QComboBox*>(binding.editor);
        const QSignalBlocker blocker(widget);
        QString targetValue = value.toString();
        if (binding.spec.key == "turb_inflow_approach" && targetValue.trimmed().isEmpty()) {
            targetValue = "vonkarman";
        }
        int index = widget->findData(targetValue);
        if (index < 0) {
            index = widget->findText(targetValue);
        }
        widget->setCurrentIndex(index);
        break;
    }
    case FieldKind::Integer: {
        auto* widget = qobject_cast<QSpinBox*>(binding.editor);
        const QSignalBlocker blocker(widget);
        if (value.metaType().id() == QMetaType::QString && value.toString().trimmed().isEmpty()) {
            widget->clear();
        } else {
            widget->setValue(value.toInt());
        }
        break;
    }
    case FieldKind::Float: {
        if (auto* precisionWidget = dynamic_cast<PrecisionNumericLineEdit*>(binding.editor)) {
            precisionWidget->setRawText(document_ ? document_->rawValue(binding.spec.key) : QString());
        } else {
            auto* spinWidget = qobject_cast<QDoubleSpinBox*>(binding.editor);
            const QSignalBlocker blocker(spinWidget);
            if (value.metaType().id() == QMetaType::QString && value.toString().trimmed().isEmpty()) {
                spinWidget->clear();
            } else {
                spinWidget->setValue(value.toDouble());
            }
        }
        break;
    }
    case FieldKind::FloatPair:
    case FieldKind::FloatTriplet:
    case FieldKind::FloatList: {
        if (auto* triplet = dynamic_cast<SplitTripletEditor*>(binding.editor)) {
            triplet->setRawText(document_ ? document_->rawValue(binding.spec.key) : QString());
        } else if (auto* precisionWidget = dynamic_cast<PrecisionNumericLineEdit*>(binding.editor)) {
            precisionWidget->setRawText(document_ ? document_->rawValue(binding.spec.key) : QString());
        } else {
            auto* lineWidget = qobject_cast<QLineEdit*>(binding.editor);
            const QSignalBlocker blocker(lineWidget);
            lineWidget->setText(renderVariantList(value.toList(), false));
            lineWidget->deselect();
            lineWidget->setCursorPosition(0);
        }
        break;
    }
    case FieldKind::UIntTriplet: {
        if (auto* triplet = dynamic_cast<SplitTripletEditor*>(binding.editor)) {
            triplet->setRawText(document_ ? document_->rawValue(binding.spec.key) : QString());
        } else {
            auto* lineWidget = qobject_cast<QLineEdit*>(binding.editor);
            const QSignalBlocker blocker(lineWidget);
            lineWidget->setText(renderVariantList(value.toList(), true));
            lineWidget->deselect();
            lineWidget->setCursorPosition(0);
        }
        break;
    }
    case FieldKind::TokenList: {
        auto* widget = qobject_cast<QLineEdit*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->setText(value.toStringList().join(", "));
        widget->deselect();
        widget->setCursorPosition(0);
        break;
    }
    case FieldKind::Multiline: {
        auto* widget = qobject_cast<QPlainTextEdit*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->setPlainText(value.toString());
        break;
    }
    case FieldKind::String: {
        auto* widget = qobject_cast<QLineEdit*>(binding.editor);
        const QSignalBlocker blocker(widget);
        const QString displayValue = binding.spec.key == "terr_voxel_height_field"
            ? displayTerrainHeightFieldValue(value.toString())
            : value.toString();
        widget->setText(displayValue);
        widget->deselect();
        widget->setCursorPosition(0);
        break;
    }
    }
}

}
