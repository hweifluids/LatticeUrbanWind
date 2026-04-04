#include "luwgui/Theme.h"

#include <algorithm>

#include <QColor>
#include <QFont>
#include <QPalette>

namespace luwgui {

namespace {

struct ThemeSpec {
    ThemeMode mode;
    QString displayName;
    QString storageKey;
    QColor window;
    QColor windowText;
    QColor base;
    QColor alternateBase;
    QColor tooltipBase;
    QColor tooltipText;
    QColor text;
    QColor button;
    QColor buttonText;
    QColor brightText;
    QColor highlight;
    QColor highlightedText;
    QColor link;
    QColor visitedLink;
    QColor disabledText;
    QColor disabledButtonText;
    QColor chrome;
    QColor inputBorder;
    QColor buttonBorder;
    QColor buttonHover;
    QColor buttonPressed;
    QColor header;
    QColor tab;
    QColor tabSelected;
    QColor splitter;
    QColor progressChunk;
    QColor muted;
    QColor emptyState;
    QColor menuHover;
    QColor focusBorder;
};

QString colorName(const QColor& color) {
    if (color.alpha() < 255) {
        return QStringLiteral("rgba(%1, %2, %3, %4)")
            .arg(color.red())
            .arg(color.green())
            .arg(color.blue())
            .arg(QString::number(color.alphaF(), 'f', 3));
    }
    return color.name(QColor::HexRgb);
}

QColor blendColors(const QColor& a, const QColor& b, qreal t) {
    const qreal u = std::clamp(t, 0.0, 1.0);
    return QColor::fromRgbF(
        a.redF() + (b.redF() - a.redF()) * u,
        a.greenF() + (b.greenF() - a.greenF()) * u,
        a.blueF() + (b.blueF() - a.blueF()) * u,
        1.0);
}

QVector<ThemeMode> allThemeModes() {
    return {
        ThemeMode::LightDefault,
        ThemeMode::DarkDefault,
        ThemeMode::White,
        ThemeMode::Black,
        ThemeMode::Pink,
        ThemeMode::Green,
        ThemeMode::Siemens,
        ThemeMode::Frieren,
        ThemeMode::Himmel,
    };
}

ThemeSpec themeSpec(ThemeMode mode) {
    switch (mode) {
    case ThemeMode::LightDefault:
        return {
            mode,
            "Light (Default)",
            "light_default",
            QColor(224, 229, 235),
            QColor(31, 40, 49),
            QColor(246, 248, 250),
            QColor(233, 238, 242),
            QColor(248, 250, 252),
            QColor(31, 40, 49),
            QColor(31, 40, 49),
            QColor(218, 225, 231),
            QColor(31, 40, 49),
            QColor(255, 255, 255),
            QColor(77, 116, 156),
            QColor(250, 251, 252),
            QColor(58, 97, 138),
            QColor(77, 116, 156),
            QColor(128, 136, 144),
            QColor(128, 136, 144),
            QColor(216, 222, 228),
            QColor(170, 181, 192),
            QColor(147, 161, 175),
            QColor(199, 211, 222),
            QColor(183, 196, 208),
            QColor(207, 216, 225),
            QColor(214, 221, 229),
            QColor(198, 210, 220),
            QColor(188, 200, 210),
            QColor(91, 133, 175),
            QColor(98, 112, 125),
            QColor(155, 165, 173),
            QColor(218, 236, 240),
            QColor(94, 127, 157),
        };
    case ThemeMode::DarkDefault:
        return {
            mode,
            "Dark (Default)",
            "dark_default",
            QColor(19, 27, 36),
            QColor(216, 224, 232),
            QColor(12, 18, 25),
            QColor(25, 34, 45),
            QColor(31, 45, 58),
            QColor(220, 228, 235),
            QColor(213, 221, 229),
            QColor(26, 36, 48),
            QColor(216, 224, 232),
            QColor(245, 245, 245),
            QColor(73, 116, 161),
            QColor(246, 248, 250),
            QColor(128, 170, 214),
            QColor(73, 116, 161),
            QColor(114, 124, 136),
            QColor(114, 124, 136),
            QColor(23, 33, 43),
            QColor(48, 64, 80),
            QColor(64, 81, 98),
            QColor(47, 67, 87),
            QColor(30, 43, 56),
            QColor(34, 49, 66),
            QColor(26, 37, 48),
            QColor(44, 65, 85),
            QColor(34, 48, 66),
            QColor(91, 133, 175),
            QColor(143, 156, 167),
            QColor(120, 129, 139),
            QColor(55, 82, 107),
            QColor(91, 133, 175),
        };
    case ThemeMode::White:
        return {
            mode,
            "White",
            "white",
            QColor(255, 255, 255),
            QColor(34, 39, 46),
            QColor(255, 255, 255),
            QColor(248, 249, 251),
            QColor(255, 255, 255),
            QColor(34, 39, 46),
            QColor(34, 39, 46),
            QColor(248, 249, 251),
            QColor(34, 39, 46),
            QColor(255, 255, 255),
            QColor(67, 102, 146),
            QColor(255, 255, 255),
            QColor(67, 102, 146),
            QColor(67, 102, 146),
            QColor(154, 160, 167),
            QColor(154, 160, 167),
            QColor(246, 247, 249),
            QColor(204, 210, 217),
            QColor(194, 201, 210),
            QColor(239, 243, 247),
            QColor(226, 232, 239),
            QColor(241, 244, 247),
            QColor(246, 247, 249),
            QColor(235, 239, 244),
            QColor(219, 225, 231),
            QColor(79, 118, 164),
            QColor(122, 132, 142),
            QColor(171, 176, 182),
            QColor(226, 236, 245),
            QColor(79, 118, 164),
        };
    case ThemeMode::Black:
        return {
            mode,
            "Black",
            "black",
            QColor(10, 10, 12),
            QColor(229, 232, 236),
            QColor(18, 19, 22),
            QColor(28, 30, 35),
            QColor(26, 27, 31),
            QColor(229, 232, 236),
            QColor(229, 232, 236),
            QColor(24, 25, 29),
            QColor(229, 232, 236),
            QColor(255, 255, 255),
            QColor(88, 145, 209),
            QColor(255, 255, 255),
            QColor(129, 181, 237),
            QColor(88, 145, 209),
            QColor(110, 119, 131),
            QColor(110, 119, 131),
            QColor(14, 15, 18),
            QColor(53, 57, 64),
            QColor(70, 74, 81),
            QColor(40, 44, 51),
            QColor(22, 24, 28),
            QColor(28, 30, 36),
            QColor(18, 20, 24),
            QColor(37, 40, 47),
            QColor(35, 38, 45),
            QColor(88, 145, 209),
            QColor(132, 141, 150),
            QColor(106, 111, 118),
            QColor(34, 55, 79),
            QColor(88, 145, 209),
        };
    case ThemeMode::Pink:
        return {
            mode,
            "Pink",
            "pink",
            QColor(248, 238, 243),
            QColor(74, 43, 57),
            QColor(255, 250, 252),
            QColor(247, 238, 242),
            QColor(255, 249, 252),
            QColor(74, 43, 57),
            QColor(74, 43, 57),
            QColor(241, 225, 232),
            QColor(74, 43, 57),
            QColor(255, 255, 255),
            QColor(194, 103, 138),
            QColor(255, 252, 254),
            QColor(166, 87, 119),
            QColor(194, 103, 138),
            QColor(166, 144, 154),
            QColor(166, 144, 154),
            QColor(243, 228, 235),
            QColor(214, 186, 197),
            QColor(199, 168, 182),
            QColor(233, 210, 221),
            QColor(223, 193, 206),
            QColor(234, 214, 223),
            QColor(239, 221, 229),
            QColor(226, 198, 211),
            QColor(213, 190, 202),
            QColor(194, 103, 138),
            QColor(143, 106, 120),
            QColor(186, 165, 174),
            QColor(243, 216, 226),
            QColor(194, 103, 138),
        };
    case ThemeMode::Green:
        return {
            mode,
            "Green",
            "green",
            QColor(234, 242, 235),
            QColor(40, 61, 48),
            QColor(248, 251, 248),
            QColor(237, 244, 238),
            QColor(248, 251, 248),
            QColor(40, 61, 48),
            QColor(40, 61, 48),
            QColor(223, 234, 224),
            QColor(40, 61, 48),
            QColor(255, 255, 255),
            QColor(87, 145, 108),
            QColor(250, 252, 250),
            QColor(78, 128, 96),
            QColor(87, 145, 108),
            QColor(132, 145, 136),
            QColor(132, 145, 136),
            QColor(228, 237, 229),
            QColor(181, 199, 185),
            QColor(157, 180, 163),
            QColor(209, 225, 212),
            QColor(192, 212, 197),
            QColor(211, 226, 214),
            QColor(218, 229, 220),
            QColor(200, 216, 203),
            QColor(189, 206, 192),
            QColor(87, 145, 108),
            QColor(110, 128, 117),
            QColor(157, 170, 160),
            QColor(214, 232, 218),
            QColor(87, 145, 108),
        };
    case ThemeMode::Siemens:
        return {
            mode,
            "Siemens",
            "siemens",
            QColor("#ffffff"),
            QColor("#203743"),
            QColor("#ffffff"),
            QColor("#f6fbfc"),
            QColor("#ffffff"),
            QColor("#203743"),
            QColor("#203743"),
            QColor("#f3fafb"),
            QColor("#203743"),
            QColor("#ffffff"),
            QColor("#01698d"),
            QColor("#ffffff"),
            QColor("#0b6f92"),
            QColor("#01698d"),
            QColor("#8ea0ac"),
            QColor("#8ea0ac"),
            QColor("#ffffff"),
            QColor("#bfd0d8"),
            QColor("#a4c0cc"),
            QColor("#e9f5f8"),
            QColor("#d9edf3"),
            QColor("#ffffff"),
            QColor("#f3fafb"),
            QColor("#ffffff"),
            QColor("#d7e7ec"),
            QColor("#01698d"),
            QColor("#5d7480"),
            QColor("#b6c3ca"),
            QColor("#dff0f5"),
            QColor("#017ca3"),
        };
    case ThemeMode::Frieren:
        return {
            mode,
            "Frieren",
            "frieren",
            QColor("#ece5d7"),
            QColor("#3f3124"),
            QColor("#f4f1e8"),
            QColor("#ebe3d4"),
            QColor("#fbf7ef"),
            QColor("#3f3124"),
            QColor("#3f3124"),
            QColor("#e6ddcc"),
            QColor("#3f3124"),
            QColor("#ffffff"),
            QColor("#c8a45a"),
            QColor("#2f2419"),
            QColor("#8e6f32"),
            QColor("#8e2a2b"),
            QColor("#9f907b"),
            QColor("#9f907b"),
            QColor("#e7dece"),
            QColor("#c7b79c"),
            QColor("#bca98a"),
            QColor("#c8a45a"),
            QColor("#b69047"),
            QColor("#e7dece"),
            QColor("#e5dac6"),
            QColor("#f4f1e8"),
            QColor("#d3c8b5"),
            QColor("#c8a45a"),
            QColor("#8a7760"),
            QColor("#9d8a73"),
            QColor("#c8a45a"),
            QColor("#c8a45a"),
        };
    case ThemeMode::Himmel:
        return {
            mode,
            "Himmel",
            "himmel",
            QColor("#f4f0e6"),
            QColor("#243b63"),
            QColor("#fbf8f1"),
            QColor("#edf1f5"),
            QColor("#f7f4ed"),
            QColor("#243b63"),
            QColor("#243b63"),
            QColor("#e2e7ee"),
            QColor("#243b63"),
            QColor("#f4f0e6"),
            QColor("#2e5fa8"),
            QColor("#f4f0e6"),
            QColor("#2e5fa8"),
            QColor("#d3b25b"),
            QColor("#8f98a6"),
            QColor("#8f98a6"),
            QColor("#e8ecf1"),
            QColor("#b9bec8"),
            QColor("#a8b2c0"),
            QColor("#eef5fd"),
            QColor("#d7e7f8"),
            QColor("#ece8df"),
            QColor("#f4f0e6"),
            QColor("#d9dee7"),
            QColor("#2e5fa8"),
            QColor("#6e7b93"),
            QColor("#8d98aa"),
            QColor("#dbe8f8"),
            QColor("#2e5fa8"),
        };
    }

    return themeSpec(ThemeMode::LightDefault);
}

QColor tooltipBaseColor(const ThemeSpec& spec) {
    const bool darkUi = spec.window.lightnessF() < 0.5;
    QColor lifted = blendColors(spec.base, QColor("#ffffff"), darkUi ? 0.62 : 0.18);
    QColor tinted = blendColors(lifted, spec.highlight, darkUi ? 0.12 : 0.08);
    tinted.setAlpha(darkUi ? 236 : 228);
    return tinted;
}

QColor tooltipTextColor(const ThemeSpec& spec) {
    return tooltipBaseColor(spec).lightnessF() > 0.58
        ? QColor("#25303a")
        : QColor("#f4f6f8");
}

QColor panelBorderColor(const ThemeSpec& spec) {
    const bool darkUi = spec.window.lightnessF() < 0.5;
    const QColor blended = blendColors(spec.base, spec.highlight, darkUi ? 0.38 : 0.20);
    const QColor hslSeed = blended.toHsl();

    qreal h = hslSeed.hslHueF();
    qreal s = hslSeed.hslSaturationF();
    qreal l = hslSeed.lightnessF();

    if (h < 0.0) {
        h = spec.highlight.hslHueF();
    }
    if (h < 0.0) {
        h = 0.58;
    }

    s = darkUi ? std::clamp(s * 0.38, 0.08, 0.16)
               : std::clamp(s * 0.32, 0.06, 0.14);
    l = darkUi ? std::clamp(l + 0.12, 0.40, 0.56)
               : std::clamp(l - 0.02, 0.58, 0.74);

    return QColor::fromHslF(h, s, l, 1.0);
}

QColor activePanelBorderColor(const ThemeSpec& spec) {
    if (spec.mode == ThemeMode::Frieren) {
        return QColor("#8e2a2b");
    }
    if (spec.mode == ThemeMode::Himmel) {
        return QColor("#a8b9cf");
    }
    return spec.focusBorder;
}

QString buildStyleSheet(const ThemeSpec& spec, FontSizePreset fontSizePreset) {
    const int mainFontPx = interfaceFontPixelSize(fontSizePreset);
    const int emptyStateFontPx = std::max(mainFontPx - 1, 11);
    const int consoleFontPx = consoleFontPixelSize(fontSizePreset);
    const int consoleToggleFontPx = std::max(mainFontPx - 1, 12);

    QString styleSheet = QString(R"(
        QWidget {
            color: %1;
            background-color: %2;
            font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
            font-size: %25px;
        }
        QMainWindow, QMenuBar, QMenu, QToolBar, QStatusBar {
            background-color: %3;
        }
        QStatusBar {
            border-top: none;
            padding: 0px;
            margin: 0px;
        }
        QStatusBar::item {
            border: none;
            padding: 0px;
            margin: 0px;
        }
        QStatusBar QLabel {
            background: transparent;
            padding: 0px;
            margin: 0px;
        }
        QMenuBar::item:disabled, QMenu::item:disabled {
            color: %17;
        }
        QMenu {
            border: 1px solid %4;
        }
        QMenu::item:selected {
            background-color: %5;
            color: %6;
        }
        QTreeWidget, QListView, QTableView, QTextEdit, QPlainTextEdit,
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTreeView {
            background-color: %7;
            selection-background-color: %9;
            selection-color: %10;
        }
        QTableView {
            gridline-color: %24;
        }
        QTreeWidget, QListView, QTableView, QTextEdit, QPlainTextEdit, QTreeView {
            border: 1px solid %24;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            border: 1px solid %8;
        }
        QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
        QTreeWidget:focus, QTreeView:focus, QTableView:focus {
            border: 1px solid %11;
        }
        QHeaderView::section {
            background-color: %12;
            color: %1;
            padding: 4px 6px;
            border: 1px solid %24;
        }
        QPushButton, QToolButton {
            background-color: %13;
            border: 1px solid %14;
            padding: 5px 9px;
            min-height: 20px;
        }
        QPushButton:hover, QToolButton:hover {
            background-color: %15;
        }
        QPushButton:pressed, QToolButton:pressed {
            background-color: %16;
        }
        QPushButton:disabled, QToolButton:disabled,
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
        QPlainTextEdit:disabled, QTextEdit:disabled, QTreeWidget:disabled, QTreeView:disabled,
        QListView:disabled, QTableView:disabled {
            color: %17;
        }
        QTabWidget::pane, QScrollArea {
            border: 1px solid %24;
        }
        QFrame#emptyPropertiesFrame {
            background: transparent;
            background-color: rgba(0, 0, 0, 0);
            border: none;
        }
        QGroupBox {
            border: 1px solid %24;
            margin-top: 14px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            background-color: %2;
            color: %18;
        }
        QTabBar::tab {
            background-color: %19;
            border: 1px solid %24;
            padding: 6px 10px;
            margin-right: 2px;
            font-weight: 400;
        }
        QTabBar::tab:selected {
            background-color: %20;
            font-weight: 700;
        }
        QSplitter {
            background-color: %7;
        }
        QSplitter::handle {
            background-color: %7;
        }
        QSplitter::handle:hover {
            background-color: %7;
        }
        QProgressBar {
            border: 1px solid %8;
            background-color: %7;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: %22;
        }
        QLabel[muted="true"] {
            color: %18;
        }
        QLabel[emptyState="true"] {
            color: %23;
            font-size: %26px;
        }
        QAbstractItemView::item:selected {
            background-color: %9;
            color: %10;
        }
    )")
        .arg(colorName(spec.windowText),
             colorName(spec.window),
             colorName(spec.chrome),
             colorName(spec.inputBorder),
             colorName(spec.menuHover),
             colorName(spec.windowText),
             colorName(spec.base),
             colorName(spec.inputBorder),
             colorName(spec.highlight),
             colorName(spec.highlightedText),
             colorName(spec.focusBorder),
             colorName(spec.header),
             colorName(spec.button),
             colorName(spec.buttonBorder),
             colorName(spec.buttonHover),
             colorName(spec.buttonPressed),
             colorName(spec.disabledText),
             colorName(spec.muted),
             colorName(spec.tab),
             colorName(spec.tabSelected),
             colorName(spec.splitter),
             colorName(spec.progressChunk),
             colorName(spec.emptyState),
             colorName(panelBorderColor(spec)))
        .arg(QString::number(mainFontPx),
             QString::number(emptyStateFontPx));

    styleSheet += QString(R"(
        QToolTip {
            color: %1;
            background-color: %2;
            border: 1px solid %3;
            padding: 5px 7px;
            border-radius: 4px;
        }
    )")
        .arg(colorName(tooltipTextColor(spec)),
             colorName(tooltipBaseColor(spec)),
             colorName(panelBorderColor(spec)));

    styleSheet += QString(R"(
        QWidget#workflowGroup,
        QWidget#workflowSeparator {
            background: transparent;
            background-color: rgba(0, 0, 0, 0);
            border: none;
        }
        QWidget#workflowBar QToolButton#workflowAction,
        QWidget#workflowBar QToolButton#workflowAction:hover,
        QWidget#workflowBar QToolButton#workflowAction:pressed,
        QWidget#workflowBar QToolButton#workflowAction:checked {
            background: transparent;
            background-color: rgba(0, 0, 0, 0);
        }
        QWidget#consoleHeader {
            background: transparent;
        }
        QLabel#consoleTitle {
            background: transparent;
        }
        QToolButton#consoleCollapseButton {
            border: none;
            background: transparent;
            padding: 0px;
            min-height: 0px;
            min-width: 0px;
            font-size: %1px;
            font-weight: 500;
        }
        QToolButton#consoleCollapseButton:hover {
            background: transparent;
        }
        QPlainTextEdit#consoleOutput {
            font-family: "Cascadia Mono", "Consolas", "Courier New", monospace;
            font-size: %2px;
        }
    )")
        .arg(QString::number(consoleToggleFontPx),
             QString::number(consoleFontPx));

    if (spec.mode == ThemeMode::Frieren) {
        styleSheet += QStringLiteral(R"(
            QWidget#consolePanel {
                background-color: #e4cdcf;
                border: 1px solid #bf9799;
            }
            QTreeWidget#projectNavTree,
            QTreeView#projectFileTree {
                selection-background-color: #8e2a2b;
                selection-color: #f4f1e8;
            }
            QTreeWidget#projectNavTree::item:selected,
            QTreeView#projectFileTree::item:selected {
                background-color: #8e2a2b;
                color: #f4f1e8;
            }
            QPlainTextEdit#consoleOutput {
                background-color: #f1e3e4;
                color: #5d2628;
                selection-background-color: #d6aeb0;
                selection-color: #341415;
                border: none;
            }
            QWidget#consoleHeader {
                background-color: #d9babd;
                border: none;
                border-bottom: 1px solid #bf9799;
            }
            QLabel#consoleTitle {
                color: #6b2c2e;
            }
            QToolButton#consoleCollapseButton {
                color: #6b2c2e;
                border-radius: 4px;
            }
            QToolButton#consoleCollapseButton:hover {
                background-color: #cdabad;
                color: #4f2021;
            }
            QPushButton[workflowDanger="true"] {
                border: 1px solid #8e2a2b;
                color: #6a2425;
                background-color: #f1e7df;
            }
            QPushButton[workflowDanger="true"]:hover {
                background-color: #8e2a2b;
                color: #f7f0e8;
            }
            QPushButton[workflowDanger="true"]:pressed {
                background-color: #742223;
                color: #f7f0e8;
            }
        )");
    }

    if (spec.mode == ThemeMode::Himmel) {
        styleSheet += QStringLiteral(R"(
            QWidget#consolePanel {
                background-color: #a9c8ec;
                border: 1px solid #2e5fa8;
            }
            QTreeWidget#projectNavTree,
            QTreeView#projectFileTree {
                selection-background-color: #d3b25b;
                selection-color: #243b63;
            }
            QTreeWidget#projectNavTree::item:selected,
            QTreeView#projectFileTree::item:selected {
                background-color: #d3b25b;
                color: #243b63;
            }
            QPlainTextEdit#consoleOutput {
                background-color: #a9c8ec;
                color: #243b63;
                selection-background-color: #2e5fa8;
                selection-color: #f4f0e6;
                border: none;
            }
            QWidget#consoleHeader {
                background-color: #d7e7f8;
                border: none;
                border-bottom: 1px solid #2e5fa8;
            }
            QLabel#consoleTitle {
                color: #243b63;
            }
            QToolButton#consoleCollapseButton {
                color: #243b63;
                border-radius: 4px;
            }
            QToolButton#consoleCollapseButton:hover {
                background-color: #2e5fa8;
                color: #f4f0e6;
            }
            QPushButton[workflowDanger="true"] {
                border: 1px solid #9c7d30;
                color: #243b63;
                background-color: #f2ead7;
            }
            QPushButton[workflowDanger="true"]:hover {
                background-color: #d3b25b;
                color: #243b63;
            }
            QPushButton[workflowDanger="true"]:pressed {
                background-color: #c09d44;
                color: #243b63;
            }
        )");
    }

    styleSheet += QString(R"(
        QWidget[panelShell="true"] {
            background: transparent;
            background-color: rgba(0, 0, 0, 0);
            border: 1px solid rgba(0, 0, 0, 0);
            border-radius: 5px;
        }
        QWidget[panelShell="true"][panelActive="true"] {
            border-color: %1;
        }
    )")
        .arg(colorName(activePanelBorderColor(spec)));

    return styleSheet;
}

} // namespace

QVector<ThemeMode> availableThemeModes() {
    return {
        ThemeMode::LightDefault,
        ThemeMode::DarkDefault,
        ThemeMode::White,
        ThemeMode::Black,
        ThemeMode::Pink,
        ThemeMode::Green,
        ThemeMode::Siemens,
        ThemeMode::Frieren,
    };
}

QString themeModeDisplayName(ThemeMode mode) {
    return themeSpec(mode).displayName;
}

QString themeModeStorageKey(ThemeMode mode) {
    return themeSpec(mode).storageKey;
}

ThemeMode themeModeFromString(const QString& text) {
    const QString normalized = text.trimmed();
    if (normalized.compare("light", Qt::CaseInsensitive) == 0
        || normalized.compare("light (default)", Qt::CaseInsensitive) == 0
        || normalized.compare("light_default", Qt::CaseInsensitive) == 0) {
        return ThemeMode::LightDefault;
    }
    if (normalized.compare("dark", Qt::CaseInsensitive) == 0
        || normalized.compare("dark (default)", Qt::CaseInsensitive) == 0
        || normalized.compare("dark_default", Qt::CaseInsensitive) == 0) {
        return ThemeMode::DarkDefault;
    }

    if (normalized.compare("frieren_himmel", Qt::CaseInsensitive) == 0
        || normalized.compare("frieren & himmel", Qt::CaseInsensitive) == 0) {
        return ThemeMode::Frieren;
    }

    for (ThemeMode mode : allThemeModes()) {
        const ThemeSpec spec = themeSpec(mode);
        if (normalized.compare(spec.displayName, Qt::CaseInsensitive) == 0
            || normalized.compare(spec.storageKey, Qt::CaseInsensitive) == 0) {
            return mode;
        }
    }
    return ThemeMode::LightDefault;
}

QString fontSizePresetDisplayName(FontSizePreset preset) {
    switch (preset) {
    case FontSizePreset::PerfectForGenZ:
        return QStringLiteral("Perfect-For-GenZ");
    case FontSizePreset::Small:
        return QStringLiteral("Small");
    case FontSizePreset::Normal:
        return QStringLiteral("Normal");
    case FontSizePreset::Large:
        return QStringLiteral("Large");
    }
    return QStringLiteral("Normal");
}

QString fontSizePresetStorageKey(FontSizePreset preset) {
    switch (preset) {
    case FontSizePreset::PerfectForGenZ:
        return QStringLiteral("perfect_for_genz");
    case FontSizePreset::Small:
        return QStringLiteral("small_mid");
    case FontSizePreset::Normal:
        return QStringLiteral("normal");
    case FontSizePreset::Large:
        return QStringLiteral("large");
    }
    return QStringLiteral("normal");
}

FontSizePreset fontSizePresetFromString(const QString& text) {
    const QString normalized = text.trimmed();
    if (normalized.compare(QStringLiteral("small"), Qt::CaseInsensitive) == 0
        || normalized.compare(QStringLiteral("perfect_for_genz"), Qt::CaseInsensitive) == 0
        || normalized.compare(QStringLiteral("perfect-for-genz"), Qt::CaseInsensitive) == 0
        || normalized.compare(QStringLiteral("perfect for genz"), Qt::CaseInsensitive) == 0) {
        return FontSizePreset::PerfectForGenZ;
    }
    if (normalized.compare(QStringLiteral("small_mid"), Qt::CaseInsensitive) == 0) {
        return FontSizePreset::Small;
    }
    if (normalized.compare(QStringLiteral("large"), Qt::CaseInsensitive) == 0) {
        return FontSizePreset::Large;
    }
    return FontSizePreset::Normal;
}

int interfaceFontPixelSize(FontSizePreset preset) {
    switch (preset) {
    case FontSizePreset::PerfectForGenZ:
        return 12;
    case FontSizePreset::Small:
        return 14;
    case FontSizePreset::Normal:
        return 15;
    case FontSizePreset::Large:
        return 17;
    }
    return 15;
}

int consoleFontPixelSize(FontSizePreset preset) {
    return std::max(interfaceFontPixelSize(preset) - 2, 10);
}

void applyTheme(QApplication& app, ThemeMode mode, FontSizePreset fontSizePreset) {
    app.setStyle("Fusion");

    const ThemeSpec spec = themeSpec(mode);
    QPalette palette;
    palette.setColor(QPalette::Window, spec.window);
    palette.setColor(QPalette::WindowText, spec.windowText);
    palette.setColor(QPalette::Base, spec.base);
    palette.setColor(QPalette::AlternateBase, spec.alternateBase);
    palette.setColor(QPalette::ToolTipBase, tooltipBaseColor(spec));
    palette.setColor(QPalette::ToolTipText, tooltipTextColor(spec));
    palette.setColor(QPalette::Text, spec.text);
    palette.setColor(QPalette::Button, spec.button);
    palette.setColor(QPalette::ButtonText, spec.buttonText);
    palette.setColor(QPalette::BrightText, spec.brightText);
    palette.setColor(QPalette::Highlight, spec.highlight);
    palette.setColor(QPalette::HighlightedText, spec.highlightedText);
    palette.setColor(QPalette::Link, spec.link);
    palette.setColor(QPalette::LinkVisited, spec.visitedLink);
    const QColor separator = panelBorderColor(spec);
    const QColor separatorSoft = blendColors(spec.base, separator, spec.window.lightnessF() < 0.5 ? 0.58 : 0.36);
    palette.setColor(QPalette::Mid, separator);
    palette.setColor(QPalette::Midlight, separatorSoft);
    palette.setColor(QPalette::Disabled, QPalette::Text, spec.disabledText);
    palette.setColor(QPalette::Disabled, QPalette::ButtonText, spec.disabledButtonText);

    QFont appFont("Segoe UI");
    appFont.setPixelSize(interfaceFontPixelSize(fontSizePreset));
    app.setFont(appFont);
    app.setPalette(palette);
    app.setStyleSheet(buildStyleSheet(spec, fontSizePreset));
}

} // namespace luwgui
