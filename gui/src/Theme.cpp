#include "luwgui/Theme.h"

#include <QColor>
#include <QPalette>

namespace luwgui {

QString themeModeDisplayName(ThemeMode mode) {
    return (mode == ThemeMode::Light) ? "Light" : "Dark";
}

ThemeMode themeModeFromString(const QString& text) {
    return text.trimmed().compare("light", Qt::CaseInsensitive) == 0 ? ThemeMode::Light : ThemeMode::Dark;
}

void applyTheme(QApplication& app, ThemeMode mode) {
    app.setStyle("Fusion");

    QPalette palette;
    QString styleSheet;

    if (mode == ThemeMode::Light) {
        palette.setColor(QPalette::Window, QColor(224, 229, 235));
        palette.setColor(QPalette::WindowText, QColor(31, 40, 49));
        palette.setColor(QPalette::Base, QColor(246, 248, 250));
        palette.setColor(QPalette::AlternateBase, QColor(233, 238, 242));
        palette.setColor(QPalette::ToolTipBase, QColor(248, 250, 252));
        palette.setColor(QPalette::ToolTipText, QColor(31, 40, 49));
        palette.setColor(QPalette::Text, QColor(31, 40, 49));
        palette.setColor(QPalette::Button, QColor(218, 225, 231));
        palette.setColor(QPalette::ButtonText, QColor(31, 40, 49));
        palette.setColor(QPalette::BrightText, QColor(255, 255, 255));
        palette.setColor(QPalette::Highlight, QColor(77, 116, 156));
        palette.setColor(QPalette::HighlightedText, QColor(250, 251, 252));
        palette.setColor(QPalette::Link, QColor(58, 97, 138));
        palette.setColor(QPalette::Disabled, QPalette::Text, QColor(128, 136, 144));
        palette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(128, 136, 144));

        styleSheet = R"(
            QWidget {
                color: #1f2831;
                background-color: #e0e5eb;
                font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
                font-size: 12px;
            }
            QMainWindow, QMenuBar, QMenu, QToolBar, QStatusBar {
                background-color: #d8dee4;
            }
            QTreeWidget, QListView, QTableView, QTextEdit, QPlainTextEdit,
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTreeView {
                background-color: #f6f8fa;
                border: 1px solid #aab5c0;
                selection-background-color: #5e7f9d;
                selection-color: #f7f9fb;
            }
            QHeaderView::section {
                background-color: #cfd8e1;
                color: #1f2831;
                padding: 4px 6px;
                border: 1px solid #aab5c0;
            }
            QPushButton, QToolButton {
                background-color: #d4dde6;
                border: 1px solid #93a1af;
                padding: 5px 9px;
                min-height: 20px;
            }
            QPushButton:hover, QToolButton:hover {
                background-color: #c7d3de;
            }
            QPushButton:pressed, QToolButton:pressed {
                background-color: #b7c4d0;
            }
            QTabWidget::pane, QScrollArea {
                border: 1px solid #aab5c0;
            }
            QGroupBox {
                border: 1px solid #aab5c0;
                margin-top: 14px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                background-color: #e0e5eb;
                color: #475766;
            }
            QTabBar::tab {
                background-color: #d6dde5;
                border: 1px solid #aab5c0;
                padding: 6px 10px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #c6d2dc;
            }
            QSplitter::handle {
                background-color: #bcc8d2;
            }
            QProgressBar {
                border: 1px solid #aab5c0;
                background-color: #f6f8fa;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #5b85af;
            }
            QLabel[muted="true"] {
                color: #62707d;
            }
        )";
    } else {
        palette.setColor(QPalette::Window, QColor(19, 27, 36));
        palette.setColor(QPalette::WindowText, QColor(216, 224, 232));
        palette.setColor(QPalette::Base, QColor(12, 18, 25));
        palette.setColor(QPalette::AlternateBase, QColor(25, 34, 45));
        palette.setColor(QPalette::ToolTipBase, QColor(31, 45, 58));
        palette.setColor(QPalette::ToolTipText, QColor(220, 228, 235));
        palette.setColor(QPalette::Text, QColor(213, 221, 229));
        palette.setColor(QPalette::Button, QColor(26, 36, 48));
        palette.setColor(QPalette::ButtonText, QColor(216, 224, 232));
        palette.setColor(QPalette::BrightText, QColor(245, 245, 245));
        palette.setColor(QPalette::Highlight, QColor(73, 116, 161));
        palette.setColor(QPalette::HighlightedText, QColor(246, 248, 250));
        palette.setColor(QPalette::Link, QColor(128, 170, 214));
        palette.setColor(QPalette::Disabled, QPalette::Text, QColor(114, 124, 136));
        palette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(114, 124, 136));

        styleSheet = R"(
            QWidget {
                color: #d8e0e8;
                background-color: #131b24;
                font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
                font-size: 12px;
            }
            QMainWindow, QMenuBar, QMenu, QToolBar, QStatusBar {
                background-color: #17212b;
            }
            QTreeWidget, QListView, QTableView, QTextEdit, QPlainTextEdit,
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTreeView {
                background-color: #0f151d;
                border: 1px solid #304050;
                selection-background-color: #496f95;
                selection-color: #f5f7fa;
            }
            QHeaderView::section {
                background-color: #223142;
                color: #d8e0e8;
                padding: 4px 6px;
                border: 1px solid #304050;
            }
            QPushButton, QToolButton {
                background-color: #243240;
                border: 1px solid #405162;
                padding: 5px 9px;
                min-height: 20px;
            }
            QPushButton:hover, QToolButton:hover {
                background-color: #2f4357;
            }
            QPushButton:pressed, QToolButton:pressed {
                background-color: #1e2b38;
            }
            QTabWidget::pane, QScrollArea {
                border: 1px solid #304050;
            }
            QGroupBox {
                border: 1px solid #304050;
                margin-top: 14px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                background-color: #131b24;
                color: #8f9ca7;
            }
            QTabBar::tab {
                background-color: #1a2530;
                border: 1px solid #304050;
                padding: 6px 10px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2c4155;
            }
            QSplitter::handle {
                background-color: #223042;
            }
            QProgressBar {
                border: 1px solid #304050;
                background-color: #0f151d;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #5b85af;
            }
            QLabel[muted="true"] {
                color: #8f9ca7;
            }
        )";
    }

    app.setPalette(palette);
    app.setStyleSheet(styleSheet);
}

}
