#include "luwgui/ConsolePanel.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QMenu>
#include <QScrollBar>
#include <QTextCursor>
#include <QVBoxLayout>

namespace luwgui {

ConsolePanel::ConsolePanel(QWidget* parent)
    : QWidget(parent) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    header_ = new QWidget(this);
    header_->setFixedHeight(20);
    auto* headerLayout = new QHBoxLayout(header_);
    headerLayout->setContentsMargins(8, 1, 8, 1);
    headerLayout->setSpacing(4);

    auto* title = new QLabel("Console", header_);
    title->setProperty("muted", true);
    headerLayout->addWidget(title);
    headerLayout->addStretch(1);

    collapseButton_ = new QToolButton(header_);
    collapseButton_->setText("-");
    collapseButton_->setToolTip("Hide console");
    collapseButton_->setFixedSize(14, 14);
    collapseButton_->setAutoRaise(true);
    collapseButton_->setStyleSheet(
        "QToolButton{border:none;background:transparent;padding:0px;font-size:14px;font-weight:500;}"
        "QToolButton:hover{background:transparent;}");
    headerLayout->addWidget(collapseButton_);
    root->addWidget(header_);

    editor_ = new QPlainTextEdit(this);
    editor_->setReadOnly(true);
    editor_->setMaximumBlockCount(12000);
    editor_->setContextMenuPolicy(Qt::CustomContextMenu);
    root->addWidget(editor_, 1);
    connect(editor_->verticalScrollBar(), &QScrollBar::valueChanged, this, [this](int) {
        updateFollowMode();
    });

    connect(editor_, &QPlainTextEdit::customContextMenuRequested, this, [this](const QPoint& pos) {
        QMenu* menu = editor_->createStandardContextMenu();
        menu->addSeparator();
        menu->addAction("Clear Console", this, &ConsolePanel::clear);
        menu->exec(editor_->mapToGlobal(pos));
        delete menu;
    });
    connect(collapseButton_, &QToolButton::clicked, this, [this] {
        collapsed_ = !collapsed_;
        editor_->setVisible(!collapsed_);
        collapseButton_->setText(collapsed_ ? "+" : "-");
        collapseButton_->setToolTip(collapsed_ ? "Show console" : "Hide console");
        setMinimumHeight(collapsed_ ? collapsedHeight() : 0);
        setMaximumHeight(collapsed_ ? collapsedHeight() : QWIDGETSIZE_MAX);
        emit collapseToggled(collapsed_);
    });
}

void ConsolePanel::appendText(const QString& text) {
    QScrollBar* scrollBar = editor_->verticalScrollBar();
    const bool followOutput = autoFollow_ || (scrollBar->maximum() - scrollBar->value() <= 20);
    QTextCursor cursor(editor_->document());
    cursor.movePosition(QTextCursor::End);
    cursor.insertText(text);
    if (followOutput) {
        editor_->moveCursor(QTextCursor::End);
        scrollBar->setValue(scrollBar->maximum());
        autoFollow_ = true;
    }
}

void ConsolePanel::clear() {
    editor_->clear();
    autoFollow_ = true;
}

int ConsolePanel::collapsedHeight() const {
    return header_ ? header_->height() + 2 : 22;
}

void ConsolePanel::updateFollowMode() {
    if (!editor_) {
        return;
    }
    QScrollBar* scrollBar = editor_->verticalScrollBar();
    autoFollow_ = (scrollBar->maximum() - scrollBar->value()) <= 20;
}

}
