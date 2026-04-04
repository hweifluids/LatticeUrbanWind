#include "luwgui/ConsolePanel.h"

#include <QEvent>
#include <QHBoxLayout>
#include <QLabel>
#include <QMenu>
#include <QScrollBar>
#include <QTextCursor>
#include <QVBoxLayout>

namespace luwgui {

namespace {

void applyCollapsedStateTo(ConsolePanel* panel,
                           QWidget* editor,
                           QToolButton* button,
                           QWidget* header,
                           bool collapsed) {
    if (!panel || !editor || !button) {
        return;
    }
    editor->setVisible(!collapsed);
    button->setText(collapsed ? "+" : "-");
    button->setToolTip(collapsed ? "Show console" : "Hide console");
    panel->setMinimumHeight(collapsed ? (header ? header->height() + 2 : 22) : 0);
    panel->setMaximumHeight(collapsed ? (header ? header->height() + 2 : 22) : QWIDGETSIZE_MAX);
}

} // namespace

ConsolePanel::ConsolePanel(QWidget* parent)
    : QWidget(parent) {
    setObjectName("consolePanel");

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    header_ = new QWidget(this);
    header_->setObjectName("consoleHeader");
    auto* headerLayout = new QHBoxLayout(header_);
    headerLayout->setContentsMargins(8, 2, 8, 2);
    headerLayout->setSpacing(4);

    titleLabel_ = new QLabel("Console", header_);
    titleLabel_->setObjectName("consoleTitle");
    headerLayout->addWidget(titleLabel_);
    headerLayout->addStretch(1);

    collapseButton_ = new QToolButton(header_);
    collapseButton_->setObjectName("consoleCollapseButton");
    collapseButton_->setText("-");
    collapseButton_->setToolTip("Hide console");
    collapseButton_->setAutoRaise(true);
    headerLayout->addWidget(collapseButton_);
    root->addWidget(header_);

    editor_ = new QPlainTextEdit(this);
    editor_->setObjectName("consoleOutput");
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
        setCollapsed(!collapsed_);
    });

    refreshHeaderMetrics();
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

bool ConsolePanel::isCollapsed() const {
    return collapsed_;
}

void ConsolePanel::setCollapsed(bool collapsed) {
    if (collapsed_ == collapsed) {
        return;
    }
    collapsed_ = collapsed;
    applyCollapsedStateTo(this, editor_, collapseButton_, header_, collapsed_);
    emit collapseToggled(collapsed_);
}

void ConsolePanel::updateFollowMode() {
    if (!editor_) {
        return;
    }
    QScrollBar* scrollBar = editor_->verticalScrollBar();
    autoFollow_ = (scrollBar->maximum() - scrollBar->value()) <= 20;
}

void ConsolePanel::changeEvent(QEvent* event) {
    QWidget::changeEvent(event);
    if (!event) {
        return;
    }
    if (event->type() == QEvent::FontChange || event->type() == QEvent::StyleChange) {
        refreshHeaderMetrics();
        applyCollapsedStateTo(this, editor_, collapseButton_, header_, collapsed_);
    }
}

void ConsolePanel::refreshHeaderMetrics() {
    if (!header_ || !collapseButton_) {
        return;
    }

    const int textHeight = fontMetrics().height();
    const int headerHeight = std::max(textHeight + 8, 22);
    header_->setFixedHeight(headerHeight);
    collapseButton_->setFixedSize(std::max(headerHeight - 8, 12), std::max(headerHeight - 8, 12));
}

}
