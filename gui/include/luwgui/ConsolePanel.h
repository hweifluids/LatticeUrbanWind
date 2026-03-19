#pragma once

#include <QPlainTextEdit>
#include <QToolButton>
#include <QWidget>

namespace luwgui {

class ConsolePanel : public QWidget {
    Q_OBJECT

public:
    explicit ConsolePanel(QWidget* parent = nullptr);

    void appendText(const QString& text);
    void clear();
    int collapsedHeight() const;

signals:
    void collapseToggled(bool collapsed);

private:
    void updateFollowMode();

    QWidget* header_ = nullptr;
    QPlainTextEdit* editor_ = nullptr;
    QToolButton* collapseButton_ = nullptr;
    bool collapsed_ = false;
    bool autoFollow_ = true;
};

}
