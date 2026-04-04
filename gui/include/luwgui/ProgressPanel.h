#pragma once

#include <QWidget>

class QLabel;
class QProgressBar;
class QGraphicsOpacityEffect;
class QPropertyAnimation;
class QTimer;

namespace luwgui {

class ProgressPanel : public QWidget {
    Q_OBJECT

public:
    explicit ProgressPanel(QWidget* parent = nullptr);

    void setIdle(const QString& summary = QString(), const QString& detail = QString());
    void setBusy(const QString& summary, const QString& detail);
    void setProgress(const QString& summary,
                     const QString& detail,
                     qint64 current,
                     qint64 total,
                     bool indeterminate);
    void showTerminalStatus(const QString& summary, const QString& detail = QString());

private:
    void changeEvent(QEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void applyProgress(qint64 current, qint64 total, bool indeterminate);
    void refreshStatusLabel();
    void updatePresentation(bool active);
    void resetFadeState();
    void clearStatusAfterFade();

    QLabel* statusLabel_ = nullptr;
    QProgressBar* bar_ = nullptr;
    QGraphicsOpacityEffect* opacityEffect_ = nullptr;
    QTimer* fadeDelayTimer_ = nullptr;
    QPropertyAnimation* fadeAnimation_ = nullptr;
    QString fullStatusText_;
};

}
