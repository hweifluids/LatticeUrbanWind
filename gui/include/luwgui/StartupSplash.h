#pragma once

#include <QPixmap>
#include <QString>
#include <QWidget>

namespace luwgui {

class StartupSplash : public QWidget {
    Q_OBJECT

public:
    explicit StartupSplash(const QString& imagePath, QWidget* parent = nullptr);

    void setStatusMessage(const QString& message);
    void showCentered();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    static QPixmap buildFallbackPixmap();
    static QPixmap prepareBackground(const QString& imagePath);
    static QString normalizeStatusMessage(QString message);

    QPixmap background_;
    QString statusMessage_;
};

}
