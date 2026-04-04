#include "luwgui/StartupSplash.h"

#include <QApplication>
#include <QEventLoop>
#include <QGuiApplication>
#include <QImage>
#include <QLinearGradient>
#include <QPainter>
#include <QPaintEvent>
#include <QPen>
#include <QScreen>

namespace luwgui {

namespace {

constexpr qreal kSplashScale = 1.414;
constexpr int kBaseSplashWidth = 640;
constexpr int kBaseSplashHeight = 360;
constexpr int kSplashWidth = qRound(static_cast<qreal>(kBaseSplashWidth) * kSplashScale);
constexpr int kSplashHeight = qRound(static_cast<qreal>(kBaseSplashHeight) * kSplashScale);
constexpr int kStatusFontPixelSize = 17;

int scaledPixels(int basePixels) {
    return qRound(static_cast<qreal>(basePixels) * kSplashScale);
}

QRect largest16By9Crop(const QSize& size) {
    if (!size.isValid()) {
        return {};
    }

    const qreal targetAspect = 16.0 / 9.0;
    int cropWidth = size.width();
    int cropHeight = qRound(static_cast<qreal>(cropWidth) / targetAspect);
    if (cropHeight > size.height()) {
        cropHeight = size.height();
        cropWidth = qRound(static_cast<qreal>(cropHeight) * targetAspect);
    }

    const int x = (size.width() - cropWidth) / 2;
    const int y = (size.height() - cropHeight) / 2;
    return {x, y, cropWidth, cropHeight};
}

void drawOutlinedText(QPainter& painter, const QRect& rect, const QString& text) {
    const int offsets[][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1},
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
    };

    painter.setPen(QColor(255, 255, 255, 230));
    for (const auto& offset : offsets) {
        painter.drawText(rect.translated(offset[0], offset[1]),
                         Qt::AlignLeft | Qt::AlignBottom | Qt::TextWordWrap,
                         text);
    }

    painter.setPen(QColor(8, 8, 8));
    painter.drawText(rect,
                     Qt::AlignLeft | Qt::AlignBottom | Qt::TextWordWrap,
                     text);
}

} // namespace

StartupSplash::StartupSplash(const QString& imagePath, QWidget* parent)
    : QWidget(parent, Qt::SplashScreen | Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint)
    , background_(prepareBackground(imagePath))
    , statusMessage_(QStringLiteral("Preparing the startup view and runtime environment self-check...")) {
    setAttribute(Qt::WA_DeleteOnClose, false);
    setFixedSize(kSplashWidth, kSplashHeight);
}

void StartupSplash::setStatusMessage(const QString& message) {
    statusMessage_ = normalizeStatusMessage(message);
    update();
    QApplication::processEvents(QEventLoop::AllEvents, 25);
}

void StartupSplash::showCentered() {
    const QList<QScreen*> screens = QGuiApplication::screens();
    const QRect screenGeometry = screens.isEmpty()
        ? QRect(0, 0, width(), height())
        : screens.front()->availableGeometry();
    move(screenGeometry.center() - QPoint(width() / 2, height() / 2));
    show();
    raise();
    QApplication::processEvents(QEventLoop::AllEvents, 25);
}

void StartupSplash::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event)

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
    painter.drawPixmap(rect(), background_);

    const int gradientHeight = scaledPixels(120);
    const int textMarginX = scaledPixels(18);
    const int textBottomMargin = scaledPixels(18);
    const int textHeight = scaledPixels(116);

    QLinearGradient gradient(0.0,
                             static_cast<qreal>(height() - gradientHeight),
                             0.0,
                             static_cast<qreal>(height()));
    gradient.setColorAt(0.0, QColor(255, 255, 255, 0));
    gradient.setColorAt(1.0, QColor(255, 255, 255, 72));
    painter.fillRect(QRect(0, height() - gradientHeight, width(), gradientHeight), gradient);

    QFont textFont(QStringLiteral("Segoe UI"));
    textFont.setPixelSize(kStatusFontPixelSize);
    textFont.setWeight(QFont::Medium);
    painter.setFont(textFont);

    const QRect textRect(textMarginX,
                         height() - textBottomMargin - textHeight,
                         width() - (textMarginX * 2),
                         textHeight);
    drawOutlinedText(painter, textRect, statusMessage_);
}

QPixmap StartupSplash::buildFallbackPixmap() {
    QPixmap pixmap(kSplashWidth, kSplashHeight);
    pixmap.fill(QColor(230, 234, 239));

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    const QRect upperRect(0, 0, kSplashWidth, kSplashHeight * 2 / 3);
    painter.fillRect(upperRect, QColor(198, 210, 220));
    painter.fillRect(QRect(0, upperRect.bottom(), kSplashWidth, kSplashHeight - upperRect.height()), QColor(222, 227, 232));

    QPen pen(QColor(120, 132, 144));
    pen.setWidth(scaledPixels(2));
    painter.setPen(pen);
    painter.drawLine(0, kSplashHeight - scaledPixels(96), kSplashWidth, kSplashHeight - scaledPixels(140));
    painter.drawLine(0, kSplashHeight - scaledPixels(48), kSplashWidth, kSplashHeight - scaledPixels(112));
    return pixmap;
}

QPixmap StartupSplash::prepareBackground(const QString& imagePath) {
    const QImage source(imagePath);
    if (source.isNull()) {
        return buildFallbackPixmap();
    }

    const QRect cropRect = largest16By9Crop(source.size());
    if (!cropRect.isValid()) {
        return buildFallbackPixmap();
    }

    const QImage cropped = source.copy(cropRect);
    const QImage scaled = cropped.scaled(kSplashWidth,
                                         kSplashHeight,
                                         Qt::IgnoreAspectRatio,
                                         Qt::SmoothTransformation);
    return QPixmap::fromImage(scaled);
}

QString StartupSplash::normalizeStatusMessage(QString message) {
    message = message.trimmed();
    if (message.isEmpty()) {
        return QStringLiteral("Preparing the startup view and runtime environment self-check...");
    }
    if (!message.endsWith(QStringLiteral("..."))) {
        message += QStringLiteral("...");
    }
    return message;
}

}
