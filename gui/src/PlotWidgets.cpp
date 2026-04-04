#include "luwgui/PlotWidgets.h"

#include <QFileInfo>
#include <QImage>
#include <QPaintEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPalette>

#include <algorithm>
#include <cmath>
#include <limits>

namespace luwgui {

namespace {

struct AxisTick {
    double value = 0.0;
    QString label;
};

QColor mix(const QColor& a, const QColor& b, double t) {
    const double u = std::clamp(t, 0.0, 1.0);
    return QColor::fromRgbF(
        a.redF() + (b.redF() - a.redF()) * u,
        a.greenF() + (b.greenF() - a.greenF()) * u,
        a.blueF() + (b.blueF() - a.blueF()) * u,
        a.alphaF() + (b.alphaF() - a.alphaF()) * u);
}

QColor gridColor(const QWidget* widget) {
    return mix(widget->palette().windowText().color(), widget->palette().window().color(), 0.7);
}

QColor accentColor(const QWidget* widget) {
    return widget->palette().highlight().color();
}

QColor secondaryColor(const QWidget* widget) {
    return mix(widget->palette().highlight().color(), widget->palette().windowText().color(), 0.45);
}

QRectF plotBoxFor(const QWidget* widget, int left, int top, int right, int bottom) {
    return widget->rect().adjusted(left, top, -right, -bottom);
}

QRectF defaultPlotBoxFor(const QWidget* widget) {
    return plotBoxFor(widget, 52, 18, 20, 34);
}

bool ensureImageSuffix(QString* filePath) {
    if (!filePath || filePath->isEmpty()) {
        return false;
    }
    if (QFileInfo(*filePath).suffix().isEmpty()) {
        *filePath += ".png";
    }
    return true;
}

QString formatLinearLabel(double value) {
    if (!std::isfinite(value)) {
        return {};
    }
    const double absValue = std::abs(value);
    if ((absValue >= 10000.0) || (absValue > 0.0 && absValue < 0.001)) {
        return QString::number(value, 'e', 1);
    }
    if (absValue >= 100.0) {
        return QString::number(value, 'f', 0);
    }
    if (absValue >= 10.0) {
        return QString::number(value, 'f', 1);
    }
    if (absValue >= 1.0) {
        return QString::number(value, 'f', 2);
    }
    return QString::number(value, 'f', 3);
}

QString formatLogLabel(double value) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        return {};
    }
    const int exponent = static_cast<int>(std::round(std::log10(value)));
    return QStringLiteral("1e%1").arg(exponent);
}

double niceStep(double rawStep) {
    if (!(rawStep > 0.0) || !std::isfinite(rawStep)) {
        return 1.0;
    }
    const double exponent = std::floor(std::log10(rawStep));
    const double fraction = rawStep / std::pow(10.0, exponent);
    double niceFraction = 1.0;
    if (fraction <= 1.0) {
        niceFraction = 1.0;
    } else if (fraction <= 2.0) {
        niceFraction = 2.0;
    } else if (fraction <= 5.0) {
        niceFraction = 5.0;
    } else {
        niceFraction = 10.0;
    }
    return niceFraction * std::pow(10.0, exponent);
}

QVector<AxisTick> makeLinearTicks(double minValue, double maxValue, int desiredCount = 5) {
    QVector<AxisTick> ticks;
    if (!std::isfinite(minValue) || !std::isfinite(maxValue) || maxValue <= minValue) {
        return ticks;
    }
    const double rawStep = (maxValue - minValue) / std::max(desiredCount - 1, 1);
    const double step = niceStep(rawStep);
    const double tickMin = std::ceil(minValue / step) * step;
    const double epsilon = step * 1.0e-6;
    for (double value = tickMin; value <= maxValue + epsilon; value += step) {
        if (value < minValue - epsilon) {
            continue;
        }
        ticks.push_back({value, formatLinearLabel(value)});
    }
    if (ticks.size() < 2) {
        ticks.push_back({minValue, formatLinearLabel(minValue)});
        ticks.push_back({maxValue, formatLinearLabel(maxValue)});
    }
    return ticks;
}

QVector<AxisTick> makeLogTicks(double minValue, double maxValue) {
    QVector<AxisTick> ticks;
    if (!(minValue > 0.0) || !(maxValue > 0.0) || !std::isfinite(minValue) || !std::isfinite(maxValue) || maxValue <= minValue) {
        return ticks;
    }
    const int expMin = static_cast<int>(std::floor(std::log10(minValue)));
    const int expMax = static_cast<int>(std::ceil(std::log10(maxValue)));
    for (int exponent = expMin; exponent <= expMax; ++exponent) {
        const double value = std::pow(10.0, exponent);
        if (value < minValue * 0.999 || value > maxValue * 1.001) {
            continue;
        }
        ticks.push_back({value, formatLogLabel(value)});
    }
    if (ticks.size() < 2) {
        const double ratio = std::pow(maxValue / minValue, 1.0 / 3.0);
        for (int i = 0; i < 4; ++i) {
            const double value = minValue * std::pow(ratio, i);
            ticks.push_back({value, formatLinearLabel(value)});
        }
    }
    return ticks;
}

QColor heatmapColor(double t) {
    const double u = std::clamp(t, 0.0, 1.0);
    const QVector<QPair<double, QColor>> stops = {
        {0.00, QColor(0, 3, 20)},
        {0.18, QColor(43, 11, 74)},
        {0.38, QColor(106, 24, 110)},
        {0.58, QColor(178, 54, 82)},
        {0.78, QColor(236, 112, 56)},
        {1.00, QColor(251, 221, 116)}
    };
    for (int i = 1; i < stops.size(); ++i) {
        if (u <= stops[i].first) {
            const double left = stops[i - 1].first;
            const double right = stops[i].first;
            const double local = (u - left) / std::max(1.0e-12, right - left);
            return mix(stops[i - 1].second, stops[i].second, local);
        }
    }
    return stops.back().second;
}

} // namespace

ExportablePlotWidget::ExportablePlotWidget(QWidget* parent)
    : QWidget(parent) {
    setMinimumHeight(180);
}

bool ExportablePlotWidget::saveImage(const QString& filePath, QString* errorMessage) {
    QString path = filePath;
    if (!ensureImageSuffix(&path)) {
        if (errorMessage) {
            *errorMessage = "Image path is empty.";
        }
        return false;
    }
    const QPixmap pixmap = grab();
    if (!pixmap.save(path)) {
        if (errorMessage) {
            *errorMessage = "Failed to save image.";
        }
        return false;
    }
    return true;
}

ProfilePlotWidget::ProfilePlotWidget(QWidget* parent)
    : ExportablePlotWidget(parent) {
}

void ProfilePlotWidget::setSamples(const QVector<QPointF>& samples) {
    samples_ = samples;
    update();
}

void ProfilePlotWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event)

    QPainter painter(this);
    painter.fillRect(rect(), palette().base());
    painter.setRenderHint(QPainter::Antialiasing, true);

    const QRectF box = defaultPlotBoxFor(this);
    const QColor grid = gridColor(this);
    painter.setPen(grid);
    painter.drawRect(box);

    for (int i = 1; i < 5; ++i) {
        const qreal x = box.left() + box.width() * (static_cast<qreal>(i) / 5.0);
        const qreal y = box.top() + box.height() * (static_cast<qreal>(i) / 5.0);
        painter.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()));
        painter.drawLine(QPointF(box.left(), y), QPointF(box.right(), y));
    }

    painter.setPen(mix(palette().windowText().color(), palette().window().color(), 0.35));
    painter.drawText(QRectF(8, 8, 120, 20), "U (m/s)");
    painter.save();
    painter.translate(10, height() / 2.0);
    painter.rotate(-90.0);
    painter.drawText(QRectF(-70, -16, 140, 20), Qt::AlignCenter, "z (m)");
    painter.restore();

    if (samples_.size() < 2) {
        painter.drawText(box, Qt::AlignCenter, "No profile.dat samples");
        return;
    }

    qreal minZ = samples_.front().x();
    qreal maxZ = samples_.front().x();
    qreal minU = samples_.front().y();
    qreal maxU = samples_.front().y();
    for (const QPointF& sample : samples_) {
        minZ = std::min(minZ, sample.x());
        maxZ = std::max(maxZ, sample.x());
        minU = std::min(minU, sample.y());
        maxU = std::max(maxU, sample.y());
    }
    if (qFuzzyCompare(minZ, maxZ)) {
        maxZ += 1.0;
    }
    if (qFuzzyCompare(minU, maxU)) {
        maxU += 1.0;
    }

    auto project = [&](const QPointF& sample) {
        const qreal zNorm = (sample.x() - minZ) / (maxZ - minZ);
        const qreal uNorm = (sample.y() - minU) / (maxU - minU);
        return QPointF(box.left() + uNorm * box.width(), box.bottom() - zNorm * box.height());
    };

    QPainterPath path;
    path.moveTo(project(samples_.front()));
    for (int i = 1; i < samples_.size(); ++i) {
        path.lineTo(project(samples_[i]));
    }

    painter.setPen(QPen(accentColor(this), 2.0));
    painter.drawPath(path);

    painter.setBrush(secondaryColor(this));
    painter.setPen(Qt::NoPen);
    for (const QPointF& sample : samples_) {
        painter.drawEllipse(project(sample), 3.0, 3.0);
    }
}

SpectrumPlotWidget::SpectrumPlotWidget(QWidget* parent)
    : ExportablePlotWidget(parent) {
    setMinimumHeight(220);
}

void SpectrumPlotWidget::setTitle(const QString& title) {
    title_ = title;
    update();
}

void SpectrumPlotWidget::setXAxisTitle(const QString& title) {
    xAxisTitle_ = title;
    update();
}

void SpectrumPlotWidget::setYAxisTitle(const QString& title) {
    yAxisTitle_ = title;
    update();
}

void SpectrumPlotWidget::setSpectrum(const QVector<QPointF>& samples, double kNyquist, double kTrust) {
    samples_ = samples;
    kNyquist_ = kNyquist;
    kTrust_ = kTrust;
    update();
}

void SpectrumPlotWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event)

    QPainter painter(this);
    painter.fillRect(rect(), palette().base());
    painter.setRenderHint(QPainter::Antialiasing, true);

    const QRectF box = plotBoxFor(this, 76, 18, 24, 50);
    painter.setPen(gridColor(this));
    painter.drawRect(box);

    painter.drawText(QRectF(10, height() - 28, width() - 20, 20), Qt::AlignHCenter | Qt::AlignVCenter, xAxisTitle_);
    painter.save();
    painter.translate(16, height() / 2.0);
    painter.rotate(-90.0);
    painter.drawText(QRectF(-110, -18, 220, 22), Qt::AlignCenter, yAxisTitle_);
    painter.restore();

    QVector<QPointF> usable;
    usable.reserve(samples_.size());
    double minX = std::numeric_limits<double>::max();
    double maxX = 0.0;
    double minY = std::numeric_limits<double>::max();
    double maxY = 0.0;
    for (const QPointF& sample : samples_) {
        if (sample.x() <= 0.0 || sample.y() <= 0.0 || !std::isfinite(sample.x()) || !std::isfinite(sample.y())) {
            continue;
        }
        usable.push_back(sample);
        minX = std::min(minX, sample.x());
        maxX = std::max(maxX, sample.x());
        minY = std::min(minY, sample.y());
        maxY = std::max(maxY, sample.y());
    }

    if (usable.size() < 2 || !(minX < maxX) || !(minY < maxY)) {
        painter.drawText(box, Qt::AlignCenter, "No valid spectrum loaded");
        return;
    }

    const double logMinX = std::log10(minX);
    const double logMaxX = std::log10(maxX);
    const double logMinY = std::log10(minY);
    const double logMaxY = std::log10(maxY);
    const auto xTicks = makeLogTicks(minX, maxX);
    const auto yTicks = makeLogTicks(minY, maxY);

    auto project = [&](const QPointF& sample) {
        const double xNorm = (std::log10(sample.x()) - logMinX) / std::max(1.0e-12, logMaxX - logMinX);
        const double yNorm = (std::log10(sample.y()) - logMinY) / std::max(1.0e-12, logMaxY - logMinY);
        return QPointF(box.left() + xNorm * box.width(), box.bottom() - yNorm * box.height());
    };
    auto projectX = [&](double value) {
        return box.left() + (std::log10(value) - logMinX) * box.width() / std::max(1.0e-12, logMaxX - logMinX);
    };
    auto projectY = [&](double value) {
        return box.bottom() - (std::log10(value) - logMinY) * box.height() / std::max(1.0e-12, logMaxY - logMinY);
    };

    painter.setPen(gridColor(this));
    for (const AxisTick& tick : xTicks) {
        const qreal x = projectX(tick.value);
        painter.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()));
        painter.drawLine(QPointF(x, box.bottom()), QPointF(x, box.bottom() + 6.0));
        painter.drawText(QRectF(x - 28.0, box.bottom() + 8.0, 56.0, 16.0), Qt::AlignHCenter | Qt::AlignTop, tick.label);
    }
    for (const AxisTick& tick : yTicks) {
        const qreal y = projectY(tick.value);
        painter.drawLine(QPointF(box.left(), y), QPointF(box.right(), y));
        painter.drawLine(QPointF(box.left() - 6.0, y), QPointF(box.left(), y));
        painter.drawText(QRectF(8.0, y - 8.0, box.left() - 14.0, 16.0), Qt::AlignRight | Qt::AlignVCenter, tick.label);
    }

    const auto drawVerticalMarker = [&](double xValue, const QColor& color, Qt::PenStyle style) {
        if (xValue <= 0.0 || xValue < minX || xValue > maxX) {
            return;
        }
        const qreal x = projectX(xValue);
        painter.setPen(QPen(color, 1.0, style));
        painter.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()));
    };

    drawVerticalMarker(kNyquist_, mix(accentColor(this), palette().windowText().color(), 0.35), Qt::DashLine);
    drawVerticalMarker(kTrust_, mix(secondaryColor(this), palette().windowText().color(), 0.25), Qt::DotLine);

    QPainterPath curve;
    curve.moveTo(project(usable.front()));
    for (int i = 1; i < usable.size(); ++i) {
        curve.lineTo(project(usable[i]));
    }
    painter.setPen(QPen(accentColor(this), 2.0));
    painter.drawPath(curve);
}

HeatmapPlotWidget::HeatmapPlotWidget(QWidget* parent)
    : ExportablePlotWidget(parent) {
    setMinimumHeight(260);
}

void HeatmapPlotWidget::setTitle(const QString& title) {
    title_ = title;
    update();
}

void HeatmapPlotWidget::setXAxisTitle(const QString& title) {
    xAxisTitle_ = title;
    update();
}

void HeatmapPlotWidget::setYAxisTitle(const QString& title) {
    yAxisTitle_ = title;
    update();
}

void HeatmapPlotWidget::setColorBarTitle(const QString& title) {
    colorBarTitle_ = title;
    update();
}

void HeatmapPlotWidget::setHeatmap(
    int columns,
    int rows,
    const QVector<double>& samples,
    double xMin,
    double xMax,
    double yMin,
    double yMax,
    double vMin,
    double vMax) {
    columns_ = columns;
    rows_ = rows;
    samples_ = samples;
    xMin_ = xMin;
    xMax_ = xMax;
    yMin_ = yMin;
    yMax_ = yMax;
    valueMin_ = vMin;
    valueMax_ = vMax;
    rebuildImage();
    update();
}

void HeatmapPlotWidget::clear() {
    columns_ = 0;
    rows_ = 0;
    samples_.clear();
    image_ = QImage();
    update();
}

void HeatmapPlotWidget::rebuildImage() {
    image_ = QImage();
    if (columns_ <= 0 || rows_ <= 0 || samples_.size() != columns_ * rows_) {
        return;
    }
    image_ = QImage(columns_, rows_, QImage::Format_ARGB32_Premultiplied);
    const double span = std::max(1.0e-12, valueMax_ - valueMin_);
    for (int row = 0; row < rows_; ++row) {
        for (int column = 0; column < columns_; ++column) {
            const int srcIndex = row * columns_ + column;
            const double value = samples_.value(srcIndex);
            QColor color = palette().base().color();
            if (std::isfinite(value)) {
                color = heatmapColor((value - valueMin_) / span);
            }
            image_.setPixelColor(column, rows_ - 1 - row, color);
        }
    }
}

void HeatmapPlotWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event)

    QPainter painter(this);
    painter.fillRect(rect(), palette().base());
    painter.setRenderHint(QPainter::Antialiasing, true);

    const QRectF box = plotBoxFor(this, 84, 18, 82, 52);
    const QRectF colorBarRect(box.right() + 18.0, box.top(), 18.0, box.height());
    painter.setPen(gridColor(this));
    painter.drawRect(box);
    painter.drawRect(colorBarRect);

    painter.drawText(QRectF(10, height() - 28, width() - 20, 20), Qt::AlignHCenter | Qt::AlignVCenter, xAxisTitle_);
    painter.save();
    painter.translate(18, height() / 2.0);
    painter.rotate(-90.0);
    painter.drawText(QRectF(-110, -18, 220, 22), Qt::AlignCenter, yAxisTitle_);
    painter.restore();

    if (image_.isNull() || !(xMax_ > xMin_) || !(yMax_ > yMin_)) {
        painter.drawText(box, Qt::AlignCenter, "No LES spectrum loaded");
        return;
    }

    painter.drawImage(box, image_);

    const auto xTicks = makeLinearTicks(xMin_, xMax_);
    const auto yTicks = makeLinearTicks(yMin_, yMax_);
    auto projectX = [&](double value) {
        return box.left() + (value - xMin_) * box.width() / std::max(1.0e-12, xMax_ - xMin_);
    };
    auto projectY = [&](double value) {
        return box.bottom() - (value - yMin_) * box.height() / std::max(1.0e-12, yMax_ - yMin_);
    };

    painter.setPen(gridColor(this));
    for (const AxisTick& tick : xTicks) {
        const qreal x = projectX(tick.value);
        painter.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()));
        painter.drawLine(QPointF(x, box.bottom()), QPointF(x, box.bottom() + 6.0));
        painter.drawText(QRectF(x - 30.0, box.bottom() + 8.0, 60.0, 16.0), Qt::AlignHCenter | Qt::AlignTop, tick.label);
    }
    for (const AxisTick& tick : yTicks) {
        const qreal y = projectY(tick.value);
        painter.drawLine(QPointF(box.left(), y), QPointF(box.right(), y));
        painter.drawLine(QPointF(box.left() - 6.0, y), QPointF(box.left(), y));
        painter.drawText(QRectF(8.0, y - 8.0, box.left() - 14.0, 16.0), Qt::AlignRight | Qt::AlignVCenter, tick.label);
    }

    QLinearGradient gradient(colorBarRect.bottomLeft(), colorBarRect.topLeft());
    gradient.setColorAt(0.0, heatmapColor(0.0));
    gradient.setColorAt(0.2, heatmapColor(0.2));
    gradient.setColorAt(0.4, heatmapColor(0.4));
    gradient.setColorAt(0.6, heatmapColor(0.6));
    gradient.setColorAt(0.8, heatmapColor(0.8));
    gradient.setColorAt(1.0, heatmapColor(1.0));
    painter.fillRect(colorBarRect, gradient);
    painter.setPen(gridColor(this));
    painter.drawRect(colorBarRect);
    painter.drawText(QRectF(colorBarRect.left() - 18.0, colorBarRect.top() - 18.0, 120.0, 16.0), Qt::AlignLeft | Qt::AlignVCenter, colorBarTitle_);
    painter.drawText(QRectF(colorBarRect.right() + 8.0, colorBarRect.top() - 8.0, 72.0, 16.0), Qt::AlignLeft | Qt::AlignVCenter, formatLinearLabel(valueMax_));
    painter.drawText(QRectF(colorBarRect.right() + 8.0, colorBarRect.bottom() - 8.0, 72.0, 16.0), Qt::AlignLeft | Qt::AlignVCenter, formatLinearLabel(valueMin_));
}

DistributionPlotWidget::DistributionPlotWidget(QWidget* parent)
    : ExportablePlotWidget(parent) {
    setMinimumHeight(220);
}

void DistributionPlotWidget::setCurves(
    const QVector<QPointF>& pdfSamples,
    const QVector<QPointF>& cdfSamples,
    const QVector<double>& guideLines) {
    pdfSamples_ = pdfSamples;
    cdfSamples_ = cdfSamples;
    guideLines_ = guideLines;
    update();
}

void DistributionPlotWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event)

    QPainter painter(this);
    painter.fillRect(rect(), palette().base());
    painter.setRenderHint(QPainter::Antialiasing, true);

    const QRectF box = defaultPlotBoxFor(this);
    const QColor grid = gridColor(this);
    painter.setPen(grid);
    painter.drawRect(box);

    painter.setPen(mix(palette().windowText().color(), palette().window().color(), 0.2));
    painter.drawText(QRectF(8, height() - 24, width() - 16, 20), Qt::AlignHCenter | Qt::AlignVCenter, "Short-side length (m)");
    painter.drawText(QRectF(8, box.top() - 2, 120, 18), Qt::AlignLeft | Qt::AlignVCenter, "PDF");
    painter.drawText(QRectF(width() - 120, box.top() - 2, 110, 18), Qt::AlignRight | Qt::AlignVCenter, "CDF");

    if (pdfSamples_.size() < 2 || cdfSamples_.size() < 2) {
        painter.drawText(box, Qt::AlignCenter, "No distribution available");
        return;
    }

    double maxX = 0.0;
    double maxPdf = 0.0;
    for (const QPointF& sample : pdfSamples_) {
        if (std::isfinite(sample.x()) && std::isfinite(sample.y())) {
            maxX = std::max(maxX, sample.x());
            maxPdf = std::max(maxPdf, sample.y());
        }
    }
    for (const QPointF& sample : cdfSamples_) {
        if (std::isfinite(sample.x())) {
            maxX = std::max(maxX, sample.x());
        }
    }
    if (maxX <= 0.0 || maxPdf <= 0.0) {
        painter.drawText(box, Qt::AlignCenter, "No distribution available");
        return;
    }

    auto projectPdf = [&](const QPointF& sample) {
        const double xNorm = 1.0 - (sample.x() / maxX);
        const double yNorm = sample.y() / maxPdf;
        return QPointF(box.left() + xNorm * box.width(), box.bottom() - yNorm * box.height());
    };
    auto projectCdf = [&](const QPointF& sample) {
        const double xNorm = 1.0 - (sample.x() / maxX);
        const double yNorm = std::clamp(sample.y(), 0.0, 1.0);
        return QPointF(box.left() + xNorm * box.width(), box.bottom() - yNorm * box.height());
    };

    for (int i = 1; i < 5; ++i) {
        const qreal x = box.left() + box.width() * (static_cast<qreal>(i) / 5.0);
        const qreal y = box.top() + box.height() * (static_cast<qreal>(i) / 5.0);
        painter.setPen(grid);
        painter.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()));
        painter.drawLine(QPointF(box.left(), y), QPointF(box.right(), y));
    }

    painter.setPen(QPen(mix(palette().windowText().color(), palette().window().color(), 0.35), 1.0, Qt::DashLine));
    for (double guide : guideLines_) {
        if (guide <= 0.0 || guide > maxX) {
            continue;
        }
        const qreal x = box.left() + (1.0 - guide / maxX) * box.width();
        painter.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()));
    }

    QPainterPath pdfPath;
    pdfPath.moveTo(projectPdf(pdfSamples_.front()));
    for (int i = 1; i < pdfSamples_.size(); ++i) {
        pdfPath.lineTo(projectPdf(pdfSamples_[i]));
    }
    painter.setPen(QPen(QColor(43, 120, 177), 2.0));
    painter.drawPath(pdfPath);

    QPainterPath cdfPath;
    cdfPath.moveTo(projectCdf(cdfSamples_.front()));
    for (int i = 1; i < cdfSamples_.size(); ++i) {
        cdfPath.lineTo(projectCdf(cdfSamples_[i]));
    }
    painter.setPen(QPen(QColor(249, 129, 44), 2.0));
    painter.drawPath(cdfPath);
}

} // namespace luwgui
