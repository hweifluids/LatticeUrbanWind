#pragma once

#include <QColor>
#include <QPointF>
#include <QString>
#include <QVector>
#include <QWidget>

namespace luwgui {

class ExportablePlotWidget : public QWidget {
    Q_OBJECT

public:
    explicit ExportablePlotWidget(QWidget* parent = nullptr);

    bool saveImage(const QString& filePath, QString* errorMessage = nullptr);
};

class ProfilePlotWidget final : public ExportablePlotWidget {
    Q_OBJECT

public:
    explicit ProfilePlotWidget(QWidget* parent = nullptr);

    void setSamples(const QVector<QPointF>& samples);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QVector<QPointF> samples_;
};

class SpectrumPlotWidget final : public ExportablePlotWidget {
    Q_OBJECT

public:
    explicit SpectrumPlotWidget(QWidget* parent = nullptr);

    void setTitle(const QString& title);
    void setXAxisTitle(const QString& title);
    void setYAxisTitle(const QString& title);
    void setSpectrum(const QVector<QPointF>& samples, double kNyquist, double kTrust);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QString title_;
    QString xAxisTitle_ = "k (rad / m)";
    QString yAxisTitle_;
    QVector<QPointF> samples_;
    double kNyquist_ = 0.0;
    double kTrust_ = 0.0;
};

class HeatmapPlotWidget final : public ExportablePlotWidget {
    Q_OBJECT

public:
    explicit HeatmapPlotWidget(QWidget* parent = nullptr);

    void setTitle(const QString& title);
    void setXAxisTitle(const QString& title);
    void setYAxisTitle(const QString& title);
    void setColorBarTitle(const QString& title);
    void setHeatmap(
        int columns,
        int rows,
        const QVector<double>& samples,
        double xMin,
        double xMax,
        double yMin,
        double yMax,
        double vMin,
        double vMax);
    void clear();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    void rebuildImage();

    QString title_;
    QString xAxisTitle_ = "k_x (rad / m)";
    QString yAxisTitle_ = "k_y (rad / m)";
    QString colorBarTitle_ = "log10(E)";
    int columns_ = 0;
    int rows_ = 0;
    QVector<double> samples_;
    double xMin_ = 0.0;
    double xMax_ = 1.0;
    double yMin_ = 0.0;
    double yMax_ = 1.0;
    double valueMin_ = 0.0;
    double valueMax_ = 1.0;
    QImage image_;
};

class DistributionPlotWidget final : public ExportablePlotWidget {
    Q_OBJECT

public:
    explicit DistributionPlotWidget(QWidget* parent = nullptr);

    void setCurves(const QVector<QPointF>& pdfSamples, const QVector<QPointF>& cdfSamples, const QVector<double>& guideLines);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QVector<QPointF> pdfSamples_;
    QVector<QPointF> cdfSamples_;
    QVector<double> guideLines_;
};

}
