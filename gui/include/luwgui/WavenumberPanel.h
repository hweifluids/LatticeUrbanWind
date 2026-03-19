#pragma once

#include <QFutureWatcher>
#include <QString>
#include <QVector>
#include <QWidget>

class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QTabWidget;

namespace luwgui {

struct WavenumberAnalysisResult;
struct LesLayerSpectrumResult {
    QString label;
    QString title;
    int columns = 0;
    int rows = 0;
    QVector<double> samples;
    double xMin = 0.0;
    double xMax = 1.0;
    double yMin = 0.0;
    double yMax = 1.0;
    double valueMin = 0.0;
    double valueMax = 1.0;
};
class SpectrumPlotWidget;
class HeatmapPlotWidget;

class WavenumberPanel : public QWidget {
    Q_OBJECT

public:
    explicit WavenumberPanel(QWidget* parent = nullptr);

    void setSuggestedFilePath(const QString& filePath, bool autoAnalyze = false);

signals:
    void statusMessage(const QString& message);
    void guiActionRequested(const QString& message);

private:
    void updateTabUi();
    void updateArrayChoices();
    void startAnalysis();
    void updateLesLayerView();
    void savePlots();

    QLineEdit* fileEdit_ = nullptr;
    QComboBox* arrayCombo_ = nullptr;
    QPushButton* analyzeButton_ = nullptr;
    QLabel* summaryLabel_ = nullptr;
    QTabWidget* plotTabs_ = nullptr;
    QWidget* lesTabCorner_ = nullptr;
    QComboBox* lesHeightCombo_ = nullptr;
    SpectrumPlotWidget* energyPlot_ = nullptr;
    SpectrumPlotWidget* compensatedPlot_ = nullptr;
    HeatmapPlotWidget* lesPlot_ = nullptr;
    QVector<LesLayerSpectrumResult> lesLayers_;
    QFutureWatcher<WavenumberAnalysisResult>* watcher_ = nullptr;
};

}
