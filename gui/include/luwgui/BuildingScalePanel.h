#pragma once

#include <QFutureWatcher>
#include <QWidget>

class QLabel;
class QLineEdit;
class QPushButton;

namespace luwgui {

struct BuildingScaleResult;
class DistributionPlotWidget;

class BuildingScalePanel : public QWidget {
    Q_OBJECT

public:
    explicit BuildingScalePanel(QWidget* parent = nullptr);

    void setSuggestedFilePath(const QString& filePath);

signals:
    void statusMessage(const QString& message);

private:
    void startAnalysis();
    void savePlot();

    QLineEdit* fileEdit_ = nullptr;
    QLabel* summaryLabel_ = nullptr;
    DistributionPlotWidget* plot_ = nullptr;
    QPushButton* analyzeButton_ = nullptr;
    QFutureWatcher<BuildingScaleResult>* watcher_ = nullptr;
};

}
