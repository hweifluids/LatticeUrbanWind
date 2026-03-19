#pragma once

#include "luwgui/ConfigDocument.h"

#include <QLineEdit>
#include <QLabel>
#include <QStackedWidget>
#include <QTableWidget>
#include <QWidget>

namespace luwgui {

class ProfilePlotWidget;

class BatchBoundaryPanel : public QWidget {
    Q_OBJECT

public:
    explicit BatchBoundaryPanel(QWidget* parent = nullptr);

    void setDocument(ConfigDocument* document);

public slots:
    void refresh();

private:
    void updateDatasetTab();
    void updateProfileTab();
    void updateProfileCurve();
    void pushDatasetEdits();
    void pushProfileEdits();

    static QVariantList parseFloatListText(const QString& text);

    ConfigDocument* document_ = nullptr;
    QStackedWidget* modeStack_ = nullptr;
    QLabel* modeSummary_ = nullptr;

    QLineEdit* dgInflowEdit_ = nullptr;
    QLineEdit* dgAngleEdit_ = nullptr;
    QTableWidget* dgMatrix_ = nullptr;

    QLineEdit* pfAngleEdit_ = nullptr;
    QTableWidget* pfCases_ = nullptr;
    ProfilePlotWidget* profilePlot_ = nullptr;
    QTableWidget* profileSamples_ = nullptr;
};

}
