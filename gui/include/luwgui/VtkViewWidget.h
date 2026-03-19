#pragma once

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QList>
#include <QSlider>
#include <QString>
#include <QWidget>

#include <vtkSmartPointer.h>

class QCheckBox;
class QPushButton;
class QVTKOpenGLNativeWidget;
class vtkActor;
class vtkAlgorithm;
class vtkAxesActor;
class vtkCutter;
class vtkContourFilter;
class vtkDataSet;
class vtkDataSetMapper;
class vtkGenericOpenGLRenderWindow;
class vtkLight;
class vtkLookupTable;
class vtkOutlineFilter;
class vtkPlane;
class vtkPolyDataMapper;
class vtkRenderer;
class vtkScalarBarRepresentation;
class vtkScalarBarActor;
class vtkScalarBarWidget;
class vtkSTLReader;
class vtkTextActor;

namespace luwgui {

class VtkViewWidget : public QWidget {
    Q_OBJECT

public:
    explicit VtkViewWidget(QWidget* parent = nullptr);

    bool loadFile(const QString& filePath, QString* errorMessage = nullptr);
    void setProjectDirectory(const QString& projectDirectory);
    QString currentFile() const;
    bool loadLatestResult(QString* errorMessage = nullptr);

public slots:
    void refreshResultCatalog();
    void handleSolverFinished();
    void resetCamera();
    void reloadCurrentFile();
    void saveImage();

signals:
    void fileLoaded(const QString& filePath);
    void statusMessage(const QString& message);
    void guiActionRequested(const QString& message);

private:
    struct ArraySelection {
        QString association;
        QString name;
    };
    struct ResultEntry {
        QString type;
        QString filePath;
        QString runStamp;
        qlonglong timeStep = -1;
        qlonglong sortKey = -1;
    };

    vtkSmartPointer<vtkDataSet> readDataSet(const QString& filePath, QString* errorMessage) const;
    void rebuildPipeline(bool resetCameraFlag = false);
    void updateArrayMenus(vtkDataSet* dataSet);
    void applyScalarState(vtkDataSet* dataSet);
    vtkSmartPointer<vtkLookupTable> buildLookupTable() const;
    ArraySelection currentScalarSelection() const;
    QList<ResultEntry> scanResultEntries() const;
    void repopulateTimeCombo(bool preserveSelection, bool autoLoadSelection);
    bool loadResultEntry(const ResultEntry& entry, QString* errorMessage = nullptr);
    ResultEntry latestEntryForType(const QString& type) const;
    ResultEntry currentSelectedEntry() const;
    void updateSliceBounds(vtkDataSet* dataSet);
    void resetQCriterionState();
    void computeQCriterion();
    void applyAxesStyle();
    void applyScalarBarStyle();
    void reloadStlOverlays();
    void updateStlOverlayVisibility();
    bool eventFilter(QObject* watched, QEvent* event) override;
    void showScalarBarEditor();

    QVTKOpenGLNativeWidget* vtkWidget_ = nullptr;
    QLabel* fileLabel_ = nullptr;
    QCheckBox* sliceCheck_ = nullptr;
    QCheckBox* qCriterionCheck_ = nullptr;
    QComboBox* resultTypeCombo_ = nullptr;
    QComboBox* resultTimeCombo_ = nullptr;
    QComboBox* scalarArrayCombo_ = nullptr;
    QComboBox* componentCombo_ = nullptr;
    QComboBox* paletteCombo_ = nullptr;
    QComboBox* sliceAxisCombo_ = nullptr;
    QComboBox* qVectorCombo_ = nullptr;
    QSlider* opacitySlider_ = nullptr;
    QDoubleSpinBox* slicePositionSpin_ = nullptr;
    QDoubleSpinBox* qIsoSpin_ = nullptr;
    QPushButton* computeQButton_ = nullptr;

    vtkSmartPointer<vtkGenericOpenGLRenderWindow> renderWindow_;
    vtkSmartPointer<vtkRenderer> renderer_;
    vtkSmartPointer<vtkDataSetMapper> mapper_;
    vtkSmartPointer<vtkActor> actor_;
    vtkSmartPointer<vtkOutlineFilter> outlineFilter_;
    vtkSmartPointer<vtkPolyDataMapper> outlineMapper_;
    vtkSmartPointer<vtkActor> outlineActor_;
    vtkSmartPointer<vtkScalarBarActor> scalarBar_;
    vtkSmartPointer<vtkScalarBarRepresentation> scalarBarRepresentation_;
    vtkSmartPointer<vtkScalarBarWidget> scalarBarWidget_;
    vtkSmartPointer<vtkAxesActor> axesActor_;
    vtkSmartPointer<vtkTextActor> watermarkActor_;
    vtkSmartPointer<vtkLight> keyLight_;
    vtkSmartPointer<vtkLight> fillLight_;
    vtkSmartPointer<vtkPlane> slicePlane_;
    vtkSmartPointer<vtkCutter> cutter_;
    vtkSmartPointer<vtkContourFilter> contourFilter_;

    vtkSmartPointer<vtkDataSet> originalData_;
    vtkSmartPointer<vtkDataSet> qCriterionData_;
    QList<vtkSmartPointer<vtkActor>> stlActors_;
    QList<vtkSmartPointer<vtkSTLReader>> stlReaders_;
    QString currentFile_;
    QString projectDirectory_;
    QList<ResultEntry> resultEntries_;
    bool blockResultSelection_ = false;
    bool qCriterionReady_ = false;
    bool scalarRangeOverrideEnabled_ = false;
    double scalarRangeOverride_[2] = {0.0, 1.0};
    double currentScalarRange_[2] = {0.0, 1.0};
    QString qCriterionVectorName_;
};

}
