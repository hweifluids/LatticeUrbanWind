#include "luwgui/VtkViewWidget.h"

#include <QColor>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QCheckBox>
#include <QDateTime>
#include <QDialog>
#include <QDialogButtonBox>
#include <QEvent>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QMouseEvent>
#include <QPushButton>
#include <QRegularExpression>
#include <QSignalBlocker>
#include <QToolButton>
#include <QVariantList>
#include <QVBoxLayout>

#include <QVTKOpenGLNativeWidget.h>

#include <vtkActor.h>
#include <vtkAxesActor.h>
#include <vtkCaptionActor2D.h>
#include <vtkCellData.h>
#include <vtkContourFilter.h>
#include <vtkCoordinate.h>
#include <vtkCutter.h>
#include <vtkDataArray.h>
#include <vtkDataObject.h>
#include <vtkDataSet.h>
#include <vtkDataSetMapper.h>
#include <vtkFieldData.h>
#include <vtkGenericDataObjectReader.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkGradientFilter.h>
#include <vtkLight.h>
#include <vtkLookupTable.h>
#include <vtkOutlineFilter.h>
#include <vtkPlane.h>
#include <vtkPointData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkScalarBarRepresentation.h>
#include <vtkSTLReader.h>
#include <vtkScalarBarActor.h>
#include <vtkScalarBarWidget.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkUnstructuredGrid.h>

#include <algorithm>
#include <numbers>
#include <set>

namespace luwgui {

namespace {

QDoubleSpinBox* createSpin(double minValue, double maxValue, double step, double value, int decimals = 3) {
    auto* spin = new QDoubleSpinBox();
    spin->setRange(minValue, maxValue);
    spin->setSingleStep(step);
    spin->setDecimals(decimals);
    spin->setValue(value);
    spin->setAccelerated(true);
    return spin;
}

QColor lerpColor(const QColor& a, const QColor& b, double t) {
    const double u = std::clamp(t, 0.0, 1.0);
    return QColor::fromRgbF(
        a.redF() + (b.redF() - a.redF()) * u,
        a.greenF() + (b.greenF() - a.greenF()) * u,
        a.blueF() + (b.blueF() - a.blueF()) * u,
        1.0);
}

qlonglong sortKeyFor(const QString& runStamp, qlonglong timeStep) {
    const QDateTime parsed = QDateTime::fromString(runStamp, "yyyyMMddhhmmss");
    const qlonglong seconds = parsed.isValid() ? parsed.toSecsSinceEpoch() : 0;
    return seconds * 1000000000LL + std::max<qlonglong>(timeStep, 0);
}

void setPlainBlackText(vtkTextProperty* property, int fontSize) {
    if (!property) {
        return;
    }
    property->SetColor(0.0, 0.0, 0.0);
    property->SetBold(false);
    property->SetItalic(false);
    property->SetShadow(false);
    property->SetFontSize(fontSize);
}

} // namespace

VtkViewWidget::VtkViewWidget(QWidget* parent)
    : QWidget(parent) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(6);

    auto* controls = new QWidget(this);
    auto* controlsLayout = new QGridLayout(controls);
    controlsLayout->setContentsMargins(0, 0, 0, 0);
    controlsLayout->setHorizontalSpacing(6);

    auto* refreshCatalogButton = new QToolButton(controls);
    refreshCatalogButton->setText("Refresh VTK");
    auto* loadButton = new QToolButton(controls);
    loadButton->setText("Reload");
    auto* resetButton = new QToolButton(controls);
    resetButton->setText("Reset View");
    auto* saveButton = new QToolButton(controls);
    saveButton->setText("Save Image");
    sliceCheck_ = new QCheckBox("Slice", controls);
    qCriterionCheck_ = new QCheckBox("Q Criterion", controls);
    qCriterionCheck_->setEnabled(false);
    computeQButton_ = new QPushButton("Compute Q", controls);
    computeQButton_->setEnabled(false);

    resultTypeCombo_ = new QComboBox(controls);
    resultTypeCombo_->addItem("Average", "average");
    resultTypeCombo_->addItem("Unsteady", "unsteady");
    resultTypeCombo_->setCurrentIndex(1);
    resultTimeCombo_ = new QComboBox(controls);
    resultTimeCombo_->setMinimumContentsLength(18);
    resultTimeCombo_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);

    fileLabel_ = new QLabel("No dataset loaded", controls);
    fileLabel_->setProperty("muted", true);
    controlsLayout->addWidget(refreshCatalogButton, 0, 0);
    controlsLayout->addWidget(loadButton, 0, 1);
    controlsLayout->addWidget(resetButton, 0, 2);
    controlsLayout->addWidget(saveButton, 0, 3);
    controlsLayout->addWidget(new QLabel("Field"), 0, 4);
    controlsLayout->addWidget(resultTypeCombo_, 0, 5);
    controlsLayout->addWidget(new QLabel("Time"), 0, 6);
    controlsLayout->addWidget(resultTimeCombo_, 0, 7, 1, 2);
    controlsLayout->addWidget(sliceCheck_, 0, 9);
    controlsLayout->addWidget(qCriterionCheck_, 0, 10);
    controlsLayout->addWidget(computeQButton_, 0, 11);
    controlsLayout->addWidget(fileLabel_, 0, 12, 1, 4);

    scalarArrayCombo_ = new QComboBox(controls);
    componentCombo_ = new QComboBox(controls);
    componentCombo_->addItems({"Magnitude", "X", "Y", "Z"});
    paletteCombo_ = new QComboBox(controls);
    paletteCombo_->addItems({"Cool Steel", "Diverging Pressure", "Thermal"});
    qVectorCombo_ = new QComboBox(controls);
    sliceAxisCombo_ = new QComboBox(controls);
    sliceAxisCombo_->addItems({"X", "Y", "Z"});
    opacitySlider_ = new QSlider(Qt::Horizontal, controls);
    opacitySlider_->setRange(5, 100);
    opacitySlider_->setValue(100);
    slicePositionSpin_ = createSpin(-1.0e9, 1.0e9, 0.1, 0.0, 3);
    slicePositionSpin_->setEnabled(false);
    qIsoSpin_ = createSpin(-1.0e9, 1.0e9, 0.01, 0.0, 4);
    qIsoSpin_->setEnabled(false);

    controlsLayout->addWidget(new QLabel("Scalar"), 1, 0);
    controlsLayout->addWidget(scalarArrayCombo_, 1, 1, 1, 3);
    controlsLayout->addWidget(new QLabel("Component"), 1, 4);
    controlsLayout->addWidget(componentCombo_, 1, 5);
    controlsLayout->addWidget(new QLabel("Palette"), 1, 6);
    controlsLayout->addWidget(paletteCombo_, 1, 7);
    controlsLayout->addWidget(new QLabel("Opacity"), 1, 8);
    controlsLayout->addWidget(opacitySlider_, 1, 9, 1, 3);

    controlsLayout->addWidget(new QLabel("Slice axis"), 2, 0);
    controlsLayout->addWidget(sliceAxisCombo_, 2, 1);
    controlsLayout->addWidget(new QLabel("Slice pos"), 2, 2);
    controlsLayout->addWidget(slicePositionSpin_, 2, 3, 1, 2);
    controlsLayout->addWidget(new QLabel("Q vector"), 2, 5);
    controlsLayout->addWidget(qVectorCombo_, 2, 6, 1, 2);
    controlsLayout->addWidget(new QLabel("Q iso"), 2, 8);
    controlsLayout->addWidget(qIsoSpin_, 2, 9, 1, 2);
    root->addWidget(controls);

    vtkWidget_ = new QVTKOpenGLNativeWidget(this);
    vtkWidget_->installEventFilter(this);
    root->addWidget(vtkWidget_, 1);

    renderWindow_ = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    renderer_ = vtkSmartPointer<vtkRenderer>::New();
    renderWindow_->AddRenderer(renderer_);
    vtkWidget_->setRenderWindow(renderWindow_);

    mapper_ = vtkSmartPointer<vtkDataSetMapper>::New();
    actor_ = vtkSmartPointer<vtkActor>::New();
    actor_->SetMapper(mapper_);
    actor_->SetVisibility(false);
    renderer_->AddActor(actor_);

    outlineFilter_ = vtkSmartPointer<vtkOutlineFilter>::New();
    outlineMapper_ = vtkSmartPointer<vtkPolyDataMapper>::New();
    outlineMapper_->SetInputConnection(outlineFilter_->GetOutputPort());
    outlineActor_ = vtkSmartPointer<vtkActor>::New();
    outlineActor_->SetMapper(outlineMapper_);
    outlineActor_->GetProperty()->SetColor(0.68, 0.74, 0.80);
    outlineActor_->GetProperty()->SetLineWidth(1.2);
    outlineActor_->SetVisibility(false);
    renderer_->AddActor(outlineActor_);

    axesActor_ = vtkSmartPointer<vtkAxesActor>::New();
    axesActor_->SetVisibility(false);
    applyAxesStyle();
    renderer_->AddActor(axesActor_);

    scalarBar_ = vtkSmartPointer<vtkScalarBarActor>::New();
    scalarBar_->SetLookupTable(buildLookupTable());
    scalarBar_->SetTitle("");
    scalarBar_->SetNumberOfLabels(5);
    scalarBar_->SetMaximumWidthInPixels(110);
    scalarBar_->SetMaximumHeightInPixels(260);
    scalarBar_->SetVisibility(false);
    applyScalarBarStyle();

    scalarBarRepresentation_ = vtkSmartPointer<vtkScalarBarRepresentation>::New();
    scalarBarRepresentation_->SetScalarBarActor(scalarBar_);
    scalarBarRepresentation_->SetAutoOrient(false);
    scalarBarRepresentation_->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
    scalarBarRepresentation_->GetPositionCoordinate()->SetValue(0.86, 0.18);
    scalarBarRepresentation_->GetPosition2Coordinate()->SetCoordinateSystemToNormalizedViewport();
    scalarBarRepresentation_->GetPosition2Coordinate()->SetValue(0.09, 0.36);

    scalarBarWidget_ = vtkSmartPointer<vtkScalarBarWidget>::New();
    scalarBarWidget_->SetInteractor(vtkWidget_->interactor());
    scalarBarWidget_->SetCurrentRenderer(renderer_);
    scalarBarWidget_->SetRepresentation(scalarBarRepresentation_);
    scalarBarWidget_->SetScalarBarActor(scalarBar_);
    scalarBarWidget_->ResizableOn();
    scalarBarWidget_->RepositionableOn();
    scalarBarWidget_->On();

    watermarkActor_ = vtkSmartPointer<vtkTextActor>::New();
    watermarkActor_->SetInput("LatticeUrbanWind");
    watermarkActor_->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
    watermarkActor_->SetPosition(0.02, 0.92);
    watermarkActor_->GetTextProperty()->SetFontFamilyToTimes();
    watermarkActor_->GetTextProperty()->SetBold(true);
    watermarkActor_->GetTextProperty()->SetItalic(false);
    watermarkActor_->GetTextProperty()->SetColor(0.45, 0.45, 0.45);
    watermarkActor_->GetTextProperty()->SetOpacity(0.14);
    watermarkActor_->GetTextProperty()->SetFontSize(24);
    renderer_->AddActor2D(watermarkActor_);

    slicePlane_ = vtkSmartPointer<vtkPlane>::New();
    cutter_ = vtkSmartPointer<vtkCutter>::New();
    contourFilter_ = vtkSmartPointer<vtkContourFilter>::New();

    auto emptyData = vtkSmartPointer<vtkUnstructuredGrid>::New();
    mapper_->SetInputData(emptyData);
    outlineFilter_->SetInputData(emptyData);

    renderer_->SetBackground(1.0, 1.0, 1.0);
    renderer_->GradientBackgroundOff();
    renderer_->AutomaticLightCreationOff();
    renderer_->RemoveAllLights();

    keyLight_ = vtkSmartPointer<vtkLight>::New();
    keyLight_->SetLightTypeToHeadlight();
    keyLight_->SetIntensity(0.45);
    renderer_->AddLight(keyLight_);

    fillLight_ = vtkSmartPointer<vtkLight>::New();
    fillLight_->SetLightTypeToCameraLight();
    fillLight_->SetPosition(-0.4, 0.7, 1.0);
    fillLight_->SetFocalPoint(0.0, 0.0, 0.0);
    fillLight_->SetIntensity(0.22);
    renderer_->AddLight(fillLight_);

    auto rebuild = [this] { rebuildPipeline(false); };
    connect(refreshCatalogButton, &QToolButton::clicked, this, &VtkViewWidget::refreshResultCatalog);
    connect(loadButton, &QToolButton::clicked, this, &VtkViewWidget::reloadCurrentFile);
    connect(resetButton, &QToolButton::clicked, this, &VtkViewWidget::resetCamera);
    connect(saveButton, &QToolButton::clicked, this, &VtkViewWidget::saveImage);
    connect(sliceCheck_, &QCheckBox::toggled, this, [this, rebuild](bool checked) {
        sliceAxisCombo_->setEnabled(checked);
        slicePositionSpin_->setEnabled(checked);
        rebuild();
    });
    connect(qCriterionCheck_, &QCheckBox::toggled, this, [this, rebuild](bool checked) {
        qIsoSpin_->setEnabled(checked && qCriterionReady_);
        rebuild();
    });
    connect(sliceAxisCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this, rebuild](int) {
        updateSliceBounds(originalData_);
        rebuild();
    });
    connect(slicePositionSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, rebuild);
    connect(qVectorCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this] {
        resetQCriterionState();
        computeQButton_->setEnabled(originalData_ && qVectorCombo_->count() > 0);
    });
    connect(qIsoSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, rebuild);
    connect(paletteCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, rebuild);
    connect(componentCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, rebuild);
    connect(scalarArrayCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, rebuild);
    connect(opacitySlider_, &QSlider::valueChanged, this, rebuild);
    connect(computeQButton_, &QPushButton::clicked, this, &VtkViewWidget::computeQCriterion);
    connect(resultTypeCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this] {
        if (blockResultSelection_) {
            return;
        }
        repopulateTimeCombo(true, true);
    });
    connect(resultTimeCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this] {
        if (blockResultSelection_) {
            return;
        }
        const ResultEntry entry = currentSelectedEntry();
        if (entry.filePath.isEmpty()) {
            return;
        }
        QString error;
        loadResultEntry(entry, &error);
        if (!error.isEmpty()) {
            emit statusMessage(error);
        }
    });
}

bool VtkViewWidget::loadFile(const QString& filePath, QString* errorMessage) {
    vtkSmartPointer<vtkDataSet> loadedData = readDataSet(filePath, errorMessage);
    if (!loadedData) {
        return false;
    }

    originalData_ = loadedData;
    currentFile_ = QFileInfo(filePath).absoluteFilePath();
    fileLabel_->setText(QFileInfo(currentFile_).fileName());
    scalarRangeOverrideEnabled_ = false;
    resetQCriterionState();
    updateSliceBounds(originalData_);
    reloadStlOverlays();
    updateStlOverlayVisibility();
    emit statusMessage("Loaded " + currentFile_);
    emit fileLoaded(currentFile_);

    const auto matchIt = std::find_if(resultEntries_.cbegin(), resultEntries_.cend(), [this](const ResultEntry& entry) {
        return QFileInfo(entry.filePath).absoluteFilePath() == currentFile_;
    });
    if (matchIt != resultEntries_.cend()) {
        const QSignalBlocker typeBlocker(resultTypeCombo_);
        resultTypeCombo_->setCurrentIndex(resultTypeCombo_->findData(matchIt->type));
        repopulateTimeCombo(true, false);
        const QSignalBlocker timeBlocker(resultTimeCombo_);
        const int timeIndex = resultTimeCombo_->findData(currentFile_);
        if (timeIndex >= 0) {
            resultTimeCombo_->setCurrentIndex(timeIndex);
        }
    }

    computeQButton_->setEnabled(qVectorCombo_->count() > 0);
    rebuildPipeline(true);
    return true;
}

void VtkViewWidget::setProjectDirectory(const QString& projectDirectory) {
    const QString normalized = QFileInfo(projectDirectory).absoluteFilePath();
    if (projectDirectory_ == normalized) {
        return;
    }
    projectDirectory_ = normalized;
    reloadStlOverlays();
    refreshResultCatalog();
}

QString VtkViewWidget::currentFile() const {
    return currentFile_;
}

bool VtkViewWidget::loadLatestResult(QString* errorMessage) {
    if (resultEntries_.isEmpty()) {
        resultEntries_ = scanResultEntries();
        repopulateTimeCombo(true, false);
    }
    ResultEntry entry = latestEntryForType("unsteady");
    if (entry.filePath.isEmpty()) {
        entry = latestEntryForType("average");
    }
    if (entry.filePath.isEmpty()) {
        if (errorMessage) {
            *errorMessage = "No VTK file found in RESULTS/vtk.";
        }
        return false;
    }
    return loadResultEntry(entry, errorMessage);
}

void VtkViewWidget::refreshResultCatalog() {
    resultEntries_ = scanResultEntries();
    repopulateTimeCombo(true, false);
    if (resultEntries_.isEmpty()) {
        emit statusMessage("No VTK file found in RESULTS/vtk.");
    }
}

void VtkViewWidget::handleSolverFinished() {
    refreshResultCatalog();
    QString error;
    if (!loadLatestResult(&error) && !error.isEmpty()) {
        emit statusMessage(error);
    }
}

void VtkViewWidget::resetCamera() {
    renderer_->ResetCamera();
    renderWindow_->Render();
}

void VtkViewWidget::reloadCurrentFile() {
    if (currentFile_.isEmpty()) {
        return;
    }
    QString error;
    loadFile(currentFile_, &error);
    if (!error.isEmpty()) {
        emit statusMessage(error);
    }
}

void VtkViewWidget::saveImage() {
    QString suggested = currentFile_.isEmpty() ? "luw_view.png" : QFileInfo(currentFile_).completeBaseName() + "_view.png";
    QString path = QFileDialog::getSaveFileName(this, "Save View Image", suggested, "PNG Image (*.png)");
    if (path.isEmpty()) {
        return;
    }
    if (QFileInfo(path).suffix().isEmpty()) {
        path += ".png";
    }

    const QImage image = vtkWidget_->grabFramebuffer();
    if (image.isNull() || !image.save(path)) {
        emit statusMessage("Failed to save current view image.");
        return;
    }
    emit statusMessage("Saved current view image to " + path);
}

vtkSmartPointer<vtkDataSet> VtkViewWidget::readDataSet(const QString& filePath, QString* errorMessage) const {
    const QString suffix = QFileInfo(filePath).suffix().toLower();

    if (suffix == "stl") {
        auto reader = vtkSmartPointer<vtkSTLReader>::New();
        reader->SetFileName(filePath.toLocal8Bit().constData());
        reader->Update();
        auto* output = reader->GetOutput();
        if (!output) {
            if (errorMessage) {
                *errorMessage = "Failed to read STL file.";
            }
            return nullptr;
        }
        auto loadedData = vtkSmartPointer<vtkDataSet>::Take(vtkDataSet::SafeDownCast(output->NewInstance()));
        loadedData->ShallowCopy(output);
        return loadedData;
    }

    auto reader = vtkSmartPointer<vtkGenericDataObjectReader>::New();
    reader->SetFileName(filePath.toLocal8Bit().constData());
    reader->Update();
    auto* output = vtkDataSet::SafeDownCast(reader->GetOutput());
    if (!output) {
        if (errorMessage) {
            *errorMessage = "Only legacy VTK dataset files and STL are supported in the embedded viewer.";
        }
        return nullptr;
    }

    auto loadedData = vtkSmartPointer<vtkDataSet>::Take(vtkDataSet::SafeDownCast(output->NewInstance()));
    loadedData->ShallowCopy(output);
    return loadedData;
}

void VtkViewWidget::rebuildPipeline(bool resetCameraFlag) {
    if (!originalData_) {
        actor_->SetVisibility(false);
        outlineActor_->SetVisibility(false);
        axesActor_->SetVisibility(false);
        scalarBar_->SetVisibility(false);
        computeQButton_->setEnabled(false);
        updateStlOverlayVisibility();
        renderWindow_->Render();
        return;
    }

    const bool sliceOn = sliceCheck_ && sliceCheck_->isChecked();
    const bool qOn = qCriterionCheck_ && qCriterionCheck_->isChecked() && qCriterionReady_ && qCriterionData_;

    vtkAlgorithm* terminalAlgorithm = nullptr;
    vtkDataSet* terminalData = nullptr;

    if (qOn) {
        contourFilter_->SetInputData(qCriterionData_);
        contourFilter_->SetValue(0, qIsoSpin_->value());
        contourFilter_->Update();
        terminalAlgorithm = contourFilter_;
        terminalData = vtkDataSet::SafeDownCast(contourFilter_->GetOutputDataObject(0));
    } else if (sliceOn) {
        double bounds[6] = {0, 1, 0, 1, 0, 1};
        originalData_->GetBounds(bounds);
        const int axisIndex = sliceAxisCombo_->currentIndex();
        double origin[3] = {0.0, 0.0, 0.0};
        origin[0] = 0.5 * (bounds[0] + bounds[1]);
        origin[1] = 0.5 * (bounds[2] + bounds[3]);
        origin[2] = 0.5 * (bounds[4] + bounds[5]);
        origin[axisIndex] = slicePositionSpin_->value();
        double normal[3] = {0.0, 0.0, 0.0};
        normal[axisIndex] = 1.0;
        slicePlane_->SetOrigin(origin);
        slicePlane_->SetNormal(normal);

        cutter_->SetInputData(originalData_);
        cutter_->SetCutFunction(slicePlane_);
        cutter_->Update();
        terminalAlgorithm = cutter_;
        terminalData = vtkDataSet::SafeDownCast(cutter_->GetOutputDataObject(0));
    } else {
        mapper_->SetInputData(originalData_);
        terminalData = originalData_;
    }

    if (terminalAlgorithm) {
        mapper_->SetInputConnection(terminalAlgorithm->GetOutputPort());
    }
    mapper_->ScalarVisibilityOn();

    auto* displayData = terminalData;
    if (displayData) {
        actor_->SetVisibility(true);
        outlineActor_->SetVisibility(true);
        scalarBar_->SetVisibility(true);
        updateArrayMenus(displayData);
        applyScalarState(displayData);
        outlineFilter_->SetInputData(displayData);

        double finalBounds[6];
        displayData->GetBounds(finalBounds);
        const double scale = std::max({finalBounds[1] - finalBounds[0], finalBounds[3] - finalBounds[2], finalBounds[5] - finalBounds[4]}) * 0.09;
        axesActor_->SetVisibility(true);
        axesActor_->SetTotalLength(scale, scale, scale);
        axesActor_->SetPosition(finalBounds[0], finalBounds[2], finalBounds[4]);
    } else {
        actor_->SetVisibility(false);
        outlineActor_->SetVisibility(false);
        axesActor_->SetVisibility(false);
        scalarBar_->SetVisibility(false);
    }

    actor_->GetProperty()->SetOpacity(opacitySlider_->value() / 100.0);
    scalarBar_->SetLookupTable(mapper_->GetLookupTable());
    computeQButton_->setEnabled(originalData_ && qVectorCombo_->count() > 0);
    updateStlOverlayVisibility();

    if (resetCameraFlag) {
        renderer_->ResetCamera();
    }
    renderWindow_->Render();
}

void VtkViewWidget::updateSliceBounds(vtkDataSet* dataSet) {
    if (!dataSet || !slicePositionSpin_) {
        return;
    }
    double bounds[6] = {0, 1, 0, 1, 0, 1};
    dataSet->GetBounds(bounds);
    const int axisIndex = std::clamp(sliceAxisCombo_->currentIndex(), 0, 2);
    const double minValue = bounds[axisIndex * 2];
    double maxValue = bounds[axisIndex * 2 + 1];
    if (!(maxValue > minValue)) {
        maxValue = minValue + 1.0;
    }

    const QSignalBlocker blocker(slicePositionSpin_);
    const double currentValue = slicePositionSpin_->value();
    slicePositionSpin_->setRange(minValue, maxValue);
    slicePositionSpin_->setSingleStep(std::max((maxValue - minValue) / 200.0, 0.001));
    if (currentValue < minValue || currentValue > maxValue) {
        slicePositionSpin_->setValue(0.5 * (minValue + maxValue));
    } else {
        slicePositionSpin_->setValue(currentValue);
    }
    sliceAxisCombo_->setEnabled(sliceCheck_->isChecked());
    slicePositionSpin_->setEnabled(sliceCheck_->isChecked());
}

void VtkViewWidget::resetQCriterionState() {
    qCriterionData_ = nullptr;
    qCriterionReady_ = false;
    qCriterionVectorName_.clear();
    qCriterionCheck_->setEnabled(false);
    qIsoSpin_->setEnabled(false);
    const QSignalBlocker blocker(qCriterionCheck_);
    qCriterionCheck_->setChecked(false);
}

void VtkViewWidget::computeQCriterion() {
    if (!originalData_ || qVectorCombo_->currentText().isEmpty()) {
        emit statusMessage("Select a point-vector field before computing Q criterion.");
        return;
    }

    const QString vectorName = qVectorCombo_->currentText();
    emit guiActionRequested("Computing Q criterion from " + vectorName);
    computeQButton_->setEnabled(false);

    auto gradient = vtkSmartPointer<vtkGradientFilter>::New();
    gradient->SetInputData(originalData_);
    gradient->SetInputScalars(vtkDataObject::FIELD_ASSOCIATION_POINTS, vectorName.toLocal8Bit().constData());
    gradient->SetComputeQCriterion(true);
    gradient->SetQCriterionArrayName("q_criterion");
    gradient->Update();

    auto* output = vtkDataSet::SafeDownCast(gradient->GetOutputDataObject(0));
    auto* qArray = output ? output->GetPointData()->GetArray("q_criterion") : nullptr;
    if (!output || !qArray) {
        computeQButton_->setEnabled(originalData_ && qVectorCombo_->count() > 0);
        emit statusMessage("Failed to compute Q criterion for the selected vector field.");
        return;
    }

    qCriterionData_ = vtkSmartPointer<vtkDataSet>::Take(vtkDataSet::SafeDownCast(output->NewInstance()));
    qCriterionData_->ShallowCopy(output);
    qCriterionReady_ = true;
    qCriterionVectorName_ = vectorName;
    qCriterionCheck_->setEnabled(true);

    double qRange[2] = {0.0, 0.0};
    qArray->GetRange(qRange);
    {
        const QSignalBlocker blocker(qIsoSpin_);
        qIsoSpin_->setRange(qRange[0], qRange[1]);
        if (qIsoSpin_->value() < qRange[0] || qIsoSpin_->value() > qRange[1] || qIsoSpin_->value() == 0.0) {
            const double suggested = (qRange[1] > 0.0) ? (qRange[1] * 0.25) : (qRange[0] + qRange[1]) * 0.5;
            qIsoSpin_->setValue(suggested);
        }
    }
    computeQButton_->setEnabled(true);
    emit statusMessage("Q criterion computed for " + vectorName + ".");
    rebuildPipeline(false);
}

void VtkViewWidget::applyAxesStyle() {
    axesActor_->SetShaftTypeToLine();
    axesActor_->SetNormalizedTipLength(0.18, 0.18, 0.18);
    axesActor_->SetNormalizedLabelPosition(1.04, 1.04, 1.04);

    if (auto* property = axesActor_->GetXAxisShaftProperty()) {
        property->SetColor(0.85, 0.15, 0.15);
        property->SetLineWidth(1.0);
    }
    if (auto* property = axesActor_->GetYAxisShaftProperty()) {
        property->SetColor(0.10, 0.62, 0.20);
        property->SetLineWidth(1.0);
    }
    if (auto* property = axesActor_->GetZAxisShaftProperty()) {
        property->SetColor(0.18, 0.36, 0.88);
        property->SetLineWidth(1.0);
    }
    if (auto* property = axesActor_->GetXAxisTipProperty()) {
        property->SetColor(0.85, 0.15, 0.15);
    }
    if (auto* property = axesActor_->GetYAxisTipProperty()) {
        property->SetColor(0.10, 0.62, 0.20);
    }
    if (auto* property = axesActor_->GetZAxisTipProperty()) {
        property->SetColor(0.18, 0.36, 0.88);
    }

    for (vtkCaptionActor2D* caption : {
             axesActor_->GetXAxisCaptionActor2D(),
             axesActor_->GetYAxisCaptionActor2D(),
             axesActor_->GetZAxisCaptionActor2D() }) {
        if (!caption) {
            continue;
        }
        caption->BorderOff();
        caption->LeaderOff();
        setPlainBlackText(caption->GetCaptionTextProperty(), 9);
    }
}

void VtkViewWidget::applyScalarBarStyle() {
    scalarBar_->SetMaximumWidthInPixels(110);
    scalarBar_->SetMaximumHeightInPixels(260);
    scalarBar_->SetTextPositionToSucceedScalarBar();
    setPlainBlackText(scalarBar_->GetTitleTextProperty(), 12);
    setPlainBlackText(scalarBar_->GetLabelTextProperty(), 11);
    scalarBar_->SetTitle("");
}

void VtkViewWidget::reloadStlOverlays() {
    for (const auto& actor : stlActors_) {
        renderer_->RemoveActor(actor);
    }
    stlActors_.clear();
    stlReaders_.clear();

    if (projectDirectory_.isEmpty()) {
        return;
    }

    std::set<QString> stlPaths;
    for (const QString& subdirectory : {"proj_temp", "building_db"}) {
        const QDir directory(QDir(projectDirectory_).filePath(subdirectory));
        const QFileInfoList files = directory.entryInfoList({"*.stl"}, QDir::Files, QDir::Name);
        for (const QFileInfo& file : files) {
            stlPaths.insert(file.absoluteFilePath());
        }
    }

    for (const QString& stlPath : stlPaths) {
        auto reader = vtkSmartPointer<vtkSTLReader>::New();
        reader->SetFileName(stlPath.toLocal8Bit().constData());
        reader->Update();

        auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputConnection(reader->GetOutputPort());

        auto actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->SetColor(0.45, 0.45, 0.45);
        actor->GetProperty()->SetOpacity(0.24);
        actor->GetProperty()->SetAmbient(0.30);
        actor->GetProperty()->SetDiffuse(0.45);
        renderer_->AddActor(actor);

        stlReaders_.push_back(reader);
        stlActors_.push_back(actor);
    }
    updateStlOverlayVisibility();
    renderWindow_->Render();
}

void VtkViewWidget::updateStlOverlayVisibility() {
    const bool visible = QFileInfo(currentFile_).suffix().compare("vtk", Qt::CaseInsensitive) == 0;
    for (const auto& actor : stlActors_) {
        actor->SetVisibility(visible);
    }
}

bool VtkViewWidget::eventFilter(QObject* watched, QEvent* event) {
    if (watched == vtkWidget_ && event->type() == QEvent::MouseButtonDblClick && scalarBarRepresentation_ && scalarBar_->GetVisibility()) {
        auto* mouseEvent = static_cast<QMouseEvent*>(event);
        const double* position = scalarBarRepresentation_->GetPositionCoordinate()->GetValue();
        const double* size = scalarBarRepresentation_->GetPosition2Coordinate()->GetValue();
        const QRectF scalarBarRect(
            position[0] * vtkWidget_->width(),
            (1.0 - position[1] - size[1]) * vtkWidget_->height(),
            size[0] * vtkWidget_->width(),
            size[1] * vtkWidget_->height());
        if (scalarBarRect.adjusted(-10.0, -10.0, 10.0, 10.0).contains(mouseEvent->position())) {
            showScalarBarEditor();
            return true;
        }
    }
    return QWidget::eventFilter(watched, event);
}

void VtkViewWidget::showScalarBarEditor() {
    QDialog dialog(this);
    dialog.setWindowTitle("Legend");
    dialog.resize(360, 220);

    auto* layout = new QVBoxLayout(&dialog);
    auto* form = new QFormLayout();

    auto* paletteEdit = new QComboBox(&dialog);
    paletteEdit->addItems({"Cool Steel", "Diverging Pressure", "Thermal"});
    paletteEdit->setCurrentIndex(paletteCombo_->currentIndex());
    form->addRow("Palette", paletteEdit);

    auto* orientationEdit = new QComboBox(&dialog);
    orientationEdit->addItem("Vertical", VTK_ORIENT_VERTICAL);
    orientationEdit->addItem("Horizontal", VTK_ORIENT_HORIZONTAL);
    orientationEdit->setCurrentIndex(scalarBar_->GetOrientation() == VTK_ORIENT_HORIZONTAL ? 1 : 0);
    form->addRow("Orientation", orientationEdit);

    auto* autoRangeCheck = new QCheckBox("Use data range", &dialog);
    autoRangeCheck->setChecked(!scalarRangeOverrideEnabled_);
    form->addRow(QString(), autoRangeCheck);

    auto* minEdit = createSpin(-1.0e12, 1.0e12, 0.1, scalarRangeOverrideEnabled_ ? scalarRangeOverride_[0] : currentScalarRange_[0], 6);
    auto* maxEdit = createSpin(-1.0e12, 1.0e12, 0.1, scalarRangeOverrideEnabled_ ? scalarRangeOverride_[1] : currentScalarRange_[1], 6);
    minEdit->setEnabled(!autoRangeCheck->isChecked());
    maxEdit->setEnabled(!autoRangeCheck->isChecked());
    form->addRow("Minimum", minEdit);
    form->addRow("Maximum", maxEdit);

    layout->addLayout(form);
    layout->addStretch(1);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    layout->addWidget(buttons);

    connect(autoRangeCheck, &QCheckBox::toggled, &dialog, [minEdit, maxEdit](bool checked) {
        minEdit->setEnabled(!checked);
        maxEdit->setEnabled(!checked);
    });
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    if (!autoRangeCheck->isChecked() && minEdit->value() >= maxEdit->value()) {
        emit statusMessage("Legend range is invalid: minimum must be smaller than maximum.");
        return;
    }

    paletteCombo_->setCurrentIndex(paletteEdit->currentIndex());
    scalarRangeOverrideEnabled_ = !autoRangeCheck->isChecked();
    if (scalarRangeOverrideEnabled_) {
        scalarRangeOverride_[0] = minEdit->value();
        scalarRangeOverride_[1] = maxEdit->value();
    }

    const int orientation = orientationEdit->currentData().toInt();
    if (orientation != scalarBar_->GetOrientation()) {
        const double* currentSize = scalarBarRepresentation_->GetPosition2Coordinate()->GetValue();
        scalarBarRepresentation_->GetPosition2Coordinate()->SetValue(currentSize[1], currentSize[0]);
    }
    scalarBar_->SetOrientation(orientation);
    scalarBarRepresentation_->SetOrientation(orientation);

    emit guiActionRequested("Updated legend settings");
    rebuildPipeline(false);
}

void VtkViewWidget::updateArrayMenus(vtkDataSet* dataSet) {
    if (!dataSet) {
        return;
    }

    const QSignalBlocker blockScalar(scalarArrayCombo_);
    const QSignalBlocker blockVector(qVectorCombo_);

    const QString previousScalar = scalarArrayCombo_->currentText();
    const QString previousVector = qVectorCombo_->currentText();

    scalarArrayCombo_->clear();
    qVectorCombo_->clear();

    auto addScalarArrays = [&](vtkFieldData* fieldData, const QString& association) {
        if (!fieldData) {
            return;
        }
        for (int i = 0; i < fieldData->GetNumberOfArrays(); ++i) {
            vtkDataArray* array = fieldData->GetArray(i);
            if (!array || !array->GetName()) {
                continue;
            }
            const QString name = QString::fromLocal8Bit(array->GetName());
            const QString display = association + ":" + name;
            scalarArrayCombo_->addItem(display, QVariantList{association, name});
        }
    };

    auto addVectorArrays = [&](vtkDataSet* sourceData) {
        if (!sourceData || !sourceData->GetPointData()) {
            return;
        }
        vtkFieldData* fieldData = sourceData->GetPointData();
        for (int i = 0; i < fieldData->GetNumberOfArrays(); ++i) {
            vtkDataArray* array = fieldData->GetArray(i);
            if (!array || !array->GetName() || array->GetNumberOfComponents() < 3) {
                continue;
            }
            qVectorCombo_->addItem(QString::fromLocal8Bit(array->GetName()));
        }
    };

    addScalarArrays(dataSet->GetPointData(), "point");
    addScalarArrays(dataSet->GetCellData(), "cell");
    addVectorArrays(originalData_ ? originalData_.GetPointer() : dataSet);

    if (!previousScalar.isEmpty()) {
        const int index = scalarArrayCombo_->findText(previousScalar);
        if (index >= 0) {
            scalarArrayCombo_->setCurrentIndex(index);
        }
    }
    if (scalarArrayCombo_->currentIndex() < 0 && scalarArrayCombo_->count() > 0) {
        scalarArrayCombo_->setCurrentIndex(0);
    }
    if (qCriterionCheck_->isChecked()) {
        const int qScalarIndex = scalarArrayCombo_->findText("point:q_criterion");
        if (qScalarIndex >= 0) {
            scalarArrayCombo_->setCurrentIndex(qScalarIndex);
        }
    }

    if (!previousVector.isEmpty()) {
        const int vectorIndex = qVectorCombo_->findText(previousVector);
        if (vectorIndex >= 0) {
            qVectorCombo_->setCurrentIndex(vectorIndex);
        }
    }
    if (qVectorCombo_->currentIndex() < 0 && qVectorCombo_->count() > 0) {
        qVectorCombo_->setCurrentIndex(0);
    }

    computeQButton_->setEnabled(originalData_ && qVectorCombo_->count() > 0);
}

void VtkViewWidget::applyScalarState(vtkDataSet* dataSet) {
    if (!dataSet || scalarArrayCombo_->count() == 0) {
        mapper_->ScalarVisibilityOff();
        return;
    }

    const ArraySelection selection = currentScalarSelection();
    vtkDataArray* array = nullptr;
    if (selection.association == "point") {
        array = dataSet->GetPointData()->GetArray(selection.name.toLocal8Bit().constData());
        mapper_->SetScalarModeToUsePointFieldData();
    } else {
        array = dataSet->GetCellData()->GetArray(selection.name.toLocal8Bit().constData());
        mapper_->SetScalarModeToUseCellFieldData();
    }

    if (!array) {
        mapper_->ScalarVisibilityOff();
        return;
    }

    mapper_->SelectColorArray(selection.name.toLocal8Bit().constData());
    mapper_->SetLookupTable(buildLookupTable());
    mapper_->ScalarVisibilityOn();

    const int component = (array->GetNumberOfComponents() > 1) ? componentCombo_->currentIndex() - 1 : 0;
    mapper_->ColorByArrayComponent(selection.name.toLocal8Bit().constData(), component);

    double range[2] = {0.0, 0.0};
    if (array->GetNumberOfComponents() > 1 && componentCombo_->currentIndex() == 0) {
        array->GetRange(range, -1);
    } else {
        array->GetRange(range, component);
    }
    currentScalarRange_[0] = range[0];
    currentScalarRange_[1] = range[1];
    if (scalarRangeOverrideEnabled_ && scalarRangeOverride_[0] < scalarRangeOverride_[1]) {
        mapper_->SetScalarRange(scalarRangeOverride_);
    } else {
        mapper_->SetScalarRange(range);
    }
    scalarBar_->SetTitle("");
}

vtkSmartPointer<vtkLookupTable> VtkViewWidget::buildLookupTable() const {
    auto lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetNumberOfTableValues(256);
    lut->Build();

    QColor a(54, 86, 122);
    QColor b(190, 204, 219);
    QColor c(208, 120, 78);
    if (paletteCombo_ && paletteCombo_->currentIndex() == 1) {
        a = QColor(44, 87, 142);
        b = QColor(222, 228, 233);
        c = QColor(167, 73, 58);
    } else if (paletteCombo_ && paletteCombo_->currentIndex() == 2) {
        a = QColor(40, 50, 98);
        b = QColor(239, 198, 97);
        c = QColor(193, 72, 46);
    }

    for (int i = 0; i < 256; ++i) {
        const double t = static_cast<double>(i) / 255.0;
        QColor color = (t < 0.5) ? lerpColor(a, b, t * 2.0) : lerpColor(b, c, (t - 0.5) * 2.0);
        lut->SetTableValue(i, color.redF(), color.greenF(), color.blueF(), 1.0);
    }
    return lut;
}

VtkViewWidget::ArraySelection VtkViewWidget::currentScalarSelection() const {
    const QVariantList payload = scalarArrayCombo_->currentData().toList();
    if (payload.size() != 2) {
        return {"point", scalarArrayCombo_->currentText()};
    }
    return {payload[0].toString(), payload[1].toString()};
}

QList<VtkViewWidget::ResultEntry> VtkViewWidget::scanResultEntries() const {
    QList<ResultEntry> entries;
    if (projectDirectory_.isEmpty()) {
        return entries;
    }

    const QDir vtkDir(QDir(projectDirectory_).filePath("RESULTS/vtk"));
    if (!vtkDir.exists()) {
        return entries;
    }

    const QFileInfoList files = vtkDir.entryInfoList({"*.vtk"}, QDir::Files, QDir::Name);
    const QRegularExpression rawPattern(R"((?:.+)?(\d{14})_raw_u-(\d+)\.vtk$)", QRegularExpression::CaseInsensitiveOption);
    const QRegularExpression avgPattern(R"((?:.+)?(\d{14})_avg-(\d+)\.vtk$)", QRegularExpression::CaseInsensitiveOption);
    for (const QFileInfo& file : files) {
        const QString name = file.fileName();
        ResultEntry entry;
        QRegularExpressionMatch match = rawPattern.match(name);
        if (match.hasMatch()) {
            entry.type = "unsteady";
            entry.runStamp = match.captured(1);
            entry.timeStep = match.captured(2).toLongLong();
        } else {
            match = avgPattern.match(name);
            if (!match.hasMatch()) {
                continue;
            }
            entry.type = "average";
            entry.runStamp = match.captured(1);
            entry.timeStep = match.captured(2).toLongLong();
        }
        entry.filePath = file.absoluteFilePath();
        entry.sortKey = sortKeyFor(entry.runStamp, entry.timeStep);
        entries.push_back(entry);
    }

    std::sort(entries.begin(), entries.end(), [](const ResultEntry& a, const ResultEntry& b) {
        if (a.sortKey != b.sortKey) {
            return a.sortKey > b.sortKey;
        }
        return a.filePath > b.filePath;
    });
    return entries;
}

void VtkViewWidget::repopulateTimeCombo(bool preserveSelection, bool autoLoadSelection) {
    const QString selectedType = resultTypeCombo_->currentData().toString();
    const QString previousFile = preserveSelection ? QFileInfo(currentFile_).absoluteFilePath() : QString();

    QSignalBlocker blocker(resultTimeCombo_);
    blockResultSelection_ = true;
    resultTimeCombo_->clear();
    for (const ResultEntry& entry : resultEntries_) {
        if (entry.type != selectedType) {
            continue;
        }
        const QString label = entry.runStamp.isEmpty()
            ? QStringLiteral("t=%1").arg(entry.timeStep)
            : QStringLiteral("%1 | t=%2").arg(entry.runStamp, QString::number(entry.timeStep));
        resultTimeCombo_->addItem(label, entry.filePath);
    }

    int targetIndex = -1;
    if (!previousFile.isEmpty()) {
        targetIndex = resultTimeCombo_->findData(previousFile);
    }
    if (targetIndex < 0 && resultTimeCombo_->count() > 0) {
        targetIndex = 0;
    }
    if (targetIndex >= 0) {
        resultTimeCombo_->setCurrentIndex(targetIndex);
    }
    blockResultSelection_ = false;

    if (autoLoadSelection && targetIndex >= 0) {
        QString error;
        loadResultEntry(currentSelectedEntry(), &error);
        if (!error.isEmpty()) {
            emit statusMessage(error);
        }
    }
}

bool VtkViewWidget::loadResultEntry(const ResultEntry& entry, QString* errorMessage) {
    if (entry.filePath.isEmpty()) {
        if (errorMessage) {
            *errorMessage = "No VTK result is available for the selected field/time.";
        }
        return false;
    }

    {
        const QSignalBlocker typeBlocker(resultTypeCombo_);
        const int typeIndex = resultTypeCombo_->findData(entry.type);
        if (typeIndex >= 0) {
            resultTypeCombo_->setCurrentIndex(typeIndex);
        }
    }
    repopulateTimeCombo(true, false);
    {
        const QSignalBlocker timeBlocker(resultTimeCombo_);
        const int timeIndex = resultTimeCombo_->findData(QFileInfo(entry.filePath).absoluteFilePath());
        if (timeIndex >= 0) {
            resultTimeCombo_->setCurrentIndex(timeIndex);
        }
    }
    return loadFile(entry.filePath, errorMessage);
}

VtkViewWidget::ResultEntry VtkViewWidget::latestEntryForType(const QString& type) const {
    for (const ResultEntry& entry : resultEntries_) {
        if (entry.type == type) {
            return entry;
        }
    }
    return {};
}

VtkViewWidget::ResultEntry VtkViewWidget::currentSelectedEntry() const {
    const QString filePath = resultTimeCombo_->currentData().toString();
    const auto it = std::find_if(resultEntries_.cbegin(), resultEntries_.cend(), [&filePath](const ResultEntry& entry) {
        return QFileInfo(entry.filePath).absoluteFilePath() == QFileInfo(filePath).absoluteFilePath();
    });
    return it != resultEntries_.cend() ? *it : ResultEntry{};
}

} // namespace luwgui
