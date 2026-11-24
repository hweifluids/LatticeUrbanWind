#include "luwgui/VtkViewWidget.h"

#include "ColorMapCatalog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSlider>
#include <QSplitter>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

namespace luwgui {

namespace {

constexpr int kObjectIdRole = Qt::UserRole + 17;

QDoubleSpinBox* createSpin(double minValue, double maxValue, double step, double value, int decimals = 4) {
    auto* spin = new QDoubleSpinBox();
    spin->setRange(minValue, maxValue);
    spin->setSingleStep(step);
    spin->setDecimals(decimals);
    spin->setValue(value);
    spin->setAccelerated(true);
    return spin;
}

QString normalizedAbsolutePath(const QString& path) {
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

bool supportedDisplaySuffix(const QString& suffix) {
    static const std::set<QString> supported{
        QStringLiteral("vtk"),
        QStringLiteral("vti"),
        QStringLiteral("vtu"),
        QStringLiteral("vts"),
        QStringLiteral("vtp"),
        QStringLiteral("vtr"),
        QStringLiteral("pvd"),
        QStringLiteral("stl"),
    };
    return supported.contains(suffix.toLower());
}

QString classifyResultFile(const QFileInfo& file, const QString& projectDirectory) {
    const QString suffix = file.suffix().toLower();
    if (suffix == QStringLiteral("stl")) {
        return QStringLiteral("geometry");
    }

    const QString name = file.fileName().toLower();
    const QString path = QDir(projectDirectory).relativeFilePath(file.absoluteFilePath()).toLower();
    if (name.contains(QStringLiteral("_avg"))
        || name.contains(QStringLiteral("avg-"))
        || name.contains(QStringLiteral("average"))
        || path.contains(QStringLiteral("season_average"))) {
        return QStringLiteral("average");
    }
    if (name.contains(QStringLiteral("_raw_"))
        || name.contains(QStringLiteral("raw_"))
        || QRegularExpression(R"((?:^|[_\-])u-\d+\.)").match(name).hasMatch()) {
        return QStringLiteral("transient");
    }
    return QStringLiteral("derived");
}

qlonglong parsedTimeStep(const QString& fileName) {
    const QRegularExpressionMatch match = QRegularExpression(R"((?:-|_)(\d+)(?:_cropped)?\.[^.]+$)").match(fileName);
    return match.hasMatch() ? match.captured(1).toLongLong() : -1;
}

QString parsedRunStamp(const QString& fileName) {
    const QRegularExpressionMatch match = QRegularExpression(R"((\d{14}))").match(fileName);
    return match.hasMatch() ? match.captured(1) : QString();
}

qlonglong resultSortKey(const QFileInfo& file, const QString& runStamp, qlonglong timeStep) {
    const QDateTime parsed = QDateTime::fromString(runStamp, QStringLiteral("yyyyMMddhhmmss"));
    if (parsed.isValid()) {
        return parsed.toSecsSinceEpoch() * 1000000000LL + std::max<qlonglong>(timeStep, 0);
    }
    return file.lastModified().toMSecsSinceEpoch();
}

QStringList displayFileNameFilters() {
    return {
        QStringLiteral("*.vtk"),
        QStringLiteral("*.vti"),
        QStringLiteral("*.vtu"),
        QStringLiteral("*.vts"),
        QStringLiteral("*.vtp"),
        QStringLiteral("*.vtr"),
        QStringLiteral("*.pvd"),
        QStringLiteral("*.stl"),
    };
}

QVector<double> parseNumberList(const QString& text) {
    QVector<double> values;
    const QStringList parts = text.split(QRegularExpression(QStringLiteral(R"([,;\s]+)")), Qt::SkipEmptyParts);
    for (const QString& part : parts) {
        bool ok = false;
        const double value = part.toDouble(&ok);
        if (ok && std::isfinite(value)) {
            values.push_back(value);
        }
    }
    return values;
}

QString formatNumberList(const QVector<double>& values) {
    QStringList parts;
    for (double value : values) {
        parts.push_back(QString::number(value, 'g', 8));
    }
    return parts.join(QStringLiteral(", "));
}

} // namespace

VtkViewWidget::VtkViewWidget(QWidget* parent)
    : QWidget(parent) {
    Streamcenter::setActiveColorMapsFile(Streamcenter::defaultColorMapsFilePath());
    buildUi();
    populateColorMaps();
}

void VtkViewWidget::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(6);

    auto* toolbar = new QWidget(this);
    auto* toolbarLayout = new QGridLayout(toolbar);
    toolbarLayout->setContentsMargins(0, 0, 0, 0);
    toolbarLayout->setHorizontalSpacing(6);
    toolbarLayout->setVerticalSpacing(4);

    auto* refreshButton = new QPushButton(tr("Refresh Results"), toolbar);
    auto* loadButton = new QPushButton(tr("Load"), toolbar);
    auto* reloadButton = new QPushButton(tr("Reload"), toolbar);
    auto* resetButton = new QPushButton(tr("Reset Camera"), toolbar);
    auto* saveButton = new QPushButton(tr("Save Image"), toolbar);

    resultTypeCombo_ = new QComboBox(toolbar);
    resultTypeCombo_->addItem(tr("All"), QStringLiteral("all"));
    resultTypeCombo_->addItem(tr("Average"), QStringLiteral("average"));
    resultTypeCombo_->addItem(tr("Transient"), QStringLiteral("transient"));
    resultTypeCombo_->addItem(tr("Derived"), QStringLiteral("derived"));
    resultTypeCombo_->addItem(tr("Geometry"), QStringLiteral("geometry"));

    resultFileCombo_ = new QComboBox(toolbar);
    resultFileCombo_->setMinimumContentsLength(42);
    resultFileCombo_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);

    fileLabel_ = new QLabel(tr("No dataset loaded"), toolbar);
    fileLabel_->setProperty("muted", true);
    resultInfoLabel_ = new QLabel(toolbar);
    resultInfoLabel_->setProperty("muted", true);

    toolbarLayout->addWidget(refreshButton, 0, 0);
    toolbarLayout->addWidget(loadButton, 0, 1);
    toolbarLayout->addWidget(reloadButton, 0, 2);
    toolbarLayout->addWidget(resetButton, 0, 3);
    toolbarLayout->addWidget(saveButton, 0, 4);
    toolbarLayout->addWidget(new QLabel(tr("Result"), toolbar), 0, 5);
    toolbarLayout->addWidget(resultTypeCombo_, 0, 6);
    toolbarLayout->addWidget(resultFileCombo_, 0, 7, 1, 3);
    toolbarLayout->addWidget(fileLabel_, 1, 0, 1, 5);
    toolbarLayout->addWidget(resultInfoLabel_, 1, 5, 1, 5);
    toolbarLayout->setColumnStretch(9, 1);
    root->addWidget(toolbar);

    auto* splitter = new QSplitter(Qt::Horizontal, this);
    viewer_ = new ViewerWidget(splitter);
    viewer_->setCanvasBorderVisible(false);
    viewer_->setParallelProjection(true);
    splitter->addWidget(viewer_);

    auto* sidePanel = new QWidget(splitter);
    auto* sideLayout = new QVBoxLayout(sidePanel);
    sideLayout->setContentsMargins(8, 0, 0, 0);
    sideLayout->setSpacing(8);

    auto* objectGroup = new QGroupBox(tr("Objects and Filters"), sidePanel);
    auto* objectLayout = new QVBoxLayout(objectGroup);
    objectTree_ = new QTreeWidget(objectGroup);
    objectTree_->setColumnCount(2);
    objectTree_->setHeaderLabels({tr("Object"), tr("Type")});
    objectTree_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    objectTree_->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    objectTree_->setMinimumHeight(170);
    objectLayout->addWidget(objectTree_);

    auto* objectButtonGrid = new QGridLayout();
    addDataButton_ = new QPushButton(tr("Data"), objectGroup);
    addGeometryButton_ = new QPushButton(tr("Geometry"), objectGroup);
    addClipButton_ = new QPushButton(tr("Clip"), objectGroup);
    addSliceButton_ = new QPushButton(tr("Slice"), objectGroup);
    addCropButton_ = new QPushButton(tr("Crop"), objectGroup);
    addContourButton_ = new QPushButton(tr("Contour"), objectGroup);
    addVolumeButton_ = new QPushButton(tr("Volume"), objectGroup);
    removeObjectButton_ = new QPushButton(tr("Remove"), objectGroup);
    objectButtonGrid->addWidget(addDataButton_, 0, 0);
    objectButtonGrid->addWidget(addGeometryButton_, 0, 1);
    objectButtonGrid->addWidget(addClipButton_, 1, 0);
    objectButtonGrid->addWidget(addSliceButton_, 1, 1);
    objectButtonGrid->addWidget(addCropButton_, 2, 0);
    objectButtonGrid->addWidget(addContourButton_, 2, 1);
    objectButtonGrid->addWidget(addVolumeButton_, 3, 0);
    objectButtonGrid->addWidget(removeObjectButton_, 3, 1);
    objectLayout->addLayout(objectButtonGrid);
    sideLayout->addWidget(objectGroup);

    auto* scroll = new QScrollArea(sidePanel);
    scroll->setWidgetResizable(true);
    auto* propertyPanel = new QWidget(scroll);
    auto* propertyLayout = new QVBoxLayout(propertyPanel);
    propertyLayout->setContentsMargins(0, 0, 0, 0);
    propertyLayout->setSpacing(8);

    auto* appearanceGroup = new QGroupBox(tr("Appearance"), propertyPanel);
    auto* appearanceForm = new QFormLayout(appearanceGroup);
    visibleCheck_ = new QCheckBox(tr("Visible"), appearanceGroup);
    surfaceCheck_ = new QCheckBox(tr("Surface"), appearanceGroup);
    outlineCheck_ = new QCheckBox(tr("Outline"), appearanceGroup);
    meshCheck_ = new QCheckBox(tr("Mesh"), appearanceGroup);
    auto* visibilityRow = new QWidget(appearanceGroup);
    auto* visibilityLayout = new QHBoxLayout(visibilityRow);
    visibilityLayout->setContentsMargins(0, 0, 0, 0);
    visibilityLayout->addWidget(visibleCheck_);
    visibilityLayout->addWidget(surfaceCheck_);
    visibilityLayout->addWidget(outlineCheck_);
    visibilityLayout->addWidget(meshCheck_);
    appearanceForm->addRow(tr("Draw"), visibilityRow);

    colorModeCombo_ = new QComboBox(appearanceGroup);
    colorModeCombo_->addItems({tr("Field"), tr("Solid color")});
    fieldCombo_ = new QComboBox(appearanceGroup);
    componentCombo_ = new QComboBox(appearanceGroup);
    colorMapCombo_ = new QComboBox(appearanceGroup);
    opacitySlider_ = new QSlider(Qt::Horizontal, appearanceGroup);
    opacitySlider_->setRange(1, 100);
    opacitySlider_->setValue(100);
    autoRangeCheck_ = new QCheckBox(tr("Use data range"), appearanceGroup);
    autoRangeCheck_->setChecked(true);
    rangeMinSpin_ = createSpin(-1.0e12, 1.0e12, 0.1, 0.0, 6);
    rangeMaxSpin_ = createSpin(-1.0e12, 1.0e12, 0.1, 1.0, 6);

    appearanceForm->addRow(tr("Color"), colorModeCombo_);
    appearanceForm->addRow(tr("Field"), fieldCombo_);
    appearanceForm->addRow(tr("Component"), componentCombo_);
    appearanceForm->addRow(tr("Color map"), colorMapCombo_);
    appearanceForm->addRow(tr("Opacity"), opacitySlider_);
    appearanceForm->addRow(QString(), autoRangeCheck_);
    appearanceForm->addRow(tr("Range min"), rangeMinSpin_);
    appearanceForm->addRow(tr("Range max"), rangeMaxSpin_);
    propertyLayout->addWidget(appearanceGroup);

    auto* filterGroup = new QGroupBox(tr("Filter Parameters"), propertyPanel);
    auto* filterLayout = new QGridLayout(filterGroup);
    const QStringList axisLabels{QStringLiteral("X"), QStringLiteral("Y"), QStringLiteral("Z")};
    filterLayout->addWidget(new QLabel(tr("Plane origin"), filterGroup), 0, 0);
    filterLayout->addWidget(new QLabel(tr("Plane normal"), filterGroup), 1, 0);
    for (int axis = 0; axis < 3; ++axis) {
        planeOriginSpin_[axis] = createSpin(-1.0e12, 1.0e12, 0.1, 0.0, 4);
        planeNormalSpin_[axis] = createSpin(-1.0, 1.0, 0.1, axis == 0 ? 1.0 : 0.0, 4);
        filterLayout->addWidget(new QLabel(axisLabels.at(axis), filterGroup), 0, 1 + axis * 2);
        filterLayout->addWidget(planeOriginSpin_[axis], 0, 2 + axis * 2);
        filterLayout->addWidget(new QLabel(axisLabels.at(axis), filterGroup), 1, 1 + axis * 2);
        filterLayout->addWidget(planeNormalSpin_[axis], 1, 2 + axis * 2);
    }

    const QStringList boundLabels{tr("X min"), tr("X max"), tr("Y min"), tr("Y max"), tr("Z min"), tr("Z max")};
    for (int index = 0; index < 6; ++index) {
        cropBoundsSpin_[index] = createSpin(-1.0e12, 1.0e12, 0.1, index % 2 == 0 ? 0.0 : 1.0, 4);
        const int row = 2 + index / 2;
        const int col = (index % 2) * 3;
        filterLayout->addWidget(new QLabel(boundLabels.at(index), filterGroup), row, col);
        filterLayout->addWidget(cropBoundsSpin_[index], row, col + 1, 1, 2);
    }
    contourValuesEdit_ = new QLineEdit(filterGroup);
    filterLayout->addWidget(new QLabel(tr("Contour values"), filterGroup), 5, 0);
    filterLayout->addWidget(contourValuesEdit_, 5, 1, 1, 6);
    propertyLayout->addWidget(filterGroup);

    auto* applyButton = new QPushButton(tr("Apply Object"), propertyPanel);
    propertyLayout->addWidget(applyButton);
    propertyLayout->addStretch(1);
    scroll->setWidget(propertyPanel);
    sideLayout->addWidget(scroll, 1);

    splitter->addWidget(sidePanel);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 0);
    splitter->setSizes({1100, 330});
    root->addWidget(splitter, 1);

    connect(refreshButton, &QPushButton::clicked, this, &VtkViewWidget::refreshResultCatalog);
    connect(loadButton, &QPushButton::clicked, this, [this] {
        QString error;
        if (!loadResultEntry(currentSelectedEntry(), &error) && !error.isEmpty()) {
            emit statusMessage(error);
        }
    });
    connect(reloadButton, &QPushButton::clicked, this, &VtkViewWidget::reloadCurrentFile);
    connect(resetButton, &QPushButton::clicked, this, &VtkViewWidget::resetCamera);
    connect(saveButton, &QPushButton::clicked, this, &VtkViewWidget::saveImage);
    connect(resultTypeCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this] {
        if (!blockUiUpdates_) {
            repopulateResultCombo(true, true);
        }
    });
    connect(resultFileCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this] {
        if (blockUiUpdates_) {
            return;
        }
        const ResultEntry entry = currentSelectedEntry();
        resultInfoLabel_->setText(resultStatusText(entry));
        QString error;
        if (!entry.filePath.isEmpty() && !loadResultEntry(entry, &error) && !error.isEmpty()) {
            emit statusMessage(error);
        }
    });

    connect(objectTree_, &QTreeWidget::itemSelectionChanged, this, &VtkViewWidget::loadSelectedObjectIntoControls);
    connect(addDataButton_, &QPushButton::clicked, this, [this] { addObject(ObjectType::Data); });
    connect(addGeometryButton_, &QPushButton::clicked, this, [this] { addObject(ObjectType::Geometry); });
    connect(addClipButton_, &QPushButton::clicked, this, [this] { addObject(ObjectType::Clip); });
    connect(addSliceButton_, &QPushButton::clicked, this, [this] { addObject(ObjectType::Slice); });
    connect(addCropButton_, &QPushButton::clicked, this, [this] { addObject(ObjectType::Crop); });
    connect(addContourButton_, &QPushButton::clicked, this, [this] { addObject(ObjectType::Contour); });
    connect(addVolumeButton_, &QPushButton::clicked, this, [this] { addObject(ObjectType::RayTracingVolume); });
    connect(removeObjectButton_, &QPushButton::clicked, this, &VtkViewWidget::removeSelectedObject);
    connect(applyButton, &QPushButton::clicked, this, &VtkViewWidget::applySelectedObjectControls);

    auto applyChange = [this] {
        if (!blockUiUpdates_) {
            applySelectedObjectControls();
        }
    };
    connect(visibleCheck_, &QCheckBox::toggled, this, applyChange);
    connect(surfaceCheck_, &QCheckBox::toggled, this, applyChange);
    connect(outlineCheck_, &QCheckBox::toggled, this, applyChange);
    connect(meshCheck_, &QCheckBox::toggled, this, applyChange);
    connect(colorModeCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, applyChange);
    connect(componentCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, applyChange);
    connect(colorMapCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, applyChange);
    connect(opacitySlider_, &QSlider::valueChanged, this, applyChange);
    connect(autoRangeCheck_, &QCheckBox::toggled, this, [this, applyChange](bool checked) {
        rangeMinSpin_->setEnabled(!checked);
        rangeMaxSpin_->setEnabled(!checked);
        applyChange();
    });
    connect(rangeMinSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, applyChange);
    connect(rangeMaxSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, applyChange);
    connect(fieldCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this] {
        if (blockUiUpdates_) {
            return;
        }
        const QString previous = componentCombo_->currentText();
        const QStringList components = fieldCombo_->currentData().toStringList();
        const QSignalBlocker blocker(componentCombo_);
        componentCombo_->clear();
        componentCombo_->addItems(components.isEmpty() ? QStringList{QStringLiteral("Magnitude")} : components);
        const int previousIndex = componentCombo_->findText(previous);
        componentCombo_->setCurrentIndex(previousIndex >= 0 ? previousIndex : 0);
        applySelectedObjectControls();
    });
    for (QDoubleSpinBox* spin : planeOriginSpin_) {
        connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, applyChange);
    }
    for (QDoubleSpinBox* spin : planeNormalSpin_) {
        connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, applyChange);
    }
    for (QDoubleSpinBox* spin : cropBoundsSpin_) {
        connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, applyChange);
    }
    connect(contourValuesEdit_, &QLineEdit::editingFinished, this, &VtkViewWidget::applySelectedObjectControls);
}

void VtkViewWidget::populateColorMaps() {
    const QSignalBlocker blocker(colorMapCombo_);
    const QString previous = colorMapCombo_->currentText();
    colorMapCombo_->clear();
    const QVector<Streamcenter::ColorMapDefinition> maps = Streamcenter::activeColorMaps();
    for (const Streamcenter::ColorMapDefinition& definition : maps) {
        colorMapCombo_->addItem(definition.name);
    }
    if (colorMapCombo_->count() == 0) {
        colorMapCombo_->addItems({QStringLiteral("Cool to Warm"), QStringLiteral("Inferno"), QStringLiteral("Viridis")});
    }
    const int previousIndex = colorMapCombo_->findText(previous);
    colorMapCombo_->setCurrentIndex(previousIndex >= 0 ? previousIndex : 0);
}

bool VtkViewWidget::loadFile(const QString& filePath, QString* errorMessage) {
    const QFileInfo info(filePath);
    if (!info.exists() || !info.isFile() || !supportedDisplaySuffix(info.suffix())) {
        if (errorMessage) {
            *errorMessage = tr("Unsupported or missing visualization file: %1").arg(filePath);
        }
        return false;
    }

    if (!resetSceneForFile(info.absoluteFilePath(), errorMessage)) {
        return false;
    }

    currentFile_ = info.absoluteFilePath();
    fileLabel_->setText(relativeProjectPath(currentFile_));
    emit statusMessage(tr("Loaded %1").arg(currentFile_));
    emit fileLoaded(currentFile_);

    const auto matchIt = std::find_if(resultEntries_.cbegin(), resultEntries_.cend(), [this](const ResultEntry& entry) {
        return normalizedAbsolutePath(entry.filePath) == normalizedAbsolutePath(currentFile_);
    });
    if (matchIt != resultEntries_.cend()) {
        const QSignalBlocker typeBlocker(resultTypeCombo_);
        const int typeIndex = resultTypeCombo_->findData(matchIt->type);
        if (typeIndex >= 0) {
            resultTypeCombo_->setCurrentIndex(typeIndex);
        }
        repopulateResultCombo(true, false);
        const QSignalBlocker fileBlocker(resultFileCombo_);
        const int fileIndex = resultFileCombo_->findData(normalizedAbsolutePath(currentFile_));
        if (fileIndex >= 0) {
            resultFileCombo_->setCurrentIndex(fileIndex);
        }
        resultInfoLabel_->setText(resultStatusText(*matchIt));
    }
    return true;
}

void VtkViewWidget::setProjectDirectory(const QString& projectDirectory) {
    const QString normalized = normalizedAbsolutePath(projectDirectory);
    if (projectDirectory_ == normalized) {
        return;
    }
    projectDirectory_ = normalized;
    Streamcenter::setActiveColorMapsFile(Streamcenter::defaultColorMapsFilePath());
    populateColorMaps();
    refreshResultCatalog();
}

QString VtkViewWidget::currentFile() const {
    return currentFile_;
}

bool VtkViewWidget::loadLatestResult(QString* errorMessage) {
    if (resultEntries_.isEmpty()) {
        resultEntries_ = scanResultEntries();
        repopulateResultCombo(true, false);
    }

    ResultEntry entry = latestEntryForType(QStringLiteral("average"));
    if (entry.filePath.isEmpty()) {
        entry = latestEntryForType(QStringLiteral("transient"));
    }
    if (entry.filePath.isEmpty()) {
        entry = latestEntryForType(QStringLiteral("derived"));
    }
    if (entry.filePath.isEmpty() && !resultEntries_.isEmpty()) {
        entry = resultEntries_.front();
    }
    if (entry.filePath.isEmpty()) {
        if (errorMessage) {
            *errorMessage = tr("No displayable VTK/VTI/PVD/STL result was found under RESULTS or proj_temp/vtk.");
        }
        return false;
    }
    return loadResultEntry(entry, errorMessage);
}

void VtkViewWidget::refreshResultCatalog() {
    resultEntries_ = scanResultEntries();
    repopulateResultCombo(true, false);
    if (resultEntries_.isEmpty()) {
        emit statusMessage(tr("No displayable result was found under RESULTS or proj_temp/vtk."));
    } else {
        emit statusMessage(tr("Found %1 displayable result file(s).").arg(resultEntries_.size()));
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
    if (viewer_) {
        viewer_->resetCameraToScene();
        viewer_->renderWhenVisible();
    }
}

void VtkViewWidget::reloadCurrentFile() {
    if (currentFile_.isEmpty()) {
        return;
    }
    QString error;
    if (!loadFile(currentFile_, &error) && !error.isEmpty()) {
        emit statusMessage(error);
    }
}

void VtkViewWidget::saveImage() {
    if (!viewer_ || !viewer_->hasScene()) {
        emit statusMessage(tr("Load a result before saving an image."));
        return;
    }
    QString suggested = currentFile_.isEmpty()
        ? QStringLiteral("luw_view.png")
        : QFileInfo(currentFile_).completeBaseName() + QStringLiteral("_view.png");
    QString path = QFileDialog::getSaveFileName(this, tr("Save View Image"), suggested, tr("PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)"));
    if (path.isEmpty()) {
        return;
    }
    if (QFileInfo(path).suffix().isEmpty()) {
        path += QStringLiteral(".png");
    }
    QString error;
    if (!viewer_->saveScreenshot(path, &error)) {
        emit statusMessage(error.isEmpty() ? tr("Failed to save current view image.") : error);
        return;
    }
    emit statusMessage(tr("Saved current view image to %1").arg(path));
}

QList<VtkViewWidget::ResultEntry> VtkViewWidget::scanResultEntries() const {
    QList<ResultEntry> entries;
    if (projectDirectory_.isEmpty() || !QFileInfo(projectDirectory_).isDir()) {
        return entries;
    }

    std::set<QString> seen;
    auto scanDirectory = [&](const QString& directoryPath, bool recursive) {
        const QDir directory(directoryPath);
        if (!directory.exists()) {
            return;
        }
        const auto flags = recursive ? QDirIterator::Subdirectories : QDirIterator::NoIteratorFlags;
        QDirIterator it(directory.absolutePath(), displayFileNameFilters(), QDir::Files, flags);
        while (it.hasNext()) {
            const QFileInfo file(it.next());
            const QString absolute = normalizedAbsolutePath(file.absoluteFilePath());
            if (seen.contains(absolute)) {
                continue;
            }
            seen.insert(absolute);
            ResultEntry entry;
            entry.filePath = absolute;
            entry.type = classifyResultFile(file, projectDirectory_);
            entry.runStamp = parsedRunStamp(file.fileName());
            entry.timeStep = parsedTimeStep(file.fileName());
            entry.modifiedMs = file.lastModified().toMSecsSinceEpoch();
            entry.sortKey = resultSortKey(file, entry.runStamp, entry.timeStep);
            entry.sourceLabel = relativeProjectPath(absolute);
            entries.push_back(entry);
        }
    };

    scanDirectory(QDir(projectDirectory_).filePath(QStringLiteral("RESULTS")), true);
    scanDirectory(QDir(projectDirectory_).filePath(QStringLiteral("proj_temp/vtk")), true);

    std::sort(entries.begin(), entries.end(), [](const ResultEntry& lhs, const ResultEntry& rhs) {
        if (lhs.sortKey != rhs.sortKey) {
            return lhs.sortKey > rhs.sortKey;
        }
        return lhs.filePath > rhs.filePath;
    });
    return entries;
}

void VtkViewWidget::repopulateResultCombo(bool preserveSelection, bool autoLoadSelection) {
    const QString selectedType = resultTypeCombo_->currentData().toString();
    const QString previousFile = preserveSelection ? normalizedAbsolutePath(currentFile_) : QString();
    QSignalBlocker blocker(resultFileCombo_);
    blockUiUpdates_ = true;
    resultFileCombo_->clear();

    for (const ResultEntry& entry : resultEntries_) {
        if (selectedType != QStringLiteral("all") && entry.type != selectedType) {
            continue;
        }
        QString label = QStringLiteral("[%1] %2").arg(resultTypeLabel(entry.type), relativeProjectPath(entry.filePath));
        if (entry.timeStep >= 0) {
            label += QStringLiteral(" | t=%1").arg(entry.timeStep);
        }
        resultFileCombo_->addItem(label, normalizedAbsolutePath(entry.filePath));
    }

    int targetIndex = -1;
    if (!previousFile.isEmpty()) {
        targetIndex = resultFileCombo_->findData(previousFile);
    }
    if (targetIndex < 0 && resultFileCombo_->count() > 0) {
        targetIndex = 0;
    }
    if (targetIndex >= 0) {
        resultFileCombo_->setCurrentIndex(targetIndex);
    }
    blockUiUpdates_ = false;

    const ResultEntry entry = currentSelectedEntry();
    resultInfoLabel_->setText(resultStatusText(entry));
    if (autoLoadSelection && targetIndex >= 0 && !entry.filePath.isEmpty()) {
        QString error;
        if (!loadResultEntry(entry, &error) && !error.isEmpty()) {
            emit statusMessage(error);
        }
    }
}

bool VtkViewWidget::loadResultEntry(const ResultEntry& entry, QString* errorMessage) {
    if (entry.filePath.isEmpty()) {
        if (errorMessage) {
            *errorMessage = tr("No result is available for the selected item.");
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
    repopulateResultCombo(true, false);
    {
        const QSignalBlocker fileBlocker(resultFileCombo_);
        const int fileIndex = resultFileCombo_->findData(normalizedAbsolutePath(entry.filePath));
        if (fileIndex >= 0) {
            resultFileCombo_->setCurrentIndex(fileIndex);
        }
    }
    resultInfoLabel_->setText(resultStatusText(entry));
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
    const QString filePath = resultFileCombo_->currentData().toString();
    const auto it = std::find_if(resultEntries_.cbegin(), resultEntries_.cend(), [&filePath](const ResultEntry& entry) {
        return normalizedAbsolutePath(entry.filePath) == normalizedAbsolutePath(filePath);
    });
    return it != resultEntries_.cend() ? *it : ResultEntry{};
}

bool VtkViewWidget::resetSceneForFile(const QString& filePath, QString* errorMessage) {
    ViewerWidget::DisplayOptions displayOptions;
    displayOptions.backgroundColor = QColor(255, 255, 255);
    displayOptions.showAxes = true;
    displayOptions.showLogo = true;
    displayOptions.lightingEnabled = true;
    displayOptions.lightingIntensity = 0.80;
    displayOptions.lightingMode = QStringLiteral("Headlight");
    displayOptions.perspectiveEnabled = false;

    viewer_->beginDisplay(displayOptions);
    objects_.clear();
    nextObjectSerial_ = 1;

    SceneObject base = makeBaseObject(filePath, errorMessage);
    if (base.id.isEmpty()) {
        viewer_->clearScene();
        return false;
    }
    objects_.push_back(base);
    if (!applyObject(base, true, errorMessage)) {
        viewer_->clearScene();
        objects_.clear();
        refreshObjectTree();
        return false;
    }

    currentFile_ = normalizedAbsolutePath(filePath);
    addProjectGeometryOverlays();
    refreshObjectTree(base.id);
    viewer_->resetCameraToScene();
    viewer_->renderWhenVisible();
    return true;
}

void VtkViewWidget::addProjectGeometryOverlays() {
    if (projectDirectory_.isEmpty()) {
        return;
    }

    std::set<QString> geometryPaths;
    for (const QString& subdirectory : {QStringLiteral("proj_temp"), QStringLiteral("building_db")}) {
        const QString root = QDir(projectDirectory_).filePath(subdirectory);
        if (!QFileInfo(root).isDir()) {
            continue;
        }
        QDirIterator it(root, {QStringLiteral("*.stl")}, QDir::Files, QDirIterator::Subdirectories);
        while (it.hasNext()) {
            const QString absolute = normalizedAbsolutePath(it.next());
            if (normalizedAbsolutePath(currentFile_) != absolute) {
                geometryPaths.insert(absolute);
            }
        }
    }

    for (const QString& path : geometryPaths) {
        SceneObject object;
        object.id = QStringLiteral("geometry_%1").arg(nextObjectSerial_++);
        object.name = QFileInfo(path).completeBaseName();
        object.inputPath = path;
        object.type = ObjectType::Geometry;
        object.visible = true;
        object.showOutline = true;
        object.showSurface = true;
        object.showMesh = false;
        object.opacity = 0.26;
        object.colorMode = QStringLiteral("Solid color");
        applyObjectTypeDefaults(&object);
        objects_.push_back(object);
        QString ignoredError;
        if (!applyObject(object, false, &ignoredError)) {
            objects_.removeLast();
        }
    }
}

VtkViewWidget::SceneObject VtkViewWidget::makeBaseObject(const QString& filePath, QString* errorMessage) const {
    SceneObject object;
    object.id = QStringLiteral("data_%1").arg(nextObjectSerial_);
    object.name = QFileInfo(filePath).completeBaseName();
    object.inputPath = normalizedAbsolutePath(filePath);
    object.type = QFileInfo(filePath).suffix().compare(QStringLiteral("stl"), Qt::CaseInsensitive) == 0
        ? ObjectType::Geometry
        : ObjectType::Data;
    applyObjectTypeDefaults(&object);

    if (object.type != ObjectType::Geometry) {
        QVector<ViewerWidget::FieldOption> fields;
        QString preferredField;
        QString preferredComponent;
        QString probeError;
        if (viewer_->probeFieldOptions(filePath, &fields, &preferredField, &preferredComponent, &probeError)) {
            object.colorField = preferredField;
            object.colorComponent = preferredComponent.isEmpty() ? QStringLiteral("Magnitude") : preferredComponent;
        } else if (errorMessage) {
            *errorMessage = probeError;
            object.id.clear();
        }
    }
    return object;
}

void VtkViewWidget::applyObjectTypeDefaults(SceneObject* object) const {
    if (!object) {
        return;
    }
    object->colorMap = defaultColorMapName();
    if (object->type == ObjectType::Geometry) {
        object->colorMode = QStringLiteral("Solid color");
        object->showSurface = true;
        object->showOutline = true;
        object->showMesh = false;
        return;
    }
    object->colorMode = QStringLiteral("Field");
    object->colorComponent = QStringLiteral("Magnitude");
    object->showSurface = true;
    object->showOutline = object->type == ObjectType::Data;
    object->showMesh = false;
    if (object->type == ObjectType::Contour) {
        object->showOutline = false;
    }
    if (object->type == ObjectType::RayTracingVolume) {
        object->showSurface = false;
        object->showOutline = true;
        object->showMesh = false;
    }
}

void VtkViewWidget::applyBoundsDefaultsFromParent(SceneObject* object) {
    if (!object || object->parentId.isEmpty()) {
        return;
    }
    const SceneObject* parent = findObject(object->parentId);
    if (!parent) {
        return;
    }
    double bounds[6] = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    QString ignoredError;
    if (!viewer_->dataObjectBounds(parent->id, viewerOptionsFor(*parent), bounds, &ignoredError)) {
        return;
    }
    for (int axis = 0; axis < 3; ++axis) {
        object->planeOrigin[axis] = 0.5 * (bounds[axis * 2] + bounds[axis * 2 + 1]);
        object->cropBounds[axis * 2] = bounds[axis * 2];
        object->cropBounds[axis * 2 + 1] = bounds[axis * 2 + 1];
    }
    object->planeNormal[0] = 1.0;
    object->planeNormal[1] = 0.0;
    object->planeNormal[2] = 0.0;

    if (object->colorField.isEmpty()) {
        object->colorField = parent->colorField;
        object->colorComponent = parent->colorComponent;
    }
    if (object->type == ObjectType::Contour && object->contourValues.isEmpty()) {
        double minValue = 0.0;
        double maxValue = 1.0;
        ViewerWidget::DataObjectOptions options = viewerOptionsFor(*object);
        if (viewer_->dataObjectScalarRange(object->id, options, &minValue, &maxValue, nullptr)) {
            object->contourValues.push_back(0.5 * (minValue + maxValue));
        } else {
            object->contourValues.push_back(0.0);
        }
    }
}

ViewerWidget::DataObjectOptions VtkViewWidget::viewerOptionsFor(const SceneObject& object) const {
    ViewerWidget::DataObjectOptions options;
    switch (object.type) {
    case ObjectType::Clip:
        options.type = ViewerWidget::DisplayObjectType::Clip;
        break;
    case ObjectType::Slice:
        options.type = ViewerWidget::DisplayObjectType::Slice;
        break;
    case ObjectType::Contour:
        options.type = ViewerWidget::DisplayObjectType::Contour;
        break;
    case ObjectType::RayTracingVolume:
        options.type = ViewerWidget::DisplayObjectType::RayTracingVolume;
        break;
    case ObjectType::Crop:
        options.type = ViewerWidget::DisplayObjectType::Crop;
        break;
    case ObjectType::Geometry:
        options.type = ViewerWidget::DisplayObjectType::Geometry;
        break;
    case ObjectType::Data:
    default:
        options.type = ViewerWidget::DisplayObjectType::Data;
        break;
    }

    options.inputPath = object.inputPath;
    options.sourceObjectId = object.parentId;
    options.visible = object.visible;
    options.showOutline = object.showOutline;
    options.showSurface = object.showSurface;
    options.showMesh = object.showMesh;
    options.showPlaneHandle = object.type == ObjectType::Clip || object.type == ObjectType::Slice;
    options.outlineOpacity = std::clamp(object.opacity, 0.0, 1.0);
    options.surfaceOpacity = std::clamp(object.opacity, 0.0, 1.0);
    options.meshOpacity = std::clamp(object.opacity, 0.0, 1.0);
    options.outlineColor = object.type == ObjectType::Geometry ? QColor(80, 80, 80) : QColor(0, 0, 0);
    options.surfaceColor = object.type == ObjectType::Geometry ? QColor(122, 122, 122) : QColor(178, 210, 235);
    options.meshColor = QColor(70, 70, 70);
    options.colorMode = object.colorMode;
    options.colorField = object.colorField;
    options.colorComponent = object.colorComponent.isEmpty() ? QStringLiteral("Magnitude") : object.colorComponent;
    options.colorMap = object.colorMap.isEmpty() ? defaultColorMapName() : object.colorMap;
    options.autoColorRange = object.autoColorRange;
    options.colorRangeMin = object.colorRangeMin;
    options.colorRangeMax = object.colorRangeMax;
    options.showLegend = object.visible && object.colorMode.compare(QStringLiteral("Field"), Qt::CaseInsensitive) == 0;
    for (int axis = 0; axis < 3; ++axis) {
        options.planeOrigin[axis] = object.planeOrigin[axis];
        options.planeNormal[axis] = object.planeNormal[axis];
    }
    for (int index = 0; index < 6; ++index) {
        options.cropBounds[index] = object.cropBounds[index];
    }
    options.contourField = object.colorField;
    options.contourComponent = object.colorComponent.isEmpty() ? QStringLiteral("Magnitude") : object.colorComponent;
    options.contourValues = object.contourValues;
    if (options.type == ViewerWidget::DisplayObjectType::Contour && options.contourValues.isEmpty()) {
        options.contourValues.push_back(0.0);
    }
    options.volumeOpacityScale = 2.0;
    options.volumeSamplingStep = 0.08;
    options.volumeFiltering = QStringLiteral("Linear");
    options.volumePreintegration = true;
    return options;
}

bool VtkViewWidget::applyObject(const SceneObject& object, bool resetCameraToObject, QString* errorMessage) {
    if (!viewer_) {
        return false;
    }
    return viewer_->addOrUpdateDataObject(object.id, viewerOptionsFor(object), resetCameraToObject, errorMessage);
}

bool VtkViewWidget::applyObjectAndDescendants(const QString& objectId, QString* errorMessage) {
    const SceneObject* object = findObject(objectId);
    if (!object) {
        return false;
    }
    if (!applyObject(*object, false, errorMessage)) {
        return false;
    }
    for (const SceneObject& child : std::as_const(objects_)) {
        if (child.parentId == objectId && !applyObjectAndDescendants(child.id, errorMessage)) {
            return false;
        }
    }
    return true;
}

void VtkViewWidget::removeObjectAndDescendants(const QString& objectId) {
    QList<QString> children;
    for (const SceneObject& object : std::as_const(objects_)) {
        if (object.parentId == objectId) {
            children.push_back(object.id);
        }
    }
    for (const QString& childId : children) {
        removeObjectAndDescendants(childId);
    }
    if (viewer_) {
        viewer_->removeDataObject(objectId);
    }
    for (int index = objects_.size() - 1; index >= 0; --index) {
        if (objects_.at(index).id == objectId) {
            objects_.removeAt(index);
        }
    }
}

void VtkViewWidget::refreshObjectTree(const QString& preferredObjectId) {
    const QSignalBlocker blocker(objectTree_);
    objectTree_->clear();
    appendObjectTreeItems(nullptr, QString());
    objectTree_->expandAll();

    const QString target = preferredObjectId.isEmpty() && !objects_.isEmpty() ? objects_.front().id : preferredObjectId;
    if (!target.isEmpty()) {
        const QList<QTreeWidgetItem*> items = objectTree_->findItems(QStringLiteral("*"), Qt::MatchWildcard | Qt::MatchRecursive, 0);
        for (QTreeWidgetItem* item : items) {
            if (item->data(0, kObjectIdRole).toString() == target) {
                objectTree_->setCurrentItem(item);
                break;
            }
        }
    }
    updateObjectControlAvailability();
}

void VtkViewWidget::appendObjectTreeItems(QTreeWidgetItem* parentItem, const QString& parentId) {
    for (const SceneObject& object : std::as_const(objects_)) {
        if (object.parentId != parentId) {
            continue;
        }
        auto* item = parentItem
            ? new QTreeWidgetItem(parentItem)
            : new QTreeWidgetItem(objectTree_);
        item->setText(0, object.name);
        item->setText(1, objectTypeLabel(object.type));
        item->setData(0, kObjectIdRole, object.id);
        item->setCheckState(0, object.visible ? Qt::Checked : Qt::Unchecked);
        appendObjectTreeItems(item, object.id);
    }
}

QString VtkViewWidget::selectedObjectId() const {
    QTreeWidgetItem* item = objectTree_->currentItem();
    return item ? item->data(0, kObjectIdRole).toString() : QString();
}

VtkViewWidget::SceneObject* VtkViewWidget::findObject(const QString& objectId) {
    for (SceneObject& object : objects_) {
        if (object.id == objectId) {
            return &object;
        }
    }
    return nullptr;
}

const VtkViewWidget::SceneObject* VtkViewWidget::findObject(const QString& objectId) const {
    for (const SceneObject& object : objects_) {
        if (object.id == objectId) {
            return &object;
        }
    }
    return nullptr;
}

VtkViewWidget::SceneObject* VtkViewWidget::selectedObject() {
    return findObject(selectedObjectId());
}

void VtkViewWidget::loadSelectedObjectIntoControls() {
    const SceneObject* object = findObject(selectedObjectId());
    blockUiUpdates_ = true;
    if (object) {
        updateControlsFromObject(*object);
    }
    blockUiUpdates_ = false;
    updateObjectControlAvailability();
}

void VtkViewWidget::updateControlsFromObject(const SceneObject& object) {
    populateFieldControls(&object);
    visibleCheck_->setChecked(object.visible);
    surfaceCheck_->setChecked(object.showSurface);
    outlineCheck_->setChecked(object.showOutline);
    meshCheck_->setChecked(object.showMesh);
    colorModeCombo_->setCurrentIndex(colorModeCombo_->findText(object.colorMode) >= 0
                                         ? colorModeCombo_->findText(object.colorMode)
                                         : 0);
    const int fieldIndex = fieldCombo_->findText(object.colorField);
    if (fieldIndex >= 0) {
        fieldCombo_->setCurrentIndex(fieldIndex);
    }
    const QStringList components = fieldCombo_->currentData().toStringList();
    componentCombo_->clear();
    componentCombo_->addItems(components.isEmpty() ? QStringList{QStringLiteral("Magnitude")} : components);
    const int componentIndex = componentCombo_->findText(object.colorComponent);
    componentCombo_->setCurrentIndex(componentIndex >= 0 ? componentIndex : 0);
    const int colorMapIndex = colorMapCombo_->findText(object.colorMap);
    if (colorMapIndex >= 0) {
        colorMapCombo_->setCurrentIndex(colorMapIndex);
    }
    opacitySlider_->setValue(static_cast<int>(std::round(std::clamp(object.opacity, 0.0, 1.0) * 100.0)));
    autoRangeCheck_->setChecked(object.autoColorRange);
    rangeMinSpin_->setValue(object.colorRangeMin);
    rangeMaxSpin_->setValue(object.colorRangeMax);
    rangeMinSpin_->setEnabled(!object.autoColorRange);
    rangeMaxSpin_->setEnabled(!object.autoColorRange);
    for (int axis = 0; axis < 3; ++axis) {
        planeOriginSpin_[axis]->setValue(object.planeOrigin[axis]);
        planeNormalSpin_[axis]->setValue(object.planeNormal[axis]);
    }
    for (int index = 0; index < 6; ++index) {
        cropBoundsSpin_[index]->setValue(object.cropBounds[index]);
    }
    contourValuesEdit_->setText(formatNumberList(object.contourValues));
}

bool VtkViewWidget::updateObjectFromControls(SceneObject* object) {
    if (!object) {
        return false;
    }
    object->visible = visibleCheck_->isChecked();
    object->showSurface = surfaceCheck_->isChecked();
    object->showOutline = outlineCheck_->isChecked();
    object->showMesh = meshCheck_->isChecked();
    object->opacity = static_cast<double>(opacitySlider_->value()) / 100.0;
    object->colorMode = colorModeCombo_->currentText();
    object->colorField = fieldCombo_->currentText();
    object->colorComponent = componentCombo_->currentText();
    object->colorMap = colorMapCombo_->currentText();
    object->autoColorRange = autoRangeCheck_->isChecked();
    object->colorRangeMin = rangeMinSpin_->value();
    object->colorRangeMax = rangeMaxSpin_->value();
    for (int axis = 0; axis < 3; ++axis) {
        object->planeOrigin[axis] = planeOriginSpin_[axis]->value();
        object->planeNormal[axis] = planeNormalSpin_[axis]->value();
    }
    for (int index = 0; index < 6; ++index) {
        object->cropBounds[index] = cropBoundsSpin_[index]->value();
    }
    object->contourValues = parseNumberList(contourValuesEdit_->text());
    if (object->type == ObjectType::Contour && object->contourValues.isEmpty()) {
        object->contourValues.push_back(0.0);
    }
    return true;
}

void VtkViewWidget::updateObjectControlAvailability() {
    const SceneObject* object = findObject(selectedObjectId());
    const bool hasObject = object != nullptr;
    removeObjectButton_->setEnabled(hasObject && object->parentId.isEmpty() ? objects_.size() > 1 : hasObject);
    addClipButton_->setEnabled(hasObject);
    addSliceButton_->setEnabled(hasObject);
    addCropButton_->setEnabled(hasObject);
    addContourButton_->setEnabled(hasObject && object->type != ObjectType::Geometry);
    const bool fieldControls = hasObject && object->type != ObjectType::Geometry;
    colorModeCombo_->setEnabled(hasObject);
    fieldCombo_->setEnabled(fieldControls && colorModeCombo_->currentText() == QStringLiteral("Field"));
    componentCombo_->setEnabled(fieldControls && colorModeCombo_->currentText() == QStringLiteral("Field"));
    colorMapCombo_->setEnabled(fieldControls && colorModeCombo_->currentText() == QStringLiteral("Field"));
    autoRangeCheck_->setEnabled(fieldControls);
    const bool rangeEnabled = fieldControls && !autoRangeCheck_->isChecked();
    rangeMinSpin_->setEnabled(rangeEnabled);
    rangeMaxSpin_->setEnabled(rangeEnabled);
    const bool planeEnabled = hasObject && (object->type == ObjectType::Clip || object->type == ObjectType::Slice);
    const bool cropEnabled = hasObject && object->type == ObjectType::Crop;
    const bool contourEnabled = hasObject && object->type == ObjectType::Contour;
    for (QDoubleSpinBox* spin : planeOriginSpin_) {
        spin->setEnabled(planeEnabled);
    }
    for (QDoubleSpinBox* spin : planeNormalSpin_) {
        spin->setEnabled(planeEnabled);
    }
    for (QDoubleSpinBox* spin : cropBoundsSpin_) {
        spin->setEnabled(cropEnabled);
    }
    contourValuesEdit_->setEnabled(contourEnabled);
}

void VtkViewWidget::populateFieldControls(const SceneObject* object) {
    const QString previousField = object ? object->colorField : fieldCombo_->currentText();
    const QSignalBlocker fieldBlocker(fieldCombo_);
    const QSignalBlocker componentBlocker(componentCombo_);
    fieldCombo_->clear();
    componentCombo_->clear();
    componentCombo_->addItem(QStringLiteral("Magnitude"));
    if (!object || object->type == ObjectType::Geometry) {
        return;
    }

    const QString path = effectiveInputPath(*object);
    QVector<ViewerWidget::FieldOption> fields;
    QString preferredField;
    QString preferredComponent;
    if (!path.isEmpty() && viewer_->probeFieldOptions(path, &fields, &preferredField, &preferredComponent, nullptr)) {
        for (const ViewerWidget::FieldOption& field : fields) {
            fieldCombo_->addItem(field.name, field.components);
        }
    }
    if (fieldCombo_->count() == 0 && !object->colorField.isEmpty()) {
        fieldCombo_->addItem(object->colorField, QStringList{QStringLiteral("Magnitude")});
    }
    int fieldIndex = fieldCombo_->findText(previousField);
    if (fieldIndex < 0) {
        fieldIndex = fieldCombo_->findText(preferredField);
    }
    if (fieldIndex < 0 && fieldCombo_->count() > 0) {
        fieldIndex = 0;
    }
    if (fieldIndex >= 0) {
        fieldCombo_->setCurrentIndex(fieldIndex);
        const QStringList components = fieldCombo_->currentData().toStringList();
        componentCombo_->clear();
        componentCombo_->addItems(components.isEmpty() ? QStringList{QStringLiteral("Magnitude")} : components);
    }
}

void VtkViewWidget::addObject(ObjectType type) {
    if (!viewer_->hasScene() && currentFile_.isEmpty()) {
        emit statusMessage(tr("Load a result before adding visualization objects."));
        return;
    }

    SceneObject object;
    object.id = QStringLiteral("object_%1").arg(nextObjectSerial_++);
    object.type = type;
    object.name = objectTypeLabel(type) + QStringLiteral(" %1").arg(nextObjectSerial_ - 1);
    applyObjectTypeDefaults(&object);

    if (type == ObjectType::Data) {
        object.inputPath = currentFile_;
        object.name = tr("Data: %1").arg(QFileInfo(currentFile_).completeBaseName());
    } else if (type == ObjectType::Geometry) {
        const QString start = projectDirectory_.isEmpty() ? QString() : QDir(projectDirectory_).filePath(QStringLiteral("building_db"));
        const QString path = QFileDialog::getOpenFileName(this, tr("Add Geometry"), start, tr("Geometry (*.stl);;All Files (*.*)"));
        if (path.isEmpty()) {
            return;
        }
        object.inputPath = normalizedAbsolutePath(path);
        object.name = QFileInfo(path).completeBaseName();
    } else if (type == ObjectType::RayTracingVolume) {
        if (QFileInfo(currentFile_).suffix().compare(QStringLiteral("vti"), Qt::CaseInsensitive) == 0) {
            object.inputPath = currentFile_;
        } else {
            const QString path = QFileDialog::getOpenFileName(this, tr("Add Volume"), projectDirectory_, tr("Image Data (*.vti);;All Files (*.*)"));
            if (path.isEmpty()) {
                return;
            }
            object.inputPath = normalizedAbsolutePath(path);
        }
        object.name = tr("Volume: %1").arg(QFileInfo(object.inputPath).completeBaseName());
    } else {
        QString parentId = selectedObjectId();
        if (parentId.isEmpty() && !objects_.isEmpty()) {
            parentId = objects_.front().id;
        }
        object.parentId = parentId;
        const SceneObject* parent = findObject(parentId);
        if (parent) {
            object.name = objectTypeLabel(type) + QStringLiteral(" of ") + parent->name;
            object.colorField = parent->colorField;
            object.colorComponent = parent->colorComponent;
            object.colorMap = parent->colorMap;
        }
        applyBoundsDefaultsFromParent(&object);
    }

    QString error;
    objects_.push_back(object);
    if (!applyObject(object, true, &error)) {
        objects_.removeLast();
        emit statusMessage(error);
        return;
    }
    refreshObjectTree(object.id);
    emit guiActionRequested(tr("Added visualization object %1").arg(object.name));
}

void VtkViewWidget::removeSelectedObject() {
    const QString objectId = selectedObjectId();
    if (objectId.isEmpty()) {
        return;
    }
    removeObjectAndDescendants(objectId);
    refreshObjectTree();
    viewer_->renderWhenVisible();
}

void VtkViewWidget::applySelectedObjectControls() {
    if (blockUiUpdates_) {
        return;
    }
    SceneObject* object = selectedObject();
    if (!object) {
        return;
    }
    const QString objectId = object->id;
    updateObjectFromControls(object);
    QString error;
    if (!applyObjectAndDescendants(objectId, &error)) {
        emit statusMessage(error);
        return;
    }
    refreshObjectTree(objectId);
    emit guiActionRequested(tr("Updated visualization object %1").arg(object->name));
}

QString VtkViewWidget::defaultColorMapName() const {
    const QString active = Streamcenter::activeDefaultColorMapName();
    return active.isEmpty() ? QStringLiteral("Cool to Warm") : active;
}

QString VtkViewWidget::effectiveInputPath(const SceneObject& object) const {
    if (!object.inputPath.trimmed().isEmpty()) {
        return object.inputPath;
    }
    QString parentId = object.parentId;
    for (int guard = 0; guard < objects_.size() && !parentId.isEmpty(); ++guard) {
        const SceneObject* parent = findObject(parentId);
        if (!parent) {
            break;
        }
        if (!parent->inputPath.trimmed().isEmpty()) {
            return parent->inputPath;
        }
        parentId = parent->parentId;
    }
    return {};
}

QString VtkViewWidget::relativeProjectPath(const QString& path) const {
    if (projectDirectory_.isEmpty()) {
        return QDir::toNativeSeparators(path);
    }
    const QString relative = QDir(projectDirectory_).relativeFilePath(path);
    return QDir::toNativeSeparators(relative.startsWith(QStringLiteral("..")) ? path : relative);
}

QString VtkViewWidget::objectTypeLabel(ObjectType type) const {
    switch (type) {
    case ObjectType::Geometry:
        return tr("Geometry");
    case ObjectType::Clip:
        return tr("Clip");
    case ObjectType::Slice:
        return tr("Slice");
    case ObjectType::Contour:
        return tr("Contour");
    case ObjectType::RayTracingVolume:
        return tr("Volume");
    case ObjectType::Crop:
        return tr("Crop");
    case ObjectType::Data:
    default:
        return tr("Data");
    }
}

QString VtkViewWidget::resultTypeLabel(const QString& type) const {
    if (type == QStringLiteral("average")) {
        return tr("Average");
    }
    if (type == QStringLiteral("transient")) {
        return tr("Transient");
    }
    if (type == QStringLiteral("geometry")) {
        return tr("Geometry");
    }
    return tr("Derived");
}

QString VtkViewWidget::resultStatusText(const ResultEntry& entry) const {
    if (entry.filePath.isEmpty()) {
        return {};
    }
    QStringList parts;
    parts << resultTypeLabel(entry.type);
    if (!entry.runStamp.isEmpty()) {
        parts << tr("run %1").arg(entry.runStamp);
    }
    if (entry.timeStep >= 0) {
        parts << tr("step %1").arg(entry.timeStep);
    }
    const QFileInfo info(entry.filePath);
    parts << tr("%1 MiB").arg(QString::number(static_cast<double>(info.size()) / (1024.0 * 1024.0), 'f', 2));
    parts << info.lastModified().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"));
    return parts.join(QStringLiteral(" | "));
}

} // namespace luwgui
