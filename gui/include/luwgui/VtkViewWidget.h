#pragma once

#include <QList>
#include <QString>
#include <QVector>
#include <QWidget>

#include "ViewerWidget.h"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSlider;
class QTreeWidget;
class QTreeWidgetItem;
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
    enum class ObjectType {
        Geometry,
        Clip,
        Slice,
        Contour,
        RayTracingVolume,
        Data,
        Crop
    };

    struct ResultEntry {
        QString type;
        QString filePath;
        QString runStamp;
        QString sourceLabel;
        qlonglong timeStep = -1;
        qint64 modifiedMs = 0;
        qlonglong sortKey = 0;
    };

    struct SceneObject {
        QString id;
        QString parentId;
        QString name;
        QString inputPath;
        ObjectType type = ObjectType::Data;
        bool visible = true;
        bool showOutline = false;
        bool showSurface = true;
        bool showMesh = false;
        double opacity = 1.0;
        QString colorMode = QStringLiteral("Field");
        QString colorField;
        QString colorComponent = QStringLiteral("Magnitude");
        QString colorMap;
        bool autoColorRange = true;
        double colorRangeMin = 0.0;
        double colorRangeMax = 1.0;
        double planeOrigin[3] = {0.0, 0.0, 0.0};
        double planeNormal[3] = {1.0, 0.0, 0.0};
        double cropBounds[6] = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
        QVector<double> contourValues;
    };

    void buildUi();
    void populateColorMaps();
    QList<ResultEntry> scanResultEntries() const;
    void repopulateResultCombo(bool preserveSelection, bool autoLoadSelection);
    bool loadResultEntry(const ResultEntry& entry, QString* errorMessage = nullptr);
    ResultEntry latestEntryForType(const QString& type) const;
    ResultEntry currentSelectedEntry() const;

    bool resetSceneForFile(const QString& filePath, QString* errorMessage);
    void addProjectGeometryOverlays();
    SceneObject makeBaseObject(const QString& filePath, QString* errorMessage) const;
    void applyObjectTypeDefaults(SceneObject* object) const;
    void applyBoundsDefaultsFromParent(SceneObject* object);
    ViewerWidget::DataObjectOptions viewerOptionsFor(const SceneObject& object) const;
    bool applyObject(const SceneObject& object, bool resetCameraToObject, QString* errorMessage = nullptr);
    bool applyObjectAndDescendants(const QString& objectId, QString* errorMessage = nullptr);
    void removeObjectAndDescendants(const QString& objectId);

    void refreshObjectTree(const QString& preferredObjectId = {});
    void appendObjectTreeItems(QTreeWidgetItem* parentItem, const QString& parentId);
    QString selectedObjectId() const;
    SceneObject* findObject(const QString& objectId);
    const SceneObject* findObject(const QString& objectId) const;
    SceneObject* selectedObject();
    void loadSelectedObjectIntoControls();
    void updateControlsFromObject(const SceneObject& object);
    bool updateObjectFromControls(SceneObject* object);
    void updateObjectControlAvailability();
    void populateFieldControls(const SceneObject* object);
    void addObject(ObjectType type);
    void removeSelectedObject();
    void applySelectedObjectControls();

    QString defaultColorMapName() const;
    QString effectiveInputPath(const SceneObject& object) const;
    QString relativeProjectPath(const QString& path) const;
    QString objectTypeLabel(ObjectType type) const;
    QString resultTypeLabel(const QString& type) const;
    QString resultStatusText(const ResultEntry& entry) const;

    ViewerWidget* viewer_ = nullptr;
    QLabel* fileLabel_ = nullptr;
    QLabel* resultInfoLabel_ = nullptr;
    QComboBox* resultTypeCombo_ = nullptr;
    QComboBox* resultFileCombo_ = nullptr;
    QTreeWidget* objectTree_ = nullptr;
    QPushButton* addDataButton_ = nullptr;
    QPushButton* addGeometryButton_ = nullptr;
    QPushButton* addClipButton_ = nullptr;
    QPushButton* addSliceButton_ = nullptr;
    QPushButton* addCropButton_ = nullptr;
    QPushButton* addContourButton_ = nullptr;
    QPushButton* addVolumeButton_ = nullptr;
    QPushButton* removeObjectButton_ = nullptr;
    QCheckBox* visibleCheck_ = nullptr;
    QCheckBox* surfaceCheck_ = nullptr;
    QCheckBox* outlineCheck_ = nullptr;
    QCheckBox* meshCheck_ = nullptr;
    QCheckBox* autoRangeCheck_ = nullptr;
    QComboBox* colorModeCombo_ = nullptr;
    QComboBox* fieldCombo_ = nullptr;
    QComboBox* componentCombo_ = nullptr;
    QComboBox* colorMapCombo_ = nullptr;
    QSlider* opacitySlider_ = nullptr;
    QDoubleSpinBox* rangeMinSpin_ = nullptr;
    QDoubleSpinBox* rangeMaxSpin_ = nullptr;
    QDoubleSpinBox* planeOriginSpin_[3] = {nullptr, nullptr, nullptr};
    QDoubleSpinBox* planeNormalSpin_[3] = {nullptr, nullptr, nullptr};
    QDoubleSpinBox* cropBoundsSpin_[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    QLineEdit* contourValuesEdit_ = nullptr;

    QString projectDirectory_;
    QString currentFile_;
    QList<ResultEntry> resultEntries_;
    QList<SceneObject> objects_;
    int nextObjectSerial_ = 1;
    bool blockUiUpdates_ = false;
};

} // namespace luwgui
