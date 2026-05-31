#pragma once

#include <QColor>
#include <QImage>
#include <QMap>
#include <QSet>
#include <QSize>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QWidget>

#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include <vtkSmartPointer.h>

class QLabel;
class QEvent;
class QFrame;
class QShowEvent;
class QStackedLayout;
class QTimer;
class QVTKOpenGLNativeWidget;
class vtkActor;
class vtkActor2D;
class vtkAxesActor;
class vtkAlgorithm;
class vtkCallbackCommand;
class vtkDataArray;
class vtkDataSet;
class vtkGenericOpenGLRenderWindow;
class vtkImplicitPlaneRepresentation;
class vtkImplicitPlaneWidget2;
class vtkLight;
class vtkLookupTable;
class vtkOrientationMarkerWidget;
class vtkPlane;
class vtkPolyData;
class vtkProp;
class vtkRenderer;
class vtkScalarBarActor;
class vtkScalarBarWidget;
class vtkSliderWidget;
class vtkTextActor;
class vtkTexture;
class vtkRenderPass;

namespace Streamcenter::Index {
class IndexVolumeBackend;
}

class ViewerWidget : public QWidget {
    Q_OBJECT

public:
    using VideoProgressCallback = std::function<bool(int currentFrame, int totalFrames, double percent, const QString& detail)>;

    enum class ScalarMode {
        Forward = 0,
        Backward = 1,
        Both = 2
    };

    enum class CameraView {
        RightPositiveY = 0,
        LeftNegativeY,
        TopPositiveZ,
        BottomNegativeZ,
        FrontNegativeX,
        BackPositiveX,
        Isometric
    };

    struct FrameInfo {
        double timestep = 0.0;
        QString path;
    };

    enum class DisplayObjectType {
        Geometry = 0,
        Clip,
        Slice,
        Contour,
        ParticleStreamline,
        RayTracingVolume,
        Data,
        Crop
    };

    struct LightingOptions {
        QColor color = QColor(255, 255, 255);
        double positionOffset[3] = {1.0, -1.0, 1.0};
        double focalOffset[3] = {0.0, 0.0, 0.0};
        double attenuation[3] = {1.0, 0.0, 0.0};
        double coneAngle = 90.0;
        double exponent = 1.0;
    };

    struct DisplayOptions {
        QColor backgroundColor = QColor(215, 215, 215);
        bool lightingEnabled = true;
        double lightingIntensity = 0.85;
        QString lightingMode = QStringLiteral("Headlight");
        LightingOptions lighting;
        bool advancedRenderingEnabled = false;
        QString advancedPreviewEngine = QStringLiteral("OpenGL PBR");
        QString advancedExportEngine = QStringLiteral("OSPRay path tracer");
        QString advancedRendererType = QStringLiteral("pathtracer");
        int advancedSamplesPerPixel = 64;
        int advancedAccumulationFrames = 8;
        int advancedMaxDepth = 8;
        bool advancedDenoise = true;
        QString advancedEnvironmentTexture;
        bool perspectiveEnabled = false;
        double perspectiveDepth = 30.0;
        bool displayTransparent = false;
        bool sharedTimeAxis = false;
        bool showLogo = true;
        bool showAxes = true;
        bool showTimeCode = false;
        bool showTimeStep = false;
        bool showCustomAxesText = false;
        QString customAxesText;
    };

    struct LegendFontOptions {
        QString family = QStringLiteral("Arial");
        int size = 18;
        QColor color = QColor(0, 0, 0);
        double opacity = 1.0;
        bool bold = false;
        bool italic = false;
        bool shadow = false;
    };

    struct LegendOptions {
        bool autoOrient = true;
        QString orientation = QStringLiteral("Vertical");
        QString windowLocation = QStringLiteral("Any Location");
        double position[2] = {0.890555, 0.0859649};
        QString title = QStringLiteral("Colors");
        bool titleEdited = false;
        QString componentTitle = QStringLiteral("Magnitude");
        QString componentFormat = QStringLiteral("Same line as title (space)");
        QString titleJustification = QStringLiteral("Centered");
        QString titleOrientation = QStringLiteral("Vertical");
        QString titlePosition = QStringLiteral("Left");
        int titlePadding = 50;
        LegendFontOptions titleFont = [] {
            LegendFontOptions options;
            options.size = 20;
            return options;
        }();
        LegendFontOptions textFont;
        int labelCount = 5;
        int colorBarThickness = 16;
        double colorBarLength = 0.33;
        bool drawBackground = false;
        QColor backgroundColor = QColor(128, 128, 128);
        int backgroundPadding = 2;
        bool drawScalarBarOutline = false;
        QColor scalarBarOutlineColor = QColor(255, 255, 255);
        double scalarBarOutlineThickness = 1.0;
        bool automaticLabelFormat = true;
        QString labelFormat = QStringLiteral("%-#6.3g");
        bool drawTickMarks = true;
        bool drawTickLabels = true;
        int tickLabelsPadding = 10;
        QString tickDirection = QStringLiteral("Outward");
        QColor tickColor = QColor(0, 0, 0);
        int tickLength = 7;
        bool useCustomLabels = false;
        bool addRangeLabels = true;
        QString rangeLabelFormat = QStringLiteral("%-#6.3g");
        bool drawDataRangeLabels = false;
        QString dataRangeLabelFormat = QStringLiteral("%-#6.1e");
        bool drawAnnotations = true;
        bool addRangeAnnotations = false;
        QColor belowRangeColor = QColor(0, 0, 0);
        QColor aboveRangeColor = QColor(255, 255, 255);
        bool automaticAnnotations = false;
        bool drawNanAnnotation = false;
        QString nanAnnotation = QStringLiteral("NaN");
        QColor nanColor = QColor(128, 128, 128);
        QString tickAnnotationPosition = QStringLiteral("Ticks left/bottom, annotations right/top");
        bool reverseLegend = false;
    };

    struct DataObjectOptions {
        DisplayObjectType type = DisplayObjectType::Geometry;
        QString inputPath;
        QString sourceObjectId;
        bool visible = true;
        bool showOutline = true;
        bool showSurface = false;
        bool showMesh = false;
        bool showPlaneHandle = true;
        double outlineOpacity = 1.0;
        double surfaceOpacity = 1.0;
        double meshOpacity = 1.0;
        QColor outlineColor = QColor(0, 0, 0);
        QColor surfaceColor = QColor(178, 210, 235);
        QColor meshColor = QColor(80, 80, 80);
        double outlineLineWidth = 1.5;
        double meshLineWidth = 0.5;
        int meshStride = 1;
        QString colorMode = QStringLiteral("Solid color");
        QString colorField;
        QString colorComponent = QStringLiteral("Magnitude");
        QString colorMap;
        bool autoColorRange = true;
        double colorRangeMin = 0.0;
        double colorRangeMax = 1.0;
        bool showLegend = true;
        double volumeSamplingStep = 0.08;
        double volumeOpacityScale = 2.0;
        QString volumeFiltering = QStringLiteral("Linear");
        bool volumePreintegration = true;
        QString materialPreset = QStringLiteral("Matte");
        QColor materialBaseColor = QColor(178, 210, 235);
        double materialMetallic = 0.0;
        double materialRoughness = 0.55;
        double materialIor = 1.5;
        double materialOpacity = 1.0;
        double materialTransmission = 0.0;
        QString osprayMaterialName;
        LegendOptions legend;
        double planeOrigin[3] = {0.0, 0.0, 0.0};
        double planeNormal[3] = {1.0, 0.0, 0.0};
        double cropBounds[6] = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
        QString contourField;
        QString contourComponent = QStringLiteral("Magnitude");
        QVector<double> contourValues;
        QVector<QColor> contourSurfaceColors;
        QVector<double> contourSurfaceOpacities;
    };

    struct FieldOption {
        QString name;
        QStringList components;
    };

    explicit ViewerWidget(QWidget* parent = nullptr);
    ~ViewerWidget() override;

    bool openFile(const QString& path, QString* errorMessage = nullptr);
    void clearScene();
    void showBlankDisplay(const QString& title);
    void beginDisplay(const DisplayOptions& options);
    void applyDisplayOptions(const DisplayOptions& options);
    void setCanvasBorderVisible(bool visible);
    void renderWhenVisible();
    bool addOrUpdateDataObject(const QString& objectId,
                               const DataObjectOptions& options,
                               bool resetCameraToObject = false,
                               QString* errorMessage = nullptr);
    void removeDataObject(const QString& objectId);
    void setActiveDataObjectHandle(const QString& objectId);
    bool probeFieldOptions(const QString& path,
                           QVector<FieldOption>* fields,
                           QString* preferredField = nullptr,
                           QString* preferredComponent = nullptr,
                           QString* errorMessage = nullptr) const;
    bool dataObjectScalarRange(const QString& objectId,
                               const DataObjectOptions& options,
                               double* minValue,
                               double* maxValue,
                               QString* errorMessage = nullptr) const;
    bool dataObjectBounds(const QString& objectId,
                          const DataObjectOptions& options,
                          double bounds[6],
                          QString* errorMessage = nullptr) const;
    bool probeScalarFields(const QString& path,
                           QStringList* fields,
                           QString* preferredField = nullptr,
                           QString* errorMessage = nullptr) const;
    QVector<double> cameraState() const;
    void restoreCameraState(const QVector<double>& values);
    void resetCameraToScene();
    void setCameraView(CameraView view);

    bool hasScene() const;
    bool hasCameraScene() const;
    bool isTimeSeries() const;
    QString currentPath() const;
    void setProjectVisualizationDirectory(const QString& directory);

    void setParallelProjection(bool enabled);
    void setTransparent(bool enabled);
    bool transparent() const;
    void setScalarMode(ScalarMode mode);
    ScalarMode scalarMode() const;
    bool scalarRange(const QString& scalarName,
                     double* minValue,
                     double* maxValue,
                     double* currentValue) const;
    void setIsoValue(const QString& scalarName, double value);
    void setInitialTimeCodeHint(double timeCode);

    QSize renderSize() const;
    bool saveScreenshot(const QString& path,
                        QString* errorMessage = nullptr,
                        bool transparentBackground = false,
                        const QSize& outputSize = QSize(),
                        QImage* capturedImage = nullptr,
                        bool requireFileWrite = true);
    bool exportVideo(const QString& path,
                     int fps,
                     int stride,
                     int startIndex,
                     int endIndex,
                     int frameInterpolationMultiplier,
                     QWidget* dialogParent,
                     QString* errorMessage = nullptr,
                     bool transparentBackground = false,
                     const QSize& outputSize = QSize(),
                     const VideoProgressCallback& progressCallback = {});

    void startAnimation();
    void stopAnimation();
    void toggleAnimation();
    bool animationPlaying() const;

    void jumpToFirstFrame();
    void jumpToPreviousFrame();
    void jumpToNextFrame();
    void jumpToLastFrame();
    bool setFrameIndex(int index, QString* errorMessage = nullptr);
    int frameCount() const;
    int frameIndex() const;
    double frameTimeCode(int index) const;
    QString frameTimeCodeText(int index) const;
    QStringList frameTimeCodeTexts() const;

    void onTimeSliderWidgetChanged(double value);

signals:
    void sceneLoaded(const QString& path, bool isTimeSeries);
    void sceneCleared();
    void timeSeriesAvailabilityChanged(bool available);
    void animationStateChanged(bool playing);
    void frameLoadRequested(int index, double timestep);
    void frameChanged(int index, double timestep);
    void cameraChanged(const QVector<double>& cameraState);
    void dataObjectPlaneChanged(const QString& objectId, const QVector<double>& origin, const QVector<double>& normal);
    void legendStyleChanged(const QString& objectId,
                            const LegendOptions& options);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void showEvent(QShowEvent* event) override;

private slots:
    void onAnimationTick();

private:
    struct ScalarVisual {
        QString scalarName;
        int slot = 0;
        int association = 0;
        double minValue = 0.0;
        double maxValue = 0.0;
        vtkSmartPointer<vtkActor> actor;
    };

    struct TimeCallbackState;
    struct PlaneCallbackState;
    struct LegendCallbackState;
    struct MaterializedDataObject;
    struct DisplayObjectTimeSeries {
        QString sourcePath;
        QVector<FrameInfo> frames;
    };
    struct DisplayActorSet {
        QVector<vtkSmartPointer<vtkProp>> props;
        vtkSmartPointer<vtkPlane> plane;
        vtkSmartPointer<vtkAlgorithm> planeAlgorithm;
        vtkSmartPointer<vtkImplicitPlaneRepresentation> planeRepresentation;
        vtkSmartPointer<vtkImplicitPlaneWidget2> planeWidget;
        vtkSmartPointer<vtkCallbackCommand> planeCallback;
        std::shared_ptr<PlaneCallbackState> planeCallbackState;
        vtkSmartPointer<vtkScalarBarActor> scalarBar;
        vtkSmartPointer<vtkActor2D> scalarBarBackground;
        vtkSmartPointer<vtkPolyData> scalarBarBackgroundPolyData;
        vtkSmartPointer<vtkActor2D> scalarBarTicks;
        vtkSmartPointer<vtkPolyData> scalarBarTicksPolyData;
        QVector<vtkSmartPointer<vtkTextActor>> scalarBarTickLabels;
        vtkSmartPointer<vtkTextActor> scalarBarTitle;
        vtkSmartPointer<vtkActor2D> scalarBarOutline;
        vtkSmartPointer<vtkPolyData> scalarBarOutlinePolyData;
        vtkSmartPointer<vtkLookupTable> scalarBarBaseLookupTable;
        vtkSmartPointer<vtkScalarBarWidget> scalarBarWidget;
        vtkSmartPointer<vtkCallbackCommand> scalarBarCallback;
        std::shared_ptr<LegendCallbackState> scalarBarCallbackState;
        QString scalarBarObjectId;
        QString scalarBarDefaultTitle;
    };

    bool loadPvdSeries(const QString& path, QString* errorMessage);
    bool loadDataSetFromPath(const QString& path, vtkSmartPointer<vtkDataSet>* outData, QString* errorMessage) const;
    bool materializeDataObject(const QString& objectId,
                               const DataObjectOptions& options,
                               MaterializedDataObject* output,
                               QSet<QString>* visiting,
                               QString* errorMessage) const;
    bool applyFrameIndex(int index, bool preserveCamera, QString* errorMessage = nullptr);
    bool applyDisplayFrameIndex(int index, bool preserveCamera, QString* errorMessage = nullptr);

    void rebuildScene(bool resetCamera);
    void rebuildScalarVisuals(bool resetCamera);
    void clearScalarVisuals();
    bool addScalarVisual(const QString& scalarName, const QColor& color, int slot, bool resetCamera);
    void clearDisplayActors();
    void removeDataObjectActors(const QString& objectId);
    bool addOrUpdateRayTracingVolume(const QString& objectId,
                                      const DataObjectOptions& options,
                                      bool resetCameraToObject,
                                      QString* errorMessage);
    void applyDisplayLighting();
    void applyAdvancedRenderingPreview();
    void applyEnvironmentTexture();
    bool exportRayTracingRequested() const;
    bool beginExportRayTracing(QString* errorMessage);
    void endExportRayTracing(vtkRenderPass* previousPass);
    void renderAccumulationFrames();
    void renderIndexVolumes();
    void rebuildDisplayOverlay();
    void scheduleRenderWhenVisible();
    void updatePlaneWidgetVisibility();

    vtkDataArray* findScalarArray(const QString& scalarName, int* association) const;
    void ensureTimeWidgets();
    void removeTimeWidgets();
    void updateTimeOverlay();
    void refreshDisplayTimeSeriesState();
    void updateDisplayTimeOverlay();
    QString displayObjectFramePathForCurrentTime(const QString& objectId, const QString& fallbackPath) const;
    void setCaptureUiVisible(bool visible);
    QImage captureCurrentImage(QString* errorMessage,
                               bool transparentBackground = false,
                               const QSize& outputSize = QSize());
    void handlePlaneInteraction(const QString& objectId, vtkImplicitPlaneWidget2* widget, vtkPlane* plane, vtkAlgorithm* algorithm);
    void handleScalarBarInteraction(const QString& objectId, vtkScalarBarWidget* widget, unsigned long eventId);
    void applyScalarBarOptions(DisplayActorSet* actorSet, const LegendOptions& options);
    void syncScalarBarActorFromRepresentation(DisplayActorSet* actorSet);
    void syncScalarBarCustomOverlays(DisplayActorSet* actorSet, const LegendOptions& options);
    void syncScalarBarBackgroundAndOutline(DisplayActorSet* actorSet, const LegendOptions& options);
    bool showScalarBarStyleDialog(DisplayActorSet* actorSet);

    QStackedLayout* stackedLayout_ = nullptr;
    QLabel* placeholderLabel_ = nullptr;
    QFrame* canvasFrame_ = nullptr;
    QVTKOpenGLNativeWidget* vtkWidget_ = nullptr;

    QTimer* animationTimer_ = nullptr;

    vtkSmartPointer<vtkGenericOpenGLRenderWindow> renderWindow_;
    vtkSmartPointer<vtkRenderer> renderer_;
    vtkSmartPointer<vtkAxesActor> axesActor_;
    vtkSmartPointer<vtkOrientationMarkerWidget> orientationWidget_;
    vtkSmartPointer<vtkDataSet> currentDataSet_;
    vtkSmartPointer<vtkActor> outlineActor_;
    vtkSmartPointer<vtkTextActor> watermarkActor_;
    vtkSmartPointer<vtkTextActor> timeLabelActor_;
    vtkSmartPointer<vtkTextActor> timeCodeActor_;
    vtkSmartPointer<vtkSliderWidget> timeSlider_;
    vtkSmartPointer<vtkLight> displayLight_;
    vtkSmartPointer<vtkTexture> environmentTexture_;
    vtkSmartPointer<vtkCallbackCommand> indexRenderCallback_;
    vtkSmartPointer<vtkCallbackCommand> cameraInteractionCallback_;
#if STREAMCENTERPLUS_ENABLE_VTK_RAYTRACING
    vtkSmartPointer<vtkRenderPass> osprayPass_;
#endif

    std::unique_ptr<TimeCallbackState> timeCallbackState_;
    std::unique_ptr<Streamcenter::Index::IndexVolumeBackend> indexVolumeBackend_;

    QVector<FrameInfo> frames_;
    QVector<ScalarVisual> scalarVisuals_;
    QMap<QString, DisplayActorSet> displayActors_;
    QMap<QString, DataObjectOptions> displayObjectOptions_;
    QMap<QString, DisplayObjectTimeSeries> displayObjectTimeSeries_;
    QMap<QString, double> isoTargets_;

    QString currentPath_;
    QString activeDataObjectHandleId_;
    QString projectVisualizationDirectory_;
    DisplayOptions displayOptions_;
    ScalarMode scalarMode_ = ScalarMode::Both;
    bool parallelProjection_ = true;
    bool transparent_ = false;
    bool isTimeSeries_ = false;
    bool customDisplayActive_ = false;
    bool animationPlaying_ = false;
    bool suppressTimeSliderCallback_ = false;
    double initialTimeCodeHint_ = std::numeric_limits<double>::quiet_NaN();
    int currentFrameIndex_ = 0;
};
