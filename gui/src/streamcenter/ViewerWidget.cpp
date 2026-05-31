#include "ViewerWidget.h"

#include "ColorMapCatalog.h"
#include "NvidiaFrucRuntime.h"
#include "index/IndexVolumeBackend.h"

#include <QColor>
#include <QColorDialog>
#include <QCheckBox>
#include <QComboBox>
#include <QApplication>
#include <QCoreApplication>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDirIterator>
#include <QDoubleSpinBox>
#include <QEventLoop>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFont>
#include <QFontDatabase>
#include <QFontMetrics>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QHash>
#include <QIcon>
#include <QImage>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QLocale>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QPainterPath>
#include <QProcess>
#include <QProcessEnvironment>
#include <QProgressDialog>
#include <QPushButton>
#include <QPixmap>
#include <QResizeEvent>
#include <QMessageBox>
#include <QRectF>
#include <QScrollArea>
#include <QShowEvent>
#include <QSignalBlocker>
#include <QSet>
#include <QStackedLayout>
#include <QStandardPaths>
#include <QStyle>
#include <QStyleOptionButton>
#include <QSpinBox>
#include <QTemporaryDir>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QXmlStreamReader>

#include <QVTKOpenGLNativeWidget.h>

#include <QAbstractButton>
#include <QAbstractSpinBox>
#include <vtkActor.h>
#include <vtkAlgorithm.h>
#include <vtkAxesActor.h>
#include <vtkActor2D.h>
#include <vtkBorderRepresentation.h>
#include <vtkBox.h>
#include <vtkCallbackCommand.h>
#include <vtkCellArray.h>
#include <vtkCell.h>
#include <vtkCaptionActor2D.h>
#include <vtkCamera.h>
#include <vtkCellData.h>
#include <vtkClipDataSet.h>
#include <vtkCommand.h>
#include <vtkContourFilter.h>
#include <vtkCoordinate.h>
#include <vtkColorTransferFunction.h>
#include <vtkCutter.h>
#include <vtkDataArray.h>
#include <vtkDataObject.h>
#include <vtkDataSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkDataSetMapper.h>
#include <vtkDataSetReader.h>
#include <vtkDataSetSurfaceFilter.h>
#include <vtkDoubleArray.h>
#include <vtkErrorCode.h>
#include <vtkExtractEdges.h>
#include <vtkFloatArray.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkGPUVolumeRayCastMapper.h>
#include <vtkImageData.h>
#include <vtkImageReader2.h>
#include <vtkImageReader2Factory.h>
#include <vtkIdList.h>
#include <vtkImplicitPlaneRepresentation.h>
#include <vtkImplicitPlaneWidget2.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkJPEGWriter.h>
#include <vtkLight.h>
#include <vtkLookupTable.h>
#include <vtkNew.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkObjectFactory.h>
#include <vtkPiecewiseFunction.h>
#include <vtkPlane.h>
#include <vtkPNGReader.h>
#include <vtkPNGWriter.h>
#include <vtkOutlineFilter.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkProp.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderPass.h>
#include <vtkRenderer.h>
#include <vtkRectilinearGrid.h>
#include <vtkScalarBarActor.h>
#include <vtkScalarBarActorInternal.h>
#include <vtkScalarBarRepresentation.h>
#include <vtkScalarBarWidget.h>
#include <vtkScalarsToColors.h>
#include <vtkSTLReader.h>
#include <vtkSliderRepresentation.h>
#include <vtkSliderRepresentation2D.h>
#include <vtkSliderWidget.h>
#include <vtkStructuredGrid.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkTexture.h>
#include <vtkThreshold.h>
#include <vtkTriangleFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkVersionMacros.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>
#include <vtkWindowToImageFilter.h>
#include <vtkXMLGenericDataObjectReader.h>
#include <vtkXMLImageDataReader.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkXMLRectilinearGridReader.h>
#include <vtkXMLStructuredGridReader.h>
#include <vtkXMLUnstructuredGridReader.h>

#if STREAMCENTERPLUS_ENABLE_VTK_RAYTRACING
#include <vtkOSPRayPass.h>
#include <vtkOSPRayRendererNode.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

namespace {

constexpr double kOpaqueOpacity = 1.0;
constexpr double kTransparentOpacity = 0.8;
constexpr int kFieldAssociationPoints = 0;
constexpr int kFieldAssociationCells = 1;
constexpr int kLegendFormLeftMargin = 8;
constexpr int kLegendFormRightMargin = 6;
constexpr int kLegendFormTopMargin = 4;
constexpr int kLegendFormRowVerticalPadding = 1;
constexpr int kLegendFormRowSpacing = 0;
constexpr int kLegendFormControlSpacing = 6;
constexpr int kLegendFormCompactControlSpacing = 3;
constexpr int kLegendColorIndicatorDiameter = 14;
constexpr int kLegendCheckRowLeftPadding = 6;
constexpr int kLegendFontRowRightPadding = 4;
constexpr int kLegendFooterTopPadding = 7;
constexpr auto kLegendImportFontLabel = "Import font file...";

std::array<double, 6> normalizedCropBounds(const double bounds[6], QString* errorMessage) {
    std::array<double, 6> normalized = {};
    for (int axis = 0; axis < 3; ++axis) {
        const int lo = axis * 2;
        const int hi = lo + 1;
        double minimum = bounds[lo];
        double maximum = bounds[hi];
        if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
            if (errorMessage != nullptr) {
                *errorMessage = QStringLiteral("Crop range contains a non-finite value.");
            }
            return {};
        }
        if (maximum < minimum) {
            std::swap(minimum, maximum);
        }
        if (!(maximum > minimum)) {
            const double scale = std::max({1.0, std::abs(minimum), std::abs(maximum)});
            const double pad = scale * 1.0e-9;
            minimum -= pad;
            maximum += pad;
        }
        normalized[static_cast<std::size_t>(lo)] = minimum;
        normalized[static_cast<std::size_t>(hi)] = maximum;
    }
    return normalized;
}

vtkSmartPointer<vtkDataSet> cropDataSetToBounds(vtkDataSet* input, const double bounds[6], QString* errorMessage) {
    if (input == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Internal error: crop input dataset is null.");
        }
        return nullptr;
    }

    QString boundsError;
    const std::array<double, 6> cropBounds = normalizedCropBounds(bounds, &boundsError);
    if (!boundsError.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = boundsError;
        }
        return nullptr;
    }

    vtkSmartPointer<vtkBox> box = vtkSmartPointer<vtkBox>::New();
    box->SetBounds(cropBounds.data());

    vtkSmartPointer<vtkClipDataSet> crop = vtkSmartPointer<vtkClipDataSet>::New();
    crop->SetInputData(input);
    crop->SetClipFunction(box);
    crop->SetValue(0.0);
    crop->InsideOutOn();
    crop->Update();

    vtkDataSet* rawOutput = vtkDataSet::SafeDownCast(crop->GetOutputDataObject(0));
    if (rawOutput == nullptr || rawOutput->GetNumberOfPoints() <= 0) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Crop range produced no visible data; check X/Y/Z ranges.");
        }
        return nullptr;
    }

    vtkDataSet* rawCopy = vtkDataSet::SafeDownCast(rawOutput->NewInstance());
    if (rawCopy == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Cannot allocate cropped visualization dataset.");
        }
        return nullptr;
    }
    vtkSmartPointer<vtkDataSet> output;
    output.TakeReference(rawCopy);
    output->DeepCopy(rawOutput);
    return output;
}

struct RenderMaterialValues {
    QColor baseColor = QColor(178, 210, 235);
    double metallic = 0.0;
    double roughness = 0.55;
    double ior = 1.5;
    double opacity = 1.0;
    double transmission = 0.0;
};

RenderMaterialValues renderMaterialValues(const ViewerWidget::DataObjectOptions& options) {
    RenderMaterialValues values;
    values.baseColor = options.materialBaseColor.isValid() ? options.materialBaseColor : options.surfaceColor;
    values.metallic = std::clamp(options.materialMetallic, 0.0, 1.0);
    values.roughness = std::clamp(options.materialRoughness, 0.0, 1.0);
    values.ior = std::max(1.0, options.materialIor);
    values.opacity = std::clamp(options.materialOpacity, 0.0, 1.0);
    values.transmission = std::clamp(options.materialTransmission, 0.0, 1.0);

    const QString preset = options.materialPreset.trimmed().toLower();
    if (preset == QStringLiteral("polished metal")) {
        values.baseColor = QColor(210, 210, 205);
        values.metallic = 1.0;
        values.roughness = 0.18;
        values.opacity = 1.0;
        values.transmission = 0.0;
    } else if (preset == QStringLiteral("rough metal")) {
        values.baseColor = QColor(180, 178, 170);
        values.metallic = 1.0;
        values.roughness = 0.68;
        values.opacity = 1.0;
        values.transmission = 0.0;
    } else if (preset == QStringLiteral("glass")) {
        values.baseColor = QColor(210, 235, 255);
        values.metallic = 0.0;
        values.roughness = 0.03;
        values.ior = 1.52;
        values.opacity = 0.32;
        values.transmission = 0.85;
    } else if (preset == QStringLiteral("thin glass")) {
        values.baseColor = QColor(225, 245, 255);
        values.metallic = 0.0;
        values.roughness = 0.02;
        values.ior = 1.45;
        values.opacity = 0.42;
        values.transmission = 0.65;
    } else if (preset == QStringLiteral("car paint")) {
        values.baseColor = QColor(180, 30, 32);
        values.metallic = 0.05;
        values.roughness = 0.28;
        values.ior = 1.55;
        values.opacity = 1.0;
        values.transmission = 0.0;
    }
    return values;
}

bool isOsprayEngine(const QString& value) {
    const QString normalized = value.trimmed().toLower();
    return normalized.contains(QStringLiteral("ospray")) || normalized.contains(QStringLiteral("path tracer"));
}

QByteArray imageFormatForPath(const QString& path) {
    const QString suffix = QFileInfo(path).suffix().toLower();
    if (suffix == QStringLiteral("jpg") || suffix == QStringLiteral("jpeg")) {
        return QByteArrayLiteral("jpg");
    }
    return QByteArrayLiteral("png");
}

QImage imageForVtkWriter(const QImage& image, const QByteArray& format) {
    if (format == QByteArrayLiteral("jpg") || format == QByteArrayLiteral("jpeg")) {
        if (image.hasAlphaChannel()) {
            QImage flattened(image.size(), QImage::Format_RGB32);
            flattened.fill(Qt::white);
            QPainter painter(&flattened);
            painter.drawImage(QPoint(0, 0), image);
            return flattened.convertToFormat(QImage::Format_RGB888);
        }
        return image.convertToFormat(QImage::Format_RGB888);
    }
    return image.convertToFormat(QImage::Format_RGBA8888);
}

int componentCountForImage(const QImage& image) {
    switch (image.format()) {
    case QImage::Format_RGB888:
        return 3;
    case QImage::Format_RGBA8888:
        return 4;
    default:
        break;
    }
    return image.hasAlphaChannel() ? 4 : 3;
}

vtkSmartPointer<vtkImageData> vtkImageFromQImage(const QImage& image, const QByteArray& format) {
    const QImage writableImage = imageForVtkWriter(image, format);
    if (writableImage.isNull() || writableImage.width() <= 0 || writableImage.height() <= 0) {
        return nullptr;
    }

    const int components = componentCountForImage(writableImage);
    if (components != 3 && components != 4) {
        return nullptr;
    }
    auto vtkImage = vtkSmartPointer<vtkImageData>::New();
    vtkImage->SetDimensions(writableImage.width(), writableImage.height(), 1);
    vtkImage->AllocateScalars(VTK_UNSIGNED_CHAR, components);

    const int rowBytes = writableImage.width() * components;
    for (int y = 0; y < writableImage.height(); ++y) {
        const uchar* src = writableImage.constScanLine(writableImage.height() - 1 - y);
        auto* dst = static_cast<unsigned char*>(vtkImage->GetScalarPointer(0, y, 0));
        std::memcpy(dst, src, rowBytes);
    }
    return vtkImage;
}

QImage qImageFromVtkImageData(vtkImageData* imageData, bool transparentBackground) {
    if (imageData == nullptr) {
        return {};
    }
    int dims[3]{};
    imageData->GetDimensions(dims);
    if (dims[0] <= 0 || dims[1] <= 0) {
        return {};
    }
    auto* src = static_cast<unsigned char*>(imageData->GetScalarPointer());
    if (src == nullptr) {
        return {};
    }

    const int components = imageData->GetNumberOfScalarComponents();
    if (components < 3) {
        return {};
    }
    QImage image(dims[0], dims[1], transparentBackground && components >= 4 ? QImage::Format_RGBA8888 : QImage::Format_RGB888);
    const int outputComponents = image.hasAlphaChannel() ? 4 : 3;
    for (int y = 0; y < dims[1]; ++y) {
        const unsigned char* srcRow = src + (static_cast<size_t>(y) * dims[0] * components);
        unsigned char* dstRow = image.scanLine(dims[1] - 1 - y);
        if (components == outputComponents) {
            std::memcpy(dstRow, srcRow, static_cast<size_t>(dims[0]) * outputComponents);
            continue;
        }
        for (int x = 0; x < dims[0]; ++x) {
            const unsigned char* srcPixel = srcRow + (x * components);
            unsigned char* dstPixel = dstRow + (x * outputComponents);
            dstPixel[0] = srcPixel[0];
            dstPixel[1] = srcPixel[1];
            dstPixel[2] = srcPixel[2];
            if (outputComponents == 4) {
                dstPixel[3] = components >= 4 ? srcPixel[3] : 255;
            }
        }
    }
    return image;
}

QString vtkWriterErrorText(unsigned long errorCode) {
    const char* text = vtkErrorCode::GetStringFromErrorCode(errorCode);
    if (text == nullptr || std::strlen(text) == 0) {
        return QObject::tr("unknown VTK writer error");
    }
    return QString::fromLatin1(text);
}

bool saveImageWithVtkWriter(const QImage& image, const QString& path, QString* errorMessage) {
    const QString trimmedPath = path.trimmed();
    if (trimmedPath.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Please choose an output image path.");
        }
        return false;
    }

    const QFileInfo info(trimmedPath);
    const QString outputDir = info.absolutePath();
    if (!outputDir.isEmpty() && !QDir().mkpath(outputDir)) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Could not create screenshot folder '%1'.").arg(outputDir);
        }
        return false;
    }

    const QByteArray format = imageFormatForPath(trimmedPath);
    vtkSmartPointer<vtkImageData> vtkImage = vtkImageFromQImage(image, format);
    if (vtkImage == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Could not prepare screenshot pixels for writing.");
        }
        return false;
    }

    const QByteArray fileName = QFile::encodeName(trimmedPath);
    unsigned long errorCode = vtkErrorCode::NoError;
    if (format == QByteArrayLiteral("jpg") || format == QByteArrayLiteral("jpeg")) {
        vtkNew<vtkJPEGWriter> writer;
        writer->SetFileName(fileName.constData());
        writer->SetInputData(vtkImage);
        writer->SetQuality(95);
        writer->Write();
        errorCode = writer->GetErrorCode();
    } else {
        vtkNew<vtkPNGWriter> writer;
        writer->SetFileName(fileName.constData());
        writer->SetInputData(vtkImage);
        writer->Write();
        errorCode = writer->GetErrorCode();
    }

    const QFileInfo writtenInfo(trimmedPath);
    if (errorCode != vtkErrorCode::NoError || !writtenInfo.exists() || writtenInfo.size() <= 0) {
        if (errorMessage != nullptr) {
            const QString detail = errorCode != vtkErrorCode::NoError
                ? vtkWriterErrorText(errorCode)
                : QObject::tr("output file was not created");
            *errorMessage = QObject::tr("Could not save screenshot to '%1' as %2: %3.")
                .arg(trimmedPath,
                     QString::fromLatin1(format).toUpper(),
                     detail);
        }
        return false;
    }
    return true;
}

class StreamcenterScalarBarActor final : public vtkScalarBarActor {
public:
    static StreamcenterScalarBarActor* New();
    vtkTypeMacro(StreamcenterScalarBarActor, vtkScalarBarActor);

    void setForegroundTickOverlay(vtkPolyData* polyData, const QColor& color, double lineWidth, bool visible) {
        if (foregroundTickActor_ == nullptr || foregroundTickMapper_ == nullptr) {
            return;
        }
        foregroundTickMapper_->SetInputData(polyData);
        foregroundTickActor_->SetVisibility(visible && polyData != nullptr);
        foregroundTickActor_->GetProperty()->SetColor(color.redF(), color.greenF(), color.blueF());
        foregroundTickActor_->GetProperty()->SetOpacity(static_cast<double>(color.alpha()) / 255.0);
        foregroundTickActor_->GetProperty()->SetLineWidth(static_cast<float>(std::max(0.0, lineWidth)));
        foregroundTickActor_->GetProperty()->SetDisplayLocationToForeground();
        foregroundTickActor_->SetLayerNumber(10);
        this->Modified();
    }

    int RenderOverlay(vtkViewport* viewport) override {
        int rendered = this->Superclass::RenderOverlay(viewport);
        if (foregroundTickActor_ != nullptr && foregroundTickActor_->GetVisibility()) {
            rendered += foregroundTickActor_->RenderOverlay(viewport);
        }
        return rendered;
    }

    bool scalarBarPixelRect(vtkViewport* viewport, QRectF* rect) {
        if (viewport == nullptr || rect == nullptr || !this->RebuildLayoutIfNeeded(viewport) || this->P == nullptr) {
            return false;
        }
        const vtkScalarBarBox& box = this->P->ScalarBarBox;
        const int* origin = this->PositionCoordinate->GetComputedViewportValue(viewport);
        const bool horizontal = this->GetOrientation() == VTK_ORIENT_HORIZONTAL;
        const int width = horizontal ? box.Size[1] : box.Size[0];
        const int height = horizontal ? box.Size[0] : box.Size[1];
        if (width <= 0 || height <= 0) {
            return false;
        }
        *rect = QRectF(origin[0] + box.Posn[0], origin[1] + box.Posn[1], width, height);
        return true;
    }

protected:
    StreamcenterScalarBarActor() {
        foregroundTickCoordinate_ = vtkSmartPointer<vtkCoordinate>::New();
        foregroundTickCoordinate_->SetCoordinateSystemToDisplay();
        foregroundTickMapper_ = vtkSmartPointer<vtkPolyDataMapper2D>::New();
        foregroundTickMapper_->SetTransformCoordinate(foregroundTickCoordinate_);
        foregroundTickActor_ = vtkSmartPointer<vtkActor2D>::New();
        foregroundTickActor_->SetMapper(foregroundTickMapper_);
        foregroundTickActor_->SetLayerNumber(10);
        foregroundTickActor_->GetProperty()->SetDisplayLocationToForeground();
        foregroundTickActor_->VisibilityOff();
    }
    ~StreamcenterScalarBarActor() override = default;

private:
    vtkSmartPointer<vtkCoordinate> foregroundTickCoordinate_;
    vtkSmartPointer<vtkPolyDataMapper2D> foregroundTickMapper_;
    vtkSmartPointer<vtkActor2D> foregroundTickActor_;
};

vtkStandardNewMacro(StreamcenterScalarBarActor);

QString gLegendProjectVisualizationDirectory;
QString gLegendProjectFontDirectory;
QVector<int> gLegendProjectFontIds;
QHash<QString, QString> gLegendProjectFontFilesByFamily;
QStringList gLegendProjectFontFamilies;

QString normalizedLegendPath(const QString& path) {
    return QDir::fromNativeSeparators(QDir::cleanPath(path));
}

QStringList legendBuiltInFontFamilies() {
    return {QStringLiteral("Arial"), QStringLiteral("Courier"), QStringLiteral("Times")};
}

QString legendFontKey(const QString& family) {
    return family.trimmed().toLower();
}

QString legendProjectFontDirectoryForVisualizationDirectory(const QString& directory) {
    if (directory.trimmed().isEmpty()) {
        return {};
    }
    return normalizedLegendPath(QDir(directory).absoluteFilePath(QStringLiteral("fonts")));
}

void clearLegendProjectFonts() {
    for (int fontId : std::as_const(gLegendProjectFontIds)) {
        QFontDatabase::removeApplicationFont(fontId);
    }
    gLegendProjectFontIds.clear();
    gLegendProjectFontFilesByFamily.clear();
    gLegendProjectFontFamilies.clear();
}

void scanLegendProjectFonts(const QString& visualizationDirectory) {
    gLegendProjectVisualizationDirectory = normalizedLegendPath(visualizationDirectory);
    gLegendProjectFontDirectory = legendProjectFontDirectoryForVisualizationDirectory(gLegendProjectVisualizationDirectory);
    clearLegendProjectFonts();
    if (gLegendProjectFontDirectory.trimmed().isEmpty() || !QDir(gLegendProjectFontDirectory).exists()) {
        return;
    }

    const QStringList nameFilters = {
        QStringLiteral("*.ttf"),
        QStringLiteral("*.otf"),
        QStringLiteral("*.ttc"),
        QStringLiteral("*.otc")
    };
    QDirIterator iterator(gLegendProjectFontDirectory, nameFilters, QDir::Files, QDirIterator::NoIteratorFlags);
    QSet<QString> seenFamilies;
    while (iterator.hasNext()) {
        const QString fontPath = normalizedLegendPath(iterator.next());
        const int fontId = QFontDatabase::addApplicationFont(fontPath);
        if (fontId < 0) {
            continue;
        }
        gLegendProjectFontIds.push_back(fontId);
        for (const QString& family : QFontDatabase::applicationFontFamilies(fontId)) {
            const QString trimmed = family.trimmed();
            if (trimmed.isEmpty()) {
                continue;
            }
            const QString key = legendFontKey(trimmed);
            if (!seenFamilies.contains(key)) {
                gLegendProjectFontFamilies.push_back(trimmed);
                seenFamilies.insert(key);
            }
            if (!gLegendProjectFontFilesByFamily.contains(key)) {
                gLegendProjectFontFilesByFamily.insert(key, fontPath);
            }
        }
    }
    gLegendProjectFontFamilies.sort(Qt::CaseInsensitive);
}

QStringList legendAvailableFontFamilies() {
    QStringList result = legendBuiltInFontFamilies();
    for (const QString& family : std::as_const(gLegendProjectFontFamilies)) {
        bool duplicate = false;
        for (const QString& existing : std::as_const(result)) {
            if (existing.compare(family, Qt::CaseInsensitive) == 0) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            result.push_back(family);
        }
    }
    return result;
}

QString legendProjectFontFile(const QString& family) {
    return gLegendProjectFontFilesByFamily.value(legendFontKey(family));
}

QString uniqueLegendFontDestinationPath(const QString& sourcePath, const QString& fontDirectory) {
    const QFileInfo sourceInfo(sourcePath);
    QDir directory(fontDirectory);
    QString baseName = sourceInfo.completeBaseName().trimmed();
    if (baseName.isEmpty()) {
        baseName = QStringLiteral("font");
    }
    const QString suffix = sourceInfo.suffix().trimmed().isEmpty()
        ? QStringLiteral("ttf")
        : sourceInfo.suffix().trimmed();
    QString candidate = normalizedLegendPath(directory.absoluteFilePath(sourceInfo.fileName()));
    int index = 1;
    while (QFileInfo::exists(candidate)
           && QFileInfo(candidate).absoluteFilePath().compare(sourceInfo.absoluteFilePath(), Qt::CaseInsensitive) != 0) {
        candidate = normalizedLegendPath(directory.absoluteFilePath(QStringLiteral("%1_%2.%3").arg(baseName).arg(index).arg(suffix)));
        ++index;
    }
    return candidate;
}

bool importLegendFontFile(const QString& sourcePath, QWidget* parent, QString* importedFamily) {
    if (importedFamily != nullptr) {
        importedFamily->clear();
    }
    const QFileInfo sourceInfo(sourcePath);
    if (!sourceInfo.exists() || !sourceInfo.isFile()) {
        QMessageBox::warning(parent,
                             QObject::tr("Import Font"),
                             QObject::tr("The selected font file does not exist."));
        return false;
    }
    if (gLegendProjectFontDirectory.trimmed().isEmpty()) {
        QMessageBox::warning(parent,
                             QObject::tr("Import Font"),
                             QObject::tr("Open or create a project before importing a legend font."));
        return false;
    }
    if (!QDir().mkpath(gLegendProjectFontDirectory)) {
        QMessageBox::warning(parent,
                             QObject::tr("Import Font"),
                             QObject::tr("Cannot create project font directory:\n%1").arg(gLegendProjectFontDirectory));
        return false;
    }

    const QString destinationPath = uniqueLegendFontDestinationPath(sourceInfo.absoluteFilePath(), gLegendProjectFontDirectory);
    const bool sameFile = QFileInfo(destinationPath).absoluteFilePath().compare(sourceInfo.absoluteFilePath(), Qt::CaseInsensitive) == 0;
    if (!sameFile) {
        if (!QFile::copy(sourceInfo.absoluteFilePath(), destinationPath)) {
            QMessageBox::warning(parent,
                                 QObject::tr("Import Font"),
                                 QObject::tr("Cannot copy font into project:\n%1").arg(destinationPath));
            return false;
        }
    }

    const int probeFontId = QFontDatabase::addApplicationFont(destinationPath);
    if (probeFontId < 0) {
        if (!sameFile) {
            QFile::remove(destinationPath);
        }
        QMessageBox::warning(parent,
                             QObject::tr("Import Font"),
                             QObject::tr("The selected file could not be loaded as a font."));
        return false;
    }
    const QStringList families = QFontDatabase::applicationFontFamilies(probeFontId);
    QFontDatabase::removeApplicationFont(probeFontId);
    if (families.isEmpty()) {
        if (!sameFile) {
            QFile::remove(destinationPath);
        }
        QMessageBox::warning(parent,
                             QObject::tr("Import Font"),
                             QObject::tr("The selected font did not expose any font family."));
        return false;
    }

    scanLegendProjectFonts(gLegendProjectVisualizationDirectory);
    if (importedFamily != nullptr) {
        *importedFamily = families.front().trimmed();
    }
    return true;
}

QColor blendLegendColors(const QColor& first, const QColor& second, double ratio) {
    const double t = std::clamp(ratio, 0.0, 1.0);
    return QColor::fromRgbF(first.redF() * (1.0 - t) + second.redF() * t,
                            first.greenF() * (1.0 - t) + second.greenF() * t,
                            first.blueF() * (1.0 - t) + second.blueF() * t,
                            first.alphaF() * (1.0 - t) + second.alphaF() * t);
}

QColor legendAccentTextColor(const QColor& color) {
    return color.lightnessF() < 0.5 ? QColor("#ffffff") : QColor("#000000");
}

QPixmap legendColorIndicatorPixmap(const QColor& color) {
    QPixmap pixmap(kLegendColorIndicatorDiameter, kLegendColorIndicatorDiameter);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(QColor(0, 0, 0, 92), 1.0));
    painter.setBrush(color.isValid() ? color : QColor(Qt::white));
    painter.drawEllipse(QRectF(0.5, 0.5, kLegendColorIndicatorDiameter - 1.0, kLegendColorIndicatorDiameter - 1.0));
    return pixmap;
}

QString legendColorCodeText(const QColor& color) {
    return color.isValid()
        ? color.name(QColor::HexArgb)
        : QStringLiteral("#ffffff");
}

void setLegendColorButton(QPushButton* button, const QColor& color) {
    if (button == nullptr) {
        return;
    }
    const QColor validColor = color.isValid() ? color : QColor(Qt::white);
    button->setProperty("storedColor", validColor);
    if (button->property("visualizationMiniColorPicker").toBool()) {
        button->setText(QString());
        button->setIcon(QIcon(legendColorIndicatorPixmap(validColor)));
        button->setIconSize(QSize(kLegendColorIndicatorDiameter, kLegendColorIndicatorDiameter));
        return;
    }
    const QString labelText = button->property("visualizationColorLabel").toString().trimmed();
    const QString colorCode = legendColorCodeText(validColor);
    button->setText(labelText.isEmpty()
        ? colorCode
        : QStringLiteral("%1 (%2)").arg(labelText, colorCode));
    button->setIcon(QIcon(legendColorIndicatorPixmap(validColor)));
    button->setIconSize(QSize(kLegendColorIndicatorDiameter, kLegendColorIndicatorDiameter));
}

QColor legendButtonColor(QPushButton* button) {
    if (button == nullptr) {
        return QColor(Qt::white);
    }
    const QColor color = button->property("storedColor").value<QColor>();
    return color.isValid() ? color : QColor(Qt::white);
}

QColor chooseLegendColor(const QColor& initialColor, QWidget* parent, const QString& title) {
    QColorDialog dialog(initialColor.isValid() ? initialColor : QColor(Qt::white), parent);
    dialog.setWindowTitle(title);
    dialog.setOption(QColorDialog::ShowAlphaChannel, true);
    auto renameAlphaLabels = [&dialog]() {
        for (QLabel* label : dialog.findChildren<QLabel*>()) {
            const QString text = label->text();
            if (text.contains(QStringLiteral("Alpha"), Qt::CaseInsensitive)) {
                QString opacityText = text;
                opacityText.replace(QStringLiteral("Alpha channel"), QStringLiteral("Opacity"), Qt::CaseInsensitive);
                opacityText.replace(QStringLiteral("Alpha"), QStringLiteral("Opacity"), Qt::CaseInsensitive);
                label->setText(opacityText);
            }
        }
    };
    renameAlphaLabels();
    QTimer::singleShot(0, &dialog, renameAlphaLabels);
    return dialog.exec() == QDialog::Accepted ? dialog.selectedColor() : QColor();
}

void makeLegendWidthShrinkable(QWidget* widget) {
    if (widget == nullptr) {
        return;
    }
    widget->setMinimumWidth(0);
    QSizePolicy policy = widget->sizePolicy();
    policy.setHorizontalPolicy(QSizePolicy::Ignored);
    widget->setSizePolicy(policy);
}

int legendFontPixelSize(const QFont& font) {
    const int explicitPixelSize = font.pixelSize();
    if (explicitPixelSize > 0) {
        return std::max(explicitPixelSize - 1, 10);
    }
    return std::max(QFontMetrics(font).height() - 3, 10);
}

int legendCompactControlHeight(const QFont& font) {
    return std::max(22, legendFontPixelSize(font) + 8);
}

int legendCompressedControlHeight(const QFont& font) {
    return std::max(20, legendCompactControlHeight(font) - 2);
}

int legendInputControlHeight(const QFont& font) {
    return legendCompressedControlHeight(font);
}

int legendComboControlHeight(const QFont& font) {
    return legendCompressedControlHeight(font);
}

int legendColorPickerControlHeight(const QFont& font) {
    return legendCompressedControlHeight(font);
}

int legendMiniColorPickerWidth(int height) {
    return std::max(1, static_cast<int>(std::round(height * 1.5)));
}

int legendFieldLabelWidth(const QFont& font) {
    const int currentWidth = std::clamp(legendFontPixelSize(font) * 8, 90, 120);
    return static_cast<int>(std::ceil(currentWidth * 1.05 * 1.10));
}

int legendRowHeight(const QFont& font) {
    return legendCompactControlHeight(font) + (kLegendFormRowVerticalPadding * 2);
}

void updateLegendFieldLabelWidthForRow(QWidget* rowWidget) {
    if (rowWidget == nullptr) {
        return;
    }

    QLabel* label = nullptr;
    for (QLabel* candidate : rowWidget->findChildren<QLabel*>(QString(), Qt::FindDirectChildrenOnly)) {
        if (candidate->property("visualizationFieldLabel").toBool()) {
            label = candidate;
            break;
        }
    }
    if (label == nullptr) {
        return;
    }

    const int baseWidth = label->property("visualizationFieldLabelBaseWidth").toInt();
    const int naturalWidth = label->fontMetrics().horizontalAdvance(label->text()) + 12;
    const int targetWidth = std::max(baseWidth, naturalWidth);
    if (label->width() != targetWidth) {
        label->setFixedWidth(targetWidth);
    }
}

QIcon makeLegendAdvancedOptionsIcon(const QColor& color) {
    QPixmap pixmap(18, 18);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.translate(9.0, 9.0);
    painter.setPen(QPen(color, 1.5, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.setBrush(Qt::NoBrush);
    for (int index = 0; index < 8; ++index) {
        painter.save();
        painter.rotate(index * 45.0);
        painter.drawLine(QPointF(0.0, -6.8), QPointF(0.0, -4.8));
        painter.restore();
    }
    painter.drawEllipse(QPointF(0.0, 0.0), 5.0, 5.0);
    painter.drawEllipse(QPointF(0.0, 0.0), 1.6, 1.6);
    return QIcon(pixmap);
}

class LegendConfigurationRowWidget final : public QWidget {
public:
    explicit LegendConfigurationRowWidget(QWidget* parent = nullptr)
        : QWidget(parent) {}

protected:
    void resizeEvent(QResizeEvent* event) override {
        QWidget::resizeEvent(event);
        updateLegendFieldLabelWidthForRow(this);
    }
};

class LegendCheckBox final : public QCheckBox {
public:
    explicit LegendCheckBox(const QString& text, QWidget* parent = nullptr)
        : QCheckBox(text, parent) {}

    QSize sizeHint() const override {
        const int indicatorSize = std::clamp(fontMetrics().height() - 2, 12, 14);
        const int width = indicatorSize + 6 + fontMetrics().horizontalAdvance(text()) + 4;
        return QSize(width, std::max(QCheckBox::sizeHint().height(), indicatorSize + 4));
    }

    QSize minimumSizeHint() const override {
        return sizeHint();
    }

protected:
    void paintEvent(QPaintEvent* event) override {
        Q_UNUSED(event)

        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, true);

        const int indicatorSize = std::clamp(fontMetrics().height() - 2, 12, 14);
        const int spacing = 6;
        const int top = std::max(0, (height() - indicatorSize) / 2);
        const QRect indicatorRect = QStyle::visualRect(
            layoutDirection(),
            rect(),
            QRect(0, top, indicatorSize, indicatorSize));
        const QRect textRect = QStyle::visualRect(
            layoutDirection(),
            rect(),
            rect().adjusted(indicatorSize + spacing, 0, 0, 0));

        const QPalette::ColorGroup colorGroup = isEnabled() ? QPalette::Active : QPalette::Disabled;
        const QColor baseColor = palette().color(colorGroup, QPalette::Base);
        const QColor contrastColor = baseColor.lightnessF() < 0.5 ? QColor("#ffffff") : QColor("#000000");
        QColor borderColor = blendLegendColors(baseColor, contrastColor, baseColor.lightnessF() < 0.5 ? 0.55 : 0.42);
        QColor fillColor = blendLegendColors(baseColor, contrastColor, baseColor.lightnessF() < 0.5 ? 0.10 : 0.03);

        if (isChecked()) {
            fillColor = palette().color(colorGroup, QPalette::Highlight);
            const QColor checkedContrast = legendAccentTextColor(fillColor);
            borderColor = blendLegendColors(fillColor, checkedContrast, fillColor.lightnessF() < 0.5 ? 0.32 : 0.28);
        } else if (underMouse() && isEnabled()) {
            fillColor = blendLegendColors(fillColor, palette().color(QPalette::Highlight), 0.10);
        }
        if (!isEnabled()) {
            borderColor = palette().color(QPalette::Disabled, QPalette::Mid);
        }

        painter.setPen(QPen(borderColor, 1.2));
        painter.setBrush(fillColor);
        painter.drawRoundedRect(QRectF(indicatorRect).adjusted(0.6, 0.6, -0.6, -0.6), 2.0, 2.0);

        if (checkState() == Qt::PartiallyChecked) {
            const QColor markColor = legendAccentTextColor(fillColor);
            painter.setPen(QPen(markColor, 1.8, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
            painter.drawLine(QPointF(indicatorRect.left() + 3.0, indicatorRect.center().y() + 0.5),
                             QPointF(indicatorRect.right() - 3.0, indicatorRect.center().y() + 0.5));
        } else if (isChecked()) {
            const QColor markColor = legendAccentTextColor(fillColor);
            QPainterPath mark;
            mark.moveTo(indicatorRect.left() + indicatorSize * 0.25, indicatorRect.top() + indicatorSize * 0.53);
            mark.lineTo(indicatorRect.left() + indicatorSize * 0.43, indicatorRect.top() + indicatorSize * 0.70);
            mark.lineTo(indicatorRect.left() + indicatorSize * 0.76, indicatorRect.top() + indicatorSize * 0.30);
            painter.setPen(QPen(markColor, 1.8, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
            painter.drawPath(mark);
        }

        painter.setPen(palette().color(colorGroup, QPalette::Text));
        painter.drawText(textRect, QStyle::visualAlignment(layoutDirection(), Qt::AlignLeft) | Qt::AlignVCenter, text());
    }
};

class LegendMiniColorButton final : public QPushButton {
public:
    explicit LegendMiniColorButton(QWidget* parent = nullptr)
        : QPushButton(parent) {}

protected:
    void paintEvent(QPaintEvent* event) override {
        Q_UNUSED(event)

        QPainter painter(this);
        QStyleOptionButton option;
        initStyleOption(&option);
        option.text.clear();
        option.icon = QIcon();
        style()->drawControl(QStyle::CE_PushButton, &option, &painter, this);

        const QColor color = legendButtonColor(this);
        const int diameter = std::clamp(std::min(width(), height()) - 8,
                                        std::min(10, kLegendColorIndicatorDiameter),
                                        kLegendColorIndicatorDiameter);
        const QRectF circleRect((width() - diameter) * 0.5,
                                (height() - diameter) * 0.5,
                                diameter,
                                diameter);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(QPen(QColor(0, 0, 0, 92), 1.0));
        painter.setBrush(color);
        painter.drawEllipse(circleRect.adjusted(0.5, 0.5, -0.5, -0.5));
    }
};

class LegendPropertiesDialog final : public QDialog {
public:
    explicit LegendPropertiesDialog(QWidget* parent = nullptr)
        : QDialog(parent) {}

protected:
    void keyPressEvent(QKeyEvent* event) override {
        if (event != nullptr
            && (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter)) {
            commitFocusedEditor();
            event->accept();
            return;
        }
        QDialog::keyPressEvent(event);
    }

private:
    void commitFocusedEditor() {
        QWidget* focused = focusWidget();
        for (QWidget* current = focused; current != nullptr; current = current->parentWidget()) {
            if (auto* spin = qobject_cast<QAbstractSpinBox*>(current)) {
                spin->interpretText();
                spin->clearFocus();
                return;
            }
            if (auto* combo = qobject_cast<QComboBox*>(current)) {
                combo->hidePopup();
                combo->clearFocus();
                return;
            }
            if (auto* edit = qobject_cast<QLineEdit*>(current)) {
                edit->clearFocus();
                return;
            }
        }
        if (focused != nullptr) {
            focused->clearFocus();
        }
    }
};

QString canonicalLegendOrientation(const QString& value) {
    return value.trimmed().compare(QStringLiteral("Horizontal"), Qt::CaseInsensitive) == 0
        ? QStringLiteral("Horizontal")
        : QStringLiteral("Vertical");
}

int vtkLegendOrientation(const QString& value) {
    return canonicalLegendOrientation(value) == QStringLiteral("Horizontal")
        ? VTK_ORIENT_HORIZONTAL
        : VTK_ORIENT_VERTICAL;
}

QString canonicalLegendWindowLocation(const QString& value) {
    const QString normalized = value.trimmed().toLower();
    if (normalized == QStringLiteral("lower left")) {
        return QStringLiteral("Lower Left");
    }
    if (normalized == QStringLiteral("lower right")) {
        return QStringLiteral("Lower Right");
    }
    if (normalized == QStringLiteral("lower center")) {
        return QStringLiteral("Lower Center");
    }
    if (normalized == QStringLiteral("upper left")) {
        return QStringLiteral("Upper Left");
    }
    if (normalized == QStringLiteral("upper right")) {
        return QStringLiteral("Upper Right");
    }
    if (normalized == QStringLiteral("upper center")) {
        return QStringLiteral("Upper Center");
    }
    return QStringLiteral("Any Location");
}

QString legendWindowLocationDisplay(const QString& value) {
    const QString canonical = canonicalLegendWindowLocation(value);
    if (canonical == QStringLiteral("Lower Left")) {
        return QStringLiteral("Lower left");
    }
    if (canonical == QStringLiteral("Lower Right")) {
        return QStringLiteral("Lower right");
    }
    if (canonical == QStringLiteral("Lower Center")) {
        return QStringLiteral("Lower center");
    }
    if (canonical == QStringLiteral("Upper Left")) {
        return QStringLiteral("Upper left");
    }
    if (canonical == QStringLiteral("Upper Right")) {
        return QStringLiteral("Upper right");
    }
    if (canonical == QStringLiteral("Upper Center")) {
        return QStringLiteral("Upper center");
    }
    return QStringLiteral("Any location");
}

int vtkLegendWindowLocation(const QString& value) {
    const QString canonical = canonicalLegendWindowLocation(value);
    if (canonical == QStringLiteral("Lower Left")) {
        return vtkBorderRepresentation::LowerLeftCorner;
    }
    if (canonical == QStringLiteral("Lower Right")) {
        return vtkBorderRepresentation::LowerRightCorner;
    }
    if (canonical == QStringLiteral("Lower Center")) {
        return vtkBorderRepresentation::LowerCenter;
    }
    if (canonical == QStringLiteral("Upper Left")) {
        return vtkBorderRepresentation::UpperLeftCorner;
    }
    if (canonical == QStringLiteral("Upper Right")) {
        return vtkBorderRepresentation::UpperRightCorner;
    }
    if (canonical == QStringLiteral("Upper Center")) {
        return vtkBorderRepresentation::UpperCenter;
    }
    return vtkBorderRepresentation::AnyLocation;
}

QString canonicalLegendJustification(const QString& value) {
    const QString normalized = value.trimmed().toLower();
    if (normalized == QStringLiteral("left")) {
        return QStringLiteral("Left");
    }
    if (normalized == QStringLiteral("right")) {
        return QStringLiteral("Right");
    }
    return QStringLiteral("Centered");
}

QString canonicalLegendTitleOrientation(const QString& value) {
    return value.trimmed().compare(QStringLiteral("Horizontal"), Qt::CaseInsensitive) == 0
        ? QStringLiteral("Horizontal")
        : QStringLiteral("Vertical");
}

QStringList legendTitlePositionItems() {
    return {QStringLiteral("Left"),
            QStringLiteral("Right"),
            QStringLiteral("Top"),
            QStringLiteral("Bottom")};
}

QString canonicalLegendTitlePosition(const QString& value) {
    const QString normalized = value.trimmed().toLower();
    if (normalized == QStringLiteral("right")) {
        return QStringLiteral("Right");
    }
    if (normalized == QStringLiteral("top") || normalized == QStringLiteral("upper")) {
        return QStringLiteral("Top");
    }
    if (normalized == QStringLiteral("bottom") || normalized == QStringLiteral("lower")) {
        return QStringLiteral("Bottom");
    }
    return QStringLiteral("Left");
}

QString canonicalLegendFontFamily(const QString& value) {
    const QString trimmed = value.trimmed();
    if (trimmed.isEmpty()) {
        return QStringLiteral("Arial");
    }
    for (const QString& family : legendAvailableFontFamilies()) {
        if (family.compare(trimmed, Qt::CaseInsensitive) == 0) {
            return family;
        }
    }
    const QString normalized = trimmed.toLower();
    if (normalized.contains(QStringLiteral("courier"))) {
        return QStringLiteral("Courier");
    }
    if (normalized.contains(QStringLiteral("times"))) {
        return QStringLiteral("Times");
    }
    if (normalized.contains(QStringLiteral("arial"))) {
        return QStringLiteral("Arial");
    }
    return trimmed;
}

QString canonicalLegendTickAnnotationPosition(const QString& value) {
    const QString normalized = value.trimmed().toLower();
    if (normalized == QStringLiteral("left")
        || normalized == QStringLiteral("bottom")
        || normalized == QStringLiteral("left/bottom")
        || normalized.startsWith(QStringLiteral("ticks left"))
        || normalized.startsWith(QStringLiteral("ticks bottom"))) {
        return QStringLiteral("Ticks left/bottom, annotations right/top");
    }
    return QStringLiteral("Ticks right/top, annotations left/bottom");
}

QStringList legendTickPositionItems(const QString& orientation) {
    return vtkLegendOrientation(orientation) == VTK_ORIENT_HORIZONTAL
        ? QStringList{QStringLiteral("Top"), QStringLiteral("Bottom")}
        : QStringList{QStringLiteral("Right"), QStringLiteral("Left")};
}

QString legendTickPositionDisplay(const QString& storedPosition, const QString& orientation) {
    const bool leftOrBottom = canonicalLegendTickAnnotationPosition(storedPosition).startsWith(QStringLiteral("Ticks left"));
    if (vtkLegendOrientation(orientation) == VTK_ORIENT_HORIZONTAL) {
        return leftOrBottom ? QStringLiteral("Bottom") : QStringLiteral("Top");
    }
    return leftOrBottom ? QStringLiteral("Left") : QStringLiteral("Right");
}

QString legendTickPositionFromDisplay(const QString& displayPosition) {
    const QString normalized = displayPosition.trimmed().toLower();
    if (normalized == QStringLiteral("left") || normalized == QStringLiteral("bottom")) {
        return QStringLiteral("Ticks left/bottom, annotations right/top");
    }
    return QStringLiteral("Ticks right/top, annotations left/bottom");
}

QString canonicalLegendTickDirection(const QString& value) {
    const QString normalized = value.trimmed().toLower();
    if (normalized.contains(QStringLiteral("through"))) {
        return QStringLiteral("Through");
    }
    if (normalized.contains(QStringLiteral("center"))) {
        return QStringLiteral("Centered");
    }
    if (normalized.contains(QStringLiteral("in"))) {
        return QStringLiteral("Inward");
    }
    return QStringLiteral("Outward");
}

int legendInteriorTickLabelCount(const ViewerWidget::LegendOptions& options) {
    return std::clamp(options.labelCount, 0, 64);
}

int legendTotalTickLabelCount(const ViewerWidget::LegendOptions& options) {
    const int interiorCount = legendInteriorTickLabelCount(options);
    return options.addRangeLabels ? interiorCount + 2 : interiorCount;
}

double legendTickFractionForIndex(int index, const ViewerWidget::LegendOptions& options) {
    const int interiorCount = legendInteriorTickLabelCount(options);
    if (options.addRangeLabels) {
        const int totalCount = interiorCount + 2;
        return totalCount <= 1 ? 0.0 : static_cast<double>(index) / static_cast<double>(totalCount - 1);
    }
    return static_cast<double>(index + 1) / static_cast<double>(interiorCount + 1);
}

QStringList legendComponentFormatItems() {
    return {QStringLiteral("Same line as title (space)"),
            QStringLiteral("Same line as title (dash)"),
            QStringLiteral("Same line as title (comma)"),
            QStringLiteral("Same line as title (bracket)"),
            QStringLiteral("New line"),
            QStringLiteral("New line (bracket)")};
}

QString canonicalLegendComponentFormat(const QString& value) {
    const QString normalized = value.trimmed().toLower();
    if (normalized == QStringLiteral("same line as title (space)")
        || normalized == QStringLiteral("same line (space)")
        || normalized == QStringLiteral("space")) {
        return QStringLiteral("Same line as title (space)");
    }
    if (normalized == QStringLiteral("same line as title (comma)")
        || normalized == QStringLiteral("same line (comma)")
        || normalized == QStringLiteral("comma")) {
        return QStringLiteral("Same line as title (comma)");
    }
    if (normalized == QStringLiteral("same line as title (bracket)")
        || normalized == QStringLiteral("same line (bracket)")
        || normalized == QStringLiteral("bracket")) {
        return QStringLiteral("Same line as title (bracket)");
    }
    if (normalized == QStringLiteral("new line (dash)")
        || normalized == QStringLiteral("new line (comma)")) {
        return QStringLiteral("New line");
    }
    if (normalized == QStringLiteral("new line (bracket)")) {
        return QStringLiteral("New line (bracket)");
    }
    if (normalized == QStringLiteral("new line")) {
        return QStringLiteral("New line");
    }
    return QStringLiteral("Same line as title (space)");
}

bool legendComponentFormatUsesNewLine(const QString& value) {
    return canonicalLegendComponentFormat(value).startsWith(QStringLiteral("New line"));
}

QString formattedLegendTitleText(const ViewerWidget::LegendOptions& options) {
    const QString title = options.title.trimmed();
    const QString component = options.componentTitle.trimmed();
    if (component.isEmpty()) {
        return title;
    }
    if (title.isEmpty()) {
        return component;
    }

    const QString format = canonicalLegendComponentFormat(options.componentFormat);
    if (format == QStringLiteral("Same line as title (space)")) {
        return QStringLiteral("%1 %2").arg(title, component);
    }
    if (format == QStringLiteral("Same line as title (comma)")) {
        return QStringLiteral("%1, %2").arg(title, component);
    }
    if (format == QStringLiteral("Same line as title (bracket)")) {
        return QStringLiteral("%1 (%2)").arg(title, component);
    }
    if (format == QStringLiteral("New line")) {
        return QStringLiteral("%1\n%2").arg(title, component);
    }
    if (format == QStringLiteral("New line (bracket)")) {
        return QStringLiteral("%1\n(%2)").arg(title, component);
    }
    return QStringLiteral("%1 - %2").arg(title, component);
}

QString vtkLegendLabelFormat(const QString& value, const QString& fallback) {
    QString format = value.trimmed();
    if (format.startsWith(QStringLiteral("{:")) && format.endsWith(QLatin1Char('}'))) {
        format = format.mid(2, format.size() - 3).trimmed();
    } else if (format.startsWith(QLatin1Char('{')) && format.endsWith(QLatin1Char('}'))) {
        format = format.mid(1, format.size() - 2).trimmed();
    }
    if (!format.contains(QLatin1Char('%'))) {
        return fallback;
    }
    return format;
}

QString nativeLegendFontFile(const QString& family, bool bold, bool italic) {
    const QString canonical = canonicalLegendFontFamily(family);
    const QString projectFontFile = legendProjectFontFile(canonical);
    if (!projectFontFile.isEmpty() && QFileInfo::exists(projectFontFile)) {
        return projectFontFile;
    }
#ifdef Q_OS_WIN
    const QString windowsFonts = QStringLiteral("C:/Windows/Fonts/");
    QStringList fileNames;
    if (canonical == QStringLiteral("Courier")) {
        if (bold && italic) {
            fileNames << QStringLiteral("courbi.ttf");
        } else if (bold) {
            fileNames << QStringLiteral("courbd.ttf");
        } else if (italic) {
            fileNames << QStringLiteral("couri.ttf");
        }
        fileNames << QStringLiteral("cour.ttf");
    } else if (canonical == QStringLiteral("Times")) {
        if (bold && italic) {
            fileNames << QStringLiteral("timesbi.ttf");
        } else if (bold) {
            fileNames << QStringLiteral("timesbd.ttf");
        } else if (italic) {
            fileNames << QStringLiteral("timesi.ttf");
        }
        fileNames << QStringLiteral("times.ttf");
    } else {
        if (bold && italic) {
            fileNames << QStringLiteral("arialbi.ttf");
        } else if (bold) {
            fileNames << QStringLiteral("arialbd.ttf");
        } else if (italic) {
            fileNames << QStringLiteral("ariali.ttf");
        }
        fileNames << QStringLiteral("arial.ttf")
                  << QStringLiteral("Arial.ttf")
                  << QStringLiteral("ARIAL.TTF");
    }
    QStringList fontDirs{windowsFonts};
    fontDirs.append(QStandardPaths::standardLocations(QStandardPaths::FontsLocation));
    for (const QString& dir : fontDirs) {
        const QDir fontDir(dir);
        for (const QString& fileName : fileNames) {
            const QString path = fontDir.absoluteFilePath(fileName);
            if (QFileInfo::exists(path)) {
                return path;
            }
        }
    }
#else
    Q_UNUSED(canonical)
    Q_UNUSED(bold)
    Q_UNUSED(italic)
#endif
    return {};
}

void setLegendTextPropertyFont(vtkTextProperty* property, const ViewerWidget::LegendFontOptions& font) {
    if (property == nullptr) {
        return;
    }
    const QString family = canonicalLegendFontFamily(font.family);
    const QString fontFile = nativeLegendFontFile(family, font.bold, font.italic);
    if (!fontFile.isEmpty()) {
        const QByteArray fontFileUtf8 = fontFile.toUtf8();
        property->SetFontFamily(VTK_FONT_FILE);
        property->SetFontFile(fontFileUtf8.constData());
    } else if (family == QStringLiteral("Courier")) {
        property->SetFontFamilyToCourier();
    } else if (family == QStringLiteral("Times")) {
        property->SetFontFamilyToTimes();
    } else {
        property->SetFontFamilyToArial();
    }
    const QColor color = font.color.isValid() ? font.color : QColor(0, 0, 0);
    property->SetFontSize(std::clamp(font.size, 1, 200));
    property->SetColor(color.redF(), color.greenF(), color.blueF());
    property->SetOpacity(std::clamp(font.opacity, 0.0, 1.0) * color.alphaF());
    property->SetBold(font.bold ? 1 : 0);
    property->SetItalic(font.italic ? 1 : 0);
    property->SetShadow(font.shadow ? 1 : 0);
    property->SetShadowOffset(2, -2);
}

void setLegendTitleJustification(vtkTextProperty* property, const QString& value) {
    if (property == nullptr) {
        return;
    }
    const QString justification = canonicalLegendJustification(value);
    if (justification == QStringLiteral("Left")) {
        property->SetJustificationToLeft();
    } else if (justification == QStringLiteral("Right")) {
        property->SetJustificationToRight();
    } else {
        property->SetJustificationToCentered();
    }
}

QString legendQtFontFamilyName(const QString& family) {
    const QString canonical = canonicalLegendFontFamily(family);
    const QStringList installedFamilies = QFontDatabase::families();
    auto findInstalledFamily = [&](const QString& requested) {
        for (const QString& candidate : installedFamilies) {
            if (candidate.compare(requested, Qt::CaseInsensitive) == 0) {
                return candidate;
            }
        }
        return requested;
    };
    if (canonical == QStringLiteral("Courier")) {
        return findInstalledFamily(QStringLiteral("Courier New"));
    }
    if (canonical == QStringLiteral("Times")) {
        return findInstalledFamily(QStringLiteral("Times New Roman"));
    }
    if (!legendProjectFontFile(canonical).isEmpty()) {
        return findInstalledFamily(canonical);
    }
    return findInstalledFamily(QStringLiteral("Arial"));
}

QFont legendQtFont(const ViewerWidget::LegendFontOptions& font) {
    QFont qtFont(legendQtFontFamilyName(font.family));
    qtFont.setPointSize(std::clamp(font.size, 1, 200));
    qtFont.setBold(font.bold);
    qtFont.setItalic(font.italic);
    return qtFont;
}

QString defaultLegendTitle(const ViewerWidget::DataObjectOptions& options) {
    const QString field = options.colorField.trimmed();
    if (field.isEmpty()) {
        return QStringLiteral("Colors");
    }
    const QString component = options.colorComponent.trimmed();
    if (!component.isEmpty() && component.compare(QStringLiteral("Magnitude"), Qt::CaseInsensitive) != 0) {
        return QStringLiteral("%1 %2").arg(field, component);
    }
    return field;
}

bool legendFormatAcceptsSingleDouble(const QString& format) {
    int conversions = 0;
    const QString trimmed = vtkLegendLabelFormat(format, QStringLiteral("%-#6.3g"));
    for (int i = 0; i < trimmed.size(); ++i) {
        if (trimmed.at(i) != QLatin1Char('%')) {
            continue;
        }
        ++i;
        if (i < trimmed.size() && trimmed.at(i) == QLatin1Char('%')) {
            continue;
        }
        while (i < trimmed.size() && QStringLiteral("-+#0 ").contains(trimmed.at(i))) {
            ++i;
        }
        while (i < trimmed.size() && trimmed.at(i).isDigit()) {
            ++i;
        }
        if (i < trimmed.size() && trimmed.at(i) == QLatin1Char('.')) {
            ++i;
            while (i < trimmed.size() && trimmed.at(i).isDigit()) {
                ++i;
            }
        }
        while (i < trimmed.size() && QStringLiteral("hlLjzt").contains(trimmed.at(i))) {
            ++i;
        }
        if (i >= trimmed.size()) {
            return false;
        }
        if (QStringLiteral("aAeEfFgG").contains(trimmed.at(i))) {
            ++conversions;
        } else {
            return false;
        }
    }
    return conversions == 1;
}

QString formattedLegendNumber(double value, const QString& format) {
    const QString effectiveFormat = vtkLegendLabelFormat(format, QStringLiteral("%-#6.3g"));
    if (legendFormatAcceptsSingleDouble(effectiveFormat)) {
        char buffer[128] = {};
        const QByteArray formatUtf8 = effectiveFormat.toUtf8();
        const int count = std::snprintf(buffer, sizeof(buffer), formatUtf8.constData(), value);
        if (count > 0) {
            return QString::fromUtf8(buffer).trimmed();
        }
    }
    return QLocale::c().toString(value, 'g', 3);
}

int estimateLegendAnnotationWidthPixels(const ViewerWidget::LegendOptions& options, vtkScalarsToColors* lookupTable) {
    QStringList labels;
    if (options.drawTickLabels) {
        double range[2] = {0.0, 1.0};
        if (lookupTable != nullptr) {
            double* lookupRange = lookupTable->GetRange();
            if (lookupRange != nullptr) {
                range[0] = lookupRange[0];
                range[1] = lookupRange[1];
            }
        }
        const int count = legendTotalTickLabelCount(options);
        const QString labelFormat = options.automaticLabelFormat
            ? QStringLiteral("%-#6.3g")
            : vtkLegendLabelFormat(options.labelFormat, QStringLiteral("%-#6.3g"));
        for (int i = 0; i < count; ++i) {
            const double t = legendTickFractionForIndex(i, options);
            labels.push_back(formattedLegendNumber(range[0] + (range[1] - range[0]) * t, labelFormat));
        }
    }
    if (options.drawAnnotations && options.drawNanAnnotation && !options.nanAnnotation.trimmed().isEmpty()) {
        labels.push_back(options.nanAnnotation.trimmed());
    }

    QFontMetrics metrics(legendQtFont(options.textFont));
    int width = 0;
    for (const QString& label : labels) {
        width = std::max(width, metrics.horizontalAdvance(label));
    }
    return std::max(0, width);
}

int defaultLegendTitlePaddingPixels(const ViewerWidget::LegendOptions& options, vtkScalarsToColors* lookupTable) {
    Q_UNUSED(options)
    Q_UNUSED(lookupTable)
    return 50;
}

int effectiveLegendTitlePaddingPixels(const ViewerWidget::LegendOptions& options, vtkScalarsToColors* lookupTable) {
    if (options.titlePadding >= 0) {
        return std::clamp(options.titlePadding, 0, 512);
    }
    return defaultLegendTitlePaddingPixels(options, lookupTable);
}

vtkSmartPointer<vtkLookupTable> reversedLookupTable(vtkLookupTable* source) {
    if (source == nullptr) {
        return nullptr;
    }
    vtkSmartPointer<vtkLookupTable> reversed = vtkSmartPointer<vtkLookupTable>::New();
    reversed->DeepCopy(source);
    const int count = source->GetNumberOfTableValues();
    for (int i = 0; i < count; ++i) {
        double rgba[4] = {0.0, 0.0, 0.0, 1.0};
        source->GetTableValue(count - 1 - i, rgba);
        reversed->SetTableValue(i, rgba);
    }
    reversed->Build();
    return reversed;
}

QString magnitudeComponentName() {
    return QStringLiteral("Magnitude");
}

QString componentNameForAxis(int axis) {
    switch (axis) {
    case 0:
        return QStringLiteral("X");
    case 1:
        return QStringLiteral("Y");
    case 2:
        return QStringLiteral("Z");
    default:
        return {};
    }
}

int axisForComponentName(const QString& component) {
    const QString normalized = component.trimmed().toLower();
    if (normalized == QStringLiteral("x")) {
        return 0;
    }
    if (normalized == QStringLiteral("y")) {
        return 1;
    }
    if (normalized == QStringLiteral("z")) {
        return 2;
    }
    return -1;
}

QString vectorMagnitudeArrayName(const QString& fieldName) {
    return fieldName + QStringLiteral("_magnitude");
}

void insertSparseImageMeshLine(vtkPoints* points,
                               vtkCellArray* lines,
                               const std::array<double, 3>& a,
                               const std::array<double, 3>& b) {
    if (points == nullptr || lines == nullptr) {
        return;
    }
    const vtkIdType first = points->InsertNextPoint(a[0], a[1], a[2]);
    const vtkIdType second = points->InsertNextPoint(b[0], b[1], b[2]);
    vtkIdType ids[2] = {first, second};
    lines->InsertNextCell(2, ids);
}

std::vector<int> sparseGridIndices(int first, int last, int stride) {
    std::vector<int> indices;
    if (last < first) {
        return indices;
    }
    stride = std::max(1, stride);
    for (int value = first; value <= last; value += stride) {
        indices.push_back(value);
    }
    if (indices.empty() || indices.back() != last) {
        indices.push_back(last);
    }
    return indices;
}

vtkSmartPointer<vtkPolyData> makeSparseImagePlaneMesh(vtkImageData* imageData, int stride) {
    if (imageData == nullptr || stride <= 1) {
        return nullptr;
    }

    int extent[6] = {};
    int dimensions[3] = {};
    double origin[3] = {};
    double spacing[3] = {};
    imageData->GetExtent(extent);
    imageData->GetDimensions(dimensions);
    imageData->GetOrigin(origin);
    imageData->GetSpacing(spacing);

    int fixedAxis = -1;
    std::array<int, 2> planeAxes = {-1, -1};
    int planeAxisCount = 0;
    for (int axis = 0; axis < 3; ++axis) {
        if (dimensions[axis] <= 1) {
            fixedAxis = axis;
        } else if (planeAxisCount < 2) {
            planeAxes[planeAxisCount++] = axis;
        }
    }
    if (fixedAxis < 0 || planeAxisCount != 2) {
        return nullptr;
    }

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
    auto coordinate = [&](int axis, int index) {
        return origin[axis] + static_cast<double>(index - extent[axis * 2]) * spacing[axis];
    };

    const int uAxis = planeAxes[0];
    const int vAxis = planeAxes[1];
    const int uFirst = extent[uAxis * 2];
    const int uLast = extent[uAxis * 2 + 1];
    const int vFirst = extent[vAxis * 2];
    const int vLast = extent[vAxis * 2 + 1];
    const double fixedValue = coordinate(fixedAxis, extent[fixedAxis * 2]);

    for (int u : sparseGridIndices(uFirst, uLast, stride)) {
        std::array<double, 3> a = {0.0, 0.0, 0.0};
        std::array<double, 3> b = {0.0, 0.0, 0.0};
        a[fixedAxis] = fixedValue;
        b[fixedAxis] = fixedValue;
        a[uAxis] = coordinate(uAxis, u);
        b[uAxis] = coordinate(uAxis, u);
        a[vAxis] = coordinate(vAxis, vFirst);
        b[vAxis] = coordinate(vAxis, vLast);
        insertSparseImageMeshLine(points, lines, a, b);
    }
    for (int v : sparseGridIndices(vFirst, vLast, stride)) {
        std::array<double, 3> a = {0.0, 0.0, 0.0};
        std::array<double, 3> b = {0.0, 0.0, 0.0};
        a[fixedAxis] = fixedValue;
        b[fixedAxis] = fixedValue;
        a[uAxis] = coordinate(uAxis, uFirst);
        b[uAxis] = coordinate(uAxis, uLast);
        a[vAxis] = coordinate(vAxis, v);
        b[vAxis] = coordinate(vAxis, v);
        insertSparseImageMeshLine(points, lines, a, b);
    }

    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(points);
    polyData->SetLines(lines);
    return polyData;
}

bool parseComponentFieldName(const QString& name, QString* baseName, int* axis, bool* upperSuffix) {
    if (name.size() < 3 || name.at(name.size() - 2) != QLatin1Char('_')) {
        return false;
    }

    const QChar suffix = name.at(name.size() - 1);
    const QChar lower = suffix.toLower();
    int parsedAxis = -1;
    if (lower == QLatin1Char('x')) {
        parsedAxis = 0;
    } else if (lower == QLatin1Char('y')) {
        parsedAxis = 1;
    } else if (lower == QLatin1Char('z')) {
        parsedAxis = 2;
    } else {
        return false;
    }

    const QString parsedBase = name.left(name.size() - 2);
    if (parsedBase.trimmed().isEmpty()) {
        return false;
    }
    if (baseName != nullptr) {
        *baseName = parsedBase;
    }
    if (axis != nullptr) {
        *axis = parsedAxis;
    }
    if (upperSuffix != nullptr) {
        *upperSuffix = suffix.isUpper();
    }
    return true;
}

struct ComponentFieldGroup {
    vtkDataSetAttributes* attributes = nullptr;
    int association = kFieldAssociationPoints;
    QString baseName;
    bool upperSuffix = false;
    vtkDataArray* arrays[3] = {nullptr, nullptr, nullptr};
    QString arrayNames[3];
    int firstOrder = std::numeric_limits<int>::max();
};

int componentCount(const ComponentFieldGroup& group) {
    int count = 0;
    for (vtkDataArray* array : group.arrays) {
        if (array != nullptr) {
            ++count;
        }
    }
    return count;
}

bool componentGroupHasConsistentTuples(const ComponentFieldGroup& group) {
    vtkIdType tupleCount = -1;
    for (vtkDataArray* array : group.arrays) {
        if (array == nullptr) {
            continue;
        }
        if (tupleCount < 0) {
            tupleCount = array->GetNumberOfTuples();
        } else if (tupleCount != array->GetNumberOfTuples()) {
            return false;
        }
    }
    return tupleCount >= 0;
}

bool isValidComponentGroup(const ComponentFieldGroup& group) {
    return componentCount(group) >= 2 && componentGroupHasConsistentTuples(group);
}

QVector<ComponentFieldGroup> collectComponentFieldGroups(vtkDataSet* dataSet) {
    QVector<ComponentFieldGroup> groups;
    if (dataSet == nullptr) {
        return groups;
    }

    int order = 0;
    auto collectFrom = [&](vtkDataSetAttributes* attributes, int association) {
        if (attributes == nullptr) {
            return;
        }
        for (int index = 0; index < attributes->GetNumberOfArrays(); ++index, ++order) {
            vtkDataArray* array = attributes->GetArray(index);
            if (array == nullptr || array->GetName() == nullptr || array->GetNumberOfComponents() != 1) {
                continue;
            }
            QString baseName;
            int axis = -1;
            bool upperSuffix = false;
            const QString name = QString::fromUtf8(array->GetName());
            if (!parseComponentFieldName(name, &baseName, &axis, &upperSuffix)) {
                continue;
            }

            ComponentFieldGroup* group = nullptr;
            for (ComponentFieldGroup& candidate : groups) {
                if (candidate.attributes == attributes
                    && candidate.association == association
                    && candidate.baseName == baseName
                    && candidate.upperSuffix == upperSuffix) {
                    group = &candidate;
                    break;
                }
            }
            if (group == nullptr) {
                ComponentFieldGroup newGroup;
                newGroup.attributes = attributes;
                newGroup.association = association;
                newGroup.baseName = baseName;
                newGroup.upperSuffix = upperSuffix;
                groups.push_back(newGroup);
                group = &groups.back();
            }
            if (group->arrays[axis] == nullptr) {
                group->arrays[axis] = array;
                group->arrayNames[axis] = name;
                group->firstOrder = std::min(group->firstOrder, order);
            }
        }
    };

    collectFrom(dataSet->GetPointData(), kFieldAssociationPoints);
    collectFrom(dataSet->GetCellData(), kFieldAssociationCells);
    return groups;
}

QStringList componentNamesForGroup(const ComponentFieldGroup& group) {
    QStringList components{magnitudeComponentName()};
    for (int axis = 0; axis < 3; ++axis) {
        if (group.arrays[axis] != nullptr) {
            components.push_back(componentNameForAxis(axis));
        }
    }
    return components;
}

QStringList componentNamesForVectorArray(vtkDataArray* array) {
    QStringList components{magnitudeComponentName()};
    if (array == nullptr) {
        return components;
    }
    const int componentCount = std::min(array->GetNumberOfComponents(), 3);
    for (int axis = 0; axis < componentCount; ++axis) {
        components.push_back(componentNameForAxis(axis));
    }
    return components;
}

vtkDataArray* arrayFromAttributes(vtkDataSetAttributes* attributes, const QString& name) {
    if (attributes == nullptr || name.trimmed().isEmpty()) {
        return nullptr;
    }
    const QByteArray key = name.toUtf8();
    return attributes->GetArray(key.constData());
}

QString uniqueArrayName(vtkDataSetAttributes* attributes, const QString& preferredName) {
    if (arrayFromAttributes(attributes, preferredName) == nullptr) {
        return preferredName;
    }
    for (int suffix = 2; suffix < 10000; ++suffix) {
        const QString candidate = QStringLiteral("%1_%2").arg(preferredName).arg(suffix);
        if (arrayFromAttributes(attributes, candidate) == nullptr) {
            return candidate;
        }
    }
    return preferredName + QStringLiteral("_copy");
}

vtkDataArray* ensureComponentMagnitudeArray(vtkDataSetAttributes* attributes, const ComponentFieldGroup& group) {
    if (attributes == nullptr || !isValidComponentGroup(group)) {
        return nullptr;
    }

    const QString magnitudeName = vectorMagnitudeArrayName(group.baseName);
    if (vtkDataArray* existing = arrayFromAttributes(attributes, magnitudeName)) {
        return existing;
    }

    vtkIdType tupleCount = -1;
    for (vtkDataArray* component : group.arrays) {
        if (component != nullptr) {
            tupleCount = component->GetNumberOfTuples();
            break;
        }
    }
    if (tupleCount < 0) {
        return nullptr;
    }

    vtkNew<vtkFloatArray> magnitude;
    magnitude->SetName(magnitudeName.toUtf8().constData());
    magnitude->SetNumberOfComponents(1);
    magnitude->SetNumberOfTuples(tupleCount);
    for (vtkIdType tuple = 0; tuple < tupleCount; ++tuple) {
        double sum = 0.0;
        for (vtkDataArray* component : group.arrays) {
            if (component == nullptr) {
                continue;
            }
            const double value = component->GetComponent(tuple, 0);
            sum += value * value;
        }
        magnitude->SetValue(tuple, static_cast<float>(std::sqrt(sum)));
    }
    attributes->AddArray(magnitude);
    return arrayFromAttributes(attributes, magnitudeName);
}

void addSyntheticVectorArrayIfPossible(vtkDataSetAttributes* attributes, const ComponentFieldGroup& group) {
    if (attributes == nullptr || !isValidComponentGroup(group) || arrayFromAttributes(attributes, group.baseName) != nullptr) {
        return;
    }

    int highestAxis = 0;
    vtkIdType tupleCount = -1;
    for (int axis = 0; axis < 3; ++axis) {
        if (group.arrays[axis] != nullptr) {
            highestAxis = axis;
            tupleCount = group.arrays[axis]->GetNumberOfTuples();
        }
    }
    if (tupleCount < 0) {
        return;
    }

    vtkNew<vtkFloatArray> vector;
    vector->SetName(group.baseName.toUtf8().constData());
    vector->SetNumberOfComponents(highestAxis + 1);
    vector->SetNumberOfTuples(tupleCount);
    for (vtkIdType tuple = 0; tuple < tupleCount; ++tuple) {
        for (int axis = 0; axis <= highestAxis; ++axis) {
            const double value = group.arrays[axis] != nullptr ? group.arrays[axis]->GetComponent(tuple, 0) : 0.0;
            vector->SetComponent(tuple, axis, value);
        }
    }
    attributes->AddArray(vector);
}

vtkDataArray* ensureNativeVectorMagnitudeArray(vtkDataSetAttributes* attributes, vtkDataArray* vector, const QString& fieldName) {
    if (attributes == nullptr || vector == nullptr) {
        return nullptr;
    }
    const int vectorComponents = std::min(vector->GetNumberOfComponents(), 3);
    if (vectorComponents < 2) {
        return nullptr;
    }

    const QString magnitudeName = vectorMagnitudeArrayName(fieldName);
    if (vtkDataArray* existing = arrayFromAttributes(attributes, magnitudeName)) {
        return existing;
    }

    vtkNew<vtkFloatArray> magnitude;
    magnitude->SetName(magnitudeName.toUtf8().constData());
    magnitude->SetNumberOfComponents(1);
    magnitude->SetNumberOfTuples(vector->GetNumberOfTuples());
    for (vtkIdType tuple = 0; tuple < vector->GetNumberOfTuples(); ++tuple) {
        double sum = 0.0;
        for (int component = 0; component < vectorComponents; ++component) {
            const double value = vector->GetComponent(tuple, component);
            sum += value * value;
        }
        magnitude->SetValue(tuple, static_cast<float>(std::sqrt(sum)));
    }
    attributes->AddArray(magnitude);
    return arrayFromAttributes(attributes, magnitudeName);
}

vtkDataArray* ensureNativeVectorComponentArray(vtkDataSetAttributes* attributes,
                                               vtkDataArray* vector,
                                               const QString& fieldName,
                                               int axis) {
    if (attributes == nullptr || vector == nullptr || axis < 0 || axis >= vector->GetNumberOfComponents() || axis >= 3) {
        return nullptr;
    }

    const QString preferredName = QStringLiteral("__scplus_%1_%2").arg(fieldName, componentNameForAxis(axis).toLower());
    if (vtkDataArray* existing = arrayFromAttributes(attributes, preferredName)) {
        if (existing->GetNumberOfComponents() == 1 && existing->GetNumberOfTuples() == vector->GetNumberOfTuples()) {
            return existing;
        }
    }

    const QString componentArrayName = uniqueArrayName(attributes, preferredName);
    vtkNew<vtkFloatArray> componentArray;
    componentArray->SetName(componentArrayName.toUtf8().constData());
    componentArray->SetNumberOfComponents(1);
    componentArray->SetNumberOfTuples(vector->GetNumberOfTuples());
    for (vtkIdType tuple = 0; tuple < vector->GetNumberOfTuples(); ++tuple) {
        componentArray->SetValue(tuple, static_cast<float>(vector->GetComponent(tuple, axis)));
    }
    attributes->AddArray(componentArray);
    return arrayFromAttributes(attributes, componentArrayName);
}

struct CameraState {
    double position[3]{};
    double focalPoint[3]{};
    double viewUp[3]{};
    double clippingRange[2]{};
    double parallelScale = 1.0;
    double viewAngle = 30.0;
};

void setTextActorViewportPosition(vtkTextActor* actor,
                                  double x,
                                  double y,
                                  int fontSize,
                                  const QColor& color,
                                  bool rightAligned = false,
                                  bool bottomAligned = false) {
    if (actor == nullptr) {
        return;
    }

    actor->SetTextScaleModeToNone();
    actor->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
    actor->SetPosition(x, y);
    actor->GetTextProperty()->SetFontSize(fontSize);
    actor->GetTextProperty()->SetColor(color.redF(), color.greenF(), color.blueF());
    actor->GetTextProperty()->SetBold(false);
    actor->GetTextProperty()->SetItalic(false);
    actor->GetTextProperty()->SetShadow(false);
    actor->GetTextProperty()->SetOpacity(color.alphaF());
    actor->GetTextProperty()->SetJustification(rightAligned ? VTK_TEXT_RIGHT : VTK_TEXT_LEFT);
    actor->GetTextProperty()->SetVerticalJustification(bottomAligned ? VTK_TEXT_BOTTOM : VTK_TEXT_TOP);
}

void applyStreamcenterLogoTextStyle(vtkTextActor* actor) {
    if (actor == nullptr || actor->GetTextProperty() == nullptr) {
        return;
    }

    vtkTextProperty* property = actor->GetTextProperty();
    property->SetBold(true);
    property->SetItalic(false);
    property->SetShadow(false);
#ifdef Q_OS_WIN
    const QString segoeBold = QStringLiteral("C:/Windows/Fonts/segoeuib.ttf");
    if (QFileInfo::exists(segoeBold)) {
        property->SetFontFamily(VTK_FONT_FILE);
        property->SetFontFile(segoeBold.toUtf8().constData());
        return;
    }
#endif
    property->SetFontFamilyAsString("Arial");
}

void configureAxisCaption(vtkCaptionActor2D* caption, const char* text) {
    if (caption == nullptr) {
        return;
    }

    caption->SetCaption(text);
    caption->BorderOff();
    caption->LeaderOff();
    caption->ThreeDimensionalLeaderOff();

    if (vtkTextActor* textActor = caption->GetTextActor()) {
        textActor->SetTextScaleModeToNone();
    }

    if (vtkTextProperty* property = caption->GetCaptionTextProperty()) {
        property->SetColor(0.0, 0.0, 0.0);
        property->SetFontSize(16);
        property->SetBold(false);
        property->SetItalic(false);
        property->SetShadow(false);
    }
}

vtkSmartPointer<vtkDataSet> cloneDataSet(vtkDataSet* source) {
    if (source == nullptr) {
        return nullptr;
    }

    vtkSmartPointer<vtkDataSet> copy;
    copy.TakeReference(vtkDataSet::SafeDownCast(source->NewInstance()));
    if (copy != nullptr) {
        copy->ShallowCopy(source);
    }
    return copy;
}

bool ghostArrayHasBit(vtkDataArray* array, vtkIdType tuple, unsigned char bit) {
    if (array == nullptr || tuple < 0 || tuple >= array->GetNumberOfTuples()) {
        return false;
    }
    const auto value = static_cast<unsigned char>(std::lround(array->GetComponent(tuple, 0)));
    return (value & bit) != 0;
}

bool cellShouldRender(vtkDataSet* dataSet,
                      vtkIdType cellId,
                      vtkDataArray* cellGhost,
                      vtkDataArray* pointGhost) {
    constexpr unsigned char kHiddenCellBits =
        static_cast<unsigned char>(vtkDataSetAttributes::HIDDENCELL | vtkDataSetAttributes::DUPLICATECELL);
    if (ghostArrayHasBit(cellGhost, cellId, kHiddenCellBits)) {
        return false;
    }
    if (dataSet == nullptr || pointGhost == nullptr) {
        return true;
    }

    vtkCell* cell = dataSet->GetCell(cellId);
    vtkIdList* pointIds = cell != nullptr ? cell->GetPointIds() : nullptr;
    if (pointIds == nullptr) {
        return true;
    }
    for (vtkIdType index = 0; index < pointIds->GetNumberOfIds(); ++index) {
        if (ghostArrayHasBit(pointGhost, pointIds->GetId(index), vtkDataSetAttributes::HIDDENPOINT)) {
            return false;
        }
    }
    return true;
}

vtkSmartPointer<vtkDataSet> renderableDataSetWithoutHiddenGhostCells(vtkDataSet* input) {
    if (input == nullptr || input->GetNumberOfCells() <= 0) {
        vtkSmartPointer<vtkDataSet> retained;
        retained = input;
        return retained;
    }

    vtkDataArray* cellGhost = input->GetCellData() != nullptr
        ? vtkDataArray::SafeDownCast(input->GetCellData()->GetArray(vtkDataSetAttributes::GhostArrayName()))
        : nullptr;
    vtkDataArray* pointGhost = input->GetPointData() != nullptr
        ? vtkDataArray::SafeDownCast(input->GetPointData()->GetArray(vtkDataSetAttributes::GhostArrayName()))
        : nullptr;
    if (cellGhost == nullptr && pointGhost == nullptr) {
        vtkSmartPointer<vtkDataSet> retained;
        retained = input;
        return retained;
    }

    constexpr const char* kRenderableCellArrayName = "__scplus_renderable_cell";
    auto visibleCells = vtkSmartPointer<vtkUnsignedCharArray>::New();
    visibleCells->SetName(kRenderableCellArrayName);
    visibleCells->SetNumberOfComponents(1);
    visibleCells->SetNumberOfTuples(input->GetNumberOfCells());

    bool hasHiddenCells = false;
    for (vtkIdType cellId = 0; cellId < input->GetNumberOfCells(); ++cellId) {
        const bool visible = cellShouldRender(input, cellId, cellGhost, pointGhost);
        visibleCells->SetValue(cellId, visible ? 1 : 0);
        hasHiddenCells = hasHiddenCells || !visible;
    }
    if (!hasHiddenCells) {
        vtkSmartPointer<vtkDataSet> retained;
        retained = input;
        return retained;
    }

    vtkSmartPointer<vtkDataSet> taggedInput = cloneDataSet(input);
    if (taggedInput == nullptr || taggedInput->GetCellData() == nullptr) {
        vtkSmartPointer<vtkDataSet> retained;
        retained = input;
        return retained;
    }
    taggedInput->GetCellData()->AddArray(visibleCells);

    vtkNew<vtkThreshold> threshold;
    threshold->SetInputData(taggedInput);
    threshold->SetInputArrayToProcess(0,
                                      0,
                                      0,
                                      vtkDataObject::FIELD_ASSOCIATION_CELLS,
                                      kRenderableCellArrayName);
    threshold->SetLowerThreshold(1.0);
    threshold->SetUpperThreshold(1.0);
    threshold->SetThresholdFunction(vtkThreshold::THRESHOLD_BETWEEN);
    threshold->Update();

    vtkSmartPointer<vtkDataSet> output = cloneDataSet(vtkDataSet::SafeDownCast(threshold->GetOutput()));
    if (output == nullptr) {
        vtkSmartPointer<vtkDataSet> retained;
        retained = input;
        return retained;
    }
    if (output->GetCellData() != nullptr) {
        output->GetCellData()->RemoveArray(kRenderableCellArrayName);
    }
    return output;
}

CameraState captureCamera(vtkRenderer* renderer) {
    CameraState state;
    if (renderer == nullptr || renderer->GetActiveCamera() == nullptr) {
        return state;
    }

    vtkCamera* camera = renderer->GetActiveCamera();
    camera->GetPosition(state.position);
    camera->GetFocalPoint(state.focalPoint);
    camera->GetViewUp(state.viewUp);
    camera->GetClippingRange(state.clippingRange);
    state.parallelScale = camera->GetParallelScale();
    state.viewAngle = camera->GetViewAngle();
    return state;
}

void restoreCamera(vtkRenderer* renderer, const CameraState& state, bool parallelProjection) {
    if (renderer == nullptr || renderer->GetActiveCamera() == nullptr) {
        return;
    }

    vtkCamera* camera = renderer->GetActiveCamera();
    camera->SetPosition(state.position);
    camera->SetFocalPoint(state.focalPoint);
    camera->SetViewUp(state.viewUp);
    camera->SetClippingRange(state.clippingRange);
    camera->SetParallelScale(state.parallelScale);
    camera->SetViewAngle(state.viewAngle);
    camera->SetParallelProjection(parallelProjection ? 1 : 0);
    renderer->ResetCameraClippingRange();
}

bool usableCameraBounds(const double bounds[6]) {
    for (int i = 0; i < 6; ++i) {
        if (!std::isfinite(bounds[i])) {
            return false;
        }
    }
    return bounds[0] <= bounds[1]
        && bounds[2] <= bounds[3]
        && bounds[4] <= bounds[5];
}

double normalizeVector(double vector[3]) {
    const double length = std::sqrt(vector[0] * vector[0]
                                    + vector[1] * vector[1]
                                    + vector[2] * vector[2]);
    if (length <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }
    vector[0] /= length;
    vector[1] /= length;
    vector[2] /= length;
    return length;
}

void crossVector(const double lhs[3], const double rhs[3], double out[3]) {
    out[0] = lhs[1] * rhs[2] - lhs[2] * rhs[1];
    out[1] = lhs[2] * rhs[0] - lhs[0] * rhs[2];
    out[2] = lhs[0] * rhs[1] - lhs[1] * rhs[0];
}

double dotVector(const double lhs[3], const double rhs[3]) {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

void cameraViewVectors(ViewerWidget::CameraView view, double direction[3], double viewUp[3]) {
    switch (view) {
    case ViewerWidget::CameraView::RightPositiveY:
        direction[0] = 0.0;
        direction[1] = 1.0;
        direction[2] = 0.0;
        viewUp[0] = 0.0;
        viewUp[1] = 0.0;
        viewUp[2] = 1.0;
        break;
    case ViewerWidget::CameraView::LeftNegativeY:
        direction[0] = 0.0;
        direction[1] = -1.0;
        direction[2] = 0.0;
        viewUp[0] = 0.0;
        viewUp[1] = 0.0;
        viewUp[2] = 1.0;
        break;
    case ViewerWidget::CameraView::TopPositiveZ:
        direction[0] = 0.0;
        direction[1] = 0.0;
        direction[2] = 1.0;
        viewUp[0] = 0.0;
        viewUp[1] = 1.0;
        viewUp[2] = 0.0;
        break;
    case ViewerWidget::CameraView::BottomNegativeZ:
        direction[0] = 0.0;
        direction[1] = 0.0;
        direction[2] = -1.0;
        viewUp[0] = 0.0;
        viewUp[1] = -1.0;
        viewUp[2] = 0.0;
        break;
    case ViewerWidget::CameraView::FrontNegativeX:
        direction[0] = -1.0;
        direction[1] = 0.0;
        direction[2] = 0.0;
        viewUp[0] = 0.0;
        viewUp[1] = 0.0;
        viewUp[2] = 1.0;
        break;
    case ViewerWidget::CameraView::BackPositiveX:
        direction[0] = 1.0;
        direction[1] = 0.0;
        direction[2] = 0.0;
        viewUp[0] = 0.0;
        viewUp[1] = 0.0;
        viewUp[2] = 1.0;
        break;
    case ViewerWidget::CameraView::Isometric:
    default:
        direction[0] = -1.0;
        direction[1] = 1.0;
        direction[2] = 1.0;
        viewUp[0] = 0.0;
        viewUp[1] = 0.0;
        viewUp[2] = 1.0;
        break;
    }
    normalizeVector(direction);
    normalizeVector(viewUp);
}

void cameraBoundsProjectionExtents(const double bounds[6],
                                   const double center[3],
                                   const double right[3],
                                   const double viewUp[3],
                                   const double directionOfProjection[3],
                                   double* halfWidth,
                                   double* halfHeight,
                                   double* halfDepth) {
    double maxWidth = 0.0;
    double maxHeight = 0.0;
    double maxDepth = 0.0;
    for (int ix = 0; ix < 2; ++ix) {
        for (int iy = 0; iy < 2; ++iy) {
            for (int iz = 0; iz < 2; ++iz) {
                const double corner[3] = {
                    bounds[ix == 0 ? 0 : 1],
                    bounds[iy == 0 ? 2 : 3],
                    bounds[iz == 0 ? 4 : 5],
                };
                const double offset[3] = {
                    corner[0] - center[0],
                    corner[1] - center[1],
                    corner[2] - center[2],
                };
                maxWidth = std::max(maxWidth, std::abs(dotVector(offset, right)));
                maxHeight = std::max(maxHeight, std::abs(dotVector(offset, viewUp)));
                maxDepth = std::max(maxDepth, std::abs(dotVector(offset, directionOfProjection)));
            }
        }
    }
    if (halfWidth != nullptr) {
        *halfWidth = maxWidth;
    }
    if (halfHeight != nullptr) {
        *halfHeight = maxHeight;
    }
    if (halfDepth != nullptr) {
        *halfDepth = maxDepth;
    }
}

void expandCameraForTargetAspect(vtkRenderer* renderer, double sourceAspect, double targetAspect) {
    if (renderer == nullptr || renderer->GetActiveCamera() == nullptr
        || sourceAspect <= 0.0 || targetAspect <= 0.0 || targetAspect >= sourceAspect) {
        return;
    }

    vtkCamera* camera = renderer->GetActiveCamera();
    const double aspectExpansion = sourceAspect / targetAspect;
    if (camera->GetParallelProjection()) {
        camera->SetParallelScale(camera->GetParallelScale() * aspectExpansion);
        return;
    }

    constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;
    constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
    const double currentAngleRadians = camera->GetViewAngle() * kDegreesToRadians;
    const double expandedAngleRadians =
        2.0 * std::atan(std::tan(currentAngleRadians * 0.5) * aspectExpansion);
    camera->SetViewAngle(std::clamp(expandedAngleRadians * kRadiansToDegrees, 1.0, 170.0));
}

QString incrementUnsignedDecimalString(QString digits) {
    if (digits.isEmpty()) {
        return QStringLiteral("1");
    }
    for (int index = digits.size() - 1; index >= 0; --index) {
        if (digits[index] < QLatin1Char('9')) {
            digits[index] = QChar(digits[index].unicode() + 1);
            return digits;
        }
        digits[index] = QLatin1Char('0');
    }
    digits.prepend(QLatin1Char('1'));
    return digits;
}

double roundedTimeCodeFromText(const QString& rawText, bool* ok) {
    if (ok != nullptr) {
        *ok = false;
    }
    QString text = rawText.trimmed();
    bool numericOk = false;
    const double numericValue = text.toDouble(&numericOk);
    if (!numericOk || !std::isfinite(numericValue)) {
        return 0.0;
    }

    if (text.contains(QLatin1Char('e'), Qt::CaseInsensitive)) {
        if (ok != nullptr) {
            *ok = true;
        }
        return std::round(numericValue * 10000000000.0) / 10000000000.0;
    }

    bool negative = false;
    if (text.startsWith(QLatin1Char('-')) || text.startsWith(QLatin1Char('+'))) {
        negative = text.startsWith(QLatin1Char('-'));
        text.remove(0, 1);
    }

    const int dotIndex = text.indexOf(QLatin1Char('.'));
    QString integerPart = dotIndex >= 0 ? text.left(dotIndex) : text;
    QString fractionPart = dotIndex >= 0 ? text.mid(dotIndex + 1) : QString();
    if (integerPart.isEmpty()) {
        integerPart = QStringLiteral("0");
    }
    while (fractionPart.size() < 11) {
        fractionPart.append(QLatin1Char('0'));
    }

    QString keptFraction = fractionPart.left(10);
    if (fractionPart.at(10) >= QLatin1Char('5')) {
        keptFraction = incrementUnsignedDecimalString(keptFraction);
        if (keptFraction.size() > 10) {
            keptFraction = keptFraction.right(10);
            integerPart = incrementUnsignedDecimalString(integerPart);
        }
    }

    QString roundedText = integerPart;
    if (!keptFraction.isEmpty()) {
        roundedText += QLatin1Char('.');
        roundedText += keptFraction;
    }
    if (negative) {
        roundedText.prepend(QLatin1Char('-'));
    }

    bool roundedOk = false;
    double roundedValue = roundedText.toDouble(&roundedOk);
    if (!roundedOk) {
        roundedValue = std::round(numericValue * 10000000000.0) / 10000000000.0;
    }
    if (std::abs(roundedValue) < 0.00000000005) {
        roundedValue = 0.0;
    }
    if (ok != nullptr) {
        *ok = true;
    }
    return roundedValue;
}

bool sameTimeCode(double lhs, double rhs) {
    return std::abs(lhs - rhs) <= 0.00000000001;
}

QString formatTimeCode(double value) {
    QString text = QString::number(value, 'f', 10);
    while (text.contains(QLatin1Char('.')) && text.endsWith(QLatin1Char('0'))) {
        text.chop(1);
    }
    if (text.endsWith(QLatin1Char('.'))) {
        text.chop(1);
    }
    if (text == QStringLiteral("-0")) {
        return QStringLiteral("0");
    }
    return text;
}

int nearestFrameIndexForTime(const QVector<ViewerWidget::FrameInfo>& frames, double timeCode) {
    if (frames.isEmpty()) {
        return -1;
    }
    int bestIndex = 0;
    double bestDistance = std::numeric_limits<double>::max();
    for (int index = 0; index < frames.size(); ++index) {
        const double distance = std::abs(frames.at(index).timestep - timeCode);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = index;
        }
    }
    return bestIndex;
}

QVector<ViewerWidget::FrameInfo> parsePvdFile(const QString& path, QString* errorMessage) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (errorMessage != nullptr) {
            *errorMessage = QString("Cannot open PVD file: %1").arg(file.errorString());
        }
        return {};
    }

    QVector<ViewerWidget::FrameInfo> frames;
    QXmlStreamReader xml(&file);
    const QFileInfo pvdInfo(path);

    while (!xml.atEnd()) {
        xml.readNext();
        if (!xml.isStartElement() || xml.name() != QStringLiteral("DataSet")) {
            continue;
        }

        ViewerWidget::FrameInfo frame;
        const auto attrs = xml.attributes();
        frame.path = QDir(pvdInfo.absolutePath()).absoluteFilePath(attrs.value("file").toString());
        bool ok = false;
        frame.timestep = roundedTimeCodeFromText(attrs.value("timestep").toString(), &ok);
        if (!ok) {
            frame.timestep = 0.0;
        }
        if (!frame.path.trimmed().isEmpty()) {
            frames.push_back(frame);
        }
    }

    if (xml.hasError()) {
        if (errorMessage != nullptr) {
            *errorMessage = QString("Cannot parse PVD file: %1").arg(xml.errorString());
        }
        return {};
    }

    std::sort(frames.begin(), frames.end(), [](const ViewerWidget::FrameInfo& a, const ViewerWidget::FrameInfo& b) {
        if (!sameTimeCode(a.timestep, b.timestep)) {
            return a.timestep < b.timestep;
        }
        return a.path.compare(b.path, Qt::CaseInsensitive) < 0;
    });

    QVector<ViewerWidget::FrameInfo> uniqueFrames;
    uniqueFrames.reserve(frames.size());
    for (const ViewerWidget::FrameInfo& frame : std::as_const(frames)) {
        if (uniqueFrames.isEmpty() || !sameTimeCode(uniqueFrames.back().timestep, frame.timestep)) {
            uniqueFrames.push_back(frame);
        }
    }
    frames = uniqueFrames;

    if (frames.isEmpty() && errorMessage != nullptr) {
        *errorMessage = "No DataSet entries were found in the PVD file.";
    }
    return frames;
}

QString preferredFallbackScalarName(vtkDataSet* dataSet) {
    if (dataSet == nullptr) {
        return {};
    }

    auto firstMatchingPointArray = [&](const QStringList& preferredNames) -> QString {
        vtkPointData* pointData = dataSet->GetPointData();
        if (pointData == nullptr) {
            return {};
        }
        for (const QString& name : preferredNames) {
            if (vtkDataArray* array = pointData->GetArray(name.toUtf8().constData())) {
                if (array->GetNumberOfComponents() == 1) {
                    return name;
                }
            }
        }
        for (int index = 0; index < pointData->GetNumberOfArrays(); ++index) {
            vtkDataArray* array = pointData->GetArray(index);
            if (array != nullptr && array->GetNumberOfComponents() == 1 && array->GetName() != nullptr) {
                return QString::fromUtf8(array->GetName());
            }
        }
        return {};
    };

    QString fallback = firstMatchingPointArray({QStringLiteral("phi_mag"),
                                                QStringLiteral("mean_mag"),
                                                QStringLiteral("valid_mask"),
                                                QStringLiteral("weight")});
    if (!fallback.isEmpty()) {
        return fallback;
    }

    vtkCellData* cellData = dataSet->GetCellData();
    if (cellData != nullptr) {
        for (int index = 0; index < cellData->GetNumberOfArrays(); ++index) {
            vtkDataArray* array = cellData->GetArray(index);
            if (array != nullptr && array->GetNumberOfComponents() == 1 && array->GetName() != nullptr) {
                return QString::fromUtf8(array->GetName());
            }
        }
    }
    return {};
}

vtkDataArray* findArray(vtkDataSet* dataSet, const QString& name, int* association) {
    if (dataSet == nullptr || name.trimmed().isEmpty()) {
        return nullptr;
    }
    const QByteArray key = name.toUtf8();
    if (dataSet->GetPointData() != nullptr) {
        if (vtkDataArray* array = dataSet->GetPointData()->GetArray(key.constData())) {
            if (association != nullptr) {
                *association = kFieldAssociationPoints;
            }
            return array;
        }
    }
    if (dataSet->GetCellData() != nullptr) {
        if (vtkDataArray* array = dataSet->GetCellData()->GetArray(key.constData())) {
            if (association != nullptr) {
                *association = kFieldAssociationCells;
            }
            return array;
        }
    }
    return nullptr;
}

void addLegacyVelocityMagnitudeIfPossible(vtkDataSet* dataSet) {
    if (dataSet == nullptr) {
        return;
    }
    auto addForAttributes = [](vtkDataSetAttributes* attributes) {
        if (attributes == nullptr || attributes->GetArray("velocity_magnitude") != nullptr) {
            return;
        }
        vtkDataArray* u = attributes->GetArray("u");
        vtkDataArray* v = attributes->GetArray("v");
        vtkDataArray* w = attributes->GetArray("w");
        if (u == nullptr || v == nullptr || w == nullptr || u->GetNumberOfTuples() != v->GetNumberOfTuples()
            || u->GetNumberOfTuples() != w->GetNumberOfTuples()) {
            return;
        }
        vtkNew<vtkFloatArray> magnitude;
        magnitude->SetName("velocity_magnitude");
        magnitude->SetNumberOfComponents(1);
        magnitude->SetNumberOfTuples(u->GetNumberOfTuples());
        for (vtkIdType i = 0; i < u->GetNumberOfTuples(); ++i) {
            const double uu = u->GetComponent(i, 0);
            const double vv = v->GetComponent(i, 0);
            const double ww = w->GetComponent(i, 0);
            magnitude->SetValue(i, static_cast<float>(std::sqrt(uu * uu + vv * vv + ww * ww)));
        }
        attributes->AddArray(magnitude);
    };
    addForAttributes(dataSet->GetPointData());
    addForAttributes(dataSet->GetCellData());
}

void prepareVectorFields(vtkDataSet* dataSet) {
    if (dataSet == nullptr) {
        return;
    }

    addLegacyVelocityMagnitudeIfPossible(dataSet);

    const QVector<ComponentFieldGroup> groups = collectComponentFieldGroups(dataSet);
    for (const ComponentFieldGroup& group : groups) {
        if (!isValidComponentGroup(group)) {
            continue;
        }
        addSyntheticVectorArrayIfPossible(group.attributes, group);
        ensureComponentMagnitudeArray(group.attributes, group);
    }

    auto addNativeVectorMagnitudes = [](vtkDataSetAttributes* attributes) {
        if (attributes == nullptr) {
            return;
        }
        for (int i = 0; i < attributes->GetNumberOfArrays(); ++i) {
            vtkDataArray* array = attributes->GetArray(i);
            if (array == nullptr || array->GetName() == nullptr) {
                continue;
            }
            const QString name = QString::fromUtf8(array->GetName());
            const int components = array->GetNumberOfComponents();
            if (components >= 2 && components <= 3) {
                ensureNativeVectorMagnitudeArray(attributes, array, name);
            }
        }
    };
    addNativeVectorMagnitudes(dataSet->GetPointData());
    addNativeVectorMagnitudes(dataSet->GetCellData());
}

struct ArrayRecord {
    vtkDataSetAttributes* attributes = nullptr;
    int association = kFieldAssociationPoints;
    vtkDataArray* array = nullptr;
    QString name;
};

QVector<ArrayRecord> collectArrayRecords(vtkDataSet* dataSet) {
    QVector<ArrayRecord> records;
    if (dataSet == nullptr) {
        return records;
    }

    auto collectFrom = [&records](vtkDataSetAttributes* attributes, int association) {
        if (attributes == nullptr) {
            return;
        }
        for (int index = 0; index < attributes->GetNumberOfArrays(); ++index) {
            vtkDataArray* array = attributes->GetArray(index);
            if (array == nullptr || array->GetName() == nullptr) {
                continue;
            }
            ArrayRecord record;
            record.attributes = attributes;
            record.association = association;
            record.array = array;
            record.name = QString::fromUtf8(array->GetName());
            records.push_back(record);
        }
    };
    collectFrom(dataSet->GetPointData(), kFieldAssociationPoints);
    collectFrom(dataSet->GetCellData(), kFieldAssociationCells);
    return records;
}

const ComponentFieldGroup* groupForComponentArray(const QVector<ComponentFieldGroup>& groups, const QString& name) {
    QString baseName;
    int axis = -1;
    bool upperSuffix = false;
    if (!parseComponentFieldName(name, &baseName, &axis, &upperSuffix)) {
        return nullptr;
    }
    for (const ComponentFieldGroup& group : groups) {
        if (group.baseName == baseName
            && group.upperSuffix == upperSuffix
            && group.arrayNames[axis] == name
            && isValidComponentGroup(group)) {
            return &group;
        }
    }
    return nullptr;
}

const ComponentFieldGroup* groupForBaseName(const QVector<ComponentFieldGroup>& groups, const QString& baseName) {
    const ComponentFieldGroup* best = nullptr;
    for (const ComponentFieldGroup& group : groups) {
        if (group.baseName != baseName || !isValidComponentGroup(group)) {
            continue;
        }
        if (best == nullptr || group.firstOrder < best->firstOrder) {
            best = &group;
        }
    }
    return best;
}

bool appendFieldOption(QVector<ViewerWidget::FieldOption>* options,
                       QStringList* seenLower,
                       const ViewerWidget::FieldOption& option) {
    if (options == nullptr || seenLower == nullptr || option.name.trimmed().isEmpty()) {
        return false;
    }
    const QString lower = option.name.toLower();
    if (seenLower->contains(lower)) {
        return false;
    }
    options->push_back(option);
    seenLower->push_back(lower);
    return true;
}

QVector<ViewerWidget::FieldOption> fieldOptions(vtkDataSet* dataSet,
                                                QString* preferredField = nullptr,
                                                QString* preferredComponent = nullptr) {
    QVector<ViewerWidget::FieldOption> options;
    if (dataSet == nullptr) {
        return options;
    }

    prepareVectorFields(dataSet);
    const QVector<ComponentFieldGroup> groups = collectComponentFieldGroups(dataSet);
    const QVector<ArrayRecord> records = collectArrayRecords(dataSet);
    QStringList seenLower;
    QStringList hiddenMagnitudeLower;

    for (const ComponentFieldGroup& group : groups) {
        if (isValidComponentGroup(group)) {
            hiddenMagnitudeLower.push_back(vectorMagnitudeArrayName(group.baseName).toLower());
        }
    }
    for (const ArrayRecord& record : records) {
        const int components = record.array->GetNumberOfComponents();
        if (components >= 2 && components <= 3) {
            hiddenMagnitudeLower.push_back(vectorMagnitudeArrayName(record.name).toLower());
        }
    }

    for (const ArrayRecord& record : records) {
        const QString lower = record.name.toLower();
        if (record.name.startsWith(QStringLiteral("__scplus_"))) {
            continue;
        }
        if (hiddenMagnitudeLower.contains(lower)) {
            continue;
        }

        if (const ComponentFieldGroup* group = groupForComponentArray(groups, record.name)) {
            ViewerWidget::FieldOption option;
            option.name = group->baseName;
            option.components = componentNamesForGroup(*group);
            appendFieldOption(&options, &seenLower, option);
            continue;
        }
        if (const ComponentFieldGroup* group = groupForBaseName(groups, record.name)) {
            ViewerWidget::FieldOption option;
            option.name = group->baseName;
            option.components = componentNamesForGroup(*group);
            appendFieldOption(&options, &seenLower, option);
            continue;
        }

        const int components = record.array->GetNumberOfComponents();
        if (components >= 2 && components <= 3) {
            ViewerWidget::FieldOption option;
            option.name = record.name;
            option.components = componentNamesForVectorArray(record.array);
            appendFieldOption(&options, &seenLower, option);
        } else if (components == 1) {
            ViewerWidget::FieldOption option;
            option.name = record.name;
            appendFieldOption(&options, &seenLower, option);
        }
    }

    if (preferredField != nullptr) {
        preferredField->clear();
    }
    if (preferredComponent != nullptr) {
        preferredComponent->clear();
    }
    for (const ViewerWidget::FieldOption& option : options) {
        if (!option.components.isEmpty()) {
            if (preferredField != nullptr) {
                *preferredField = option.name;
            }
            if (preferredComponent != nullptr) {
                *preferredComponent = magnitudeComponentName();
            }
            return options;
        }
        QString ignoredBase;
        if (parseComponentFieldName(option.name, &ignoredBase, nullptr, nullptr)) {
            continue;
        }
        if (preferredField != nullptr) {
            *preferredField = option.name;
        }
        if (preferredComponent != nullptr) {
            *preferredComponent = magnitudeComponentName();
        }
        return options;
    }
    return options;
}

struct ResolvedScalarField {
    vtkDataArray* array = nullptr;
    int association = kFieldAssociationPoints;
    QString arrayName;
};

ResolvedScalarField resolveScalarField(vtkDataSet* dataSet, const QString& fieldName, const QString& componentName) {
    ResolvedScalarField resolved;
    if (dataSet == nullptr || fieldName.trimmed().isEmpty()) {
        return resolved;
    }

    prepareVectorFields(dataSet);
    const int requestedAxis = axisForComponentName(componentName);
    const QVector<ComponentFieldGroup> groups = collectComponentFieldGroups(dataSet);
    if (const ComponentFieldGroup* group = groupForBaseName(groups, fieldName)) {
        if (requestedAxis >= 0 && group->arrays[requestedAxis] != nullptr) {
            resolved.array = group->arrays[requestedAxis];
            resolved.association = group->association;
            resolved.arrayName = group->arrayNames[requestedAxis];
            return resolved;
        }

        resolved.array = ensureComponentMagnitudeArray(group->attributes, *group);
        resolved.association = group->association;
        resolved.arrayName = vectorMagnitudeArrayName(group->baseName);
        return resolved;
    }

    int association = kFieldAssociationPoints;
    vtkDataArray* array = findArray(dataSet, fieldName, &association);
    if (array == nullptr) {
        return resolved;
    }

    if (array->GetNumberOfComponents() == 1) {
        resolved.array = array;
        resolved.association = association;
        resolved.arrayName = fieldName;
        return resolved;
    }

    vtkDataSetAttributes* attributes = nullptr;
    if (association == kFieldAssociationPoints) {
        attributes = dataSet->GetPointData();
    } else {
        attributes = dataSet->GetCellData();
    }
    if (requestedAxis >= 0) {
        resolved.array = ensureNativeVectorComponentArray(attributes, array, fieldName, requestedAxis);
        resolved.association = association;
        if (resolved.array != nullptr && resolved.array->GetName() != nullptr) {
            resolved.arrayName = QString::fromUtf8(resolved.array->GetName());
        }
        return resolved;
    }

    resolved.array = ensureNativeVectorMagnitudeArray(attributes, array, fieldName);
    resolved.association = association;
    resolved.arrayName = vectorMagnitudeArrayName(fieldName);
    return resolved;
}

vtkSmartPointer<vtkLookupTable> makeLookupTable(const QString& name) {
    vtkSmartPointer<vtkLookupTable> table = vtkSmartPointer<vtkLookupTable>::New();
    if (const Streamcenter::ColorMapDefinition* definition = Streamcenter::activeColorMapByName(name)) {
        Streamcenter::fillLookupTable(table.GetPointer(), *definition, 256);
        return table;
    }

    const QString fallbackName = Streamcenter::activeDefaultColorMapName();
    if (const Streamcenter::ColorMapDefinition* fallback = Streamcenter::activeColorMapByName(fallbackName)) {
        Streamcenter::fillLookupTable(table.GetPointer(), *fallback, 256);
        return table;
    }

    table->SetNumberOfTableValues(2);
    table->SetTableValue(0, 0.0, 0.0, 0.0, 1.0);
    table->SetTableValue(1, 1.0, 1.0, 1.0, 1.0);
    table->Build();
    return table;
}

template <typename MapperT>
vtkSmartPointer<vtkLookupTable> configureMapperScalars(MapperT* mapper,
                                                       vtkDataSet* dataSet,
                                                       const ViewerWidget::DataObjectOptions& options) {
    if (mapper == nullptr || dataSet == nullptr || options.colorMode.compare(QStringLiteral("Field"), Qt::CaseInsensitive) != 0) {
        if (mapper != nullptr) {
            mapper->ScalarVisibilityOff();
        }
        return nullptr;
    }
    const ResolvedScalarField scalar = resolveScalarField(dataSet, options.colorField, options.colorComponent);
    if (scalar.array == nullptr || scalar.arrayName.trimmed().isEmpty()) {
        mapper->ScalarVisibilityOff();
        return nullptr;
    }
    double range[2] = {options.colorRangeMin, options.colorRangeMax};
    if (options.autoColorRange || !(range[1] > range[0])) {
        scalar.array->GetRange(range);
    }
    vtkSmartPointer<vtkLookupTable> table = makeLookupTable(options.colorMap);
    table->SetTableRange(range);
    table->Build();
    mapper->ScalarVisibilityOn();
    mapper->SetColorModeToMapScalars();
    mapper->SetLookupTable(table);
    mapper->SetScalarRange(range);
    if (scalar.association == kFieldAssociationCells) {
        mapper->SetScalarModeToUseCellFieldData();
    } else {
        mapper->SetScalarModeToUsePointFieldData();
    }
    mapper->SelectColorArray(scalar.arrayName.toUtf8().constData());
    return table;
}

bool isPerContourSurfaceColorMode(const QString& mode) {
    return mode.compare(QStringLiteral("Solid color (per contour)"), Qt::CaseInsensitive) == 0;
}

vtkSmartPointer<vtkColorTransferFunction> makeVolumeColorTransferFunction(const QString& colorMap,
                                                                          const double range[2]) {
    vtkSmartPointer<vtkLookupTable> table = makeLookupTable(colorMap);
    vtkSmartPointer<vtkColorTransferFunction> transfer = vtkSmartPointer<vtkColorTransferFunction>::New();
    const double minValue = range[0];
    const double maxValue = range[1] > range[0] ? range[1] : range[0] + 1.0;
    for (int i = 0; i < 256; ++i) {
        double rgba[4] = {0.0, 0.0, 0.0, 1.0};
        table->GetTableValue(i, rgba);
        const double t = static_cast<double>(i) / 255.0;
        transfer->AddRGBPoint(minValue + t * (maxValue - minValue), rgba[0], rgba[1], rgba[2]);
    }
    return transfer;
}

vtkSmartPointer<vtkPiecewiseFunction> makeVolumeOpacityFunction(const double range[2], double opacityScale) {
    vtkSmartPointer<vtkPiecewiseFunction> opacity = vtkSmartPointer<vtkPiecewiseFunction>::New();
    const double minValue = range[0];
    const double maxValue = range[1] > range[0] ? range[1] : range[0] + 1.0;
    const double span = maxValue - minValue;
    const double scale = std::clamp(opacityScale, 0.0, 8.0);
    opacity->AddPoint(minValue, 0.0);
    opacity->AddPoint(minValue + 0.08 * span, 0.0);
    opacity->AddPoint(minValue + 0.35 * span, std::clamp(0.04 * scale, 0.0, 1.0));
    opacity->AddPoint(minValue + 0.70 * span, std::clamp(0.16 * scale, 0.0, 1.0));
    opacity->AddPoint(maxValue, std::clamp(0.36 * scale, 0.0, 1.0));
    return opacity;
}

QString firstPointFieldName(vtkImageData* imageData) {
    vtkPointData* pointData = imageData != nullptr ? imageData->GetPointData() : nullptr;
    if (pointData == nullptr) {
        return {};
    }
    for (int index = 0; index < pointData->GetNumberOfArrays(); ++index) {
        vtkDataArray* array = pointData->GetArray(index);
        if (array != nullptr
            && array->GetName() != nullptr
            && array->GetNumberOfComponents() >= 1
            && array->GetNumberOfTuples() == imageData->GetNumberOfPoints()) {
            const QString name = QString::fromUtf8(array->GetName());
            if (!name.startsWith(QStringLiteral("__scplus_"))) {
                return name;
            }
        }
    }
    return {};
}

}  // namespace

struct ViewerWidget::TimeCallbackState {
    ViewerWidget* viewer = nullptr;
    bool swallowFirstCallback = true;
    vtkSmartPointer<vtkCallbackCommand> command;
};

struct ViewerWidget::PlaneCallbackState {
    ViewerWidget* viewer = nullptr;
    QString objectId;
    vtkSmartPointer<vtkPlane> plane;
    vtkSmartPointer<vtkAlgorithm> algorithm;
};

struct ViewerWidget::LegendCallbackState {
    ViewerWidget* viewer = nullptr;
    QString objectId;
};

struct ViewerWidget::MaterializedDataObject {
    vtkSmartPointer<vtkDataSet> sourceData;
    vtkSmartPointer<vtkDataSet> visualData;
    vtkSmartPointer<vtkDataSet> outputData;
    vtkSmartPointer<vtkAlgorithm> algorithm;
    vtkSmartPointer<vtkPlane> plane;
};

ViewerWidget::ViewerWidget(QWidget* parent)
    : QWidget(parent) {
    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);

    canvasFrame_ = new QFrame(this);
    canvasFrame_->setObjectName(QStringLiteral("vtk_canvas_frame"));
    canvasFrame_->setProperty("displayCanvasBorder", false);
    canvasFrame_->setFrameShape(QFrame::NoFrame);
    auto* canvasFrameLayout = new QVBoxLayout(canvasFrame_);
    canvasFrameLayout->setContentsMargins(0, 0, 0, 0);
    canvasFrameLayout->setSpacing(0);
    rootLayout->addWidget(canvasFrame_);

    stackedLayout_ = new QStackedLayout();
    canvasFrameLayout->addLayout(stackedLayout_);

    placeholderLabel_ = new QLabel(tr("Visualization will appear here after opening a display."), this);
    placeholderLabel_->setAlignment(Qt::AlignCenter);
    placeholderLabel_->setStyleSheet("QLabel{color:#707070; background:#ffffff;}");
    stackedLayout_->addWidget(placeholderLabel_);

    vtkWidget_ = new QVTKOpenGLNativeWidget(canvasFrame_);
    vtkWidget_->setStyleSheet("QWidget{background:#ffffff;}");
    vtkWidget_->installEventFilter(this);
    stackedLayout_->addWidget(vtkWidget_);
    stackedLayout_->setCurrentIndex(0);

    renderWindow_ = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    renderer_ = vtkSmartPointer<vtkRenderer>::New();
    renderer_->SetBackground(215.0 / 255.0, 215.0 / 255.0, 215.0 / 255.0);
    renderer_->AutomaticLightCreationOff();
    renderWindow_->AddRenderer(renderer_);
    vtkWidget_->setRenderWindow(renderWindow_);
    vtkWidget_->interactor()->SetInteractorStyle(vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New());
    cameraInteractionCallback_ = vtkSmartPointer<vtkCallbackCommand>::New();
    cameraInteractionCallback_->SetClientData(this);
    cameraInteractionCallback_->SetCallback([](vtkObject*, unsigned long, void* clientData, void*) {
        auto* viewer = static_cast<ViewerWidget*>(clientData);
        if (viewer != nullptr) {
            emit viewer->cameraChanged(viewer->cameraState());
        }
    });
    vtkWidget_->interactor()->AddObserver(vtkCommand::InteractionEvent, cameraInteractionCallback_);
    vtkWidget_->interactor()->AddObserver(vtkCommand::EndInteractionEvent, cameraInteractionCallback_);
    applyDisplayLighting();
    indexVolumeBackend_ = std::make_unique<Streamcenter::Index::IndexVolumeBackend>();
    indexRenderCallback_ = vtkSmartPointer<vtkCallbackCommand>::New();
    indexRenderCallback_->SetClientData(this);
    indexRenderCallback_->SetCallback([](vtkObject*, unsigned long, void* clientData, void*) {
        auto* viewer = static_cast<ViewerWidget*>(clientData);
        if (viewer != nullptr) {
            viewer->renderIndexVolumes();
        }
    });
    renderer_->AddObserver(vtkCommand::EndEvent, indexRenderCallback_);

    axesActor_ = vtkSmartPointer<vtkAxesActor>::New();
    axesActor_->SetTotalLength(0.78, 0.78, 0.78);
    axesActor_->SetNormalizedShaftLength(0.72, 0.72, 0.72);
    axesActor_->SetNormalizedTipLength(0.28, 0.28, 0.28);
    axesActor_->SetNormalizedLabelPosition(1.10, 1.10, 1.10);
    configureAxisCaption(axesActor_->GetXAxisCaptionActor2D(), "X");
    configureAxisCaption(axesActor_->GetYAxisCaptionActor2D(), "Y");
    configureAxisCaption(axesActor_->GetZAxisCaptionActor2D(), "Z");
    orientationWidget_ = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
    orientationWidget_->SetOrientationMarker(axesActor_);
    orientationWidget_->SetInteractor(vtkWidget_->interactor());
    orientationWidget_->SetViewport(0.015, 0.015, 0.145, 0.155);
    orientationWidget_->SetEnabled(1);
    orientationWidget_->InteractiveOff();

    animationTimer_ = new QTimer(this);
    connect(animationTimer_, &QTimer::timeout, this, &ViewerWidget::onAnimationTick);
}

ViewerWidget::~ViewerWidget() {
    clearScene();
}

bool ViewerWidget::eventFilter(QObject* watched, QEvent* event) {
    if (watched == vtkWidget_ && event != nullptr) {
        if (event->type() == QEvent::Show
            || event->type() == QEvent::ShowToParent
            || event->type() == QEvent::Resize) {
            scheduleRenderWhenVisible();
        }
        if (event->type() == QEvent::MouseButtonDblClick) {
            auto* mouseEvent = static_cast<QMouseEvent*>(event);
            if (mouseEvent->button() == Qt::LeftButton) {
                const QSize widgetSize = vtkWidget_->size();
                const double normalizedX = widgetSize.width() > 0
                    ? mouseEvent->position().x() / static_cast<double>(widgetSize.width())
                    : 0.0;
                const double normalizedY = widgetSize.height() > 0
                    ? 1.0 - mouseEvent->position().y() / static_cast<double>(widgetSize.height())
                    : 0.0;
                auto containsPointer = [normalizedX, normalizedY](const DisplayActorSet& actorSet) {
                    vtkCoordinate* position = nullptr;
                    vtkCoordinate* position2 = nullptr;
                    if (actorSet.scalarBarWidget != nullptr
                        && actorSet.scalarBarWidget->GetScalarBarRepresentation() != nullptr) {
                        vtkScalarBarRepresentation* representation = actorSet.scalarBarWidget->GetScalarBarRepresentation();
                        position = representation->GetPositionCoordinate();
                        position2 = representation->GetPosition2Coordinate();
                    } else if (actorSet.scalarBar != nullptr) {
                        position = actorSet.scalarBar->GetPositionCoordinate();
                        position2 = actorSet.scalarBar->GetPosition2Coordinate();
                    }
                    if (position == nullptr || position2 == nullptr) {
                        return false;
                    }
                    const double* origin = position->GetValue();
                    const double* size = position2->GetValue();
                    constexpr double kLegendPickPadding = 0.04;
                    const double minX = std::min(origin[0], origin[0] + size[0]) - kLegendPickPadding;
                    const double maxX = std::max(origin[0], origin[0] + size[0]) + kLegendPickPadding;
                    const double minY = std::min(origin[1], origin[1] + size[1]) - kLegendPickPadding;
                    const double maxY = std::max(origin[1], origin[1] + size[1]) + kLegendPickPadding;
                    return normalizedX >= minX && normalizedX <= maxX && normalizedY >= minY && normalizedY <= maxY;
                };
                for (auto it = displayActors_.begin(); it != displayActors_.end(); ++it) {
                    if (it->scalarBar != nullptr && containsPointer(*it) && showScalarBarStyleDialog(&(*it))) {
                        return true;
                    }
                }
            }
        }
    }
    return QWidget::eventFilter(watched, event);
}

void ViewerWidget::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    scheduleRenderWhenVisible();
}

bool ViewerWidget::openFile(const QString& path, QString* errorMessage) {
    clearScene();
    customDisplayActive_ = false;

    const QString absolutePath = QFileInfo(path).absoluteFilePath();
    const QString suffix = QFileInfo(absolutePath).suffix().toLower();

    if (suffix == "pvd") {
        const QVector<FrameInfo> parsed = parsePvdFile(absolutePath, errorMessage);
        if (parsed.isEmpty()) {
            return false;
        }
        if (parsed.size() > 1) {
            frames_ = parsed;
            currentPath_ = absolutePath;
            currentFrameIndex_ = 0;
            isTimeSeries_ = true;
            if (!applyFrameIndex(0, false, errorMessage)) {
                clearScene();
                return false;
            }
        } else {
            vtkSmartPointer<vtkDataSet> data;
            if (!loadDataSetFromPath(parsed.front().path, &data, errorMessage)) {
                return false;
            }
            currentDataSet_ = data;
            currentPath_ = absolutePath;
            isTimeSeries_ = false;
            currentFrameIndex_ = 0;
            frames_.clear();
        }
    } else {
        vtkSmartPointer<vtkDataSet> data;
        if (!loadDataSetFromPath(absolutePath, &data, errorMessage)) {
            return false;
        }
        currentDataSet_ = data;
        currentPath_ = absolutePath;
        isTimeSeries_ = false;
        currentFrameIndex_ = 0;
        frames_.clear();
    }

    rebuildScene(true);
    stackedLayout_->setCurrentIndex(1);
    renderWindow_->Render();

    emit timeSeriesAvailabilityChanged(isTimeSeries_);
    emit sceneLoaded(currentPath_, isTimeSeries_);
    if (isTimeSeries_ && currentFrameIndex_ >= 0 && currentFrameIndex_ < frames_.size()) {
        emit frameChanged(currentFrameIndex_, frames_[currentFrameIndex_].timestep);
    }
    return true;
}

void ViewerWidget::clearScene() {
    stopAnimation();

    clearScalarVisuals();
    clearDisplayActors();
    removeTimeWidgets();

    currentPath_.clear();
    currentDataSet_ = nullptr;
    frames_.clear();
    isoTargets_.clear();
    displayObjectOptions_.clear();
    displayObjectTimeSeries_.clear();
    activeDataObjectHandleId_.clear();
    watermarkActor_ = nullptr;
    outlineActor_ = nullptr;
    currentFrameIndex_ = 0;
    isTimeSeries_ = false;
    customDisplayActive_ = false;
    suppressTimeSliderCallback_ = false;
    initialTimeCodeHint_ = std::numeric_limits<double>::quiet_NaN();

    if (renderer_ != nullptr) {
        renderer_->RemoveAllViewProps();
    }
    if (stackedLayout_ != nullptr) {
        stackedLayout_->setCurrentIndex(0);
    }
    if (renderWindow_ != nullptr) {
        renderWindow_->Render();
    }

    emit timeSeriesAvailabilityChanged(false);
    emit sceneCleared();
}

void ViewerWidget::showBlankDisplay(const QString& title) {
    clearScene();
    customDisplayActive_ = true;
    if (placeholderLabel_ != nullptr) {
        placeholderLabel_->setText(title);
    }
    if (stackedLayout_ != nullptr) {
        stackedLayout_->setCurrentIndex(0);
    }
}

void ViewerWidget::beginDisplay(const DisplayOptions& options) {
    stopAnimation();
    clearScalarVisuals();
    clearDisplayActors();
    removeTimeWidgets();
    currentDataSet_ = nullptr;
    frames_.clear();
    isoTargets_.clear();
    displayObjectOptions_.clear();
    displayObjectTimeSeries_.clear();
    activeDataObjectHandleId_.clear();
    currentPath_.clear();
    currentFrameIndex_ = 0;
    isTimeSeries_ = false;
    customDisplayActive_ = true;
    displayOptions_ = options;
    initialTimeCodeHint_ = std::numeric_limits<double>::quiet_NaN();

    if (renderer_ != nullptr) {
        renderer_->RemoveAllViewProps();
    }
    applyDisplayOptions(options);
    rebuildDisplayOverlay();
    if (stackedLayout_ != nullptr) {
        stackedLayout_->setCurrentIndex(1);
    }
    if (renderWindow_ != nullptr) {
        renderWindow_->Render();
    }
    emit timeSeriesAvailabilityChanged(false);
}

void ViewerWidget::applyDisplayOptions(const DisplayOptions& options) {
    displayOptions_ = options;
    if (renderer_ == nullptr) {
        return;
    }
    const bool rebuildObjects = customDisplayActive_ && !displayObjectOptions_.isEmpty();
    const CameraState camera = captureCamera(renderer_);
    const QMap<QString, DataObjectOptions> objectOptions = displayObjectOptions_;
    renderer_->SetBackground(options.backgroundColor.redF(),
                             options.backgroundColor.greenF(),
                             options.backgroundColor.blueF());
    applyDisplayLighting();
    applyAdvancedRenderingPreview();
    parallelProjection_ = !options.perspectiveEnabled;
    if (renderer_->GetActiveCamera() != nullptr) {
        renderer_->GetActiveCamera()->SetParallelProjection(options.perspectiveEnabled ? 0 : 1);
        if (options.perspectiveEnabled) {
            renderer_->GetActiveCamera()->SetViewAngle(std::clamp(options.perspectiveDepth, 5.0, 120.0));
        }
    }
    if (orientationWidget_ != nullptr) {
        if (options.showAxes) {
            orientationWidget_->EnabledOn();
        } else {
            orientationWidget_->EnabledOff();
        }
    }
    if (rebuildObjects) {
        clearDisplayActors();
        rebuildDisplayOverlay();
        displayObjectOptions_ = objectOptions;
        for (auto it = objectOptions.constBegin(); it != objectOptions.constEnd(); ++it) {
            QString ignoredError;
            addOrUpdateDataObject(it.key(), it.value(), false, &ignoredError);
        }
        restoreCamera(renderer_, camera, parallelProjection_);
        applyDisplayLighting();
    } else {
        rebuildDisplayOverlay();
    }
    if (renderWindow_ != nullptr) {
        renderWindow_->Render();
    }
}

void ViewerWidget::applyEnvironmentTexture() {
    if (renderer_ == nullptr) {
        return;
    }

    const QString texturePath = displayOptions_.advancedRenderingEnabled
        ? displayOptions_.advancedEnvironmentTexture.trimmed()
        : QString();
    if (texturePath.isEmpty()) {
        renderer_->UseImageBasedLightingOff();
        renderer_->SetEnvironmentTexture(nullptr);
        environmentTexture_ = nullptr;
        return;
    }

    const QString absolutePath = QFileInfo(texturePath).absoluteFilePath();
    if (!QFileInfo::exists(absolutePath)) {
        renderer_->UseImageBasedLightingOff();
        renderer_->SetEnvironmentTexture(nullptr);
        environmentTexture_ = nullptr;
        return;
    }

    const QByteArray texturePathBytes = QFile::encodeName(absolutePath);
    vtkSmartPointer<vtkImageReader2> reader;
    reader.TakeReference(vtkImageReader2Factory::CreateImageReader2(texturePathBytes.constData()));
    if (reader == nullptr) {
        renderer_->UseImageBasedLightingOff();
        renderer_->SetEnvironmentTexture(nullptr);
        environmentTexture_ = nullptr;
        return;
    }

    reader->SetFileName(texturePathBytes.constData());
    reader->Update();
    vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
    texture->SetInputConnection(reader->GetOutputPort());
    texture->InterpolateOn();
    texture->MipmapOn();
    texture->UseSRGBColorSpaceOn();
    environmentTexture_ = texture;
    renderer_->SetEnvironmentTexture(environmentTexture_, true);
    renderer_->UseImageBasedLightingOn();
}

void ViewerWidget::applyAdvancedRenderingPreview() {
    if (renderer_ == nullptr) {
        return;
    }
    applyEnvironmentTexture();

#if STREAMCENTERPLUS_ENABLE_VTK_RAYTRACING
    const bool useOsprayPreview = displayOptions_.advancedRenderingEnabled
        && isOsprayEngine(displayOptions_.advancedPreviewEngine);
    if (useOsprayPreview) {
        if (osprayPass_ == nullptr) {
            osprayPass_ = vtkSmartPointer<vtkOSPRayPass>::New();
        }
        vtkOSPRayRendererNode::SetRendererType(displayOptions_.advancedRendererType.trimmed().isEmpty()
                                                   ? std::string("pathtracer")
                                                   : displayOptions_.advancedRendererType.toStdString(),
                                               renderer_);
        vtkOSPRayRendererNode::SetSamplesPerPixel(std::max(1, displayOptions_.advancedSamplesPerPixel), renderer_);
        vtkOSPRayRendererNode::SetMaxFrames(std::max(1, displayOptions_.advancedAccumulationFrames), renderer_);
        vtkOSPRayRendererNode::SetAmbientSamples(1, renderer_);
#if VTK_VERSION_NUMBER >= VTK_VERSION_CHECK(9, 3, 0)
        vtkOSPRayRendererNode::SetMaxDepth(std::max(1, displayOptions_.advancedMaxDepth), renderer_);
#endif
#if VTK_VERSION_NUMBER >= VTK_VERSION_CHECK(9, 4, 0)
        vtkOSPRayRendererNode::SetEnableDenoiser(displayOptions_.advancedDenoise ? 1 : 0, renderer_);
#endif
        renderer_->SetPass(osprayPass_);
        return;
    }
#endif

    renderer_->SetPass(nullptr);
}

bool ViewerWidget::exportRayTracingRequested() const {
    return displayOptions_.advancedRenderingEnabled
        && isOsprayEngine(displayOptions_.advancedExportEngine);
}

bool ViewerWidget::beginExportRayTracing(QString* errorMessage) {
    if (!exportRayTracingRequested()) {
        if (renderer_ != nullptr) {
            renderer_->SetPass(nullptr);
        }
        return true;
    }

#if STREAMCENTERPLUS_ENABLE_VTK_RAYTRACING
    if (renderer_ == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = tr("OSPRay export is unavailable because the renderer is not initialized.");
        }
        return false;
    }
    if (osprayPass_ == nullptr) {
        osprayPass_ = vtkSmartPointer<vtkOSPRayPass>::New();
    }
    vtkOSPRayRendererNode::SetRendererType(displayOptions_.advancedRendererType.trimmed().isEmpty()
                                               ? std::string("pathtracer")
                                               : displayOptions_.advancedRendererType.toStdString(),
                                           renderer_);
    vtkOSPRayRendererNode::SetSamplesPerPixel(std::max(1, displayOptions_.advancedSamplesPerPixel), renderer_);
    vtkOSPRayRendererNode::SetMaxFrames(std::max(1, displayOptions_.advancedAccumulationFrames), renderer_);
    vtkOSPRayRendererNode::SetAmbientSamples(1, renderer_);
#if VTK_VERSION_NUMBER >= VTK_VERSION_CHECK(9, 3, 0)
    vtkOSPRayRendererNode::SetMaxDepth(std::max(1, displayOptions_.advancedMaxDepth), renderer_);
#endif
#if VTK_VERSION_NUMBER >= VTK_VERSION_CHECK(9, 4, 0)
    vtkOSPRayRendererNode::SetEnableDenoiser(displayOptions_.advancedDenoise ? 1 : 0, renderer_);
#endif
    renderer_->SetPass(osprayPass_);
    return true;
#else
    if (errorMessage != nullptr) {
        *errorMessage = tr("OSPRay export is unavailable because this GUI was built without VTK RenderingRayTracing. Rebuild with -EnableVtkRayTracing and a VTK install that includes RenderingRayTracing/OSPRay.");
    }
    return false;
#endif
}

void ViewerWidget::endExportRayTracing(vtkRenderPass* previousPass) {
    if (renderer_ == nullptr) {
        return;
    }
    renderer_->SetPass(previousPass);
}

void ViewerWidget::renderAccumulationFrames() {
    if (renderWindow_ == nullptr) {
        return;
    }
    const int frames = exportRayTracingRequested()
        ? std::max(1, displayOptions_.advancedAccumulationFrames)
        : 1;
    for (int index = 0; index < frames; ++index) {
        renderWindow_->Render();
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    }
}

void ViewerWidget::setCanvasBorderVisible(bool visible) {
    if (canvasFrame_ == nullptr) {
        return;
    }

    if (canvasFrame_->property("displayCanvasBorder").toBool() == visible) {
        return;
    }

    canvasFrame_->setProperty("displayCanvasBorder", visible);
    if (QLayout* frameLayout = canvasFrame_->layout()) {
        const int borderMargin = visible ? 1 : 0;
        frameLayout->setContentsMargins(borderMargin, borderMargin, borderMargin, borderMargin);
    }
    canvasFrame_->style()->unpolish(canvasFrame_);
    canvasFrame_->style()->polish(canvasFrame_);
    canvasFrame_->update();
}

void ViewerWidget::renderWhenVisible() {
    scheduleRenderWhenVisible();
}

bool ViewerWidget::materializeDataObject(const QString& objectId,
                                         const DataObjectOptions& options,
                                         MaterializedDataObject* output,
                                         QSet<QString>* visiting,
                                         QString* errorMessage) const {
    if (output == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Internal error: materialized data output pointer is null.");
        }
        return false;
    }
    if (visiting == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Internal error: materialized data recursion state is null.");
        }
        return false;
    }

    const QString normalizedObjectId = objectId.trimmed();
    if (!normalizedObjectId.isEmpty()) {
        if (visiting->contains(normalizedObjectId)) {
            if (errorMessage != nullptr) {
                *errorMessage = QStringLiteral("Filter source cycle detected at object %1.").arg(normalizedObjectId);
            }
            return false;
        }
        visiting->insert(normalizedObjectId);
    }

    auto finish = [&]() {
        if (!normalizedObjectId.isEmpty()) {
            visiting->remove(normalizedObjectId);
        }
    };
    auto fail = [&](const QString& message) {
        if (errorMessage != nullptr) {
            *errorMessage = message;
        }
        finish();
        return false;
    };

    vtkSmartPointer<vtkDataSet> sourceData;
    const QString sourceObjectId = options.sourceObjectId.trimmed();
    if (!sourceObjectId.isEmpty()) {
        if (sourceObjectId == normalizedObjectId) {
            return fail(QStringLiteral("Filter cannot use itself as source: %1.").arg(normalizedObjectId));
        }
        const auto sourceIt = displayObjectOptions_.constFind(sourceObjectId);
        if (sourceIt == displayObjectOptions_.constEnd()) {
            return fail(QStringLiteral("Filter source object is not available in the display: %1.").arg(sourceObjectId));
        }
        MaterializedDataObject sourceOutput;
        if (!materializeDataObject(sourceObjectId, sourceIt.value(), &sourceOutput, visiting, errorMessage)) {
            finish();
            return false;
        }
        sourceData = sourceOutput.outputData;
    } else {
        if (options.inputPath.trimmed().isEmpty()) {
            return fail(QStringLiteral("Please select an input VTK or STL file before applying this object."));
        }

        const QString absoluteInputPath = QFileInfo(options.inputPath).absoluteFilePath();
        const QString sourceSuffix = QFileInfo(absoluteInputPath).suffix().toLower();
        QString loadPath = absoluteInputPath;
        if (sourceSuffix == QStringLiteral("pvd")) {
            const QVector<FrameInfo> parsed = parsePvdFile(absoluteInputPath, errorMessage);
            if (parsed.isEmpty()) {
                finish();
                return false;
            }
            loadPath = displayObjectFramePathForCurrentTime(normalizedObjectId, parsed.front().path);
        }

        const QString suffix = QFileInfo(loadPath).suffix().toLower();
        if (options.type != DisplayObjectType::Geometry && suffix == QStringLiteral("stl")) {
            return fail(QStringLiteral("Clip, Slice, Data, Crop, and Contour objects require VTK-family input, not STL."));
        }
        if (!loadDataSetFromPath(loadPath, &sourceData, errorMessage)) {
            finish();
            return false;
        }
    }

    if (sourceData == nullptr) {
        return fail(QStringLiteral("Filter source produced no VTK dataset."));
    }
    prepareVectorFields(sourceData);

    vtkSmartPointer<vtkDataSet> visualInput = renderableDataSetWithoutHiddenGhostCells(sourceData);
    if (options.type == DisplayObjectType::Crop) {
        vtkSmartPointer<vtkDataSet> croppedInput = cropDataSetToBounds(visualInput, options.cropBounds, errorMessage);
        if (croppedInput == nullptr) {
            finish();
            return false;
        }
        visualInput = croppedInput;
        prepareVectorFields(visualInput);
    }

    vtkSmartPointer<vtkAlgorithm> algorithm;
    vtkSmartPointer<vtkPlane> plane;
    if (options.type == DisplayObjectType::Clip) {
        plane = vtkSmartPointer<vtkPlane>::New();
        plane->SetOrigin(options.planeOrigin);
        plane->SetNormal(options.planeNormal);
        vtkSmartPointer<vtkClipDataSet> clip = vtkSmartPointer<vtkClipDataSet>::New();
        clip->SetInputData(visualInput);
        clip->SetClipFunction(plane);
        clip->InsideOutOn();
        clip->Update();
        algorithm = clip;
    } else if (options.type == DisplayObjectType::Slice) {
        plane = vtkSmartPointer<vtkPlane>::New();
        plane->SetOrigin(options.planeOrigin);
        plane->SetNormal(options.planeNormal);
        vtkSmartPointer<vtkCutter> cutter = vtkSmartPointer<vtkCutter>::New();
        cutter->SetInputData(visualInput);
        cutter->SetCutFunction(plane);
        cutter->Update();
        algorithm = cutter;
    } else if (options.type == DisplayObjectType::Contour) {
        const QString field = options.contourField.trimmed().isEmpty() ? options.colorField : options.contourField;
        const ResolvedScalarField contourField = resolveScalarField(visualInput, field, options.contourComponent);
        if (contourField.array == nullptr || contourField.arrayName.trimmed().isEmpty()) {
            return fail(QStringLiteral("Contour field was not found: %1").arg(field));
        }
        vtkSmartPointer<vtkContourFilter> contour = vtkSmartPointer<vtkContourFilter>::New();
        contour->SetInputData(visualInput);
        contour->SetInputArrayToProcess(0,
                                        0,
                                        0,
                                        contourField.association == kFieldAssociationPoints ? vtkDataObject::FIELD_ASSOCIATION_POINTS
                                                                                            : vtkDataObject::FIELD_ASSOCIATION_CELLS,
                                        contourField.arrayName.toUtf8().constData());
        QVector<double> values = options.contourValues;
        if (values.isEmpty()) {
            double range[2]{};
            contourField.array->GetRange(range);
            values.push_back(range[0] + 0.5 * (range[1] - range[0]));
        }
        contour->SetNumberOfContours(values.size());
        for (int i = 0; i < values.size(); ++i) {
            contour->SetValue(i, values[i]);
        }
        contour->Update();
        algorithm = contour;
    }

    vtkSmartPointer<vtkDataSet> resultData;
    if (algorithm != nullptr) {
        resultData = cloneDataSet(vtkDataSet::SafeDownCast(algorithm->GetOutputDataObject(0)));
    } else {
        resultData = visualInput;
    }
    if (resultData == nullptr || resultData->GetNumberOfPoints() <= 0) {
        return fail(QStringLiteral("Visualization object produced no visible data."));
    }
    prepareVectorFields(resultData);

    output->sourceData = sourceData;
    output->visualData = visualInput;
    output->outputData = resultData;
    output->algorithm = algorithm;
    output->plane = plane;
    finish();
    return true;
}

bool ViewerWidget::addOrUpdateDataObject(const QString& objectId,
                                         const DataObjectOptions& options,
                                         bool resetCameraToObject,
                                         QString* errorMessage) {
    if (objectId.trimmed().isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = "Internal error: visualization object id is empty.";
        }
        return false;
    }
    if (options.type == DisplayObjectType::ParticleStreamline) {
        displayObjectOptions_.insert(objectId, options);
        removeDataObjectActors(objectId);
        return true;
    }
    if (options.type == DisplayObjectType::RayTracingVolume) {
        return addOrUpdateRayTracingVolume(objectId, options, resetCameraToObject, errorMessage);
    }
    if (options.sourceObjectId.trimmed().isEmpty() && options.inputPath.trimmed().isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = "Please select an input VTK or STL file before applying this object.";
        }
        return false;
    }

    if (options.sourceObjectId.trimmed().isEmpty()) {
        const QString absoluteInputPath = QFileInfo(options.inputPath).absoluteFilePath();
        const QString sourceSuffix = QFileInfo(absoluteInputPath).suffix().toLower();
        if (sourceSuffix == QStringLiteral("pvd")) {
            const QVector<FrameInfo> parsed = parsePvdFile(absoluteInputPath, errorMessage);
            if (parsed.isEmpty()) {
                return false;
            }
            if (parsed.size() > 1) {
                DisplayObjectTimeSeries series;
                series.sourcePath = absoluteInputPath;
                series.frames = parsed;
                displayObjectTimeSeries_.insert(objectId, series);
                refreshDisplayTimeSeriesState();
            } else {
                displayObjectTimeSeries_.remove(objectId);
                refreshDisplayTimeSeriesState();
            }
        } else if (displayObjectTimeSeries_.remove(objectId) > 0) {
            refreshDisplayTimeSeriesState();
        }
    } else if (displayObjectTimeSeries_.remove(objectId) > 0) {
        refreshDisplayTimeSeriesState();
    }

    if (indexVolumeBackend_ != nullptr) {
        indexVolumeBackend_->remove(objectId);
    }
    displayObjectOptions_.insert(objectId, options);
    removeDataObjectActors(objectId);
    DisplayActorSet actorSet;
    if (!options.visible) {
        displayActors_.insert(objectId, actorSet);
        applyDisplayLighting();
        return true;
    }

    MaterializedDataObject materialized;
    QSet<QString> visiting;
    if (!materializeDataObject(objectId, options, &materialized, &visiting, errorMessage)) {
        return false;
    }
    vtkSmartPointer<vtkDataSet> input = materialized.sourceData;
    vtkSmartPointer<vtkDataSet> visualInput = materialized.visualData;
    vtkSmartPointer<vtkDataSet> outputData = materialized.outputData;
    vtkSmartPointer<vtkAlgorithm> algorithm = materialized.algorithm;
    actorSet.plane = materialized.plane;
    actorSet.planeAlgorithm = algorithm;

    if (input == nullptr || visualInput == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = "Visualization object produced no VTK dataset.";
        }
        return false;
    }
    const QString absoluteInputPath = options.sourceObjectId.trimmed().isEmpty()
        ? QFileInfo(options.inputPath).absoluteFilePath()
        : QString();
    if (options.sourceObjectId.trimmed().isEmpty()
        && options.type != DisplayObjectType::Geometry
        && QFileInfo(absoluteInputPath).suffix().compare(QStringLiteral("stl"), Qt::CaseInsensitive) == 0) {
        if (errorMessage != nullptr) {
            *errorMessage = "Clip, Slice, Data, Crop, and Contour objects require VTK-family input, not STL.";
        }
        return false;
    }

    auto addProp = [&](vtkProp* prop) {
        if (prop == nullptr || renderer_ == nullptr) {
            return;
        }
        renderer_->AddViewProp(prop);
        actorSet.props.push_back(prop);
    };

    constexpr double kDisplayTransparentOpacityFactor = 0.7;
    const double displayAlpha = displayOptions_.displayTransparent ? kDisplayTransparentOpacityFactor : 1.0;
    const bool advancedMaterialEnabled = displayOptions_.advancedRenderingEnabled
        && options.type != DisplayObjectType::RayTracingVolume
        && options.type != DisplayObjectType::ParticleStreamline;
    const RenderMaterialValues material = renderMaterialValues(options);
    DataObjectOptions renderOptions = options;
    if (advancedMaterialEnabled) {
        renderOptions.colorMode = QStringLiteral("Solid color");
        renderOptions.surfaceColor = material.baseColor;
        renderOptions.surfaceOpacity = material.opacity * (1.0 - (0.75 * material.transmission));
        renderOptions.showLegend = false;
        renderOptions.showMesh = false;
        renderOptions.showOutline = false;
    }
    const bool surfaceEnabled = renderOptions.type == DisplayObjectType::Contour ? true : renderOptions.showSurface;
    vtkSmartPointer<vtkLookupTable> surfaceLookupTable;
    if (surfaceEnabled) {
        auto applySurfaceProperty = [&](vtkActor* actor, const QColor& color, double opacity) {
            if (actor == nullptr) {
                return;
            }
            vtkProperty* property = actor->GetProperty();
            if (property == nullptr) {
                return;
            }
            property->SetColor(color.redF(), color.greenF(), color.blueF());
            property->SetOpacity(std::clamp(opacity, 0.0, 1.0) * displayAlpha);
            property->SetLighting(displayOptions_.lightingEnabled ? 1 : 0);
            property->SetAmbient(std::clamp(1.0 - displayOptions_.lightingIntensity, 0.0, 1.0));
            property->SetDiffuse(std::clamp(displayOptions_.lightingIntensity, 0.0, 1.0));
            if (advancedMaterialEnabled) {
                property->SetInterpolationToPBR();
                property->SetMetallic(material.metallic);
                property->SetRoughness(material.roughness);
                property->SetBaseIOR(material.ior);
                property->SetSpecular(1.0);
                if (!options.osprayMaterialName.trimmed().isEmpty()) {
                    property->SetMaterialName(options.osprayMaterialName.trimmed().toUtf8().constData());
                }
            } else {
                property->SetInterpolationToPhong();
            }
        };

        if (renderOptions.type == DisplayObjectType::Contour && isPerContourSurfaceColorMode(renderOptions.colorMode)) {
            const QString field = renderOptions.contourField.trimmed().isEmpty() ? renderOptions.colorField : renderOptions.contourField;
            const ResolvedScalarField contourField = resolveScalarField(visualInput, field, renderOptions.contourComponent);
            QVector<double> values = renderOptions.contourValues;
            if (values.isEmpty() && contourField.array != nullptr) {
                double range[2]{};
                contourField.array->GetRange(range);
                values.push_back(range[0] + 0.5 * (range[1] - range[0]));
            }
            for (int i = 0; i < values.size(); ++i) {
                vtkSmartPointer<vtkContourFilter> contour = vtkSmartPointer<vtkContourFilter>::New();
                contour->SetInputData(visualInput);
                contour->SetInputArrayToProcess(0,
                                                0,
                                                0,
                                                contourField.association == kFieldAssociationPoints ? vtkDataObject::FIELD_ASSOCIATION_POINTS
                                                                                                    : vtkDataObject::FIELD_ASSOCIATION_CELLS,
                                                contourField.arrayName.toUtf8().constData());
                contour->SetNumberOfContours(1);
                contour->SetValue(0, values[i]);
                contour->Update();

                vtkNew<vtkDataSetMapper> mapper;
                mapper->SetInputConnection(contour->GetOutputPort());
                mapper->ScalarVisibilityOff();
                vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
                actor->SetMapper(mapper);
                const QColor color = i < renderOptions.contourSurfaceColors.size() && renderOptions.contourSurfaceColors[i].isValid()
                    ? renderOptions.contourSurfaceColors[i]
                    : renderOptions.surfaceColor;
                const double opacity = i < renderOptions.contourSurfaceOpacities.size()
                    ? renderOptions.contourSurfaceOpacities[i]
                    : renderOptions.surfaceOpacity;
                applySurfaceProperty(actor, color, opacity);
                addProp(actor);
            }
        } else {
            vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
            if (advancedMaterialEnabled) {
                vtkNew<vtkDataSetSurfaceFilter> surface;
                if (algorithm != nullptr) {
                    surface->SetInputConnection(algorithm->GetOutputPort());
                } else {
                    surface->SetInputData(visualInput);
                }
                vtkNew<vtkTriangleFilter> triangles;
                triangles->SetInputConnection(surface->GetOutputPort());
                vtkNew<vtkPolyDataMapper> mapper;
                mapper->SetInputConnection(triangles->GetOutputPort());
                mapper->ScalarVisibilityOff();
                actor->SetMapper(mapper);
            } else if (algorithm != nullptr) {
                vtkNew<vtkDataSetMapper> mapper;
                mapper->SetInputConnection(algorithm->GetOutputPort());
                surfaceLookupTable = configureMapperScalars(mapper.GetPointer(),
                                                            outputData != nullptr ? outputData.GetPointer() : visualInput.GetPointer(),
                                                            renderOptions);
                actor->SetMapper(mapper);
            } else {
                vtkNew<vtkDataSetMapper> mapper;
                mapper->SetInputData(visualInput);
                surfaceLookupTable = configureMapperScalars(mapper.GetPointer(), visualInput, renderOptions);
                actor->SetMapper(mapper);
            }
            applySurfaceProperty(actor, renderOptions.surfaceColor, renderOptions.surfaceOpacity);
            addProp(actor);
        }
    }

    if (renderOptions.showLegend && surfaceLookupTable != nullptr && renderer_ != nullptr) {
        actorSet.scalarBarBackgroundPolyData = vtkSmartPointer<vtkPolyData>::New();
        vtkSmartPointer<vtkPolyDataMapper2D> legendBackgroundMapper = vtkSmartPointer<vtkPolyDataMapper2D>::New();
        legendBackgroundMapper->SetInputData(actorSet.scalarBarBackgroundPolyData);
        vtkSmartPointer<vtkCoordinate> legendBackgroundCoordinate = vtkSmartPointer<vtkCoordinate>::New();
        legendBackgroundCoordinate->SetCoordinateSystemToNormalizedViewport();
        legendBackgroundMapper->SetTransformCoordinate(legendBackgroundCoordinate);
        actorSet.scalarBarBackground = vtkSmartPointer<vtkActor2D>::New();
        actorSet.scalarBarBackground->SetMapper(legendBackgroundMapper);
        actorSet.scalarBarBackground->VisibilityOff();
        addProp(actorSet.scalarBarBackground);

        vtkSmartPointer<StreamcenterScalarBarActor> scalarBar = vtkSmartPointer<StreamcenterScalarBarActor>::New();
        actorSet.scalarBar = scalarBar;
        actorSet.scalarBarObjectId = objectId;
        actorSet.scalarBarDefaultTitle = defaultLegendTitle(options);
        actorSet.scalarBarBaseLookupTable = surfaceLookupTable;
        LegendOptions legendOptions = options.legend;
        if (!legendOptions.titleEdited || legendOptions.title.trimmed().isEmpty()) {
            legendOptions.title = actorSet.scalarBarDefaultTitle;
            legendOptions.titleEdited = false;
        }
        vtkSmartPointer<vtkLookupTable> legendLookupTable = legendOptions.reverseLegend
            ? reversedLookupTable(surfaceLookupTable)
            : surfaceLookupTable;
        actorSet.scalarBar->SetLookupTable(legendLookupTable);
        addProp(actorSet.scalarBar);

        actorSet.scalarBarTicksPolyData = vtkSmartPointer<vtkPolyData>::New();
        vtkSmartPointer<vtkPolyDataMapper2D> legendTicksMapper = vtkSmartPointer<vtkPolyDataMapper2D>::New();
        legendTicksMapper->SetInputData(actorSet.scalarBarTicksPolyData);
        vtkSmartPointer<vtkCoordinate> legendTicksCoordinate = vtkSmartPointer<vtkCoordinate>::New();
        legendTicksCoordinate->SetCoordinateSystemToDisplay();
        legendTicksMapper->SetTransformCoordinate(legendTicksCoordinate);
        actorSet.scalarBarTicks = vtkSmartPointer<vtkActor2D>::New();
        actorSet.scalarBarTicks->SetMapper(legendTicksMapper);
        actorSet.scalarBarTicks->SetLayerNumber(10);
        actorSet.scalarBarTicks->GetProperty()->SetDisplayLocationToForeground();
        actorSet.scalarBarTicks->VisibilityOff();
        addProp(actorSet.scalarBarTicks);

        actorSet.scalarBarTitle = vtkSmartPointer<vtkTextActor>::New();
        actorSet.scalarBarTitle->SetTextScaleModeToNone();
        actorSet.scalarBarTitle->SetLayerNumber(2);
        actorSet.scalarBarTitle->VisibilityOff();
        addProp(actorSet.scalarBarTitle);

        actorSet.scalarBarOutlinePolyData = vtkSmartPointer<vtkPolyData>::New();
        vtkSmartPointer<vtkPolyDataMapper2D> legendOutlineMapper = vtkSmartPointer<vtkPolyDataMapper2D>::New();
        legendOutlineMapper->SetInputData(actorSet.scalarBarOutlinePolyData);
        vtkSmartPointer<vtkCoordinate> legendOutlineCoordinate = vtkSmartPointer<vtkCoordinate>::New();
        legendOutlineCoordinate->SetCoordinateSystemToNormalizedViewport();
        legendOutlineMapper->SetTransformCoordinate(legendOutlineCoordinate);
        actorSet.scalarBarOutline = vtkSmartPointer<vtkActor2D>::New();
        actorSet.scalarBarOutline->SetMapper(legendOutlineMapper);
        actorSet.scalarBarOutline->VisibilityOff();
        addProp(actorSet.scalarBarOutline);

        if (vtkWidget_ != nullptr && vtkWidget_->interactor() != nullptr) {
            actorSet.scalarBarWidget = vtkSmartPointer<vtkScalarBarWidget>::New();
            actorSet.scalarBarWidget->SetInteractor(vtkWidget_->interactor());
            actorSet.scalarBarWidget->SetScalarBarActor(actorSet.scalarBar);
            actorSet.scalarBarWidget->On();

            actorSet.scalarBarCallbackState = std::make_shared<LegendCallbackState>();
            actorSet.scalarBarCallbackState->viewer = this;
            actorSet.scalarBarCallbackState->objectId = objectId;
            actorSet.scalarBarCallback = vtkSmartPointer<vtkCallbackCommand>::New();
            actorSet.scalarBarCallback->SetClientData(actorSet.scalarBarCallbackState.get());
            actorSet.scalarBarCallback->SetCallback([](vtkObject* caller, unsigned long eventId, void* clientData, void*) {
                auto* state = static_cast<ViewerWidget::LegendCallbackState*>(clientData);
                auto* widget = vtkScalarBarWidget::SafeDownCast(caller);
                if (state == nullptr || state->viewer == nullptr || widget == nullptr) {
                    return;
                }
                state->viewer->handleScalarBarInteraction(state->objectId, widget, eventId);
            });
            actorSet.scalarBarWidget->AddObserver(vtkCommand::InteractionEvent, actorSet.scalarBarCallback);
            actorSet.scalarBarWidget->AddObserver(vtkCommand::EndInteractionEvent, actorSet.scalarBarCallback);
        }
        applyScalarBarOptions(&actorSet, legendOptions);
    }

    if (options.showOutline) {
        vtkNew<vtkOutlineFilter> outlineFilter;
        outlineFilter->SetInputData(options.type == DisplayObjectType::Crop ? visualInput : input);
        outlineFilter->Update();
        vtkNew<vtkDataSetMapper> mapper;
        mapper->SetInputConnection(outlineFilter->GetOutputPort());
        mapper->ScalarVisibilityOff();
        vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->SetColor(options.outlineColor.redF(), options.outlineColor.greenF(), options.outlineColor.blueF());
        actor->GetProperty()->SetOpacity(std::clamp(options.outlineOpacity, 0.0, 1.0) * displayAlpha);
        actor->GetProperty()->SetLineWidth(std::max(0.1, options.outlineLineWidth));
        addProp(actor);
    }

    if (options.showMesh) {
        vtkSmartPointer<vtkPolyData> sparseMesh;
        if (algorithm == nullptr) {
            sparseMesh = makeSparseImagePlaneMesh(vtkImageData::SafeDownCast(visualInput),
                                                  std::max(1, options.meshStride));
        }

        vtkNew<vtkPolyDataMapper> mapper;
        if (sparseMesh != nullptr) {
            mapper->SetInputData(sparseMesh);
        } else {
            vtkNew<vtkExtractEdges> edges;
            if (algorithm != nullptr) {
                vtkNew<vtkDataSetSurfaceFilter> surface;
                surface->SetInputConnection(algorithm->GetOutputPort());
                surface->Update();
                edges->SetInputConnection(surface->GetOutputPort());
            } else {
                edges->SetInputData(visualInput);
            }
            edges->Update();
            mapper->SetInputConnection(edges->GetOutputPort());
        }
        mapper->ScalarVisibilityOff();
        vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->SetColor(options.meshColor.redF(), options.meshColor.greenF(), options.meshColor.blueF());
        actor->GetProperty()->SetOpacity(std::clamp(options.meshOpacity, 0.0, 1.0) * displayAlpha);
        actor->GetProperty()->SetLineWidth(std::max(0.1, options.meshLineWidth));
        addProp(actor);
    }

    if (actorSet.plane != nullptr && actorSet.planeAlgorithm != nullptr && vtkWidget_ != nullptr && vtkWidget_->interactor() != nullptr) {
        actorSet.planeRepresentation = vtkSmartPointer<vtkImplicitPlaneRepresentation>::New();
        double bounds[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        input->GetBounds(bounds);
        actorSet.planeRepresentation->SetPlaceFactor(1.0);
        actorSet.planeRepresentation->PlaceWidget(bounds);
        double planeOrigin[3] = {options.planeOrigin[0], options.planeOrigin[1], options.planeOrigin[2]};
        double planeNormal[3] = {options.planeNormal[0], options.planeNormal[1], options.planeNormal[2]};
        actorSet.planeRepresentation->SetOrigin(planeOrigin);
        actorSet.planeRepresentation->SetNormal(planeNormal);
        actorSet.planeRepresentation->DrawPlaneOn();
        actorSet.planeRepresentation->OutlineTranslationOff();
        actorSet.planeRepresentation->GetPlane(actorSet.plane);

        actorSet.planeWidget = vtkSmartPointer<vtkImplicitPlaneWidget2>::New();
        actorSet.planeWidget->SetInteractor(vtkWidget_->interactor());
        actorSet.planeWidget->SetRepresentation(actorSet.planeRepresentation);

        actorSet.planeCallbackState = std::make_shared<PlaneCallbackState>();
        actorSet.planeCallbackState->viewer = this;
        actorSet.planeCallbackState->objectId = objectId;
        actorSet.planeCallbackState->plane = actorSet.plane;
        actorSet.planeCallbackState->algorithm = actorSet.planeAlgorithm;
        actorSet.planeCallback = vtkSmartPointer<vtkCallbackCommand>::New();
        actorSet.planeCallback->SetClientData(actorSet.planeCallbackState.get());
        actorSet.planeCallback->SetCallback([](vtkObject* caller, unsigned long, void* clientData, void*) {
            auto* state = static_cast<ViewerWidget::PlaneCallbackState*>(clientData);
            auto* widget = vtkImplicitPlaneWidget2::SafeDownCast(caller);
            if (state == nullptr || state->viewer == nullptr || widget == nullptr) {
                return;
            }
            state->viewer->handlePlaneInteraction(state->objectId, widget, state->plane, state->algorithm);
        });
        actorSet.planeWidget->AddObserver(vtkCommand::InteractionEvent, actorSet.planeCallback);
        actorSet.planeWidget->AddObserver(vtkCommand::EndInteractionEvent, actorSet.planeCallback);
        actorSet.planeWidget->Off();
    }

    displayActors_.insert(objectId, actorSet);
    updatePlaneWidgetVisibility();
    if (renderer_ != nullptr) {
        renderer_->ResetCameraClippingRange();
        if (resetCameraToObject) {
            renderer_->ResetCamera();
        }
    }
    applyDisplayLighting();
    if (renderWindow_ != nullptr) {
        renderWindow_->Render();
    }
    return true;
}

bool ViewerWidget::addOrUpdateRayTracingVolume(const QString& objectId,
                                               const DataObjectOptions& options,
                                               bool resetCameraToObject,
                                               QString* errorMessage) {
    if (options.inputPath.trimmed().isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = "Please select a .vti file or a .pvd file referencing .vti frames before applying this volume.";
        }
        return false;
    }

    const QString absoluteInputPath = QFileInfo(options.inputPath).absoluteFilePath();
    const QString sourceSuffix = QFileInfo(absoluteInputPath).suffix().toLower();
    QVector<FrameInfo> parsedFrames;
    QVector<Streamcenter::Index::VolumeFrame> indexFrames;
    QString loadPath = absoluteInputPath;

    if (sourceSuffix == QStringLiteral("pvd")) {
        parsedFrames = parsePvdFile(absoluteInputPath, errorMessage);
        if (parsedFrames.isEmpty()) {
            return false;
        }
        indexFrames.reserve(parsedFrames.size());
        for (const FrameInfo& frame : std::as_const(parsedFrames)) {
            if (QFileInfo(frame.path).suffix().compare(QStringLiteral("vti"), Qt::CaseInsensitive) != 0) {
                if (errorMessage != nullptr) {
                    *errorMessage = QString("Ray-tracing volume accepts only .pvd files that reference .vti frames; rejected frame: %1")
                                        .arg(frame.path);
                }
                return false;
            }
            Streamcenter::Index::VolumeFrame indexFrame;
            indexFrame.timestep = frame.timestep;
            indexFrame.path = frame.path;
            indexFrames.push_back(indexFrame);
        }
        if (parsedFrames.size() > 1 && isTimeSeries_ && currentFrameIndex_ >= 0 && currentFrameIndex_ < frames_.size()) {
            const int frameIndex = nearestFrameIndexForTime(parsedFrames, frames_.at(currentFrameIndex_).timestep);
            loadPath = frameIndex >= 0 ? parsedFrames.at(frameIndex).path : parsedFrames.front().path;
        } else {
            loadPath = parsedFrames.front().path;
        }
    } else if (sourceSuffix == QStringLiteral("vti")) {
        Streamcenter::Index::VolumeFrame indexFrame;
        indexFrame.timestep = 0.0;
        indexFrame.path = absoluteInputPath;
        indexFrames.push_back(indexFrame);
    } else {
        if (errorMessage != nullptr) {
            *errorMessage = QString("Ray-tracing volume accepts only .vti input or .pvd time-series input, not: %1")
                                .arg(absoluteInputPath);
        }
        return false;
    }

    Streamcenter::Index::VolumeOptions volumeOptions;
    volumeOptions.fieldName = options.colorField;
    volumeOptions.componentName = options.colorComponent;
    volumeOptions.colorMap = options.colorMap;
    volumeOptions.autoColorRange = options.autoColorRange;
    volumeOptions.colorRangeMin = options.colorRangeMin;
    volumeOptions.colorRangeMax = options.colorRangeMax;
    volumeOptions.opacityScale = options.volumeOpacityScale;
    volumeOptions.samplingStep = options.volumeSamplingStep;
    volumeOptions.filtering = options.volumeFiltering;
    volumeOptions.preintegration = options.volumePreintegration;

    vtkSmartPointer<vtkDataSet> input;
    if (!loadDataSetFromPath(loadPath, &input, errorMessage)) {
        return false;
    }
    vtkImageData* imageData = vtkImageData::SafeDownCast(input);
    if (imageData == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QString("Ray-tracing volume accepts only vtkImageData/.vti input, not: %1").arg(loadPath);
        }
        return false;
    }

    QString fieldName = options.colorField.trimmed();
    QString componentName = options.colorComponent;
    if (fieldName.isEmpty()) {
        fieldName = firstPointFieldName(imageData);
        componentName = QStringLiteral("Magnitude");
    }
    const ResolvedScalarField scalar = resolveScalarField(imageData, fieldName, componentName);
    if (scalar.array == nullptr || scalar.arrayName.trimmed().isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QString("Point scalar field was not found in the VTI file: %1").arg(fieldName);
        }
        return false;
    }
    if (scalar.association != kFieldAssociationPoints) {
        if (errorMessage != nullptr) {
            *errorMessage = QString("Ray-tracing volume supports only point data arrays; selected field is not point data: %1")
                                .arg(fieldName);
        }
        return false;
    }
    if (imageData->GetPointData() == nullptr
        || imageData->GetPointData()->SetActiveScalars(scalar.arrayName.toUtf8().constData()) < 0) {
        if (errorMessage != nullptr) {
            *errorMessage = QString("Could not activate point scalar field for volume rendering: %1").arg(scalar.arrayName);
        }
        return false;
    }

    double scalarRange[2] = {options.colorRangeMin, options.colorRangeMax};
    if (options.autoColorRange || !(scalarRange[1] > scalarRange[0])) {
        scalar.array->GetRange(scalarRange);
    }
    if (!(scalarRange[1] > scalarRange[0])) {
        scalarRange[1] = scalarRange[0] + 1.0;
    }

    bool indexBackendOwnsRendering = false;
    if (indexVolumeBackend_ != nullptr) {
        const Streamcenter::Index::RuntimeStatus status = indexVolumeBackend_->runtimeStatus();
        if (status.available) {
            QString indexError;
            indexBackendOwnsRendering = indexVolumeBackend_->loadOrUpdate(objectId, indexFrames, volumeOptions, &indexError);
            if (!indexBackendOwnsRendering) {
                indexVolumeBackend_->remove(objectId);
            }
        }
    }

    displayObjectOptions_.insert(objectId, options);
    removeDataObjectActors(objectId);
    if (sourceSuffix == QStringLiteral("pvd") && parsedFrames.size() > 1) {
        DisplayObjectTimeSeries series;
        series.sourcePath = absoluteInputPath;
        series.frames = parsedFrames;
        displayObjectTimeSeries_.insert(objectId, series);
        refreshDisplayTimeSeriesState();
    } else if (displayObjectTimeSeries_.remove(objectId) > 0) {
        refreshDisplayTimeSeriesState();
    }

    DisplayActorSet actorSet;
    if (!options.visible) {
        if (indexVolumeBackend_ != nullptr) {
            indexVolumeBackend_->remove(objectId);
        }
        displayActors_.insert(objectId, actorSet);
        applyDisplayLighting();
        return true;
    }

    auto addProp = [&](vtkProp* prop) {
        if (prop == nullptr || renderer_ == nullptr) {
            return;
        }
        renderer_->AddViewProp(prop);
        actorSet.props.push_back(prop);
    };

    if (!indexBackendOwnsRendering) {
        vtkSmartPointer<vtkGPUVolumeRayCastMapper> volumeMapper = vtkSmartPointer<vtkGPUVolumeRayCastMapper>::New();
        if (volumeMapper == nullptr) {
            if (errorMessage != nullptr) {
                *errorMessage = QStringLiteral(
                    "VTK GPU volume ray tracing is unavailable: vtkGPUVolumeRayCastMapper has no OpenGL backend. "
                    "Install or rebuild VTK with the RenderingVolumeOpenGL2/opengl module.");
            }
            return false;
        }

        double spacing[3] = {1.0, 1.0, 1.0};
        imageData->GetSpacing(spacing);
        const double minSpacing = std::max(0.0001,
                                           std::min({std::abs(spacing[0]), std::abs(spacing[1]), std::abs(spacing[2])}));
        const double requestedStep = options.volumeSamplingStep > 0.0 ? options.volumeSamplingStep : minSpacing;
        const double sampleDistance = std::max(0.0001, requestedStep);

        volumeMapper->SetInputData(imageData);
        volumeMapper->SetScalarModeToUsePointData();
        volumeMapper->SetBlendModeToComposite();
        volumeMapper->AutoAdjustSampleDistancesOff();
        volumeMapper->LockSampleDistanceToInputSpacingOff();
        volumeMapper->SetSampleDistance(sampleDistance);
        volumeMapper->SetImageSampleDistance(1.0);
        volumeMapper->UseJitteringOn();

        vtkSmartPointer<vtkVolumeProperty> volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
        volumeProperty->SetIndependentComponents(1);
        volumeProperty->SetColor(makeVolumeColorTransferFunction(options.colorMap, scalarRange));
        volumeProperty->SetScalarOpacity(makeVolumeOpacityFunction(scalarRange, options.volumeOpacityScale));
        volumeProperty->SetScalarOpacityUnitDistance(sampleDistance);
        volumeProperty->SetTransferFunctionModeTo1D();
        if (options.volumeFiltering.compare(QStringLiteral("Nearest"), Qt::CaseInsensitive) == 0) {
            volumeProperty->SetInterpolationTypeToNearest();
        } else {
            volumeProperty->SetInterpolationTypeToLinear();
        }
        if (displayOptions_.lightingEnabled) {
            volumeProperty->ShadeOn();
            volumeProperty->SetAmbient(std::clamp(1.0 - displayOptions_.lightingIntensity, 0.0, 1.0));
            volumeProperty->SetDiffuse(std::clamp(displayOptions_.lightingIntensity, 0.0, 1.0));
            volumeProperty->SetSpecular(0.10);
        } else {
            volumeProperty->ShadeOff();
        }

        vtkSmartPointer<vtkVolume> volume = vtkSmartPointer<vtkVolume>::New();
        volume->SetMapper(volumeMapper);
        volume->SetProperty(volumeProperty);
        addProp(volume);
    }

    if (options.showOutline) {
        vtkNew<vtkOutlineFilter> outlineFilter;
        outlineFilter->SetInputData(input);
        outlineFilter->Update();
        vtkNew<vtkDataSetMapper> mapper;
        mapper->SetInputConnection(outlineFilter->GetOutputPort());
        mapper->ScalarVisibilityOff();
        vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->SetColor(options.outlineColor.redF(), options.outlineColor.greenF(), options.outlineColor.blueF());
        actor->GetProperty()->SetOpacity(std::clamp(options.outlineOpacity, 0.0, 1.0));
        actor->GetProperty()->SetLineWidth(std::max(0.1, options.outlineLineWidth));
        addProp(actor);
    }

    displayActors_.insert(objectId, actorSet);
    updatePlaneWidgetVisibility();
    if (renderer_ != nullptr) {
        renderer_->ResetCameraClippingRange();
        if (resetCameraToObject) {
            renderer_->ResetCamera();
        }
    }
    applyDisplayLighting();
    if (renderWindow_ != nullptr) {
        renderWindow_->Render();
    }
    return true;
}

void ViewerWidget::removeDataObject(const QString& objectId) {
    displayObjectOptions_.remove(objectId);
    displayObjectTimeSeries_.remove(objectId);
    if (indexVolumeBackend_ != nullptr) {
        indexVolumeBackend_->remove(objectId);
    }
    refreshDisplayTimeSeriesState();
    if (activeDataObjectHandleId_ == objectId) {
        activeDataObjectHandleId_.clear();
    }
    removeDataObjectActors(objectId);
    updatePlaneWidgetVisibility();
    applyDisplayLighting();
}

void ViewerWidget::applyDisplayLighting() {
    if (renderer_ == nullptr) {
        return;
    }

    if (displayLight_ == nullptr) {
        displayLight_ = vtkSmartPointer<vtkLight>::New();
    }
    renderer_->AutomaticLightCreationOff();
    renderer_->RemoveAllLights();
    renderer_->AddLight(displayLight_);

    displayLight_->SetSwitch(displayOptions_.lightingEnabled ? 1 : 0);
    displayLight_->SetIntensity(std::clamp(displayOptions_.lightingIntensity, 0.0, 2.0));
    const QColor lightColor = displayOptions_.lighting.color.isValid()
        ? displayOptions_.lighting.color
        : QColor(Qt::white);
    displayLight_->SetAmbientColor(lightColor.redF(), lightColor.greenF(), lightColor.blueF());
    displayLight_->SetDiffuseColor(lightColor.redF(), lightColor.greenF(), lightColor.blueF());
    displayLight_->SetSpecularColor(lightColor.redF(), lightColor.greenF(), lightColor.blueF());
    displayLight_->SetAttenuationValues(std::max(0.0, displayOptions_.lighting.attenuation[0]),
                                        std::max(0.0, displayOptions_.lighting.attenuation[1]),
                                        std::max(0.0, displayOptions_.lighting.attenuation[2]));
    displayLight_->SetConeAngle(std::clamp(displayOptions_.lighting.coneAngle, 0.0, 180.0));
    displayLight_->SetExponent(std::clamp(displayOptions_.lighting.exponent, 0.0, 128.0));

    double bounds[6] = {0.0, -1.0, 0.0, -1.0, 0.0, -1.0};
    renderer_->ComputeVisiblePropBounds(bounds);
    bool validBounds = bounds[0] <= bounds[1] && bounds[2] <= bounds[3] && bounds[4] <= bounds[5];
    for (double value : bounds) {
        validBounds = validBounds && std::isfinite(value);
    }

    double center[3] = {0.0, 0.0, 0.0};
    double distance = 1.0;
    if (validBounds) {
        center[0] = 0.5 * (bounds[0] + bounds[1]);
        center[1] = 0.5 * (bounds[2] + bounds[3]);
        center[2] = 0.5 * (bounds[4] + bounds[5]);
        const double spanX = bounds[1] - bounds[0];
        const double spanY = bounds[3] - bounds[2];
        const double spanZ = bounds[5] - bounds[4];
        const double radius = 0.5 * std::sqrt(spanX * spanX + spanY * spanY + spanZ * spanZ);
        distance = std::max(1.0, radius * 3.0);
    }

    const QString mode = displayOptions_.lightingMode.trimmed();
    if (mode.compare(QStringLiteral("Scene light"), Qt::CaseInsensitive) == 0
        || mode.compare(QStringLiteral("SceneLight"), Qt::CaseInsensitive) == 0) {
        displayLight_->SetLightTypeToSceneLight();
        displayLight_->PositionalOn();
    } else if (mode.compare(QStringLiteral("Directional light"), Qt::CaseInsensitive) == 0
               || mode.compare(QStringLiteral("Directional"), Qt::CaseInsensitive) == 0) {
        displayLight_->SetLightTypeToSceneLight();
        displayLight_->PositionalOff();
    } else {
        displayLight_->SetLightTypeToHeadlight();
        displayLight_->PositionalOff();
    }

    displayLight_->SetPosition(center[0] + displayOptions_.lighting.positionOffset[0] * distance,
                               center[1] + displayOptions_.lighting.positionOffset[1] * distance,
                               center[2] + displayOptions_.lighting.positionOffset[2] * distance);
    displayLight_->SetFocalPoint(center[0] + displayOptions_.lighting.focalOffset[0] * distance,
                                 center[1] + displayOptions_.lighting.focalOffset[1] * distance,
                                 center[2] + displayOptions_.lighting.focalOffset[2] * distance);
}

void ViewerWidget::setActiveDataObjectHandle(const QString& objectId) {
    const QString normalized = objectId.trimmed();
    if (activeDataObjectHandleId_ == normalized) {
        updatePlaneWidgetVisibility();
        return;
    }
    activeDataObjectHandleId_ = normalized;
    updatePlaneWidgetVisibility();
}

void ViewerWidget::removeDataObjectActors(const QString& objectId) {
    auto it = displayActors_.find(objectId);
    if (it == displayActors_.end()) {
        return;
    }
    if (it->planeWidget != nullptr) {
        it->planeWidget->Off();
    }
    if (it->scalarBarWidget != nullptr) {
        it->scalarBarWidget->Off();
    }
    if (renderer_ != nullptr) {
        for (const vtkSmartPointer<vtkProp>& prop : it->props) {
            renderer_->RemoveViewProp(prop);
        }
    }
    displayActors_.erase(it);
    if (renderWindow_ != nullptr) {
        renderWindow_->Render();
    }
}

void ViewerWidget::renderIndexVolumes() {
    if (indexVolumeBackend_ == nullptr || renderer_ == nullptr || renderWindow_ == nullptr) {
        return;
    }
    QString ignoredError;
    indexVolumeBackend_->render(renderer_, renderWindow_, currentFrameIndex_, &ignoredError);
}

void ViewerWidget::handlePlaneInteraction(const QString& objectId,
                                          vtkImplicitPlaneWidget2* widget,
                                          vtkPlane* plane,
                                          vtkAlgorithm* algorithm) {
    if (widget == nullptr || plane == nullptr) {
        return;
    }
    if (objectId != activeDataObjectHandleId_ || !displayObjectOptions_.value(objectId).showPlaneHandle) {
        return;
    }
    auto* representation = vtkImplicitPlaneRepresentation::SafeDownCast(widget->GetRepresentation());
    if (representation == nullptr) {
        return;
    }
    representation->GetPlane(plane);
    if (algorithm != nullptr) {
        algorithm->Modified();
        algorithm->Update();
    }
    double originValues[3] = {0.0, 0.0, 0.0};
    double normalValues[3] = {1.0, 0.0, 0.0};
    plane->GetOrigin(originValues);
    plane->GetNormal(normalValues);
    if (renderer_ != nullptr) {
        renderer_->ResetCameraClippingRange();
    }
    if (renderWindow_ != nullptr) {
        renderWindow_->Render();
    }
    emit dataObjectPlaneChanged(objectId,
                                QVector<double>{originValues[0], originValues[1], originValues[2]},
                                QVector<double>{normalValues[0], normalValues[1], normalValues[2]});
}

void ViewerWidget::handleScalarBarInteraction(const QString& objectId, vtkScalarBarWidget* widget, unsigned long eventId) {
    if (widget == nullptr) {
        return;
    }
    auto actorIt = displayActors_.find(objectId);
    auto optionIt = displayObjectOptions_.find(objectId);
    if (actorIt == displayActors_.end() || optionIt == displayObjectOptions_.end()) {
        return;
    }

    vtkScalarBarRepresentation* representation = widget->GetScalarBarRepresentation();
    if (representation != nullptr) {
        const double* position = representation->GetPositionCoordinate()->GetValue();
        optionIt->legend.position[0] = position[0];
        optionIt->legend.position[1] = position[1];
        optionIt->legend.windowLocation = QStringLiteral("Any Location");
        syncScalarBarActorFromRepresentation(&(*actorIt));
    }

    syncScalarBarCustomOverlays(&(*actorIt), optionIt->legend);
    syncScalarBarBackgroundAndOutline(&(*actorIt), optionIt->legend);
    if (renderWindow_ != nullptr) {
        renderWindow_->Render();
    }
    if (eventId == vtkCommand::EndInteractionEvent) {
        emit legendStyleChanged(objectId, optionIt->legend);
    }
}

void ViewerWidget::syncScalarBarActorFromRepresentation(DisplayActorSet* actorSet) {
    if (actorSet == nullptr || actorSet->scalarBar == nullptr || actorSet->scalarBarWidget == nullptr) {
        return;
    }
    vtkScalarBarRepresentation* representation = actorSet->scalarBarWidget->GetScalarBarRepresentation();
    if (representation == nullptr) {
        return;
    }

    const double* position = representation->GetPositionCoordinate()->GetValue();
    const double* size = representation->GetPosition2Coordinate()->GetValue();
    actorSet->scalarBar->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
    actorSet->scalarBar->GetPositionCoordinate()->SetValue(position[0], position[1]);
    actorSet->scalarBar->GetPosition2Coordinate()->SetCoordinateSystemToNormalizedViewport();
    actorSet->scalarBar->GetPosition2Coordinate()->SetValue(size[0], size[1]);
    actorSet->scalarBar->Modified();
}

void ViewerWidget::syncScalarBarCustomOverlays(DisplayActorSet* actorSet, const LegendOptions& options) {
    if (actorSet == nullptr || actorSet->scalarBar == nullptr) {
        return;
    }

    QRectF barRect;
    if (auto* scalarBar = StreamcenterScalarBarActor::SafeDownCast(actorSet->scalarBar.GetPointer())) {
        scalarBar->scalarBarPixelRect(renderer_, &barRect);
    }
    const int viewportWidth = vtkWidget_ != nullptr ? std::max(1, vtkWidget_->width()) : 800;
    const int viewportHeight = vtkWidget_ != nullptr ? std::max(1, vtkWidget_->height()) : 600;
    if (barRect.isEmpty()) {
        vtkCoordinate* positionCoordinate = nullptr;
        vtkCoordinate* sizeCoordinate = nullptr;
        if (actorSet->scalarBarWidget != nullptr
            && actorSet->scalarBarWidget->GetScalarBarRepresentation() != nullptr) {
            vtkScalarBarRepresentation* representation = actorSet->scalarBarWidget->GetScalarBarRepresentation();
            positionCoordinate = representation->GetPositionCoordinate();
            sizeCoordinate = representation->GetPosition2Coordinate();
        } else {
            positionCoordinate = actorSet->scalarBar->GetPositionCoordinate();
            sizeCoordinate = actorSet->scalarBar->GetPosition2Coordinate();
        }
        if (positionCoordinate != nullptr && sizeCoordinate != nullptr) {
            const double* position = positionCoordinate->GetValue();
            const double* size = sizeCoordinate->GetValue();
            const double frameX = position[0] * viewportWidth;
            const double frameY = position[1] * viewportHeight;
            const double frameWidth = size[0] * viewportWidth;
            const double frameHeight = size[1] * viewportHeight;
            const double thickness = std::max(1, options.colorBarThickness);
            if (vtkLegendOrientation(options.orientation) == VTK_ORIENT_HORIZONTAL) {
                barRect = QRectF(frameX,
                                 frameY + std::max(0.0, frameHeight - thickness) * 0.5,
                                 frameWidth,
                                 std::min(thickness, frameHeight));
            } else {
                barRect = QRectF(frameX + std::max(0.0, frameWidth - thickness) * 0.5,
                                 frameY,
                                 std::min(thickness, frameWidth),
                                 frameHeight);
            }
        }
    }

    if (actorSet->scalarBarTitle != nullptr) {
        const QString titleText = formattedLegendTitleText(options);
        if (titleText.isEmpty() || barRect.isEmpty()) {
            actorSet->scalarBarTitle->VisibilityOff();
        } else {
            const QByteArray titleUtf8 = titleText.toUtf8();
            actorSet->scalarBarTitle->SetInput(titleUtf8.constData());
            actorSet->scalarBarTitle->SetTextScaleModeToNone();
            actorSet->scalarBarTitle->GetPositionCoordinate()->SetCoordinateSystemToDisplay();
            setLegendTextPropertyFont(actorSet->scalarBarTitle->GetTextProperty(), options.titleFont);

            vtkTextProperty* property = actorSet->scalarBarTitle->GetTextProperty();
            const bool verticalTitle = canonicalLegendTitleOrientation(options.titleOrientation) == QStringLiteral("Vertical");
            const QString titlePosition = canonicalLegendTitlePosition(options.titlePosition);
            const QString titleJustification = canonicalLegendJustification(options.titleJustification);
            const int componentPadding = !options.componentTitle.trimmed().isEmpty()
                    && legendComponentFormatUsesNewLine(options.componentFormat)
                ? 10
                : 0;
            const int titlePadding = effectiveLegendTitlePaddingPixels(options, actorSet->scalarBar->GetLookupTable())
                + componentPadding;

            auto justifiedBarX = [&]() {
                if (titleJustification == QStringLiteral("Left")) {
                    return barRect.left();
                }
                if (titleJustification == QStringLiteral("Right")) {
                    return barRect.right();
                }
                return barRect.left() + barRect.width() * 0.5;
            };
            auto centeredVerticalTitleX = [&](double y, bool topSide) {
                QFontMetrics metrics(legendQtFont(options.titleFont));
                const QStringList lines = titleText.split(QLatin1Char('\n'));
                const int lineCount = std::max(1, static_cast<int>(lines.size()));
                int textWidth = 0;
                for (const QString& line : lines) {
                    textWidth = std::max(textWidth, metrics.horizontalAdvance(line));
                }
                const double halfRotatedWidth = std::max(1, metrics.height() * lineCount) * 0.5;
                const double halfRotatedHeight = std::max(1, textWidth) * 0.5;
                double x = barRect.left() + barRect.width() * 0.5;
                if (titleJustification == QStringLiteral("Left")) {
                    x = barRect.left() + halfRotatedWidth;
                } else if (titleJustification == QStringLiteral("Right")) {
                    x = barRect.right() - halfRotatedWidth;
                }
                return QPointF(x, y + (topSide ? halfRotatedHeight : -halfRotatedHeight));
            };

            if (verticalTitle) {
                actorSet->scalarBarTitle->SetOrientation(90.0f);
                property->SetJustificationToCentered();
                property->SetVerticalJustificationToCentered();
                const int titleSideFootprint = QFontMetrics(legendQtFont(options.titleFont)).height();
                if (titlePosition == QStringLiteral("Right")) {
                    actorSet->scalarBarTitle->SetPosition(barRect.right() + titlePadding + titleSideFootprint * 0.5,
                                                          barRect.top() + barRect.height() * 0.5);
                } else if (titlePosition == QStringLiteral("Top")) {
                    const QPointF position = centeredVerticalTitleX(barRect.top() + barRect.height() + titlePadding, true);
                    actorSet->scalarBarTitle->SetPosition(position.x(), position.y());
                } else if (titlePosition == QStringLiteral("Bottom")) {
                    const QPointF position = centeredVerticalTitleX(barRect.top() - titlePadding, false);
                    actorSet->scalarBarTitle->SetPosition(position.x(), position.y());
                } else {
                    actorSet->scalarBarTitle->SetPosition(barRect.left() - titlePadding - titleSideFootprint * 0.5,
                                                          barRect.top() + barRect.height() * 0.5);
                }
            } else {
                actorSet->scalarBarTitle->SetOrientation(0.0f);
                if (titlePosition == QStringLiteral("Right")) {
                    property->SetJustificationToLeft();
                    property->SetVerticalJustificationToCentered();
                    actorSet->scalarBarTitle->SetPosition(barRect.right() + titlePadding,
                                                          barRect.top() + barRect.height() * 0.5);
                } else if (titlePosition == QStringLiteral("Bottom")) {
                    setLegendTitleJustification(property, options.titleJustification);
                    property->SetVerticalJustificationToTop();
                    actorSet->scalarBarTitle->SetPosition(justifiedBarX(), barRect.top() - titlePadding);
                } else if (titlePosition == QStringLiteral("Left")) {
                    property->SetJustificationToRight();
                    property->SetVerticalJustificationToCentered();
                    actorSet->scalarBarTitle->SetPosition(barRect.left() - titlePadding,
                                                          barRect.top() + barRect.height() * 0.5);
                } else {
                    setLegendTitleJustification(property, options.titleJustification);
                    property->SetVerticalJustificationToBottom();
                    actorSet->scalarBarTitle->SetPosition(justifiedBarX(), barRect.top() + barRect.height() + titlePadding);
                }
            }
            actorSet->scalarBarTitle->VisibilityOn();
        }
    }

    if (actorSet->scalarBarTicks == nullptr || actorSet->scalarBarTicksPolyData == nullptr) {
        return;
    }
    const int tickCount = legendTotalTickLabelCount(options);
    auto hideTickLabels = [&]() {
        for (vtkSmartPointer<vtkTextActor>& labelActor : actorSet->scalarBarTickLabels) {
            if (labelActor != nullptr) {
                labelActor->VisibilityOff();
            }
        }
    };
    if (options.drawTickLabels && tickCount > 0 && !barRect.isEmpty() && actorSet->scalarBar->GetLookupTable() != nullptr) {
        while (actorSet->scalarBarTickLabels.size() < tickCount) {
            vtkSmartPointer<vtkTextActor> labelActor = vtkSmartPointer<vtkTextActor>::New();
            labelActor->SetTextScaleModeToNone();
            labelActor->SetLayerNumber(2);
            labelActor->GetPositionCoordinate()->SetCoordinateSystemToDisplay();
            if (renderer_ != nullptr) {
                renderer_->AddActor2D(labelActor);
                actorSet->props.push_back(labelActor.GetPointer());
            }
            actorSet->scalarBarTickLabels.push_back(labelActor);
        }
        for (int index = tickCount; index < actorSet->scalarBarTickLabels.size(); ++index) {
            if (actorSet->scalarBarTickLabels[index] != nullptr) {
                actorSet->scalarBarTickLabels[index]->VisibilityOff();
            }
        }

        double range[2] = {0.0, 1.0};
        if (vtkScalarsToColors* lookupTable = actorSet->scalarBar->GetLookupTable()) {
            double* lookupRange = lookupTable->GetRange();
            if (lookupRange != nullptr) {
                range[0] = lookupRange[0];
                range[1] = lookupRange[1];
            }
        }
        const QString labelFormat = options.automaticLabelFormat
            ? QStringLiteral("%-#6.3g")
            : vtkLegendLabelFormat(options.labelFormat, QStringLiteral("%-#6.3g"));
        const bool horizontal = vtkLegendOrientation(options.orientation) == VTK_ORIENT_HORIZONTAL;
        const bool precede = canonicalLegendTickAnnotationPosition(options.tickAnnotationPosition).startsWith(QStringLiteral("Ticks left"));
        const double labelPadding = std::max(0, options.tickLabelsPadding);
        for (int i = 0; i < tickCount; ++i) {
            vtkTextActor* labelActor = actorSet->scalarBarTickLabels[i];
            if (labelActor == nullptr) {
                continue;
            }
            const double t = legendTickFractionForIndex(i, options);
            const double value = range[0] + (range[1] - range[0]) * t;
            const QByteArray labelUtf8 = formattedLegendNumber(value, labelFormat).toUtf8();
            labelActor->SetInput(labelUtf8.constData());
            labelActor->SetTextScaleModeToNone();
            labelActor->GetPositionCoordinate()->SetCoordinateSystemToDisplay();
            setLegendTextPropertyFont(labelActor->GetTextProperty(), options.textFont);
            labelActor->SetOrientation(0.0f);
            if (horizontal) {
                labelActor->GetTextProperty()->SetJustificationToCentered();
                if (precede) {
                    labelActor->GetTextProperty()->SetVerticalJustificationToTop();
                    labelActor->SetPosition(barRect.left() + barRect.width() * t,
                                            barRect.top() - labelPadding);
                } else {
                    labelActor->GetTextProperty()->SetVerticalJustificationToBottom();
                    labelActor->SetPosition(barRect.left() + barRect.width() * t,
                                            barRect.top() + barRect.height() + labelPadding);
                }
            } else {
                labelActor->GetTextProperty()->SetVerticalJustificationToCentered();
                if (precede) {
                    labelActor->GetTextProperty()->SetJustificationToRight();
                    labelActor->SetPosition(barRect.left() - labelPadding,
                                            barRect.top() + barRect.height() * t);
                } else {
                    labelActor->GetTextProperty()->SetJustificationToLeft();
                    labelActor->SetPosition(barRect.left() + barRect.width() + labelPadding,
                                            barRect.top() + barRect.height() * t);
                }
            }
            labelActor->VisibilityOn();
        }
    } else {
        hideTickLabels();
    }

    if (!options.drawTickMarks || tickCount <= 0 || barRect.isEmpty()) {
        actorSet->scalarBarTicks->VisibilityOff();
        if (auto* scalarBar = StreamcenterScalarBarActor::SafeDownCast(actorSet->scalarBar.GetPointer())) {
            scalarBar->setForegroundTickOverlay(nullptr, QColor(0, 0, 0), 0.0, false);
        }
        return;
    }

    vtkNew<vtkPoints> points;
    vtkNew<vtkCellArray> lines;
    const bool horizontal = vtkLegendOrientation(options.orientation) == VTK_ORIENT_HORIZONTAL;
    const bool precede = canonicalLegendTickAnnotationPosition(options.tickAnnotationPosition).startsWith(QStringLiteral("Ticks left"));
    const QString tickDirection = canonicalLegendTickDirection(options.tickDirection);
    const double tickLength = std::max(0, options.tickLength);
    double outsideLength = tickLength;
    double insideLength = 0.0;
    if (tickDirection == QStringLiteral("Centered")) {
        outsideLength = tickLength * 0.5;
        insideLength = tickLength * 0.5;
    } else if (tickDirection == QStringLiteral("Inward")) {
        outsideLength = 0.0;
        insideLength = tickLength;
    }
    for (int i = 0; i < tickCount; ++i) {
        const double t = legendTickFractionForIndex(i, options);
        double x1 = 0.0;
        double y1 = 0.0;
        double x2 = 0.0;
        double y2 = 0.0;
        if (horizontal) {
            const double x = barRect.left() + barRect.width() * t;
            const double edgeY = precede ? barRect.top() : barRect.top() + barRect.height();
            const double directionSign = precede ? -1.0 : 1.0;
            x1 = x;
            x2 = x;
            if (tickDirection == QStringLiteral("Through")) {
                y1 = barRect.top();
                y2 = barRect.top() + barRect.height();
            } else {
                y1 = edgeY - directionSign * insideLength;
                y2 = edgeY + directionSign * outsideLength;
            }
        } else {
            const double y = barRect.top() + barRect.height() * t;
            const double edgeX = precede ? barRect.left() : barRect.left() + barRect.width();
            const double directionSign = precede ? -1.0 : 1.0;
            y1 = y;
            y2 = y;
            if (tickDirection == QStringLiteral("Through")) {
                x1 = barRect.left();
                x2 = barRect.left() + barRect.width();
            } else {
                x1 = edgeX - directionSign * insideLength;
                x2 = edgeX + directionSign * outsideLength;
            }
        }
        const vtkIdType first = points->InsertNextPoint(x1, y1, 0.0);
        const vtkIdType second = points->InsertNextPoint(x2, y2, 0.0);
        vtkIdType ids[2] = {first, second};
        lines->InsertNextCell(2, ids);
    }

    actorSet->scalarBarTicksPolyData->SetPoints(points);
    actorSet->scalarBarTicksPolyData->SetLines(lines);
    actorSet->scalarBarTicksPolyData->Modified();
    const QColor color = options.tickColor.isValid() ? options.tickColor : QColor(0, 0, 0);
    actorSet->scalarBarTicks->GetProperty()->SetColor(color.redF(), color.greenF(), color.blueF());
    actorSet->scalarBarTicks->GetProperty()->SetOpacity(static_cast<double>(color.alpha()) / 255.0);
    actorSet->scalarBarTicks->GetProperty()->SetDisplayLocationToForeground();
    actorSet->scalarBarTicks->SetLayerNumber(10);
    actorSet->scalarBarTicks->GetProperty()->SetLineWidth(1.75f);
    if (auto* scalarBar = StreamcenterScalarBarActor::SafeDownCast(actorSet->scalarBar.GetPointer())) {
        scalarBar->setForegroundTickOverlay(actorSet->scalarBarTicksPolyData, color, 1.75, true);
        actorSet->scalarBarTicks->VisibilityOff();
    } else {
        actorSet->scalarBarTicks->VisibilityOn();
    }
}

void ViewerWidget::syncScalarBarBackgroundAndOutline(DisplayActorSet* actorSet, const LegendOptions& options) {
    if (actorSet == nullptr || actorSet->scalarBar == nullptr) {
        return;
    }

    auto hideBackgroundAndOutline = [&]() {
        if (actorSet->scalarBarBackground != nullptr) {
            actorSet->scalarBarBackground->VisibilityOff();
        }
        if (actorSet->scalarBarOutline != nullptr) {
            actorSet->scalarBarOutline->VisibilityOff();
        }
    };

    const bool drawBackground = options.drawBackground
        && actorSet->scalarBarBackground != nullptr
        && actorSet->scalarBarBackgroundPolyData != nullptr;
    const bool drawOutline = options.drawScalarBarOutline
        && actorSet->scalarBarOutline != nullptr
        && actorSet->scalarBarOutlinePolyData != nullptr;
    if (!drawBackground && !drawOutline) {
        hideBackgroundAndOutline();
        return;
    }

    vtkCoordinate* positionCoordinate = nullptr;
    vtkCoordinate* sizeCoordinate = nullptr;
    if (actorSet->scalarBarWidget != nullptr
        && actorSet->scalarBarWidget->GetScalarBarRepresentation() != nullptr) {
        vtkScalarBarRepresentation* representation = actorSet->scalarBarWidget->GetScalarBarRepresentation();
        positionCoordinate = representation->GetPositionCoordinate();
        sizeCoordinate = representation->GetPosition2Coordinate();
    } else {
        positionCoordinate = actorSet->scalarBar->GetPositionCoordinate();
        sizeCoordinate = actorSet->scalarBar->GetPosition2Coordinate();
    }
    if (positionCoordinate == nullptr || sizeCoordinate == nullptr) {
        hideBackgroundAndOutline();
        return;
    }

    const double* position = positionCoordinate->GetValue();
    const double* size = sizeCoordinate->GetValue();
    const int viewportWidth = vtkWidget_ != nullptr ? std::max(1, vtkWidget_->width()) : 800;
    const int viewportHeight = vtkWidget_ != nullptr ? std::max(1, vtkWidget_->height()) : 600;
    const double padX = static_cast<double>(std::max(0, options.backgroundPadding)) / viewportWidth;
    const double padY = static_cast<double>(std::max(0, options.backgroundPadding)) / viewportHeight;
    const double minX = std::clamp(std::min(position[0], position[0] + size[0]) - padX, -1.0, 2.0);
    const double maxX = std::clamp(std::max(position[0], position[0] + size[0]) + padX, -1.0, 2.0);
    const double minY = std::clamp(std::min(position[1], position[1] + size[1]) - padY, -1.0, 2.0);
    const double maxY = std::clamp(std::max(position[1], position[1] + size[1]) + padY, -1.0, 2.0);
    if (maxX <= minX || maxY <= minY) {
        hideBackgroundAndOutline();
        return;
    }

    auto updateRectanglePoints = [&](vtkPolyData* polyData) {
        vtkNew<vtkPoints> points;
        points->SetNumberOfPoints(4);
        points->SetPoint(0, minX, minY, 0.0);
        points->SetPoint(1, maxX, minY, 0.0);
        points->SetPoint(2, maxX, maxY, 0.0);
        points->SetPoint(3, minX, maxY, 0.0);
        polyData->SetPoints(points);
    };

    if (drawBackground) {
        updateRectanglePoints(actorSet->scalarBarBackgroundPolyData);
        vtkNew<vtkCellArray> polys;
        vtkIdType ids[4] = {0, 1, 2, 3};
        polys->InsertNextCell(4, ids);
        actorSet->scalarBarBackgroundPolyData->SetPolys(polys);
        actorSet->scalarBarBackgroundPolyData->Modified();

        const QColor color = options.backgroundColor.isValid() ? options.backgroundColor : QColor(128, 128, 128);
        actorSet->scalarBarBackground->GetProperty()->SetColor(color.redF(), color.greenF(), color.blueF());
        actorSet->scalarBarBackground->GetProperty()->SetOpacity(color.alphaF());
        actorSet->scalarBarBackground->VisibilityOn();
    } else if (actorSet->scalarBarBackground != nullptr) {
        actorSet->scalarBarBackground->VisibilityOff();
    }

    if (drawOutline) {
        updateRectanglePoints(actorSet->scalarBarOutlinePolyData);
        vtkNew<vtkCellArray> lines;
        vtkIdType ids[5] = {0, 1, 2, 3, 0};
        lines->InsertNextCell(5, ids);
        actorSet->scalarBarOutlinePolyData->SetLines(lines);
        actorSet->scalarBarOutlinePolyData->Modified();

        const QColor color = options.scalarBarOutlineColor.isValid() ? options.scalarBarOutlineColor : QColor(255, 255, 255);
        actorSet->scalarBarOutline->GetProperty()->SetColor(color.redF(), color.greenF(), color.blueF());
        actorSet->scalarBarOutline->GetProperty()->SetOpacity(color.alphaF());
        actorSet->scalarBarOutline->GetProperty()->SetLineWidth(static_cast<float>(std::max(0.0, options.scalarBarOutlineThickness)));
        actorSet->scalarBarOutline->VisibilityOn();
    } else if (actorSet->scalarBarOutline != nullptr) {
        actorSet->scalarBarOutline->VisibilityOff();
    }
}

void ViewerWidget::applyScalarBarOptions(DisplayActorSet* actorSet, const LegendOptions& options) {
    if (actorSet == nullptr || actorSet->scalarBar == nullptr) {
        return;
    }

    vtkScalarBarActor* actor = actorSet->scalarBar;
    vtkLookupTable* baseLookupTable = actorSet->scalarBarBaseLookupTable != nullptr
        ? actorSet->scalarBarBaseLookupTable.GetPointer()
        : vtkLookupTable::SafeDownCast(actor->GetLookupTable());
    if (baseLookupTable != nullptr) {
        vtkSmartPointer<vtkLookupTable> legendLookupTable = options.reverseLegend
            ? reversedLookupTable(baseLookupTable)
            : actorSet->scalarBarBaseLookupTable;
        actor->SetLookupTable(legendLookupTable != nullptr ? legendLookupTable.GetPointer() : baseLookupTable);
    }
    if (vtkLookupTable* lookupTable = vtkLookupTable::SafeDownCast(actor->GetLookupTable())) {
        const QColor nanColor = options.nanColor.isValid() ? options.nanColor : QColor(128, 128, 128);
        lookupTable->SetNanColor(nanColor.redF(), nanColor.greenF(), nanColor.blueF(), nanColor.alphaF());
        const QColor belowColor = options.belowRangeColor.isValid() ? options.belowRangeColor : QColor(0, 0, 0);
        const QColor aboveColor = options.aboveRangeColor.isValid() ? options.aboveRangeColor : QColor(255, 255, 255);
        lookupTable->SetBelowRangeColor(belowColor.redF(), belowColor.greenF(), belowColor.blueF(), belowColor.alphaF());
        lookupTable->SetAboveRangeColor(aboveColor.redF(), aboveColor.greenF(), aboveColor.blueF(), aboveColor.alphaF());
        lookupTable->SetUseBelowRangeColor(options.addRangeAnnotations ? 1 : 0);
        lookupTable->SetUseAboveRangeColor(options.addRangeAnnotations ? 1 : 0);
        lookupTable->Build();
    }
    actor->SetTitle("");
    actor->SetComponentTitle("");
    actor->SetOrientation(vtkLegendOrientation(options.orientation));
    actor->SetUnconstrainedFontSize(true);

    setLegendTextPropertyFont(actor->GetTitleTextProperty(), options.titleFont);
    setLegendTitleJustification(actor->GetTitleTextProperty(), options.titleJustification);
    setLegendTextPropertyFont(actor->GetLabelTextProperty(), options.textFont);
    setLegendTextPropertyFont(actor->GetAnnotationTextProperty(), options.textFont);

    actor->SetNumberOfLabels(0);
    const QString labelFormat = options.automaticLabelFormat
        ? QStringLiteral("%-#6.3g")
        : vtkLegendLabelFormat(options.labelFormat, QStringLiteral("%-#6.3g"));
    const QByteArray labelFormatUtf8 = labelFormat.toUtf8();
    actor->SetLabelFormat(labelFormatUtf8.constData());
    actor->SetDrawTickLabels(0);
    actor->SetDrawAnnotations((options.drawAnnotations || options.drawNanAnnotation || options.addRangeAnnotations) ? 1 : 0);
    actor->SetDrawNanAnnotation(options.drawNanAnnotation ? 1 : 0);
    actor->SetAnnotationLeaderPadding(8.0);
    const QByteArray nanAnnotation = options.nanAnnotation.toUtf8();
    actor->SetNanAnnotation(nanAnnotation.constData());
    actor->SetTextPosition(canonicalLegendTickAnnotationPosition(options.tickAnnotationPosition).startsWith(QStringLiteral("Ticks left"))
                               ? vtkScalarBarActor::PrecedeScalarBar
                               : vtkScalarBarActor::SucceedScalarBar);

    actor->SetDrawBackground(0);
    if (actor->GetBackgroundProperty() != nullptr) {
        const QColor color = options.backgroundColor.isValid() ? options.backgroundColor : QColor(128, 128, 128);
        actor->GetBackgroundProperty()->SetColor(color.redF(), color.greenF(), color.blueF());
        actor->GetBackgroundProperty()->SetOpacity(color.alphaF());
    }
    actor->SetTextPad(1);
    actor->SetDrawFrame(0);
    if (actor->GetFrameProperty() != nullptr) {
        const QColor color = options.scalarBarOutlineColor.isValid() ? options.scalarBarOutlineColor : QColor(255, 255, 255);
        actor->GetFrameProperty()->SetColor(color.redF(), color.greenF(), color.blueF());
        actor->GetFrameProperty()->SetOpacity(color.alphaF());
        actor->GetFrameProperty()->SetLineWidth(static_cast<float>(std::max(0.0, options.scalarBarOutlineThickness)));
    }
    actor->SetDrawBelowRangeSwatch(options.addRangeAnnotations);
    actor->SetDrawAboveRangeSwatch(options.addRangeAnnotations);

    actor->SetUseCustomLabels(false);
    actor->SetCustomLabels(nullptr);

    const int orientation = vtkLegendOrientation(options.orientation);
    const double length = std::clamp(options.colorBarLength, 0.01, 1.0);
    const int viewportWidth = vtkWidget_ != nullptr ? std::max(1, vtkWidget_->width()) : 800;
    const int viewportHeight = vtkWidget_ != nullptr ? std::max(1, vtkWidget_->height()) : 600;

    const double barPixelThickness = std::max(1, options.colorBarThickness);
    const QFontMetrics labelMetrics(legendQtFont(options.textFont));
    actor->SetVerticalTitleSeparation(0);
    const int labelSideFootprint = options.drawTickLabels
        ? estimateLegendAnnotationWidthPixels(options, actor->GetLookupTable()) + std::max(0, options.tickLabelsPadding) + 8
        : 0;
    const int labelTopFootprint = options.drawTickLabels
        ? labelMetrics.height() + std::max(0, options.tickLabelsPadding) + 8
        : 0;
    const int frameWidthPixels = orientation == VTK_ORIENT_HORIZONTAL
        ? std::max(1, static_cast<int>(std::round(length * viewportWidth)))
        : std::clamp(static_cast<int>(std::round(barPixelThickness)) + labelSideFootprint,
                     48,
                     std::max(48, static_cast<int>(std::round(viewportWidth * 0.45))));
    const int frameHeightPixels = orientation == VTK_ORIENT_HORIZONTAL
        ? std::clamp(static_cast<int>(std::round(barPixelThickness)) + labelTopFootprint,
                     42,
                     std::max(42, static_cast<int>(std::round(viewportHeight * 0.35))))
        : std::max(1, static_cast<int>(std::round(length * viewportHeight)));
    const double sizeX = orientation == VTK_ORIENT_HORIZONTAL
        ? length
        : std::clamp(static_cast<double>(frameWidthPixels) / viewportWidth, 0.01, 0.98);
    const double sizeY = orientation == VTK_ORIENT_HORIZONTAL
        ? std::clamp(static_cast<double>(frameHeightPixels) / viewportHeight, 0.01, 0.98)
        : length;
    const double barRatio = orientation == VTK_ORIENT_HORIZONTAL
        ? barPixelThickness / std::max(1.0, sizeY * viewportHeight)
        : barPixelThickness / std::max(1.0, sizeX * viewportWidth);
    actor->SetBarRatio(std::clamp(barRatio, 0.035, 0.45));
    actor->SetMaximumWidthInPixels(std::max(1, static_cast<int>(std::round(sizeX * viewportWidth))));
    actor->SetMaximumHeightInPixels(std::max(1, static_cast<int>(std::round(sizeY * viewportHeight))));
    actor->SetWidth(sizeX);
    actor->SetHeight(sizeY);
    const double effectiveX = std::clamp(options.position[0], -10.0, 0.98 - sizeX);
    const double effectiveY = std::clamp(options.position[1], -10.0, 0.98 - sizeY);
    actor->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
    actor->GetPositionCoordinate()->SetValue(effectiveX, effectiveY);
    actor->GetPosition2Coordinate()->SetCoordinateSystemToNormalizedViewport();
    actor->GetPosition2Coordinate()->SetValue(sizeX, sizeY);

    if (actorSet->scalarBarWidget != nullptr) {
        vtkScalarBarRepresentation* representation = actorSet->scalarBarWidget->GetScalarBarRepresentation();
        if (representation != nullptr) {
            representation->SetAutoOrient(options.autoOrient);
            representation->SetOrientation(orientation);
            representation->SetWindowLocation(vtkLegendWindowLocation(options.windowLocation));
            representation->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
            representation->GetPositionCoordinate()->SetValue(effectiveX, effectiveY);
            representation->GetPosition2Coordinate()->SetCoordinateSystemToNormalizedViewport();
            representation->GetPosition2Coordinate()->SetValue(sizeX, sizeY);
            representation->UpdateWindowLocation();
            syncScalarBarActorFromRepresentation(actorSet);
        }
    }

    actor->Modified();
    syncScalarBarCustomOverlays(actorSet, options);
    syncScalarBarBackgroundAndOutline(actorSet, options);
    if (renderWindow_ != nullptr) {
        renderWindow_->Render();
    }
}

bool ViewerWidget::showScalarBarStyleDialog(DisplayActorSet* actorSet) {
    if (actorSet == nullptr || actorSet->scalarBar == nullptr) {
        return false;
    }

    LegendOptions initialOptions;
    auto objectIt = displayObjectOptions_.find(actorSet->scalarBarObjectId);
    if (objectIt != displayObjectOptions_.end()) {
        initialOptions = objectIt->legend;
    }
    const QString currentDefaultTitle = !actorSet->scalarBarDefaultTitle.trimmed().isEmpty()
        ? actorSet->scalarBarDefaultTitle.trimmed()
        : objectIt != displayObjectOptions_.end()
            ? defaultLegendTitle(objectIt.value())
            : QStringLiteral("Colors");
    if (!initialOptions.titleEdited || initialOptions.title.trimmed().isEmpty()) {
        initialOptions.title = currentDefaultTitle;
        initialOptions.titleEdited = false;
    }
    if (initialOptions.titlePadding < 0) {
        initialOptions.titlePadding = defaultLegendTitlePaddingPixels(initialOptions, actorSet->scalarBar->GetLookupTable());
    }

    LegendPropertiesDialog dialog(this);
    dialog.setObjectName(QStringLiteral("legend_properties_dialog"));
    dialog.setWindowTitle(tr("Edit Color Legend Properties"));
    constexpr int kLegendDialogWidth = 486;
    dialog.setMinimumWidth(kLegendDialogWidth);
    dialog.resize(kLegendDialogWidth, 760);
    const QFont dialogFont = dialog.font();
    const int controlHeight = legendCompactControlHeight(dialogFont);
    const int inputControlHeight = legendInputControlHeight(dialogFont);
    const int comboControlHeight = legendComboControlHeight(dialogFont);
    const int colorPickerControlHeight = legendColorPickerControlHeight(dialogFont);
    const int miniColorPickerWidth = legendMiniColorPickerWidth(inputControlHeight);
    const int rowHeight = legendRowHeight(dialogFont);
    const int fieldLabelBaseWidth = legendFieldLabelWidth(dialogFont);
    const int legendFontPx = legendFontPixelSize(dialogFont);
    const auto cssColor = [](QColor color) {
        return QStringLiteral("rgba(%1, %2, %3, %4)")
            .arg(color.red())
            .arg(color.green())
            .arg(color.blue())
            .arg(QString::number(color.alphaF(), 'f', 3));
    };
    QColor checkedBackground = dialog.palette().color(QPalette::Highlight);
    checkedBackground.setAlpha(42);
    QColor checkedHoverBackground = dialog.palette().color(QPalette::Highlight);
    checkedHoverBackground.setAlpha(58);
    QColor checkedBorder = dialog.palette().color(QPalette::Highlight);
    checkedBorder.setAlpha(150);
    dialog.setStyleSheet(QStringLiteral(
        "QDialog#legend_properties_dialog QLabel,"
        "QDialog#legend_properties_dialog QLineEdit,"
        "QDialog#legend_properties_dialog QComboBox,"
        "QDialog#legend_properties_dialog QSpinBox,"
        "QDialog#legend_properties_dialog QDoubleSpinBox,"
        "QDialog#legend_properties_dialog QCheckBox,"
        "QDialog#legend_properties_dialog QPushButton,"
        "QDialog#legend_properties_dialog QToolButton{font-size:%1px;}"
        "QDialog#legend_properties_dialog QLabel[visualizationFieldLabel=\"true\"]{padding:0px 6px 0px 2px;}"
        "QDialog#legend_properties_dialog QLineEdit,"
        "QDialog#legend_properties_dialog QSpinBox,"
        "QDialog#legend_properties_dialog QDoubleSpinBox{padding:0px 6px 0px 6px;min-height:%2px;max-height:%2px;}"
        "QDialog#legend_properties_dialog QComboBox{padding:0px 14px 0px 6px;min-height:%3px;max-height:%3px;}"
        "QDialog#legend_properties_dialog QPushButton#visualization_color_picker[visualizationColorPicker=\"true\"]{padding:0px 10px 0px 5px;min-height:%4px;max-height:%4px;}"
        "QDialog#legend_properties_dialog QPushButton#visualization_color_picker[visualizationMiniColorPicker=\"true\"]{padding:0px;margin:0px;min-width:%9px;min-height:%2px;max-width:%9px;max-height:%2px;}"
        "QDialog#legend_properties_dialog QCheckBox[visualizationCheckField=\"true\"]{padding:0px;margin:0px;}"
        "QDialog#legend_properties_dialog QToolButton#visualization_object_advanced_button{padding:0px;margin:0px;border:1px solid %7;border-radius:3px;background-color:transparent;border-image:none;min-width:%5px;min-height:%5px;max-width:%5px;max-height:%5px;}"
        "QDialog#legend_properties_dialog QToolButton#visualization_object_advanced_button:checked{background-color:%6;border:1px solid %7;border-image:none;}"
        "QDialog#legend_properties_dialog QToolButton#visualization_object_advanced_button:checked:hover,"
        "QDialog#legend_properties_dialog QToolButton#visualization_object_advanced_button:checked:pressed{background-color:%8;border:1px solid %7;border-image:none;}"
        "QDialog#legend_properties_dialog QToolButton[legendFontStyleButton=\"true\"]{padding:0px;margin:0px;min-width:%2px;min-height:%2px;max-width:%2px;max-height:%2px;}")
        .arg(QString::number(legendFontPx),
             QString::number(inputControlHeight),
             QString::number(comboControlHeight),
             QString::number(colorPickerControlHeight),
             QString::number(inputControlHeight),
             cssColor(checkedBackground),
             cssColor(checkedBorder),
             cssColor(checkedHoverBackground),
             QString::number(miniColorPickerWidth)));

    auto* rootLayout = new QVBoxLayout(&dialog);
    rootLayout->setContentsMargins(kLegendFormLeftMargin,
                                   kLegendFormTopMargin,
                                   kLegendFormRightMargin,
                                   8);
    rootLayout->setSpacing(0);

    auto* filterRow = new LegendConfigurationRowWidget(&dialog);
    filterRow->setProperty("visualizationConfigurationRow", true);
    filterRow->setFixedHeight(rowHeight);
    filterRow->setMinimumWidth(0);
    filterRow->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    auto* filterLayout = new QHBoxLayout(filterRow);
    filterLayout->setContentsMargins(0,
                                     kLegendFormRowVerticalPadding,
                                     4,
                                     kLegendFormRowVerticalPadding);
    filterLayout->setSpacing(kLegendFormControlSpacing);
    auto* searchEdit = new QLineEdit(filterRow);
    searchEdit->setObjectName(QStringLiteral("visualization_object_filter_edit"));
    searchEdit->setFixedHeight(inputControlHeight);
    makeLegendWidthShrinkable(searchEdit);
    searchEdit->setPlaceholderText(tr("Search ... (use Esc to clear text)"));
    auto* advancedButton = new QToolButton(filterRow);
    advancedButton->setObjectName(QStringLiteral("visualization_object_advanced_button"));
    advancedButton->setCheckable(true);
    advancedButton->setToolTip(tr("Advanced options"));
    advancedButton->setIcon(makeLegendAdvancedOptionsIcon(dialog.palette().color(QPalette::ButtonText)));
    advancedButton->setFixedSize(inputControlHeight, inputControlHeight);
    advancedButton->setIconSize(QSize(16, 16));
    advancedButton->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    advancedButton->setToolButtonStyle(Qt::ToolButtonIconOnly);
    advancedButton->setAutoRaise(true);
    advancedButton->setCursor(Qt::PointingHandCursor);
    advancedButton->setFocusPolicy(Qt::NoFocus);
    filterLayout->addWidget(searchEdit, 1);
    filterLayout->addWidget(advancedButton, 0, Qt::AlignRight | Qt::AlignVCenter);
    rootLayout->addWidget(filterRow);
    auto* filterBottomPad = new QWidget(&dialog);
    filterBottomPad->setFixedHeight(2);
    filterBottomPad->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    filterBottomPad->setAutoFillBackground(true);
    QPalette filterBottomPalette = filterBottomPad->palette();
    filterBottomPalette.setColor(QPalette::Window, dialog.palette().color(QPalette::Window));
    filterBottomPad->setPalette(filterBottomPalette);
    rootLayout->addWidget(filterBottomPad);

    auto* scrollArea = new QScrollArea(&dialog);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    auto* content = new QWidget(scrollArea);
    content->setObjectName(QStringLiteral("configuration_page"));
    content->setProperty("visualizationConfigurationPage", true);
    content->setMinimumWidth(0);
    content->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    auto* contentLayout = new QVBoxLayout(content);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentLayout->setSpacing(kLegendFormRowSpacing);
    scrollArea->setWidget(content);
    rootLayout->addWidget(scrollArea, 1);

    struct FilterItem {
        QWidget* widget = nullptr;
        QString searchText;
        bool advancedOnly = false;
        std::function<bool()> dependency;
    };
    QVector<FilterItem> filterItems;

    auto addFilterItem = [&](QWidget* widget,
                             const QString& searchText,
                             bool advancedOnly = false,
                             std::function<bool()> dependency = {}) {
        filterItems.push_back({widget, searchText, advancedOnly, std::move(dependency)});
    };

    auto makeSection = [&](const QString& title, bool advancedOnly = false) {
        auto* section = new QWidget(content);
        section->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        auto* sectionLayout = new QVBoxLayout(section);
        sectionLayout->setContentsMargins(0, 0, 0, 1);
        sectionLayout->setSpacing(1);
        auto* label = new QLabel(title, section);
        QFont sectionFont = label->font();
        sectionFont.setBold(false);
        sectionFont.setItalic(true);
        label->setFont(sectionFont);
        label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        QColor sectionTextColor = label->palette().color(QPalette::WindowText);
        sectionTextColor.setAlphaF(0.80);
        label->setStyleSheet(QStringLiteral("QLabel{color:%1;}").arg(cssColor(sectionTextColor)));
        const int sectionLabelHeight = std::max(label->fontMetrics().height() + 1, legendFontPx + 3) + 1;
        section->setFixedHeight(sectionLabelHeight + 3);
        label->setFixedHeight(sectionLabelHeight);
        label->setContentsMargins(0, 2, 0, 0);
        auto* line = new QFrame(section);
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Plain);
        line->setFixedHeight(1);
        line->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        line->setStyleSheet(QStringLiteral("QFrame{color:rgba(128,128,128,0.30);}"));
        sectionLayout->addWidget(label);
        sectionLayout->addWidget(line);
        contentLayout->addWidget(section);
        addFilterItem(section, title, advancedOnly);
        return section;
    };

    auto makeRowWidget = [&](const QString& labelText,
                             const QString& searchText,
                             bool advancedOnly = false,
                             std::function<bool()> dependency = {}) {
        auto* row = new LegendConfigurationRowWidget(content);
        row->setProperty("visualizationConfigurationRow", true);
        row->setMinimumWidth(0);
        row->setFixedHeight(rowHeight);
        row->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        auto* rowLayout = new QHBoxLayout(row);
        rowLayout->setContentsMargins(0,
                                      kLegendFormRowVerticalPadding,
                                      4,
                                      kLegendFormRowVerticalPadding);
        rowLayout->setSpacing(kLegendFormControlSpacing);
        if (!labelText.isEmpty()) {
            auto* label = new QLabel(labelText, row);
            label->setProperty("visualizationFieldLabel", true);
            label->setProperty("visualizationFieldLabelBaseWidth", fieldLabelBaseWidth);
            label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
            label->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
            label->setFixedHeight(controlHeight);
            label->setToolTip(labelText);
            rowLayout->addWidget(label, 0, Qt::AlignVCenter);
            updateLegendFieldLabelWidthForRow(row);
        }
        contentLayout->addWidget(row);
        addFilterItem(row, labelText + QLatin1Char(' ') + searchText, advancedOnly, std::move(dependency));
        return rowLayout;
    };

    auto makeCompactControlGroup = [](QHBoxLayout* rowLayout) {
        auto* groupWidget = new QWidget(rowLayout->parentWidget());
        groupWidget->setMinimumWidth(0);
        groupWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        auto* groupLayout = new QHBoxLayout(groupWidget);
        groupLayout->setContentsMargins(0, 0, 0, 0);
        groupLayout->setSpacing(kLegendFormCompactControlSpacing);
        rowLayout->addWidget(groupWidget, 1);
        return groupLayout;
    };

    auto configureControl = [&](QWidget* widget, int height = -1) {
        if (widget == nullptr) {
            return;
        }
        widget->setMinimumWidth(0);
        widget->setFixedHeight(height > 0 ? height : controlHeight);
        widget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        makeLegendWidthShrinkable(widget);
    };

    auto makeLineEdit = [&](const QString& text) {
        auto* edit = new QLineEdit(text, content);
        configureControl(edit, inputControlHeight);
        return edit;
    };

    auto makeSpin = [&](double minValue, double maxValue, double value, double step, int decimals) {
        auto* spin = new QDoubleSpinBox(content);
        spin->setRange(minValue, maxValue);
        spin->setSingleStep(step);
        spin->setDecimals(decimals);
        spin->setButtonSymbols(QAbstractSpinBox::NoButtons);
        spin->setKeyboardTracking(false);
        spin->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        configureControl(spin, inputControlHeight);
        spin->setValue(value);
        return spin;
    };

    auto setColorButtonColor = [](QPushButton* button, const QColor& color) {
        setLegendColorButton(button, color);
    };
    auto colorButtonColor = [](QPushButton* button) {
        return legendButtonColor(button);
    };
    auto makeColorButton = [&](const QColor& color, const QString& title) {
        auto* button = new QPushButton(content);
        button->setObjectName(QStringLiteral("visualization_color_picker"));
        button->setProperty("visualizationColorPicker", true);
        button->setProperty("visualizationColorLabel", title);
        button->setAutoDefault(false);
        button->setMinimumWidth(0);
        button->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        button->setFixedHeight(colorPickerControlHeight);
        makeLegendWidthShrinkable(button);
        button->setToolTip(title);
        setColorButtonColor(button, color);
        connect(button, &QPushButton::clicked, &dialog, [button, title, colorButtonColor, setColorButtonColor, &dialog]() {
            const QColor chosen = chooseLegendColor(colorButtonColor(button), &dialog, title);
            if (chosen.isValid()) {
                setColorButtonColor(button, chosen);
            }
        });
        return button;
    };
    auto makeMiniColorButton = [&](const QColor& color, const QString& title) {
        auto* button = new LegendMiniColorButton(content);
        button->setObjectName(QStringLiteral("visualization_color_picker"));
        button->setProperty("visualizationColorPicker", true);
        button->setProperty("visualizationMiniColorPicker", true);
        button->setAutoDefault(false);
        button->setFixedSize(miniColorPickerWidth, inputControlHeight);
        button->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        button->setToolTip(title);
        setColorButtonColor(button, color);
        connect(button, &QPushButton::clicked, &dialog, [button, title, colorButtonColor, setColorButtonColor, &dialog]() {
            const QColor chosen = chooseLegendColor(colorButtonColor(button), &dialog, title);
            if (chosen.isValid()) {
                setColorButtonColor(button, chosen);
            }
        });
        return button;
    };

    auto makeCombo = [&](const QStringList& items, const QString& value) {
        auto* combo = new QComboBox(content);
        combo->addItems(items);
        const int index = combo->findText(value);
        combo->setCurrentIndex(index >= 0 ? index : 0);
        combo->setMinimumContentsLength(0);
        combo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
        combo->setProperty("visualizationComboBox", true);
        configureControl(combo, comboControlHeight);
        return combo;
    };

    auto makeCheckRow = [&](const QString& labelText,
                            bool checked,
                            bool advancedOnly = false,
                            std::function<bool()> dependency = {}) {
        auto* check = new LegendCheckBox(labelText, content);
        check->setChecked(checked);
        check->setProperty("visualizationCheckField", true);
        check->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
        check->setMinimumWidth(check->sizeHint().width());
        check->setFixedHeight(controlHeight);
        auto* rowLayout = makeRowWidget(QString(), labelText, advancedOnly, std::move(dependency));
        const QMargins margins = rowLayout->contentsMargins();
        rowLayout->setContentsMargins(kLegendCheckRowLeftPadding,
                                      margins.top(),
                                      margins.right(),
                                      margins.bottom());
        rowLayout->addWidget(check, 1, Qt::AlignVCenter);
        return check;
    };

    struct FontControls {
        QComboBox* family = nullptr;
        QSpinBox* size = nullptr;
        QPushButton* color = nullptr;
        QDoubleSpinBox* opacity = nullptr;
        QToolButton* bold = nullptr;
        QToolButton* italic = nullptr;
        QToolButton* shadow = nullptr;
    };

    auto makeStyleButton = [&](const QString& text, const QString& toolTip, bool checked) {
        auto* button = new QToolButton(content);
        button->setText(text);
        button->setToolTip(toolTip);
        button->setCheckable(true);
        button->setChecked(checked);
        button->setProperty("legendFontStyleButton", true);
        button->setToolButtonStyle(Qt::ToolButtonTextOnly);
        button->setFixedSize(inputControlHeight, inputControlHeight);
        button->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        if (text == QStringLiteral("B")) {
            QFont font = button->font();
            font.setBold(true);
            button->setFont(font);
        } else if (text == QStringLiteral("I")) {
            QFont font = button->font();
            font.setItalic(true);
            button->setFont(font);
        }
        return button;
    };

    QVector<QComboBox*> fontFamilyCombos;
    auto populateFontFamilyCombo = [&](QComboBox* combo, const QString& selectedFamily) {
        if (combo == nullptr) {
            return;
        }
        const QSignalBlocker blocker(combo);
        combo->clear();
        for (const QString& family : legendAvailableFontFamilies()) {
            combo->addItem(family);
            const int itemIndex = combo->count() - 1;
            combo->setItemData(itemIndex, QFont(legendQtFontFamilyName(family)), Qt::FontRole);
        }
        combo->insertSeparator(combo->count());
        combo->addItem(QObject::tr(kLegendImportFontLabel));

        const QString canonical = canonicalLegendFontFamily(selectedFamily);
        const int index = combo->findText(canonical);
        combo->setCurrentIndex(index >= 0 ? index : combo->findText(QStringLiteral("Arial")));
        const QString currentFamily = combo->currentText();
        combo->setFont(QFont(legendQtFontFamilyName(currentFamily)));
        combo->setProperty("legendPreviousFontFamily", currentFamily);
    };
    auto refreshFontFamilyCombos = [&](QComboBox* importingCombo, const QString& importedFamily) {
        for (QComboBox* combo : std::as_const(fontFamilyCombos)) {
            const QString previous = combo != nullptr
                ? combo->property("legendPreviousFontFamily").toString()
                : QString();
            populateFontFamilyCombo(combo, combo == importingCombo ? importedFamily : previous);
        }
    };

    auto makeFontControls = [&](const LegendFontOptions& font, const QString& labelText, const QString& searchText) {
        FontControls controls;
        auto* rowLayout = makeRowWidget(labelText, searchText);
        auto* controlLayout = makeCompactControlGroup(rowLayout);
        controlLayout->setContentsMargins(0, 0, kLegendFontRowRightPadding, 0);
        controls.family = new QComboBox(content);
        controls.family->setMinimumContentsLength(0);
        controls.family->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
        controls.family->setProperty("visualizationComboBox", true);
        configureControl(controls.family, comboControlHeight);
        fontFamilyCombos.push_back(controls.family);
        populateFontFamilyCombo(controls.family, canonicalLegendFontFamily(font.family));
        connect(controls.family, &QComboBox::currentTextChanged, &dialog, [&, combo = controls.family](const QString& text) {
            if (text == QObject::tr(kLegendImportFontLabel)) {
                const QString previousFamily = combo->property("legendPreviousFontFamily").toString();
                const QString startDirectory = !gLegendProjectFontDirectory.isEmpty()
                    ? gLegendProjectFontDirectory
                    : QStandardPaths::writableLocation(QStandardPaths::FontsLocation);
                const QString fontPath = QFileDialog::getOpenFileName(
                    &dialog,
                    tr("Import Font File"),
                    startDirectory,
                    tr("Font files (*.ttf *.otf *.ttc *.otc);;All files (*.*)"));
                QString importedFamily;
                if (!fontPath.isEmpty() && importLegendFontFile(fontPath, &dialog, &importedFamily)) {
                    refreshFontFamilyCombos(combo, importedFamily);
                } else {
                    populateFontFamilyCombo(combo, previousFamily);
                }
                return;
            }
            combo->setFont(QFont(legendQtFontFamilyName(text)));
            combo->setProperty("legendPreviousFontFamily", text);
        });
        controls.size = new QSpinBox(content);
        controls.size->setRange(1, 200);
        controls.size->setValue(std::clamp(font.size, 1, 200));
        controls.size->setButtonSymbols(QAbstractSpinBox::UpDownArrows);
        controls.size->setKeyboardTracking(false);
        configureControl(controls.size, inputControlHeight);
        controls.color = makeMiniColorButton(font.color, tr("Color"));
        controls.opacity = makeSpin(0.0, 1.0, std::clamp(font.opacity, 0.0, 1.0), 0.05, 2);
        controls.bold = makeStyleButton(QStringLiteral("B"), tr("Bold"), font.bold);
        controls.italic = makeStyleButton(QStringLiteral("I"), tr("Italic"), font.italic);
        controls.shadow = makeStyleButton(QStringLiteral("S"), tr("Shadow"), font.shadow);
        controlLayout->addWidget(controls.family, 1, Qt::AlignVCenter);
        controlLayout->addWidget(controls.size, 1, Qt::AlignVCenter);
        controlLayout->addWidget(controls.color, 0, Qt::AlignVCenter);
        controlLayout->addWidget(controls.opacity, 0, Qt::AlignVCenter);
        controlLayout->addWidget(controls.bold, 0, Qt::AlignVCenter);
        controlLayout->addWidget(controls.italic, 0, Qt::AlignVCenter);
        controlLayout->addWidget(controls.shadow, 0, Qt::AlignVCenter);
        return controls;
    };

    auto readFont = [&](const FontControls& controls) {
        LegendFontOptions font;
        const QString selectedFamily = controls.family->currentText();
        font.family = selectedFamily == QObject::tr(kLegendImportFontLabel)
            ? controls.family->property("legendPreviousFontFamily").toString()
            : selectedFamily;
        font.size = controls.size->value();
        font.color = colorButtonColor(controls.color);
        font.opacity = controls.opacity->value();
        font.bold = controls.bold->isChecked();
        font.italic = controls.italic->isChecked();
        font.shadow = controls.shadow->isChecked();
        return font;
    };

    auto writeFont = [&](const FontControls& controls, const LegendFontOptions& font) {
        populateFontFamilyCombo(controls.family, canonicalLegendFontFamily(font.family));
        controls.size->setValue(std::clamp(font.size, 1, 200));
        setColorButtonColor(controls.color, font.color);
        controls.opacity->setValue(std::clamp(font.opacity, 0.0, 1.0));
        controls.bold->setChecked(font.bold);
        controls.italic->setChecked(font.italic);
        controls.shadow->setChecked(font.shadow);
    };

    makeSection(tr("Legend layout"));
    auto* autoOrientCheck = makeCheckRow(tr("Auto orient"), initialOptions.autoOrient);
    auto* orientationCombo = makeCombo({QStringLiteral("Vertical"), QStringLiteral("Horizontal")},
                                       canonicalLegendOrientation(initialOptions.orientation));
    makeRowWidget(tr("Orientation"), tr("Orientation"), false, [autoOrientCheck]() {
        return !autoOrientCheck->isChecked();
    })->addWidget(orientationCombo, 1);
    auto* locationCombo = makeCombo({QStringLiteral("Any location"),
                                     QStringLiteral("Lower left"),
                                     QStringLiteral("Lower right"),
                                     QStringLiteral("Lower center"),
                                     QStringLiteral("Upper left"),
                                     QStringLiteral("Upper right"),
                                     QStringLiteral("Upper center")},
                                    legendWindowLocationDisplay(initialOptions.windowLocation));
    makeRowWidget(tr("Legend location"), tr("Legend location Window location"))->addWidget(locationCombo, 1);
    auto* positionRow = makeRowWidget(tr("Position"), tr("Position"));
    auto* positionXSpin = makeSpin(-10.0, 10.0, initialOptions.position[0], 0.01, 6);
    auto* positionYSpin = makeSpin(-10.0, 10.0, initialOptions.position[1], 0.01, 6);
    auto* positionControls = makeCompactControlGroup(positionRow);
    positionControls->addWidget(positionXSpin, 1, Qt::AlignVCenter);
    positionControls->addWidget(positionYSpin, 1, Qt::AlignVCenter);

    makeSection(tr("Title"));
    auto* titleEdit = makeLineEdit(initialOptions.title);
    makeRowWidget(tr("Title content"), tr("Title content"))->addWidget(titleEdit, 1);
    auto* componentTitleEdit = makeLineEdit(initialOptions.componentTitle);
    makeRowWidget(tr("Component title"), tr("Component title"))->addWidget(componentTitleEdit, 1);
    auto* componentFormatCombo = makeCombo(legendComponentFormatItems(),
                                           canonicalLegendComponentFormat(initialOptions.componentFormat));
    makeRowWidget(tr("Component format"),
                  tr("Component format"),
                  true,
                  [componentTitleEdit]() {
                      return !componentTitleEdit->text().trimmed().isEmpty();
                  })->addWidget(componentFormatCombo, 1);
    auto* justificationCombo = makeCombo({QStringLiteral("Centered"), QStringLiteral("Left"), QStringLiteral("Right")},
                                         canonicalLegendJustification(initialOptions.titleJustification));
    makeRowWidget(tr("Title justification"), tr("Title justification"))->addWidget(justificationCombo, 1);
    auto* titleOrientationCombo = makeCombo({QStringLiteral("Vertical"), QStringLiteral("Horizontal")},
                                            canonicalLegendTitleOrientation(initialOptions.titleOrientation));
    makeRowWidget(tr("Title orientation"), tr("Title orientation"))->addWidget(titleOrientationCombo, 1);
    auto* titlePositionCombo = makeCombo(legendTitlePositionItems(),
                                         canonicalLegendTitlePosition(initialOptions.titlePosition));
    makeRowWidget(tr("Title position"), tr("Title position"), true)->addWidget(titlePositionCombo, 1);
    auto* titlePaddingSpin = makeSpin(0.0, 512.0, initialOptions.titlePadding, 1.0, 0);
    makeRowWidget(tr("Title padding"), tr("Title padding"), true)->addWidget(titlePaddingSpin, 1);
    FontControls titleFontControls = makeFontControls(initialOptions.titleFont, tr("Title fonts"), tr("Title font"));

    makeSection(tr("Colorbar"));
    auto* thicknessSpin = makeSpin(1.0, 512.0, initialOptions.colorBarThickness, 1.0, 0);
    makeRowWidget(tr("Colorbar thickness"), tr("Colorbar thickness"))->addWidget(thicknessSpin, 1);
    auto* lengthSpin = makeSpin(0.01, 1.0, initialOptions.colorBarLength, 0.01, 3);
    makeRowWidget(tr("Colorbar length"), tr("Colorbar length"))->addWidget(lengthSpin, 1);
    auto* addRangeAnnotationsCheck = makeCheckRow(tr("Draw range annotations"), initialOptions.addRangeAnnotations, true);
    auto* rangeColorRow = makeRowWidget(QString(), tr("Below color Above color"), true, [addRangeAnnotationsCheck]() {
        return addRangeAnnotationsCheck->isChecked();
    });
    auto* rangeColorControls = makeCompactControlGroup(rangeColorRow);
    auto* belowRangeColorButton = makeColorButton(initialOptions.belowRangeColor, tr("Below color"));
    auto* aboveRangeColorButton = makeColorButton(initialOptions.aboveRangeColor, tr("Above color"));
    rangeColorControls->addWidget(belowRangeColorButton, 1, Qt::AlignVCenter);
    rangeColorControls->addWidget(aboveRangeColorButton, 1, Qt::AlignVCenter);
    auto* drawNanAnnotationCheck = makeCheckRow(tr("Draw NaN annotations"), initialOptions.drawNanAnnotation, true);
    auto* nanAnnotationEdit = makeLineEdit(initialOptions.nanAnnotation);
    makeRowWidget(tr("NaN annotation"), tr("NaN annotation"), true, [drawNanAnnotationCheck]() {
        return drawNanAnnotationCheck->isChecked();
    })->addWidget(nanAnnotationEdit, 1);
    auto* nanColorButton = makeColorButton(initialOptions.nanColor, tr("NaN color"));
    makeRowWidget(QString(), tr("NaN color"), true, [drawNanAnnotationCheck]() {
        return drawNanAnnotationCheck->isChecked();
    })->addWidget(nanColorButton, 1);
    auto* reverseLegendCheck = makeCheckRow(tr("Reverse colorbar"), initialOptions.reverseLegend, true);

    makeSection(tr("Ticks and labels"));
    auto* drawTickMarksCheck = makeCheckRow(tr("Draw ticks"), initialOptions.drawTickMarks);
    auto* drawTickLabelsCheck = makeCheckRow(tr("Draw tick labels"), initialOptions.drawTickLabels);
    auto* labelCountSpin = makeSpin(0.0, 64.0, initialOptions.labelCount, 1.0, 0);
    labelCountSpin->setButtonSymbols(QAbstractSpinBox::UpDownArrows);
    makeRowWidget(tr("Tick label number"), tr("Tick label number Number of tick labels"))->addWidget(labelCountSpin, 1);
    auto* tickFormatRow = makeRowWidget(tr("Tick format"),
                                        tr("Tick format Tick direction Tick color Tick length"),
                                        false,
                                        [drawTickMarksCheck]() {
                                            return drawTickMarksCheck->isChecked();
                                        });
    auto* tickFormatControls = makeCompactControlGroup(tickFormatRow);
    auto* tickDirectionCombo = makeCombo({QStringLiteral("Outward"),
                                          QStringLiteral("Centered"),
                                          QStringLiteral("Inward"),
                                          QStringLiteral("Through")},
                                         canonicalLegendTickDirection(initialOptions.tickDirection));
    auto* tickColorButton = makeMiniColorButton(initialOptions.tickColor, tr("Tick color"));
    auto* tickLengthSpin = makeSpin(0.0,
                                    128.0,
                                    std::max(0, initialOptions.tickLength),
                                    1.0,
                                    0);
    tickLengthSpin->setButtonSymbols(QAbstractSpinBox::UpDownArrows);
    tickFormatControls->addWidget(tickDirectionCombo, 1, Qt::AlignVCenter);
    tickFormatControls->addWidget(tickLengthSpin, 1, Qt::AlignVCenter);
    tickFormatControls->addWidget(tickColorButton, 0, Qt::AlignRight | Qt::AlignVCenter);
    auto* tickLabelsPaddingSpin = makeSpin(0.0, 64.0, initialOptions.tickLabelsPadding, 1.0, 0);
    makeRowWidget(tr("Labels padding"), tr("Labels padding Tick labels padding"), true, [drawTickLabelsCheck]() {
        return drawTickLabelsCheck->isChecked();
    })->addWidget(tickLabelsPaddingSpin, 1);
    auto* autoLabelFormatCheck = makeCheckRow(tr("Automatic label format"), initialOptions.automaticLabelFormat, true, [drawTickLabelsCheck]() {
        return drawTickLabelsCheck->isChecked();
    });
    auto* addRangeLabelsCheck = makeCheckRow(tr("Add range labels"), initialOptions.addRangeLabels, true, [drawTickMarksCheck, drawTickLabelsCheck]() {
        return drawTickMarksCheck->isChecked() || drawTickLabelsCheck->isChecked();
    });
    auto* labelFormatEdit = makeLineEdit(initialOptions.labelFormat);
    makeRowWidget(tr("Label format"), tr("Label format"), true, [drawTickLabelsCheck, autoLabelFormatCheck]() {
        return drawTickLabelsCheck->isChecked() && !autoLabelFormatCheck->isChecked();
    })->addWidget(labelFormatEdit, 1);
    auto* rangeLabelFormatEdit = makeLineEdit(initialOptions.rangeLabelFormat);
    makeRowWidget(tr("Range label format"), tr("Range label format"), true, [drawTickMarksCheck, addRangeLabelsCheck, autoLabelFormatCheck]() {
        return drawTickMarksCheck->isChecked() && addRangeLabelsCheck->isChecked() && !autoLabelFormatCheck->isChecked();
    })->addWidget(rangeLabelFormatEdit, 1);
    auto* tickPositionCombo = makeCombo(legendTickPositionItems(initialOptions.orientation),
                                        legendTickPositionDisplay(initialOptions.tickAnnotationPosition,
                                                                  initialOptions.orientation));
    makeRowWidget(tr("Tick position"), tr("Tick position Tick and annotation positions"))->addWidget(tickPositionCombo, 1);
    FontControls textFontControls = makeFontControls(initialOptions.textFont, tr("Label fonts"), tr("Label font Text annotation font"));

    makeSection(tr("Background and outline"), true);
    auto* drawBackgroundCheck = makeCheckRow(tr("Draw background"), initialOptions.drawBackground, true);
    auto* backgroundColorButton = makeColorButton(initialOptions.backgroundColor, tr("Background color"));
    makeRowWidget(QString(), tr("Background color"), true, [drawBackgroundCheck]() {
        return drawBackgroundCheck->isChecked();
    })->addWidget(backgroundColorButton, 1);
    auto* drawOutlineCheck = makeCheckRow(tr("Draw outline"), initialOptions.drawScalarBarOutline, true);
    auto* backgroundPaddingSpin = makeSpin(0.0, 64.0, initialOptions.backgroundPadding, 1.0, 0);
    makeRowWidget(tr("Background padding"), tr("Background padding"), true, [drawBackgroundCheck, drawOutlineCheck]() {
        return drawBackgroundCheck->isChecked() || drawOutlineCheck->isChecked();
    })->addWidget(backgroundPaddingSpin, 1);
    auto* outlineColorButton = makeColorButton(initialOptions.scalarBarOutlineColor, tr("Outline color"));
    makeRowWidget(QString(), tr("Outline color"), true, [drawOutlineCheck]() {
        return drawOutlineCheck->isChecked();
    })->addWidget(outlineColorButton, 1);
    auto* outlineThicknessSpin = makeSpin(0.0, 64.0, initialOptions.scalarBarOutlineThickness, 0.5, 1);
    makeRowWidget(tr("Outline thickness"), tr("Outline thickness"), true, [drawOutlineCheck]() {
        return drawOutlineCheck->isChecked();
    })->addWidget(outlineThicknessSpin, 1);
    contentLayout->addStretch(1);

    const LegendOptions originalOptions = initialOptions;
    LegendOptions dialogOptions = initialOptions;
    bool controlsUpdating = false;
    auto populateTickPositionCombo = [&](const QString& storedPosition, const QString& orientation) {
        const QSignalBlocker blocker(tickPositionCombo);
        tickPositionCombo->clear();
        tickPositionCombo->addItems(legendTickPositionItems(orientation));
        tickPositionCombo->setCurrentText(legendTickPositionDisplay(storedPosition, orientation));
    };
    auto writeOptionsToControls = [&](const LegendOptions& options) {
        controlsUpdating = true;
        dialogOptions = options;
        autoOrientCheck->setChecked(options.autoOrient);
        orientationCombo->setCurrentText(canonicalLegendOrientation(options.orientation));
        locationCombo->setCurrentText(legendWindowLocationDisplay(options.windowLocation));
        positionXSpin->setValue(options.position[0]);
        positionYSpin->setValue(options.position[1]);
        titleEdit->setText(options.title);
        componentTitleEdit->setText(options.componentTitle);
        componentFormatCombo->setCurrentText(canonicalLegendComponentFormat(options.componentFormat));
        justificationCombo->setCurrentText(canonicalLegendJustification(options.titleJustification));
        titleOrientationCombo->setCurrentText(canonicalLegendTitleOrientation(options.titleOrientation));
        titlePositionCombo->setCurrentText(canonicalLegendTitlePosition(options.titlePosition));
        titlePaddingSpin->setValue(std::clamp(options.titlePadding, 0, 512));
        writeFont(titleFontControls, options.titleFont);
        writeFont(textFontControls, options.textFont);
        thicknessSpin->setValue(options.colorBarThickness);
        lengthSpin->setValue(options.colorBarLength);
        addRangeAnnotationsCheck->setChecked(options.addRangeAnnotations);
        setColorButtonColor(belowRangeColorButton, options.belowRangeColor);
        setColorButtonColor(aboveRangeColorButton, options.aboveRangeColor);
        drawNanAnnotationCheck->setChecked(options.drawNanAnnotation);
        nanAnnotationEdit->setText(options.nanAnnotation);
        setColorButtonColor(nanColorButton, options.nanColor);
        reverseLegendCheck->setChecked(options.reverseLegend);
        drawTickMarksCheck->setChecked(options.drawTickMarks);
        labelCountSpin->setValue(options.labelCount);
        drawTickLabelsCheck->setChecked(options.drawTickLabels);
        tickLabelsPaddingSpin->setValue(options.tickLabelsPadding);
        tickDirectionCombo->setCurrentText(canonicalLegendTickDirection(options.tickDirection));
        setColorButtonColor(tickColorButton, options.tickColor);
        tickLengthSpin->setValue(std::clamp(options.tickLength, 0, 128));
        autoLabelFormatCheck->setChecked(options.automaticLabelFormat);
        addRangeLabelsCheck->setChecked(options.addRangeLabels);
        labelFormatEdit->setText(options.labelFormat);
        rangeLabelFormatEdit->setText(options.rangeLabelFormat);
        populateTickPositionCombo(options.tickAnnotationPosition, options.orientation);
        drawBackgroundCheck->setChecked(options.drawBackground);
        setColorButtonColor(backgroundColorButton, options.backgroundColor);
        backgroundPaddingSpin->setValue(options.backgroundPadding);
        drawOutlineCheck->setChecked(options.drawScalarBarOutline);
        setColorButtonColor(outlineColorButton, options.scalarBarOutlineColor);
        outlineThicknessSpin->setValue(options.scalarBarOutlineThickness);
        controlsUpdating = false;
    };

    auto readOptionsFromControls = [&]() {
        LegendOptions options = dialogOptions;
        options.autoOrient = autoOrientCheck->isChecked();
        options.orientation = orientationCombo->currentText();
        options.windowLocation = canonicalLegendWindowLocation(locationCombo->currentText());
        options.position[0] = positionXSpin->value();
        options.position[1] = positionYSpin->value();
        options.title = titleEdit->text();
        options.titleEdited = options.title.trimmed().compare(currentDefaultTitle, Qt::CaseSensitive) != 0;
        options.componentTitle = componentTitleEdit->text();
        options.componentFormat = componentFormatCombo->currentText();
        options.titleJustification = justificationCombo->currentText();
        options.titleOrientation = titleOrientationCombo->currentText();
        options.titlePosition = titlePositionCombo->currentText();
        options.titlePadding = static_cast<int>(std::round(titlePaddingSpin->value()));
        options.titleFont = readFont(titleFontControls);
        options.textFont = readFont(textFontControls);
        options.colorBarThickness = static_cast<int>(std::round(thicknessSpin->value()));
        options.colorBarLength = lengthSpin->value();
        options.addRangeAnnotations = addRangeAnnotationsCheck->isChecked();
        options.belowRangeColor = colorButtonColor(belowRangeColorButton);
        options.aboveRangeColor = colorButtonColor(aboveRangeColorButton);
        options.drawNanAnnotation = drawNanAnnotationCheck->isChecked();
        options.nanAnnotation = nanAnnotationEdit->text();
        options.nanColor = colorButtonColor(nanColorButton);
        options.reverseLegend = reverseLegendCheck->isChecked();
        options.drawTickMarks = drawTickMarksCheck->isChecked();
        options.labelCount = static_cast<int>(std::round(labelCountSpin->value()));
        options.drawTickLabels = drawTickLabelsCheck->isChecked();
        options.tickLabelsPadding = static_cast<int>(std::round(tickLabelsPaddingSpin->value()));
        options.tickDirection = tickDirectionCombo->currentText();
        options.tickColor = colorButtonColor(tickColorButton);
        options.tickLength = static_cast<int>(std::round(tickLengthSpin->value()));
        options.automaticLabelFormat = autoLabelFormatCheck->isChecked();
        options.addRangeLabels = addRangeLabelsCheck->isChecked();
        options.labelFormat = labelFormatEdit->text().trimmed();
        options.rangeLabelFormat = rangeLabelFormatEdit->text().trimmed();
        options.tickAnnotationPosition = legendTickPositionFromDisplay(tickPositionCombo->currentText());
        options.drawBackground = drawBackgroundCheck->isChecked();
        options.backgroundColor = colorButtonColor(backgroundColorButton);
        options.backgroundPadding = static_cast<int>(std::round(backgroundPaddingSpin->value()));
        options.drawScalarBarOutline = drawOutlineCheck->isChecked();
        options.scalarBarOutlineColor = colorButtonColor(outlineColorButton);
        options.scalarBarOutlineThickness = outlineThicknessSpin->value();
        return options;
    };

    auto writePositionControls = [&](const LegendOptions& options) {
        const QSignalBlocker blockerX(positionXSpin);
        const QSignalBlocker blockerY(positionYSpin);
        positionXSpin->setValue(options.position[0]);
        positionYSpin->setValue(options.position[1]);
    };

    auto syncAppliedPosition = [&](LegendOptions& options) {
        vtkCoordinate* positionCoordinate = nullptr;
        if (actorSet->scalarBarWidget != nullptr
            && actorSet->scalarBarWidget->GetScalarBarRepresentation() != nullptr) {
            positionCoordinate = actorSet->scalarBarWidget->GetScalarBarRepresentation()->GetPositionCoordinate();
        } else if (actorSet->scalarBar != nullptr) {
            positionCoordinate = actorSet->scalarBar->GetPositionCoordinate();
        }
        if (positionCoordinate == nullptr) {
            return;
        }
        const double* position = positionCoordinate->GetValue();
        options.position[0] = position[0];
        options.position[1] = position[1];
        writePositionControls(options);
    };

    auto updateRowVisibility = [&]() {
        const QString filter = searchEdit->text().trimmed().toLower();
        const bool showAdvanced = advancedButton->isChecked();
        for (const FilterItem& item : filterItems) {
            if (item.widget == nullptr) {
                continue;
            }
            const bool visibleByMode = !item.advancedOnly || showAdvanced;
            const bool visibleBySearch = filter.isEmpty() || item.searchText.toLower().contains(filter);
            const bool visibleByDependency = !item.dependency || item.dependency();
            item.widget->setVisible(visibleByMode && visibleBySearch && visibleByDependency);
        }
    };

    auto applyControls = [&]() {
        if (controlsUpdating) {
            return;
        }
        LegendOptions updated = readOptionsFromControls();
        updated.componentFormat = canonicalLegendComponentFormat(updated.componentFormat);
        updated.titlePosition = canonicalLegendTitlePosition(updated.titlePosition);
        updated.tickAnnotationPosition = canonicalLegendTickAnnotationPosition(updated.tickAnnotationPosition);
        dialogOptions = updated;
        applyScalarBarOptions(actorSet, updated);
        syncAppliedPosition(updated);
        dialogOptions = updated;
        auto it = displayObjectOptions_.find(actorSet->scalarBarObjectId);
        if (it != displayObjectOptions_.end()) {
            it->legend = updated;
        }
        emit legendStyleChanged(actorSet->scalarBarObjectId, updated);
    };

    auto applyOriginalOptions = [&]() {
        controlsUpdating = true;
        applyScalarBarOptions(actorSet, originalOptions);
        auto it = displayObjectOptions_.find(actorSet->scalarBarObjectId);
        if (it != displayObjectOptions_.end()) {
            it->legend = originalOptions;
        }
        emit legendStyleChanged(actorSet->scalarBarObjectId, originalOptions);
        controlsUpdating = false;
    };

    auto refreshAndApply = [&]() {
        updateRowVisibility();
        applyControls();
    };

    auto connectCheck = [&](QCheckBox* check) {
        connect(check, &QCheckBox::toggled, &dialog, refreshAndApply);
    };
    auto connectCombo = [&](QComboBox* combo) {
        connect(combo, &QComboBox::currentTextChanged, &dialog, applyControls);
    };
    auto connectLine = [&](QLineEdit* edit) {
        connect(edit, &QLineEdit::editingFinished, &dialog, applyControls);
    };
    auto connectDoubleSpin = [&](QDoubleSpinBox* spin) {
        connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), &dialog, applyControls);
    };
    auto connectSpin = [&](QSpinBox* spin) {
        connect(spin, qOverload<int>(&QSpinBox::valueChanged), &dialog, applyControls);
    };
    auto connectButton = [&](QAbstractButton* button) {
        connect(button, &QAbstractButton::clicked, &dialog, applyControls);
    };
    auto connectFont = [&](const FontControls& controls) {
        connectCombo(controls.family);
        connectSpin(controls.size);
        connectButton(controls.color);
        connectDoubleSpin(controls.opacity);
        connectButton(controls.bold);
        connectButton(controls.italic);
        connectButton(controls.shadow);
    };

    connectCheck(autoOrientCheck);
    connectCheck(addRangeAnnotationsCheck);
    connectButton(belowRangeColorButton);
    connectButton(aboveRangeColorButton);
    connectCheck(drawNanAnnotationCheck);
    connectLine(nanAnnotationEdit);
    connectButton(nanColorButton);
    connectCheck(reverseLegendCheck);
    connectCheck(drawTickMarksCheck);
    connectDoubleSpin(labelCountSpin);
    connectCheck(drawTickLabelsCheck);
    connectDoubleSpin(tickLabelsPaddingSpin);
    connectCombo(tickDirectionCombo);
    connectButton(tickColorButton);
    connectDoubleSpin(tickLengthSpin);
    connectCheck(autoLabelFormatCheck);
    connectCheck(addRangeLabelsCheck);
    connectLine(labelFormatEdit);
    connectLine(rangeLabelFormatEdit);
    connectCombo(tickPositionCombo);
    connectCheck(drawBackgroundCheck);
    connectButton(backgroundColorButton);
    connectDoubleSpin(backgroundPaddingSpin);
    connectCheck(drawOutlineCheck);
    connectButton(outlineColorButton);
    connectDoubleSpin(outlineThicknessSpin);
    connect(orientationCombo, &QComboBox::currentTextChanged, &dialog, [&](const QString& text) {
        if (!controlsUpdating) {
            const QString storedTickPosition = legendTickPositionFromDisplay(tickPositionCombo->currentText());
            populateTickPositionCombo(storedTickPosition, text);
        }
        applyControls();
    });
    connectCombo(locationCombo);
    auto connectPositionSpin = [&](QDoubleSpinBox* spin) {
        connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), &dialog, [&]() {
            if (!controlsUpdating) {
                const QSignalBlocker blocker(locationCombo);
                locationCombo->setCurrentText(legendWindowLocationDisplay(QStringLiteral("Any Location")));
            }
            applyControls();
        });
    };
    connectPositionSpin(positionXSpin);
    connectPositionSpin(positionYSpin);
    connectLine(titleEdit);
    connect(componentTitleEdit, &QLineEdit::editingFinished, &dialog, refreshAndApply);
    connectCombo(componentFormatCombo);
    connectCombo(justificationCombo);
    connectCombo(titleOrientationCombo);
    connectCombo(titlePositionCombo);
    connectDoubleSpin(titlePaddingSpin);
    connectFont(titleFontControls);
    connectFont(textFontControls);

    connect(advancedButton, &QToolButton::toggled, &dialog, updateRowVisibility);
    connect(searchEdit, &QLineEdit::textChanged, &dialog, updateRowVisibility);

    writeOptionsToControls(initialOptions);
    updateRowVisibility();

    auto* buttons = new QDialogButtonBox(Qt::Horizontal, &dialog);
    auto* resetButton = buttons->addButton(tr("Reset"), QDialogButtonBox::ResetRole);
    auto* cancelButton = buttons->addButton(QDialogButtonBox::Cancel);
    auto* okButton = buttons->addButton(QDialogButtonBox::Ok);
    Q_UNUSED(cancelButton);
    for (QPushButton* button : buttons->findChildren<QPushButton*>()) {
        button->setAutoDefault(false);
        button->setDefault(false);
    }
    rootLayout->addSpacing(kLegendFooterTopPadding);
    rootLayout->addWidget(buttons);

    connect(resetButton, &QPushButton::clicked, &dialog, [&]() {
        LegendOptions resetOptions;
        resetOptions.title = currentDefaultTitle;
        resetOptions.titleEdited = false;
        resetOptions.titlePadding = defaultLegendTitlePaddingPixels(resetOptions, actorSet->scalarBar->GetLookupTable());
        writeOptionsToControls(resetOptions);
        updateRowVisibility();
        applyControls();
    });
    connect(okButton, &QPushButton::clicked, &dialog, [&]() {
        applyControls();
        dialog.accept();
    });
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

    const int result = dialog.exec();
    if (result != QDialog::Accepted) {
        applyOriginalOptions();
    }
    return true;
}

bool ViewerWidget::probeScalarFields(const QString& path,
                                     QStringList* fields,
                                     QString* preferredField,
                                     QString* errorMessage) const {
    QVector<FieldOption> options;
    QString preferredComponent;
    if (!probeFieldOptions(path, &options, preferredField, &preferredComponent, errorMessage)) {
        return false;
    }
    if (fields != nullptr) {
        fields->clear();
        for (const FieldOption& option : options) {
            fields->push_back(option.name);
        }
    }
    return true;
}

bool ViewerWidget::probeFieldOptions(const QString& path,
                                     QVector<FieldOption>* fields,
                                     QString* preferredField,
                                     QString* preferredComponent,
                                     QString* errorMessage) const {
    vtkSmartPointer<vtkDataSet> data;
    if (!loadDataSetFromPath(path, &data, errorMessage)) {
        return false;
    }
    const QVector<FieldOption> options = fieldOptions(data, preferredField, preferredComponent);
    if (fields != nullptr) {
        *fields = options;
    }
    return true;
}

bool ViewerWidget::dataObjectScalarRange(const QString& objectId,
                                         const DataObjectOptions& options,
                                         double* minValue,
                                         double* maxValue,
                                         QString* errorMessage) const {
    if (minValue == nullptr || maxValue == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Internal error: scalar range output pointer is null.");
        }
        return false;
    }
    if (options.sourceObjectId.trimmed().isEmpty() && options.inputPath.trimmed().isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Please select an input VTK file before applying Auto range.");
        }
        return false;
    }

    vtkSmartPointer<vtkDataSet> data;
    if (!options.sourceObjectId.trimmed().isEmpty()) {
        MaterializedDataObject materialized;
        QSet<QString> visiting;
        if (!materializeDataObject(objectId, options, &materialized, &visiting, errorMessage)) {
            return false;
        }
        data = materialized.outputData;
    } else {
        const QString absoluteInputPath = QFileInfo(options.inputPath).absoluteFilePath();
        const QString sourceSuffix = QFileInfo(absoluteInputPath).suffix().toLower();
        QString loadPath = absoluteInputPath;
        if (sourceSuffix == QStringLiteral("pvd")) {
            const QVector<FrameInfo> parsed = parsePvdFile(absoluteInputPath, errorMessage);
            if (parsed.isEmpty()) {
                return false;
            }
            loadPath = parsed.front().path;
            if (parsed.size() > 1 && isTimeSeries_ && currentFrameIndex_ >= 0 && currentFrameIndex_ < frames_.size()) {
                const int frameIndex = nearestFrameIndexForTime(parsed, frames_.at(currentFrameIndex_).timestep);
                if (frameIndex >= 0) {
                    loadPath = parsed.at(frameIndex).path;
                }
            } else if (!objectId.trimmed().isEmpty()) {
                loadPath = displayObjectFramePathForCurrentTime(objectId, loadPath);
            }
        }

        if (!loadDataSetFromPath(loadPath, &data, errorMessage)) {
            return false;
        }
        prepareVectorFields(data);
        if (options.type == DisplayObjectType::Crop) {
            vtkSmartPointer<vtkDataSet> croppedData = cropDataSetToBounds(data, options.cropBounds, errorMessage);
            if (croppedData == nullptr) {
                return false;
            }
            data = croppedData;
            prepareVectorFields(data);
        }
    }

    QString fieldName = options.colorField.trimmed();
    QString componentName = options.colorComponent;
    if (fieldName.isEmpty() && options.type == DisplayObjectType::RayTracingVolume) {
        if (auto* imageData = vtkImageData::SafeDownCast(data)) {
            fieldName = firstPointFieldName(imageData);
            componentName = QStringLiteral("Magnitude");
        }
    }
    const ResolvedScalarField scalar = resolveScalarField(data, fieldName, componentName);
    if (scalar.array == nullptr || scalar.arrayName.trimmed().isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Scalar field was not found: %1").arg(fieldName);
        }
        return false;
    }
    if (options.type == DisplayObjectType::RayTracingVolume && scalar.association != kFieldAssociationPoints) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Ray-tracing volume supports only point data arrays; selected field is not point data: %1")
                                .arg(fieldName);
        }
        return false;
    }

    double range[2]{};
    scalar.array->GetRange(range);
    *minValue = range[0];
    *maxValue = range[1];
    return true;
}

bool ViewerWidget::dataObjectBounds(const QString& objectId,
                                    const DataObjectOptions& options,
                                    double bounds[6],
                                    QString* errorMessage) const {
    if (bounds == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Internal error: bounds output pointer is null.");
        }
        return false;
    }
    if (options.sourceObjectId.trimmed().isEmpty() && options.inputPath.trimmed().isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Please select an input VTK file before reading object bounds.");
        }
        return false;
    }

    MaterializedDataObject materialized;
    QSet<QString> visiting;
    if (!materializeDataObject(objectId, options, &materialized, &visiting, errorMessage)) {
        return false;
    }
    if (materialized.outputData == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Visualization object produced no VTK dataset.");
        }
        return false;
    }

    materialized.outputData->GetBounds(bounds);
    for (int index = 0; index < 6; ++index) {
        if (!std::isfinite(bounds[index])) {
            if (errorMessage != nullptr) {
                *errorMessage = QStringLiteral("Visualization object produced non-finite bounds.");
            }
            return false;
        }
    }
    if (bounds[0] > bounds[1] || bounds[2] > bounds[3] || bounds[4] > bounds[5]) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Visualization object produced invalid bounds.");
        }
        return false;
    }
    return true;
}

QVector<double> ViewerWidget::cameraState() const {
    QVector<double> values;
    if (renderer_ == nullptr || renderer_->GetActiveCamera() == nullptr) {
        return values;
    }
    const CameraState state = captureCamera(renderer_);
    values.reserve(11);
    for (double value : state.position) {
        values.push_back(value);
    }
    for (double value : state.focalPoint) {
        values.push_back(value);
    }
    for (double value : state.viewUp) {
        values.push_back(value);
    }
    values.push_back(state.parallelScale);
    values.push_back(state.viewAngle);
    return values;
}

void ViewerWidget::restoreCameraState(const QVector<double>& values) {
    if (renderer_ == nullptr || renderer_->GetActiveCamera() == nullptr || values.size() < 10) {
        return;
    }
    CameraState state;
    for (int i = 0; i < 3; ++i) {
        state.position[i] = values[i];
        state.focalPoint[i] = values[i + 3];
        state.viewUp[i] = values[i + 6];
    }
    state.parallelScale = values[9];
    state.viewAngle = values.size() > 10 ? values[10] : renderer_->GetActiveCamera()->GetViewAngle();
    double clipping[2]{};
    renderer_->GetActiveCamera()->GetClippingRange(clipping);
    state.clippingRange[0] = clipping[0];
    state.clippingRange[1] = clipping[1];
    restoreCamera(renderer_, state, !displayOptions_.perspectiveEnabled);
    if (renderWindow_ != nullptr) {
        renderWindow_->Render();
    }
    scheduleRenderWhenVisible();
    emit cameraChanged(cameraState());
}

void ViewerWidget::resetCameraToScene() {
    if (renderer_ == nullptr) {
        return;
    }

    renderer_->ResetCamera();
    renderer_->ResetCameraClippingRange();
    if (vtkCamera* camera = renderer_->GetActiveCamera()) {
        camera->SetParallelProjection(parallelProjection_ ? 1 : 0);
        if (!parallelProjection_) {
            camera->SetViewAngle(std::clamp(displayOptions_.perspectiveDepth, 5.0, 120.0));
        }
    }
    if (renderWindow_ != nullptr) {
        renderWindow_->Render();
    }
    scheduleRenderWhenVisible();
    emit cameraChanged(cameraState());
}

void ViewerWidget::setCameraView(CameraView view) {
    if (!hasCameraScene() || renderer_->GetActiveCamera() == nullptr) {
        return;
    }

    double direction[3]{};
    double viewUp[3]{};
    cameraViewVectors(view, direction, viewUp);

    vtkCamera* camera = renderer_->GetActiveCamera();
    double bounds[6]{};
    renderer_->ComputeVisiblePropBounds(bounds);
    const bool hasBounds = usableCameraBounds(bounds);

    double center[3]{};
    double radius = 1.0;
    if (hasBounds) {
        center[0] = (bounds[0] + bounds[1]) * 0.5;
        center[1] = (bounds[2] + bounds[3]) * 0.5;
        center[2] = (bounds[4] + bounds[5]) * 0.5;
        const double dx = bounds[1] - bounds[0];
        const double dy = bounds[3] - bounds[2];
        const double dz = bounds[5] - bounds[4];
        radius = std::max(std::sqrt(dx * dx + dy * dy + dz * dz) * 0.5, 1.0);
    } else {
        camera->GetFocalPoint(center);
        radius = std::max(camera->GetDistance(), 1.0);
    }

    const double distance = std::max(radius * 2.5, 1.0);
    camera->SetFocalPoint(center);
    camera->SetPosition(center[0] + direction[0] * distance,
                        center[1] + direction[1] * distance,
                        center[2] + direction[2] * distance);
    camera->SetViewUp(viewUp);
    camera->OrthogonalizeViewUp();
    camera->SetParallelProjection(parallelProjection_ ? 1 : 0);
    if (!parallelProjection_) {
        camera->SetViewAngle(std::clamp(displayOptions_.perspectiveDepth, 5.0, 120.0));
    }

    if (hasBounds) {
        double directionOfProjection[3]{};
        double currentViewUp[3]{};
        camera->GetDirectionOfProjection(directionOfProjection);
        camera->GetViewUp(currentViewUp);
        normalizeVector(directionOfProjection);
        normalizeVector(currentViewUp);
        double right[3]{};
        crossVector(directionOfProjection, currentViewUp, right);
        if (normalizeVector(right) <= std::numeric_limits<double>::epsilon()) {
            right[0] = 1.0;
            right[1] = 0.0;
            right[2] = 0.0;
        }

        double halfWidth = 0.0;
        double halfHeight = 0.0;
        double halfDepth = 0.0;
        cameraBoundsProjectionExtents(bounds,
                                      center,
                                      right,
                                      currentViewUp,
                                      directionOfProjection,
                                      &halfWidth,
                                      &halfHeight,
                                      &halfDepth);

        double aspect = 1.0;
        if (vtkWidget_ != nullptr && vtkWidget_->width() > 0 && vtkWidget_->height() > 0) {
            aspect = static_cast<double>(vtkWidget_->width()) / static_cast<double>(vtkWidget_->height());
        } else if (renderWindow_ != nullptr) {
            const int* renderSize = renderWindow_->GetSize();
            if (renderSize != nullptr && renderSize[0] > 0 && renderSize[1] > 0) {
                aspect = static_cast<double>(renderSize[0]) / static_cast<double>(renderSize[1]);
            }
        }
        aspect = std::max(aspect, 0.01);

        constexpr double kViewFitMargin = 1.035;
        const double viewHalfExtent = std::max({halfHeight, halfWidth / aspect, 0.001});
        if (parallelProjection_) {
            camera->SetParallelScale(viewHalfExtent * kViewFitMargin);
            const double clippedDistance = std::max(radius * 2.5, halfDepth + 1.0);
            camera->SetPosition(center[0] - directionOfProjection[0] * clippedDistance,
                                center[1] - directionOfProjection[1] * clippedDistance,
                                center[2] - directionOfProjection[2] * clippedDistance);
            camera->SetFocalPoint(center);
        } else {
            constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;
            const double viewAngle = std::clamp(camera->GetViewAngle(), 5.0, 120.0);
            const double halfAngleRadians = viewAngle * 0.5 * kDegreesToRadians;
            const double perspectiveDistance = halfDepth
                + (viewHalfExtent * kViewFitMargin) / std::max(std::tan(halfAngleRadians), 0.001);
            camera->SetPosition(center[0] - directionOfProjection[0] * perspectiveDistance,
                                center[1] - directionOfProjection[1] * perspectiveDistance,
                                center[2] - directionOfProjection[2] * perspectiveDistance);
            camera->SetFocalPoint(center);
        }
        renderer_->ResetCameraClippingRange(bounds);
    } else {
        renderer_->ResetCameraClippingRange();
    }

    if (renderWindow_ != nullptr) {
        renderWindow_->Render();
    }
    scheduleRenderWhenVisible();
    emit cameraChanged(cameraState());
}

bool ViewerWidget::hasScene() const {
    return currentDataSet_ != nullptr || customDisplayActive_;
}

bool ViewerWidget::hasCameraScene() const {
    return hasScene()
        && renderer_ != nullptr
        && renderer_->GetActiveCamera() != nullptr
        && vtkWidget_ != nullptr
        && stackedLayout_ != nullptr
        && stackedLayout_->currentWidget() == vtkWidget_;
}

bool ViewerWidget::isTimeSeries() const {
    return isTimeSeries_;
}

QString ViewerWidget::currentPath() const {
    return currentPath_;
}

void ViewerWidget::setProjectVisualizationDirectory(const QString& directory) {
    const QString normalized = normalizedLegendPath(directory);
    if (projectVisualizationDirectory_ == normalized
        && gLegendProjectVisualizationDirectory == normalized) {
        return;
    }
    projectVisualizationDirectory_ = normalized;
    Streamcenter::setActiveColorMapsFile(normalized.trimmed().isEmpty()
                                             ? QString()
                                             : Streamcenter::projectColorMapsFilePath(normalized));
    scanLegendProjectFonts(projectVisualizationDirectory_);
}

void ViewerWidget::setParallelProjection(bool enabled) {
    parallelProjection_ = enabled;
    if (renderer_ != nullptr && renderer_->GetActiveCamera() != nullptr) {
        renderer_->GetActiveCamera()->SetParallelProjection(enabled ? 1 : 0);
        renderWindow_->Render();
    }
}

void ViewerWidget::setTransparent(bool enabled) {
    transparent_ = enabled;
    const double opacity = transparent_ ? kTransparentOpacity : kOpaqueOpacity;
    for (ScalarVisual& visual : scalarVisuals_) {
        if (visual.actor != nullptr) {
            visual.actor->GetProperty()->SetOpacity(opacity);
        }
    }
    renderWindow_->Render();
}

bool ViewerWidget::transparent() const {
    return transparent_;
}

void ViewerWidget::setScalarMode(ScalarMode mode) {
    scalarMode_ = mode;
    if (hasScene()) {
        rebuildScalarVisuals(false);
        renderWindow_->Render();
    }
}

ViewerWidget::ScalarMode ViewerWidget::scalarMode() const {
    return scalarMode_;
}

bool ViewerWidget::scalarRange(const QString& scalarName,
                               double* minValue,
                               double* maxValue,
                               double* currentValue) const {
    int association = kFieldAssociationPoints;
    vtkDataArray* array = findScalarArray(scalarName, &association);
    if (array == nullptr) {
        return false;
    }

    double range[2]{};
    array->GetRange(range);
    const double defaultValue = range[0] + (0.75 * (range[1] - range[0]));
    const double targetValue = isoTargets_.contains(scalarName)
        ? isoTargets_.value(scalarName)
        : defaultValue;
    const double clampedValue = std::clamp(targetValue, range[0], range[1]);

    if (minValue != nullptr) {
        *minValue = range[0];
    }
    if (maxValue != nullptr) {
        *maxValue = range[1];
    }
    if (currentValue != nullptr) {
        *currentValue = clampedValue;
    }
    return true;
}

void ViewerWidget::setIsoValue(const QString& scalarName, double value) {
    double minValue = 0.0;
    double maxValue = 0.0;
    if (!scalarRange(scalarName, &minValue, &maxValue, nullptr)) {
        return;
    }

    isoTargets_[scalarName] = std::clamp(value, minValue, maxValue);
    if (!hasScene()) {
        return;
    }

    const CameraState camera = captureCamera(renderer_);
    rebuildScalarVisuals(false);
    restoreCamera(renderer_, camera, parallelProjection_);
    renderWindow_->Render();
}

void ViewerWidget::setInitialTimeCodeHint(double timeCode) {
    initialTimeCodeHint_ = std::isfinite(timeCode) ? timeCode : std::numeric_limits<double>::quiet_NaN();
}

void ViewerWidget::startAnimation() {
    if (!isTimeSeries_ || frames_.size() <= 1 || animationPlaying_) {
        return;
    }
    animationPlaying_ = true;
    animationTimer_->start(100);
    emit animationStateChanged(true);
}

void ViewerWidget::stopAnimation() {
    animationTimer_->stop();
    if (!animationPlaying_) {
        return;
    }
    animationPlaying_ = false;
    emit animationStateChanged(false);
}

void ViewerWidget::toggleAnimation() {
    if (animationPlaying_) {
        stopAnimation();
    } else {
        startAnimation();
    }
}

bool ViewerWidget::animationPlaying() const {
    return animationPlaying_;
}

void ViewerWidget::jumpToFirstFrame() {
    if (!isTimeSeries_ || frames_.isEmpty()) {
        return;
    }
    stopAnimation();
    QString error;
    applyFrameIndex(0, true, &error);
}

void ViewerWidget::jumpToPreviousFrame() {
    if (!isTimeSeries_ || frames_.isEmpty()) {
        return;
    }
    stopAnimation();
    QString error;
    applyFrameIndex(currentFrameIndex_ - 1, true, &error);
}

void ViewerWidget::jumpToNextFrame() {
    if (!isTimeSeries_ || frames_.isEmpty()) {
        return;
    }
    stopAnimation();
    QString error;
    applyFrameIndex(currentFrameIndex_ + 1, true, &error);
}

void ViewerWidget::jumpToLastFrame() {
    if (!isTimeSeries_ || frames_.isEmpty()) {
        return;
    }
    stopAnimation();
    QString error;
    applyFrameIndex(frames_.size() - 1, true, &error);
}

bool ViewerWidget::setFrameIndex(int index, QString* errorMessage) {
    if (!isTimeSeries_ || frames_.isEmpty()) {
        return false;
    }
    stopAnimation();
    return applyFrameIndex(index, true, errorMessage);
}

int ViewerWidget::frameCount() const {
    return isTimeSeries_ ? frames_.size() : 0;
}

int ViewerWidget::frameIndex() const {
    return currentFrameIndex_;
}

double ViewerWidget::frameTimeCode(int index) const {
    if (!isTimeSeries_ || index < 0 || index >= frames_.size()) {
        return 0.0;
    }
    return frames_.at(index).timestep;
}

QString ViewerWidget::frameTimeCodeText(int index) const {
    if (!isTimeSeries_ || index < 0 || index >= frames_.size()) {
        return {};
    }
    return formatTimeCode(frames_.at(index).timestep);
}

QStringList ViewerWidget::frameTimeCodeTexts() const {
    QStringList labels;
    if (!isTimeSeries_) {
        return labels;
    }
    labels.reserve(frames_.size());
    for (const FrameInfo& frame : frames_) {
        labels << formatTimeCode(frame.timestep);
    }
    return labels;
}

void ViewerWidget::onTimeSliderWidgetChanged(double value) {
    if (suppressTimeSliderCallback_) {
        suppressTimeSliderCallback_ = false;
        return;
    }

    if (!isTimeSeries_) {
        return;
    }

    QString error;
    applyFrameIndex(static_cast<int>(std::lround(value)), true, &error);
}

void ViewerWidget::onAnimationTick() {
    if (!animationPlaying_ || !isTimeSeries_ || frames_.isEmpty()) {
        stopAnimation();
        return;
    }

    if (currentFrameIndex_ >= frames_.size() - 1) {
        stopAnimation();
        return;
    }

    QString error;
    if (!applyFrameIndex(currentFrameIndex_ + 1, true, &error)) {
        stopAnimation();
    }
}

bool ViewerWidget::loadPvdSeries(const QString& path, QString* errorMessage) {
    const QVector<FrameInfo> parsed = parsePvdFile(path, errorMessage);
    if (parsed.isEmpty()) {
        return false;
    }
    if (parsed.size() <= 1) {
        if (errorMessage != nullptr) {
            *errorMessage = "The PVD file contains only one distinct time step.";
        }
        return false;
    }

    frames_ = parsed;
    currentPath_ = QFileInfo(path).absoluteFilePath();
    currentFrameIndex_ = 0;
    isTimeSeries_ = true;
    return true;
}

bool ViewerWidget::loadDataSetFromPath(const QString& path,
                                       vtkSmartPointer<vtkDataSet>* outData,
                                       QString* errorMessage) const {
    if (outData == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = "Internal error: dataset output pointer is null.";
        }
        return false;
    }

    const QFileInfo inputInfo(path);
    if (!inputInfo.exists() || !inputInfo.isFile()) {
        if (errorMessage != nullptr) {
            *errorMessage = QString("Result file does not exist: %1").arg(path);
        }
        return false;
    }

    const QString suffix = QFileInfo(path).suffix().toLower();
    const QByteArray pathBytes = QFile::encodeName(path);
    vtkSmartPointer<vtkDataSet> data;

    if (suffix == "pvd") {
        const QVector<FrameInfo> parsed = parsePvdFile(path, errorMessage);
        if (parsed.isEmpty()) {
            return false;
        }
        return loadDataSetFromPath(parsed.front().path, outData, errorMessage);
    } else if (suffix == "vts") {
        vtkNew<vtkXMLStructuredGridReader> reader;
        reader->SetFileName(pathBytes.constData());
        reader->Update();
        data = cloneDataSet(vtkDataSet::SafeDownCast(reader->GetOutput()));
    } else if (suffix == "vti") {
        vtkNew<vtkXMLImageDataReader> reader;
        reader->SetFileName(pathBytes.constData());
        reader->Update();
        data = cloneDataSet(vtkDataSet::SafeDownCast(reader->GetOutput()));
    } else if (suffix == "vtr") {
        vtkNew<vtkXMLRectilinearGridReader> reader;
        reader->SetFileName(pathBytes.constData());
        reader->Update();
        data = cloneDataSet(vtkDataSet::SafeDownCast(reader->GetOutput()));
    } else if (suffix == "vtu") {
        vtkNew<vtkXMLUnstructuredGridReader> reader;
        reader->SetFileName(pathBytes.constData());
        reader->Update();
        data = cloneDataSet(vtkDataSet::SafeDownCast(reader->GetOutput()));
    } else if (suffix == "vtp") {
        vtkNew<vtkXMLPolyDataReader> reader;
        reader->SetFileName(pathBytes.constData());
        reader->Update();
        data = cloneDataSet(vtkDataSet::SafeDownCast(reader->GetOutput()));
    } else if (suffix == "stl") {
        vtkNew<vtkSTLReader> reader;
        reader->SetFileName(pathBytes.constData());
        reader->Update();
        data = cloneDataSet(vtkDataSet::SafeDownCast(reader->GetOutput()));
    } else if (suffix == "vtk") {
        vtkNew<vtkDataSetReader> reader;
        reader->SetFileName(pathBytes.constData());
        reader->ReadAllScalarsOn();
        reader->ReadAllVectorsOn();
        reader->ReadAllFieldsOn();
        reader->Update();
        data = cloneDataSet(reader->GetOutput());
    } else if (suffix.startsWith("vt")) {
        vtkNew<vtkXMLGenericDataObjectReader> reader;
        reader->SetFileName(pathBytes.constData());
        reader->Update();
        data = cloneDataSet(vtkDataSet::SafeDownCast(reader->GetOutput()));
    } else {
        if (errorMessage != nullptr) {
            *errorMessage = QString("Unsupported file type: %1").arg(path);
        }
        return false;
    }

    if (data == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QString("Could not read result file: %1").arg(path);
        }
        return false;
    }

    prepareVectorFields(data);
    *outData = data;
    return true;
}

bool ViewerWidget::applyFrameIndex(int index, bool preserveCamera, QString* errorMessage) {
    if (customDisplayActive_) {
        return applyDisplayFrameIndex(index, preserveCamera, errorMessage);
    }
    if (!isTimeSeries_ || frames_.isEmpty()) {
        return false;
    }

    const int maxFrame = static_cast<int>(frames_.size()) - 1;
    const int clamped = std::clamp(index, 0, maxFrame);
    emit frameLoadRequested(clamped, frames_.at(clamped).timestep);
    vtkSmartPointer<vtkDataSet> data;
    if (!loadDataSetFromPath(frames_[clamped].path, &data, errorMessage)) {
        return false;
    }

    CameraState camera = captureCamera(renderer_);
    currentDataSet_ = data;
    currentFrameIndex_ = clamped;

    rebuildScene(!preserveCamera);
    if (preserveCamera) {
        restoreCamera(renderer_, camera, parallelProjection_);
        renderWindow_->Render();
    }

    emit frameChanged(currentFrameIndex_, frames_[currentFrameIndex_].timestep);
    return true;
}

bool ViewerWidget::applyDisplayFrameIndex(int index, bool preserveCamera, QString* errorMessage) {
    if (!customDisplayActive_ || !isTimeSeries_ || frames_.isEmpty()) {
        return false;
    }

    const int maxFrame = static_cast<int>(frames_.size()) - 1;
    const int clamped = std::clamp(index, 0, maxFrame);
    emit frameLoadRequested(clamped, frames_.at(clamped).timestep);
    const CameraState camera = captureCamera(renderer_);
    const QMap<QString, DataObjectOptions> objectOptions = displayObjectOptions_;
    currentFrameIndex_ = clamped;

    clearDisplayActors();
    for (auto it = objectOptions.constBegin(); it != objectOptions.constEnd(); ++it) {
        if (!addOrUpdateDataObject(it.key(), it.value(), false, errorMessage)) {
            return false;
        }
    }

    if (preserveCamera) {
        restoreCamera(renderer_, camera, parallelProjection_);
    }
    updateDisplayTimeOverlay();
    if (renderWindow_ != nullptr) {
        renderWindow_->Render();
    }
    emit frameChanged(currentFrameIndex_, frames_.at(currentFrameIndex_).timestep);
    return true;
}

void ViewerWidget::rebuildScene(bool resetCamera) {
    if (currentDataSet_ == nullptr) {
        return;
    }

    clearScalarVisuals();
    removeTimeWidgets();
    renderer_->RemoveAllViewProps();
    outlineActor_ = nullptr;
    watermarkActor_ = nullptr;

    vtkNew<vtkOutlineFilter> outlineFilter;
    outlineFilter->SetInputData(currentDataSet_);
    outlineFilter->Update();

    vtkNew<vtkDataSetMapper> outlineMapper;
    outlineMapper->SetInputConnection(outlineFilter->GetOutputPort());
    outlineActor_ = vtkSmartPointer<vtkActor>::New();
    outlineActor_->SetMapper(outlineMapper);
    outlineActor_->GetProperty()->SetColor(0.83, 0.83, 0.83);
    outlineActor_->GetProperty()->SetLineWidth(1.0);
    renderer_->AddActor(outlineActor_);

    watermarkActor_ = vtkSmartPointer<vtkTextActor>::New();
    watermarkActor_->SetInput("StreamCenter+");
    QColor resultLogoColor(166, 166, 166);
    resultLogoColor.setAlphaF(0.28);
    setTextActorViewportPosition(watermarkActor_, 0.015, 0.965, 24, resultLogoColor);
    applyStreamcenterLogoTextStyle(watermarkActor_);
    renderer_->AddActor2D(watermarkActor_);

    rebuildScalarVisuals(resetCamera);

    renderer_->GetActiveCamera()->SetParallelProjection(parallelProjection_ ? 1 : 0);
    if (resetCamera) {
        renderer_->ResetCamera();
    }

    if (isTimeSeries_) {
        updateTimeOverlay();
    }
}

void ViewerWidget::rebuildScalarVisuals(bool resetCamera) {
    clearScalarVisuals();

    bool addedAny = false;
    if (scalarMode_ == ScalarMode::Forward) {
        addedAny = addScalarVisual("FTLE_forward", QColor("orange"), 0, resetCamera);
    } else if (scalarMode_ == ScalarMode::Backward) {
        addedAny = addScalarVisual("FTLE_backward", QColor("royalblue"), 0, resetCamera);
    } else {
        const bool forwardOk = addScalarVisual("FTLE_forward", QColor("orange"), 0, resetCamera);
        const bool backwardOk = addScalarVisual("FTLE_backward", QColor("royalblue"), 1, false);
        addedAny = forwardOk || backwardOk;
    }

    if (!addedAny) {
        const QString fallbackScalar = preferredFallbackScalarName(currentDataSet_);
        if (!fallbackScalar.isEmpty()) {
            addedAny = addScalarVisual(fallbackScalar, QColor("#2f8f66"), 0, resetCamera);
        }
    }

    if (!addedAny) {
        removeTimeWidgets();
    }
}

void ViewerWidget::clearScalarVisuals() {
    for (ScalarVisual& visual : scalarVisuals_) {
        if (visual.actor != nullptr) {
            renderer_->RemoveActor(visual.actor);
        }
    }
    scalarVisuals_.clear();
}

void ViewerWidget::clearDisplayActors() {
    if (indexVolumeBackend_ != nullptr) {
        indexVolumeBackend_->clear();
    }
    if (renderer_ != nullptr) {
        for (auto it = displayActors_.begin(); it != displayActors_.end(); ++it) {
            if (it->planeWidget != nullptr) {
                it->planeWidget->Off();
            }
            if (it->scalarBarWidget != nullptr) {
                it->scalarBarWidget->Off();
            }
            for (const vtkSmartPointer<vtkProp>& prop : it->props) {
                renderer_->RemoveViewProp(prop);
            }
        }
    }
    displayActors_.clear();
}

void ViewerWidget::updatePlaneWidgetVisibility() {
    bool changed = false;
    for (auto it = displayActors_.begin(); it != displayActors_.end(); ++it) {
        if (it->planeWidget == nullptr) {
            continue;
        }
        const DataObjectOptions options = displayObjectOptions_.value(it.key());
        const bool enabled = !activeDataObjectHandleId_.isEmpty()
            && it.key() == activeDataObjectHandleId_
            && options.showPlaneHandle;
        if (enabled) {
            it->planeWidget->On();
        } else {
            it->planeWidget->Off();
        }
        changed = true;
    }
    if (changed && renderWindow_ != nullptr) {
        renderWindow_->Render();
    }
}

void ViewerWidget::rebuildDisplayOverlay() {
    if (renderer_ == nullptr || !customDisplayActive_) {
        return;
    }
    if (watermarkActor_ != nullptr) {
        renderer_->RemoveActor2D(watermarkActor_);
        watermarkActor_ = nullptr;
    }
    if (timeCodeActor_ != nullptr) {
        renderer_->RemoveActor2D(timeCodeActor_);
        timeCodeActor_ = nullptr;
    }
    if (timeLabelActor_ != nullptr) {
        renderer_->RemoveActor2D(timeLabelActor_);
        timeLabelActor_ = nullptr;
    }
    if (displayOptions_.showLogo) {
        watermarkActor_ = vtkSmartPointer<vtkTextActor>::New();
        watermarkActor_->SetInput("StreamCenter+");
        QColor logoColor(105, 105, 105);
        logoColor.setAlphaF(0.28);
        setTextActorViewportPosition(watermarkActor_, 0.015, 0.965, 24, logoColor);
        applyStreamcenterLogoTextStyle(watermarkActor_);
        renderer_->AddActor2D(watermarkActor_);
    }
    if (displayOptions_.showTimeCode || displayOptions_.showTimeStep) {
        timeCodeActor_ = vtkSmartPointer<vtkTextActor>::New();
        setTextActorViewportPosition(timeCodeActor_, 0.98, 0.03, 12, Qt::black, true, true);
        renderer_->AddActor2D(timeCodeActor_);
        updateDisplayTimeOverlay();
    }
    if (displayOptions_.showCustomAxesText && !displayOptions_.customAxesText.trimmed().isEmpty()) {
        timeLabelActor_ = vtkSmartPointer<vtkTextActor>::New();
        timeLabelActor_->SetInput(displayOptions_.customAxesText.toUtf8().constData());
        setTextActorViewportPosition(timeLabelActor_, 0.015, 0.08, 12, Qt::black);
        renderer_->AddActor2D(timeLabelActor_);
    }
}

void ViewerWidget::scheduleRenderWhenVisible() {
    if (renderWindow_ == nullptr) {
        return;
    }

    auto requestRender = [this]() {
        if (renderWindow_ == nullptr || vtkWidget_ == nullptr || stackedLayout_ == nullptr
            || stackedLayout_->currentWidget() != vtkWidget_ || !vtkWidget_->isVisible()
            || vtkWidget_->width() <= 0 || vtkWidget_->height() <= 0) {
            return;
        }
        if (vtkWidget_->interactor() != nullptr && !vtkWidget_->interactor()->GetInitialized()) {
            vtkWidget_->interactor()->Initialize();
        }
        for (auto it = displayActors_.begin(); it != displayActors_.end(); ++it) {
            const auto optionsIt = displayObjectOptions_.constFind(it.key());
            if (optionsIt != displayObjectOptions_.constEnd() && it->scalarBar != nullptr) {
                applyScalarBarOptions(&(*it), optionsIt->legend);
            }
        }
        renderWindow_->Render();
        vtkWidget_->update();
    };
    for (int delayMs : {0, 16, 80, 160, 320, 640}) {
        QTimer::singleShot(delayMs, this, requestRender);
    }
}

bool ViewerWidget::addScalarVisual(const QString& scalarName, const QColor& color, int slot, bool resetCamera) {
    int association = kFieldAssociationPoints;
    vtkDataArray* array = findScalarArray(scalarName, &association);
    if (array == nullptr) {
        return false;
    }

    double range[2]{};
    array->GetRange(range);
    const double minValue = range[0];
    const double maxValue = range[1];
    const double target = isoTargets_.contains(scalarName)
        ? isoTargets_.value(scalarName)
        : (minValue + 0.75 * (maxValue - minValue));
    const double effective = std::clamp(target, minValue, maxValue);
    isoTargets_[scalarName] = effective;

    vtkNew<vtkContourFilter> contour;
    contour->SetInputData(currentDataSet_);
    contour->SetInputArrayToProcess(
        0,
        0,
        0,
        association == kFieldAssociationPoints ? vtkDataObject::FIELD_ASSOCIATION_POINTS
                                               : vtkDataObject::FIELD_ASSOCIATION_CELLS,
        scalarName.toUtf8().constData());
    contour->SetValue(0, effective);
    contour->Update();

    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(contour->GetOutputPort());
    mapper->ScalarVisibilityOff();

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(color.redF(), color.greenF(), color.blueF());
    actor->GetProperty()->SetOpacity(transparent_ ? kTransparentOpacity : kOpaqueOpacity);
    renderer_->AddActor(actor);
    if (resetCamera) {
        renderer_->ResetCamera();
    }

    ScalarVisual visual;
    visual.scalarName = scalarName;
    visual.slot = slot;
    visual.association = association;
    visual.minValue = minValue;
    visual.maxValue = maxValue;
    visual.actor = actor;

    scalarVisuals_.push_back(visual);
    return true;
}

vtkDataArray* ViewerWidget::findScalarArray(const QString& scalarName, int* association) const {
    if (currentDataSet_ == nullptr) {
        return nullptr;
    }

    if (currentDataSet_->GetPointData() != nullptr) {
        if (vtkDataArray* array = currentDataSet_->GetPointData()->GetArray(scalarName.toUtf8().constData())) {
            if (association != nullptr) {
                *association = kFieldAssociationPoints;
            }
            return array;
        }
    }

    if (currentDataSet_->GetCellData() != nullptr) {
        if (vtkDataArray* array = currentDataSet_->GetCellData()->GetArray(scalarName.toUtf8().constData())) {
            if (association != nullptr) {
                *association = kFieldAssociationCells;
            }
            return array;
        }
    }

    return nullptr;
}

void ViewerWidget::ensureTimeWidgets() {
    if (!isTimeSeries_) {
        return;
    }

    if (timeSlider_ == nullptr) {
        vtkNew<vtkSliderRepresentation2D> representation;
        representation->SetTitleText("");
        representation->GetPoint1Coordinate()->SetCoordinateSystemToNormalizedDisplay();
        representation->GetPoint2Coordinate()->SetCoordinateSystemToNormalizedDisplay();
        representation->GetPoint1Coordinate()->SetValue(0.02, 0.16);
        representation->GetPoint2Coordinate()->SetValue(0.15, 0.16);
        representation->SetSliderLength(0.01);
        representation->SetSliderWidth(0.02);
        representation->SetTubeWidth(0.005);
        representation->SetLabelHeight(0.02);
        representation->SetLabelFormat("%0.0f");
        representation->GetLabelProperty()->SetColor(0.0, 0.0, 0.0);

        timeSlider_ = vtkSmartPointer<vtkSliderWidget>::New();
        timeSlider_->SetInteractor(vtkWidget_->interactor());
        timeSlider_->SetRepresentation(representation);
        timeSlider_->SetAnimationModeToAnimate();
        timeSlider_->EnabledOn();

        timeCallbackState_ = std::make_unique<TimeCallbackState>();
        timeCallbackState_->viewer = this;
        timeCallbackState_->command = vtkSmartPointer<vtkCallbackCommand>::New();
        timeCallbackState_->command->SetClientData(timeCallbackState_.get());
        timeCallbackState_->command->SetCallback([](vtkObject* caller, unsigned long, void* clientData, void*) {
            auto* state = static_cast<TimeCallbackState*>(clientData);
            auto* sliderWidget = vtkSliderWidget::SafeDownCast(caller);
            if (state == nullptr || state->viewer == nullptr || sliderWidget == nullptr) {
                return;
            }
            if (state->swallowFirstCallback) {
                state->swallowFirstCallback = false;
                return;
            }
            auto* representationInner = vtkSliderRepresentation::SafeDownCast(sliderWidget->GetRepresentation());
            if (representationInner == nullptr) {
                return;
            }
            state->viewer->onTimeSliderWidgetChanged(representationInner->GetValue());
        });
        timeSlider_->AddObserver(vtkCommand::InteractionEvent, timeCallbackState_->command);
    }

    auto* representation = vtkSliderRepresentation::SafeDownCast(timeSlider_->GetRepresentation());
    if (representation != nullptr) {
        suppressTimeSliderCallback_ = true;
        representation->SetMinimumValue(0.0);
        representation->SetMaximumValue(std::max(0, static_cast<int>(frames_.size()) - 1));
        representation->SetValue(currentFrameIndex_);
        representation->SetLabelFormat("%0.0f");
    }
    timeSlider_->EnabledOn();

    if (timeLabelActor_ == nullptr) {
        timeLabelActor_ = vtkSmartPointer<vtkTextActor>::New();
        timeLabelActor_->SetInput("Time");
        setTextActorViewportPosition(timeLabelActor_, 0.15, 0.147, 9, Qt::black);
        renderer_->AddActor2D(timeLabelActor_);
    }
}

void ViewerWidget::removeTimeWidgets() {
    if (timeSlider_ != nullptr) {
        timeSlider_->EnabledOff();
        timeSlider_ = nullptr;
    }
    timeCallbackState_.reset();

    if (timeLabelActor_ != nullptr) {
        renderer_->RemoveActor2D(timeLabelActor_);
        timeLabelActor_ = nullptr;
    }

    if (timeCodeActor_ != nullptr) {
        renderer_->RemoveActor2D(timeCodeActor_);
        timeCodeActor_ = nullptr;
    }
}

void ViewerWidget::updateTimeOverlay() {
    if (!isTimeSeries_ || currentFrameIndex_ < 0 || currentFrameIndex_ >= frames_.size()) {
        return;
    }

    if (timeCodeActor_ == nullptr) {
        timeCodeActor_ = vtkSmartPointer<vtkTextActor>::New();
        renderer_->AddActor2D(timeCodeActor_);
    }

    const QString label = QStringLiteral("t = %1").arg(formatTimeCode(frames_[currentFrameIndex_].timestep));
    timeCodeActor_->SetInput(label.toUtf8().constData());
    setTextActorViewportPosition(timeCodeActor_, 0.98, 0.03, 12, Qt::black, true, true);
}

void ViewerWidget::refreshDisplayTimeSeriesState() {
    if (!customDisplayActive_) {
        return;
    }

    const bool wasTimeSeries = isTimeSeries_;
    const double previousTime =
        isTimeSeries_ && currentFrameIndex_ >= 0 && currentFrameIndex_ < frames_.size()
            ? frames_.at(currentFrameIndex_).timestep
            : std::numeric_limits<double>::quiet_NaN();

    QVector<FrameInfo> mergedFrames;
    for (auto it = displayObjectTimeSeries_.constBegin(); it != displayObjectTimeSeries_.constEnd(); ++it) {
        for (const FrameInfo& frame : it->frames) {
            FrameInfo merged;
            merged.timestep = frame.timestep;
            mergedFrames.push_back(merged);
        }
    }

    std::sort(mergedFrames.begin(), mergedFrames.end(), [](const FrameInfo& a, const FrameInfo& b) {
        return a.timestep < b.timestep;
    });
    QVector<FrameInfo> uniqueFrames;
    uniqueFrames.reserve(mergedFrames.size());
    for (const FrameInfo& frame : std::as_const(mergedFrames)) {
        if (uniqueFrames.isEmpty() || !sameTimeCode(uniqueFrames.back().timestep, frame.timestep)) {
            uniqueFrames.push_back(frame);
        }
    }

    if (uniqueFrames.size() > 1) {
        int nextIndex = 0;
        if (std::isfinite(previousTime)) {
            nextIndex = std::max(0, nearestFrameIndexForTime(uniqueFrames, previousTime));
        } else if (std::isfinite(initialTimeCodeHint_)) {
            nextIndex = std::max(0, nearestFrameIndexForTime(uniqueFrames, initialTimeCodeHint_));
            initialTimeCodeHint_ = std::numeric_limits<double>::quiet_NaN();
        }
        frames_ = uniqueFrames;
        currentFrameIndex_ = std::clamp(nextIndex, 0, static_cast<int>(frames_.size()) - 1);
        isTimeSeries_ = true;
    } else {
        frames_.clear();
        currentFrameIndex_ = 0;
        isTimeSeries_ = false;
    }

    updateDisplayTimeOverlay();
    if (wasTimeSeries != isTimeSeries_) {
        emit timeSeriesAvailabilityChanged(isTimeSeries_);
    }
    if (isTimeSeries_ && currentFrameIndex_ >= 0 && currentFrameIndex_ < frames_.size()) {
        emit frameChanged(currentFrameIndex_, frames_.at(currentFrameIndex_).timestep);
    }
}

void ViewerWidget::updateDisplayTimeOverlay() {
    if (!customDisplayActive_ || timeCodeActor_ == nullptr) {
        return;
    }

    QStringList lines;
    if (displayOptions_.showTimeCode) {
        lines << (isTimeSeries_ && currentFrameIndex_ >= 0 && currentFrameIndex_ < frames_.size()
                      ? QStringLiteral("t = %1").arg(formatTimeCode(frames_.at(currentFrameIndex_).timestep))
                      : QStringLiteral("t = --"));
    }
    if (displayOptions_.showTimeStep) {
        lines << (isTimeSeries_ && currentFrameIndex_ >= 0 && currentFrameIndex_ < frames_.size()
                      ? QStringLiteral("step = %1 / %2").arg(currentFrameIndex_ + 1).arg(frames_.size())
                      : QStringLiteral("step = --"));
    }
    timeCodeActor_->SetInput(lines.join(QLatin1Char('\n')).toUtf8().constData());
}

QString ViewerWidget::displayObjectFramePathForCurrentTime(const QString& objectId, const QString& fallbackPath) const {
    const auto it = displayObjectTimeSeries_.constFind(objectId);
    if (it == displayObjectTimeSeries_.constEnd() || it->frames.isEmpty()) {
        return fallbackPath;
    }
    if (!isTimeSeries_ || currentFrameIndex_ < 0 || currentFrameIndex_ >= frames_.size()) {
        return it->frames.front().path;
    }
    const int index = nearestFrameIndexForTime(it->frames, frames_.at(currentFrameIndex_).timestep);
    return index >= 0 ? it->frames.at(index).path : it->frames.front().path;
}

void ViewerWidget::setCaptureUiVisible(bool visible) {
    if (timeSlider_ != nullptr) {
        if (visible) {
            timeSlider_->EnabledOn();
        } else {
            timeSlider_->EnabledOff();
        }
    }

    if (timeLabelActor_ != nullptr) {
        timeLabelActor_->SetVisibility(visible ? 1 : 0);
    }

    renderWindow_->Render();
}

QImage ViewerWidget::captureCurrentImage(QString* errorMessage,
                                         bool transparentBackground,
                                         const QSize& outputSize) {
    if (!hasScene()) {
        if (errorMessage != nullptr) {
            *errorMessage = "Please load a result file first.";
        }
        return {};
    }

    const int* originalWindowSize = renderWindow_ != nullptr ? renderWindow_->GetSize() : nullptr;
    const int originalWidth = originalWindowSize != nullptr ? std::max(1, originalWindowSize[0]) : std::max(1, width());
    const int originalHeight = originalWindowSize != nullptr ? std::max(1, originalWindowSize[1]) : std::max(1, height());
    const bool customOutputSize = outputSize.isValid() && outputSize.width() > 0 && outputSize.height() > 0;
    const QSize captureSize = customOutputSize ? outputSize : QSize(originalWidth, originalHeight);
    const CameraState cameraState = captureCamera(renderer_);
    const bool previousParallelProjection =
        renderer_ != nullptr && renderer_->GetActiveCamera() != nullptr
            ? renderer_->GetActiveCamera()->GetParallelProjection() != 0
            : parallelProjection_;
    vtkSmartPointer<vtkRenderPass> previousRenderPass = renderer_ != nullptr ? renderer_->GetPass() : nullptr;
    const int previousAlphaBitPlanes = renderWindow_ != nullptr ? renderWindow_->GetAlphaBitPlanes() : 0;
    double previousBackgroundAlpha = 1.0;
    if (renderer_ != nullptr) {
        previousBackgroundAlpha = renderer_->GetBackgroundAlpha();
        renderer_->SetBackgroundAlpha(transparentBackground ? 0.0 : 1.0);
    }
    const bool useRayTracingExport = exportRayTracingRequested();
    const int previousWatermarkVisibility = watermarkActor_ != nullptr ? watermarkActor_->GetVisibility() : 0;
    const int previousTimeCodeVisibility = timeCodeActor_ != nullptr ? timeCodeActor_->GetVisibility() : 0;
    const int previousTimeLabelVisibility = timeLabelActor_ != nullptr ? timeLabelActor_->GetVisibility() : 0;
    const int previousOrientationEnabled = orientationWidget_ != nullptr ? orientationWidget_->GetEnabled() : 0;
    auto setRayTracingOverlaysVisible = [&](bool visible) {
        if (!useRayTracingExport) {
            return;
        }
        if (watermarkActor_ != nullptr) {
            watermarkActor_->SetVisibility(visible ? previousWatermarkVisibility : 0);
        }
        if (timeCodeActor_ != nullptr) {
            timeCodeActor_->SetVisibility(visible ? previousTimeCodeVisibility : 0);
        }
        if (timeLabelActor_ != nullptr) {
            timeLabelActor_->SetVisibility(visible ? previousTimeLabelVisibility : 0);
        }
        if (orientationWidget_ != nullptr) {
            orientationWidget_->SetEnabled(visible ? previousOrientationEnabled : 0);
        }
    };
    if (renderWindow_ != nullptr) {
        renderWindow_->SetAlphaBitPlanes(transparentBackground ? 1 : 0);
        if (customOutputSize && (captureSize.width() != originalWidth || captureSize.height() != originalHeight)) {
            const double sourceAspect = static_cast<double>(originalWidth) / static_cast<double>(originalHeight);
            const double targetAspect = static_cast<double>(captureSize.width()) / static_cast<double>(captureSize.height());
            expandCameraForTargetAspect(renderer_, sourceAspect, targetAspect);
            if (!useRayTracingExport) {
                renderWindow_->SetSize(captureSize.width(), captureSize.height());
            }
        }
    }

    auto restoreCaptureState = [&]() {
        endExportRayTracing(previousRenderPass);
        if (renderer_ != nullptr) {
            renderer_->SetBackgroundAlpha(previousBackgroundAlpha);
            restoreCamera(renderer_, cameraState, previousParallelProjection);
        }
        setRayTracingOverlaysVisible(true);
        if (renderWindow_ != nullptr) {
            renderWindow_->SetAlphaBitPlanes(previousAlphaBitPlanes);
            if (!useRayTracingExport
                && customOutputSize
                && (captureSize.width() != originalWidth || captureSize.height() != originalHeight)) {
                renderWindow_->SetSize(originalWidth, originalHeight);
            }
            renderWindow_->Render();
        }
    };
    auto grabFramebufferFallback = [&]() -> QImage {
        if (vtkWidget_ == nullptr || vtkWidget_->width() <= 0 || vtkWidget_->height() <= 0) {
            return {};
        }
        QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        QImage image = vtkWidget_->grab().toImage();
        if (image.isNull()) {
            return {};
        }
        if (customOutputSize && image.size() != captureSize) {
            image = image.scaled(captureSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
        }
        return transparentBackground
            ? image.convertToFormat(QImage::Format_RGBA8888)
            : image.convertToFormat(QImage::Format_RGB888);
    };

    setRayTracingOverlaysVisible(false);
    if (useRayTracingExport) {
#if STREAMCENTERPLUS_ENABLE_VTK_RAYTRACING
        QTemporaryDir exportTempDir;
        if (!exportTempDir.isValid()) {
            restoreCaptureState();
            if (errorMessage != nullptr) {
                *errorMessage = "Could not create a temporary folder for ray-traced export.";
            }
            return {};
        }

        QJsonObject scene;
        scene.insert(QStringLiteral("width"), captureSize.width());
        scene.insert(QStringLiteral("height"), captureSize.height());
        scene.insert(QStringLiteral("parallel_projection"), previousParallelProjection);
        scene.insert(QStringLiteral("renderer_type"),
                     displayOptions_.advancedRendererType.trimmed().isEmpty()
                         ? QStringLiteral("pathtracer")
                         : displayOptions_.advancedRendererType.trimmed());
        scene.insert(QStringLiteral("samples_per_pixel"), std::max(1, displayOptions_.advancedSamplesPerPixel));
        scene.insert(QStringLiteral("accumulation_frames"), std::max(1, displayOptions_.advancedAccumulationFrames));
        scene.insert(QStringLiteral("max_depth"), std::max(1, displayOptions_.advancedMaxDepth));
        QJsonArray camera;
        for (double value : cameraState.position) {
            camera.append(value);
        }
        for (double value : cameraState.focalPoint) {
            camera.append(value);
        }
        for (double value : cameraState.viewUp) {
            camera.append(value);
        }
        camera.append(cameraState.parallelScale);
        camera.append(cameraState.viewAngle);
        scene.insert(QStringLiteral("camera"), camera);
        QJsonArray background;
        if (renderer_ != nullptr) {
            double rgb[3]{};
            renderer_->GetBackground(rgb);
            background.append(rgb[0]);
            background.append(rgb[1]);
            background.append(rgb[2]);
        }
        scene.insert(QStringLiteral("background"), background);
        scene.insert(QStringLiteral("lighting_intensity"), std::clamp(displayOptions_.lightingIntensity, 0.0, 2.0));

        QJsonArray objects;
        for (auto it = displayObjectOptions_.constBegin(); it != displayObjectOptions_.constEnd(); ++it) {
            const DataObjectOptions& objectOptions = it.value();
            if (!objectOptions.visible
                || objectOptions.inputPath.trimmed().isEmpty()
                || objectOptions.type == DisplayObjectType::ParticleStreamline
                || objectOptions.type == DisplayObjectType::RayTracingVolume) {
                continue;
            }

            const bool surfaceEnabled = objectOptions.type == DisplayObjectType::Contour ? true : objectOptions.showSurface;
            if (!surfaceEnabled) {
                continue;
            }

            const QString objectId = it.key();
            const QString absoluteInputPath = QFileInfo(objectOptions.inputPath).absoluteFilePath();
            QString loadPath = absoluteInputPath;
            if (QFileInfo(absoluteInputPath).suffix().compare(QStringLiteral("pvd"), Qt::CaseInsensitive) == 0) {
                loadPath = displayObjectFramePathForCurrentTime(objectId, absoluteInputPath);
            }

            const RenderMaterialValues material = renderMaterialValues(objectOptions);
            QJsonObject object;
            object.insert(QStringLiteral("input_path"), loadPath);
            switch (objectOptions.type) {
            case DisplayObjectType::Clip:
                object.insert(QStringLiteral("type"), QStringLiteral("clip"));
                break;
            case DisplayObjectType::Slice:
                object.insert(QStringLiteral("type"), QStringLiteral("slice"));
                break;
            case DisplayObjectType::Data:
                object.insert(QStringLiteral("type"), QStringLiteral("geometry"));
                break;
            case DisplayObjectType::Crop:
                object.insert(QStringLiteral("type"), QStringLiteral("crop"));
                {
                    QJsonArray cropBounds;
                    for (double value : objectOptions.cropBounds) {
                        cropBounds.append(value);
                    }
                    object.insert(QStringLiteral("crop_bounds"), cropBounds);
                }
                break;
            case DisplayObjectType::Contour:
                object.insert(QStringLiteral("type"), QStringLiteral("contour"));
                object.insert(QStringLiteral("contour_field"),
                              objectOptions.contourField.trimmed().isEmpty()
                                  ? objectOptions.colorField
                                  : objectOptions.contourField);
                {
                    QJsonArray values;
                    for (double contourValue : objectOptions.contourValues) {
                        values.append(contourValue);
                    }
                    object.insert(QStringLiteral("contour_values"), values);
                }
                break;
            default:
                object.insert(QStringLiteral("type"), QStringLiteral("geometry"));
                break;
            }
            QJsonArray planeOrigin;
            QJsonArray planeNormal;
            for (double value : objectOptions.planeOrigin) {
                planeOrigin.append(value);
            }
            for (double value : objectOptions.planeNormal) {
                planeNormal.append(value);
            }
            object.insert(QStringLiteral("plane_origin"), planeOrigin);
            object.insert(QStringLiteral("plane_normal"), planeNormal);
            QJsonArray baseColor;
            baseColor.append(material.baseColor.redF());
            baseColor.append(material.baseColor.greenF());
            baseColor.append(material.baseColor.blueF());
            object.insert(QStringLiteral("base_color"), baseColor);
            object.insert(QStringLiteral("metallic"), material.metallic);
            object.insert(QStringLiteral("roughness"), material.roughness);
            object.insert(QStringLiteral("ior"), material.ior);
            object.insert(QStringLiteral("opacity"), std::clamp(material.opacity * (1.0 - (0.75 * material.transmission)), 0.0, 1.0));
            const QString osprayMaterialName = objectOptions.osprayMaterialName.trimmed();
            if (!osprayMaterialName.isEmpty()) {
                object.insert(QStringLiteral("ospray_material_name"), osprayMaterialName);
            }
            objects.append(object);
        }

        if (objects.isEmpty()) {
            restoreCaptureState();
            if (errorMessage != nullptr) {
                *errorMessage = "No geometry surface is available for ray-traced export.";
            }
            return {};
        }
        scene.insert(QStringLiteral("objects"), objects);

        const QString scenePath = exportTempDir.filePath(QStringLiteral("raytrace_scene.json"));
        const QString imagePath = exportTempDir.filePath(QStringLiteral("raytrace.png"));
        QFile sceneFile(scenePath);
        if (!sceneFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            restoreCaptureState();
            if (errorMessage != nullptr) {
                *errorMessage = QString("Could not write ray-traced export scene: %1").arg(scenePath);
            }
            return {};
        }
        sceneFile.write(QJsonDocument(scene).toJson(QJsonDocument::Compact));
        sceneFile.close();

        QString helperName = QStringLiteral("StreamcenterPlusRayTraceExport");
#ifdef Q_OS_WIN
        helperName += QStringLiteral(".exe");
#endif
        const QString helperPath = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(helperName);
        QProcess process;
        QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
        const QString appDir = QCoreApplication::applicationDirPath();
#ifdef Q_OS_WIN
        environment.insert(QStringLiteral("PATH"), appDir + QLatin1Char(';') + environment.value(QStringLiteral("PATH")));
#else
        environment.insert(QStringLiteral("PATH"), appDir + QLatin1Char(':') + environment.value(QStringLiteral("PATH")));
#endif
        if (environment.value(QStringLiteral("OSPRAY_MODULE_PATH")).isEmpty()
            && QFileInfo::exists(QDir(appDir).absoluteFilePath(QStringLiteral("ospray_module_cpu.dll")))) {
            environment.insert(QStringLiteral("OSPRAY_MODULE_PATH"), appDir);
        }
        process.setProcessEnvironment(environment);
        process.start(helperPath, {scenePath, imagePath});
        if (!process.waitForStarted() || !process.waitForFinished(-1)) {
            restoreCaptureState();
            if (errorMessage != nullptr) {
                *errorMessage = QString("Could not run ray-traced export helper: %1").arg(helperPath);
            }
            return {};
        }
        if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
            const QString helperError = QString::fromLocal8Bit(process.readAllStandardError()).trimmed();
            restoreCaptureState();
            if (errorMessage != nullptr) {
                *errorMessage = helperError.isEmpty()
                    ? QString("Ray-traced export helper failed with exit code %1.").arg(process.exitCode())
                    : helperError;
            }
            return {};
        }
        QImageReader imageReader(imagePath);
        QImage exportImage = imageReader.read();
        if (exportImage.isNull()) {
            const QByteArray imageFileName = QFile::encodeName(imagePath);
            vtkNew<vtkPNGReader> pngReader;
            pngReader->SetFileName(imageFileName.constData());
            pngReader->Update();
            exportImage = qImageFromVtkImageData(pngReader->GetOutput(), transparentBackground);
        }
        restoreCaptureState();
        if (!exportImage.isNull()) {
            return transparentBackground ? exportImage.convertToFormat(QImage::Format_RGBA8888)
                                         : exportImage.convertToFormat(QImage::Format_RGB888);
        }
        if (errorMessage != nullptr) {
            const QFileInfo imageInfo(imagePath);
            QStringList supportedFormats;
            const QList<QByteArray> formats = QImageReader::supportedImageFormats();
            supportedFormats.reserve(formats.size());
            for (const QByteArray& format : formats) {
                supportedFormats.push_back(QString::fromLatin1(format));
            }
            *errorMessage = QString("Ray-traced export helper completed without producing a readable image. Output exists: %1; size: %2 bytes; reader error: %3; supported formats: %4.")
                .arg(imageInfo.exists() ? QStringLiteral("yes") : QStringLiteral("no"))
                .arg(imageInfo.exists() ? imageInfo.size() : 0)
                .arg(imageReader.errorString())
                .arg(supportedFormats.join(QStringLiteral(", ")));
        }
        return {};
#else
        restoreCaptureState();
        if (errorMessage != nullptr) {
            *errorMessage = tr("OSPRay export is unavailable because this GUI was built without VTK RenderingRayTracing. Rebuild with -EnableVtkRayTracing and a VTK install that includes RenderingRayTracing/OSPRay.");
        }
        return {};
#endif
    }
    if (!beginExportRayTracing(errorMessage)) {
        restoreCaptureState();
        return {};
    }
    renderAccumulationFrames();
    if (renderWindow_ != nullptr) {
        renderWindow_->Modified();
    }

    vtkNew<vtkWindowToImageFilter> filter;
    filter->SetInput(renderWindow_);
    filter->ReadFrontBufferOff();
    filter->SetScale(1);
    if (transparentBackground) {
        filter->SetInputBufferTypeToRGBA();
    } else {
        filter->SetInputBufferTypeToRGB();
    }
    filter->Update();

    vtkImageData* imageData = filter->GetOutput();
    if (imageData == nullptr) {
        const QImage fallbackImage = grabFramebufferFallback();
        restoreCaptureState();
        if (!fallbackImage.isNull()) {
            return fallbackImage;
        }
        if (errorMessage != nullptr) {
            *errorMessage = "Could not capture the render window.";
        }
        return {};
    }

    int dims[3]{};
    imageData->GetDimensions(dims);
    if (dims[0] <= 0 || dims[1] <= 0) {
        const QImage fallbackImage = grabFramebufferFallback();
        restoreCaptureState();
        if (!fallbackImage.isNull()) {
            return fallbackImage;
        }
        if (errorMessage != nullptr) {
            *errorMessage = "Captured image has invalid dimensions.";
        }
        return {};
    }
    if (customOutputSize && (dims[0] != captureSize.width() || dims[1] != captureSize.height())) {
        restoreCaptureState();
        if (errorMessage != nullptr) {
            *errorMessage = tr("Could not render screenshot at %1 x %2. VTK returned %3 x %4 instead.")
                .arg(captureSize.width())
                .arg(captureSize.height())
                .arg(dims[0])
                .arg(dims[1]);
        }
        return {};
    }

    unsigned char* src = static_cast<unsigned char*>(imageData->GetScalarPointer());
    if (src == nullptr) {
        const QImage fallbackImage = grabFramebufferFallback();
        restoreCaptureState();
        if (!fallbackImage.isNull()) {
            return fallbackImage;
        }
        if (errorMessage != nullptr) {
            *errorMessage = "Captured pixel buffer is not available.";
        }
        return {};
    }

    const int components = transparentBackground ? 4 : 3;
    QImage image(dims[0], dims[1], transparentBackground ? QImage::Format_RGBA8888 : QImage::Format_RGB888);
    const int rowBytes = dims[0] * components;
    for (int y = 0; y < dims[1]; ++y) {
        std::memcpy(image.scanLine(dims[1] - 1 - y), src + (y * rowBytes), rowBytes);
    }
    restoreCaptureState();
    return image;
}

QSize ViewerWidget::renderSize() const {
    const int* windowSize = renderWindow_ != nullptr ? renderWindow_->GetSize() : nullptr;
    if (windowSize == nullptr || windowSize[0] <= 0 || windowSize[1] <= 0) {
        return size();
    }
    return QSize(windowSize[0], windowSize[1]);
}

bool ViewerWidget::saveScreenshot(const QString& path,
                                  QString* errorMessage,
                                  bool transparentBackground,
                                  const QSize& outputSize,
                                  QImage* capturedImage,
                                  bool requireFileWrite) {
    if (!hasScene()) {
        if (errorMessage != nullptr) {
            *errorMessage = "Please load a result file first.";
        }
        return false;
    }

    setCaptureUiVisible(false);
    const QImage image = captureCurrentImage(errorMessage, transparentBackground, outputSize);
    setCaptureUiVisible(true);
    if (image.isNull()) {
        return false;
    }

    if (capturedImage != nullptr) {
        *capturedImage = image;
    }
    QString writeError;
    if (!saveImageWithVtkWriter(image, path, &writeError)) {
        if (errorMessage != nullptr) {
            *errorMessage = writeError;
        }
        return !requireFileWrite;
    }
    if (errorMessage != nullptr) {
        errorMessage->clear();
    }
    return true;
}

bool ViewerWidget::exportVideo(const QString& path,
                               int fps,
                               int stride,
                               int startIndex,
                               int endIndex,
                               int frameInterpolationMultiplier,
                               QWidget* dialogParent,
                               QString* errorMessage,
                               bool transparentBackground,
                               const QSize& outputSize,
                               const VideoProgressCallback& progressCallback) {
    Q_UNUSED(dialogParent);
    if (!isTimeSeries_ || frames_.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = "The current result is not a PVD time series.";
        }
        return false;
    }

    const int normalizedInterpolationMultiplier =
        (frameInterpolationMultiplier == 2 || frameInterpolationMultiplier == 4) ? frameInterpolationMultiplier : 1;
    if (transparentBackground && normalizedInterpolationMultiplier > 1) {
        if (errorMessage != nullptr) {
            *errorMessage = tr("Frame interpolation does not support transparent background. Disable transparent background or select Disabled.");
        }
        return false;
    }
    if (normalizedInterpolationMultiplier > 1) {
        const Streamcenter::Fruc::RuntimeStatus frucStatus = Streamcenter::Fruc::NvidiaFrucRuntime::status();
        if (!frucStatus.available) {
            if (errorMessage != nullptr) {
                *errorMessage = frucStatus.message;
            }
            return false;
        }
    }

    const QString ffmpegPath = QStandardPaths::findExecutable("ffmpeg");
    if (ffmpegPath.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = "Cannot find ffmpeg in PATH.";
        }
        return false;
    }
    const QString suffix = QFileInfo(path).suffix().toLower();
    const bool requiresEvenDimensions = suffix != QStringLiteral("gif");
    auto roundUpEven = [](int value) {
        const int clamped = std::max(1, value);
        return (clamped % 2 == 0) ? clamped : clamped + 1;
    };
    QSize videoOutputSize = outputSize;
    if (requiresEvenDimensions) {
        const QSize baseSize = videoOutputSize.isValid() && videoOutputSize.width() > 0 && videoOutputSize.height() > 0
            ? videoOutputSize
            : renderSize();
        videoOutputSize = QSize(roundUpEven(baseSize.width()), roundUpEven(baseSize.height()));
    }

    const int maxFrame = static_cast<int>(frames_.size()) - 1;
    const int first = std::clamp(startIndex, 0, maxFrame);
    const int last = std::clamp(endIndex, 0, maxFrame);
    const int begin = std::min(first, last);
    const int finish = std::max(first, last);
    const int frameStep = std::max(1, stride);
    const int previousFrame = currentFrameIndex_;
    const bool wasPlaying = animationPlaying_;

    stopAnimation();

    QTemporaryDir tempDir;
    if (!tempDir.isValid()) {
        if (errorMessage != nullptr) {
            *errorMessage = "Could not create a temporary directory for the video export.";
        }
        return false;
    }

    QVector<int> exportFrames;
    for (int i = begin; i <= finish; i += frameStep) {
        exportFrames.push_back(i);
    }

    setCaptureUiVisible(false);

    bool cancelled = false;
    QString frameError;
    QVector<QImage> capturedFrames;
    if (normalizedInterpolationMultiplier > 1) {
        capturedFrames.reserve(exportFrames.size());
    }
    for (int i = 0; i < exportFrames.size(); ++i) {
        const double percent = exportFrames.isEmpty()
            ? 0.0
            : ((normalizedInterpolationMultiplier > 1 ? 55.0 : 90.0) *
               static_cast<double>(i) / static_cast<double>(exportFrames.size()));
        if (progressCallback && !progressCallback(i, exportFrames.size(), percent,
                                                  tr("Rendering frame %1 of %2...")
                                                      .arg(i + 1)
                                                      .arg(exportFrames.size()))) {
            cancelled = true;
            break;
        }

        if (!applyFrameIndex(exportFrames[i], true, &frameError)) {
            break;
        }

        const QImage image = captureCurrentImage(&frameError, transparentBackground, videoOutputSize);
        if (image.isNull()) {
            break;
        }

        if (normalizedInterpolationMultiplier > 1) {
            capturedFrames.push_back(image.convertToFormat(QImage::Format_RGBA8888));
        } else {
            const QString fileName = tempDir.filePath(QString("frame_%1.png").arg(i, 5, 10, QLatin1Char('0')));
            if (!saveImageWithVtkWriter(image, fileName, &frameError)) {
                if (frameError.trimmed().isEmpty()) {
                    frameError = QString("Could not write temporary frame '%1'.").arg(fileName);
                }
                break;
            }
        }

        if (progressCallback && !progressCallback(i + 1, exportFrames.size(), percent,
                                                  tr("Rendered frame %1 of %2.")
                                                      .arg(i + 1)
                                                      .arg(exportFrames.size()))) {
            cancelled = true;
            break;
        }
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    }

    bool ok = frameError.isEmpty() && !cancelled;
    int encodedFrameCount = exportFrames.size();
    int encodedFps = std::max(1, fps);
    if (ok && normalizedInterpolationMultiplier > 1) {
        if (progressCallback && !progressCallback(0, exportFrames.size(), 55.0, tr("Interpolating frames..."))) {
            ok = false;
            cancelled = true;
        }
    }
    if (ok && normalizedInterpolationMultiplier > 1) {
        QVector<QImage> interpolatedFrames;
        QString interpolationError;
        ok = Streamcenter::Fruc::NvidiaFrucRuntime::interpolateFrames(
            capturedFrames,
            normalizedInterpolationMultiplier,
            &interpolatedFrames,
            &interpolationError,
            [progressCallback](double localPercent, const QString& detail) {
                const double percent = 55.0 + (30.0 * std::clamp(localPercent, 0.0, 100.0) / 100.0);
                return !progressCallback || progressCallback(0, 0, percent, detail);
            });
        if (!ok) {
            frameError = interpolationError;
        }

        if (ok) {
            encodedFrameCount = static_cast<int>(interpolatedFrames.size());
            encodedFps = std::max(1, fps) * normalizedInterpolationMultiplier;
            const int interpolatedFrameCount = static_cast<int>(interpolatedFrames.size());
            for (int i = 0; i < interpolatedFrameCount; ++i) {
                const double percent = 85.0 + (7.0 * static_cast<double>(i) /
                                               static_cast<double>(std::max(1, interpolatedFrameCount)));
                if (progressCallback && !progressCallback(i, interpolatedFrameCount, percent,
                                                          tr("Writing interpolated frame %1 of %2...")
                                                              .arg(i + 1)
                                                              .arg(interpolatedFrameCount))) {
                    ok = false;
                    cancelled = true;
                    break;
                }

                const QString fileName = tempDir.filePath(QString("frame_%1.png").arg(i, 5, 10, QLatin1Char('0')));
                if (!saveImageWithVtkWriter(interpolatedFrames[i], fileName, &frameError)) {
                    if (frameError.trimmed().isEmpty()) {
                        frameError = QString("Could not write temporary frame '%1'.").arg(fileName);
                    }
                    ok = false;
                    break;
                }
                QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
            }
        }
    }
    if (ok) {
        if (progressCallback && !progressCallback(encodedFrameCount, encodedFrameCount, 92.0, tr("Encoding animation..."))) {
            ok = false;
            cancelled = true;
        }
    }
    if (ok) {
        QProcess ffmpeg;
        QStringList args{
            "-y",
            "-framerate", QString::number(encodedFps),
            "-i", tempDir.filePath("frame_%05d.png"),
        };
        if (suffix == QStringLiteral("gif")) {
            args << "-filter_complex"
                 << "[0:v]split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
                 << "-loop" << "0";
        } else if (transparentBackground && suffix == QStringLiteral("webm")) {
            args << "-c:v" << "libvpx-vp9"
                 << "-vf" << "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=black@0"
                 << "-pix_fmt" << "yuva420p"
                 << "-auto-alt-ref" << "0";
        } else {
            args << "-c:v" << "libx264"
                 << "-vf" << "pad=ceil(iw/2)*2:ceil(ih/2)*2"
                 << "-pix_fmt" << "yuv420p";
        };
        args << path;
        ffmpeg.start(ffmpegPath, args);
        ok = ffmpeg.waitForStarted() && ffmpeg.waitForFinished(-1) &&
             ffmpeg.exitStatus() == QProcess::NormalExit && ffmpeg.exitCode() == 0;
        if (!ok && errorMessage != nullptr) {
            *errorMessage = QString::fromUtf8(ffmpeg.readAllStandardError());
            if (errorMessage->trimmed().isEmpty()) {
                *errorMessage = "ffmpeg failed while encoding the animation file.";
            }
        }
        if (ok && progressCallback) {
            progressCallback(encodedFrameCount, encodedFrameCount, 100.0, tr("Animation encoded."));
        }
    } else if (cancelled && errorMessage != nullptr) {
        *errorMessage = tr("Animation export interrupted.");
    } else if (!frameError.isEmpty() && errorMessage != nullptr) {
        *errorMessage = frameError;
    }

    setCaptureUiVisible(true);
    QString restoreError;
    applyFrameIndex(previousFrame, true, &restoreError);
    if (wasPlaying) {
        startAnimation();
    }
    return ok;
}
