#include "index/IndexVolumeData.h"

#include <QFileInfo>

#include <vtkDataArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkXMLImageDataReader.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <utility>

namespace Streamcenter::Index {
namespace {

constexpr double kGeometryTolerance = 1.0e-10;

bool sameDouble(double a, double b) {
    return std::abs(a - b) <= kGeometryTolerance * std::max({1.0, std::abs(a), std::abs(b)});
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

QString componentNameForAxis(int axis) {
    switch (axis) {
    case 0:
        return QStringLiteral("X");
    case 1:
        return QStringLiteral("Y");
    case 2:
        return QStringLiteral("Z");
    default:
        return QStringLiteral("Magnitude");
    }
}

bool parseComponentFieldName(const QString& name, QString* baseName, int* axis) {
    if (name.size() < 3 || name.at(name.size() - 2) != QLatin1Char('_')) {
        return false;
    }

    int parsedAxis = -1;
    const QChar suffix = name.at(name.size() - 1).toLower();
    if (suffix == QLatin1Char('x')) {
        parsedAxis = 0;
    } else if (suffix == QLatin1Char('y')) {
        parsedAxis = 1;
    } else if (suffix == QLatin1Char('z')) {
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
    return true;
}

struct ComponentGroup {
    QString baseName;
    vtkDataArray* arrays[3] = {nullptr, nullptr, nullptr};
    int firstOrder = std::numeric_limits<int>::max();
};

QVector<ComponentGroup> collectComponentGroups(vtkPointData* pointData) {
    QVector<ComponentGroup> groups;
    if (pointData == nullptr) {
        return groups;
    }

    for (int index = 0; index < pointData->GetNumberOfArrays(); ++index) {
        vtkDataArray* array = pointData->GetArray(index);
        if (array == nullptr || array->GetName() == nullptr || array->GetNumberOfComponents() != 1) {
            continue;
        }
        QString baseName;
        int axis = -1;
        if (!parseComponentFieldName(QString::fromUtf8(array->GetName()), &baseName, &axis)) {
            continue;
        }

        ComponentGroup* group = nullptr;
        for (ComponentGroup& candidate : groups) {
            if (candidate.baseName == baseName) {
                group = &candidate;
                break;
            }
        }
        if (group == nullptr) {
            ComponentGroup newGroup;
            newGroup.baseName = baseName;
            groups.push_back(newGroup);
            group = &groups.back();
        }
        if (group->arrays[axis] == nullptr) {
            group->arrays[axis] = array;
            group->firstOrder = std::min(group->firstOrder, index);
        }
    }
    return groups;
}

bool validComponentGroup(const ComponentGroup& group, vtkIdType expectedTuples) {
    int present = 0;
    for (vtkDataArray* array : group.arrays) {
        if (array != nullptr) {
            ++present;
            if (array->GetNumberOfTuples() != expectedTuples) {
                return false;
            }
        }
    }
    return present >= 2;
}

vtkDataArray* firstUsablePointArray(vtkPointData* pointData) {
    if (pointData == nullptr) {
        return nullptr;
    }
    for (int index = 0; index < pointData->GetNumberOfArrays(); ++index) {
        vtkDataArray* array = pointData->GetArray(index);
        if (array != nullptr && array->GetName() != nullptr && array->GetNumberOfComponents() >= 1) {
            return array;
        }
    }
    return nullptr;
}

struct ResolvedArray {
    vtkDataArray* directArray = nullptr;
    ComponentGroup componentGroup;
    bool usesComponentGroup = false;
    QString scalarName;
};

ResolvedArray resolvePointArray(vtkImageData* image, const VolumeOptions& options, QString* errorMessage) {
    ResolvedArray resolved;
    vtkPointData* pointData = image != nullptr ? image->GetPointData() : nullptr;
    if (pointData == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("The VTI file does not contain point data.");
        }
        return resolved;
    }

    const QString requestedField = options.fieldName.trimmed();
    if (!requestedField.isEmpty()) {
        const QByteArray key = requestedField.toUtf8();
        if (vtkDataArray* array = pointData->GetArray(key.constData())) {
            resolved.directArray = array;
            resolved.scalarName = requestedField;
            return resolved;
        }

        const QVector<ComponentGroup> groups = collectComponentGroups(pointData);
        for (const ComponentGroup& group : groups) {
            if (group.baseName == requestedField && validComponentGroup(group, image->GetNumberOfPoints())) {
                resolved.componentGroup = group;
                resolved.usesComponentGroup = true;
                resolved.scalarName = requestedField;
                return resolved;
            }
        }

        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Point scalar field was not found in the VTI file: %1").arg(requestedField);
        }
        return resolved;
    }

    if (vtkDataArray* array = firstUsablePointArray(pointData)) {
        resolved.directArray = array;
        resolved.scalarName = QString::fromUtf8(array->GetName());
        return resolved;
    }

    if (errorMessage != nullptr) {
        *errorMessage = QStringLiteral("The VTI file does not contain any numeric point arrays.");
    }
    return resolved;
}

bool appendDirectArrayScalars(vtkDataArray* array,
                              const QString& componentName,
                              std::vector<float>* values,
                              QString* resolvedComponent,
                              QString* errorMessage) {
    if (array == nullptr || values == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Internal error: scalar array is null.");
        }
        return false;
    }

    const vtkIdType tupleCount = array->GetNumberOfTuples();
    const int componentCount = array->GetNumberOfComponents();
    if (tupleCount <= 0 || componentCount <= 0) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("The selected point array is empty.");
        }
        return false;
    }

    values->resize(static_cast<size_t>(tupleCount));
    const int requestedAxis = axisForComponentName(componentName);
    if (componentCount == 1) {
        for (vtkIdType tuple = 0; tuple < tupleCount; ++tuple) {
            (*values)[static_cast<size_t>(tuple)] = static_cast<float>(array->GetComponent(tuple, 0));
        }
        if (resolvedComponent != nullptr) {
            *resolvedComponent = QStringLiteral("Magnitude");
        }
        return true;
    }

    if (requestedAxis >= 0) {
        if (requestedAxis >= componentCount) {
            if (errorMessage != nullptr) {
                *errorMessage = QStringLiteral("The selected vector field does not have a %1 component.")
                                    .arg(componentName.trimmed());
            }
            return false;
        }
        for (vtkIdType tuple = 0; tuple < tupleCount; ++tuple) {
            (*values)[static_cast<size_t>(tuple)] = static_cast<float>(array->GetComponent(tuple, requestedAxis));
        }
        if (resolvedComponent != nullptr) {
            *resolvedComponent = componentNameForAxis(requestedAxis);
        }
        return true;
    }

    for (vtkIdType tuple = 0; tuple < tupleCount; ++tuple) {
        double sum = 0.0;
        for (int component = 0; component < componentCount; ++component) {
            const double value = array->GetComponent(tuple, component);
            sum += value * value;
        }
        (*values)[static_cast<size_t>(tuple)] = static_cast<float>(std::sqrt(sum));
    }
    if (resolvedComponent != nullptr) {
        *resolvedComponent = QStringLiteral("Magnitude");
    }
    return true;
}

bool appendComponentGroupScalars(const ComponentGroup& group,
                                 vtkIdType tupleCount,
                                 const QString& componentName,
                                 std::vector<float>* values,
                                 QString* resolvedComponent,
                                 QString* errorMessage) {
    if (!validComponentGroup(group, tupleCount) || values == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("The selected component field group is incomplete.");
        }
        return false;
    }

    values->resize(static_cast<size_t>(tupleCount));
    const int requestedAxis = axisForComponentName(componentName);
    if (requestedAxis >= 0) {
        vtkDataArray* component = group.arrays[requestedAxis];
        if (component == nullptr) {
            if (errorMessage != nullptr) {
                *errorMessage = QStringLiteral("The selected component field group does not have a %1 component.")
                                    .arg(componentName.trimmed());
            }
            return false;
        }
        for (vtkIdType tuple = 0; tuple < tupleCount; ++tuple) {
            (*values)[static_cast<size_t>(tuple)] = static_cast<float>(component->GetComponent(tuple, 0));
        }
        if (resolvedComponent != nullptr) {
            *resolvedComponent = componentNameForAxis(requestedAxis);
        }
        return true;
    }

    for (vtkIdType tuple = 0; tuple < tupleCount; ++tuple) {
        double sum = 0.0;
        for (vtkDataArray* component : group.arrays) {
            if (component == nullptr) {
                continue;
            }
            const double value = component->GetComponent(tuple, 0);
            sum += value * value;
        }
        (*values)[static_cast<size_t>(tuple)] = static_cast<float>(std::sqrt(sum));
    }
    if (resolvedComponent != nullptr) {
        *resolvedComponent = QStringLiteral("Magnitude");
    }
    return true;
}

void rangeForValues(const std::vector<float>& values, double outRange[2]) {
    if (values.empty()) {
        outRange[0] = 0.0;
        outRange[1] = 1.0;
        return;
    }

    double minValue = std::numeric_limits<double>::infinity();
    double maxValue = -std::numeric_limits<double>::infinity();
    for (float value : values) {
        const double asDouble = static_cast<double>(value);
        if (!std::isfinite(asDouble)) {
            continue;
        }
        minValue = std::min(minValue, asDouble);
        maxValue = std::max(maxValue, asDouble);
    }
    if (!std::isfinite(minValue) || !std::isfinite(maxValue)) {
        minValue = 0.0;
        maxValue = 1.0;
    }
    if (!(maxValue > minValue)) {
        maxValue = minValue + 1.0;
    }
    outRange[0] = minValue;
    outRange[1] = maxValue;
}

bool readVtiFrame(const VolumeFrame& frame,
                  const VolumeOptions& options,
                  VolumeFrameData* outFrame,
                  VolumeSeriesData* seriesTemplate,
                  QString* errorMessage) {
    const QFileInfo info(frame.path);
    if (!info.exists() || !info.isFile()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("VTI frame does not exist: %1").arg(frame.path);
        }
        return false;
    }
    if (info.suffix().compare(QStringLiteral("vti"), Qt::CaseInsensitive) != 0) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Ray-tracing volume accepts only .vti frames; rejected: %1").arg(frame.path);
        }
        return false;
    }

    vtkNew<vtkXMLImageDataReader> reader;
    reader->SetFileName(info.absoluteFilePath().toUtf8().constData());
    reader->Update();
    vtkImageData* image = reader->GetOutput();
    if (image == nullptr || image->GetNumberOfPoints() <= 0) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Could not read VTI image data: %1").arg(frame.path);
        }
        return false;
    }

    ResolvedArray resolved = resolvePointArray(image, options, errorMessage);
    if (resolved.directArray == nullptr && !resolved.usesComponentGroup) {
        return false;
    }

    VolumeFrameData frameData;
    frameData.timestep = frame.timestep;
    frameData.path = info.absoluteFilePath();
    QString resolvedComponent;
    const bool copied = resolved.usesComponentGroup
        ? appendComponentGroupScalars(resolved.componentGroup,
                                      image->GetNumberOfPoints(),
                                      options.componentName,
                                      &frameData.scalars,
                                      &resolvedComponent,
                                      errorMessage)
        : appendDirectArrayScalars(resolved.directArray,
                                   options.componentName,
                                   &frameData.scalars,
                                   &resolvedComponent,
                                   errorMessage);
    if (!copied) {
        return false;
    }

    if (static_cast<vtkIdType>(frameData.scalars.size()) != image->GetNumberOfPoints()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Selected point array tuple count does not match VTI point count: %1").arg(frame.path);
        }
        return false;
    }

    int dimensions[3] = {0, 0, 0};
    int extent[6] = {0, -1, 0, -1, 0, -1};
    double origin[3] = {0.0, 0.0, 0.0};
    double spacing[3] = {1.0, 1.0, 1.0};
    image->GetDimensions(dimensions);
    image->GetExtent(extent);
    image->GetOrigin(origin);
    image->GetSpacing(spacing);

    if (seriesTemplate->frames.isEmpty()) {
        std::copy(std::begin(dimensions), std::end(dimensions), std::begin(seriesTemplate->dimensions));
        std::copy(std::begin(extent), std::end(extent), std::begin(seriesTemplate->extent));
        std::copy(std::begin(origin), std::end(origin), std::begin(seriesTemplate->origin));
        std::copy(std::begin(spacing), std::end(spacing), std::begin(seriesTemplate->spacing));
        seriesTemplate->scalarName = resolved.scalarName;
        seriesTemplate->componentName = resolvedComponent;
    } else {
        for (int axis = 0; axis < 3; ++axis) {
            if (seriesTemplate->dimensions[axis] != dimensions[axis]
                || !sameDouble(seriesTemplate->origin[axis], origin[axis])
                || !sameDouble(seriesTemplate->spacing[axis], spacing[axis])) {
                if (errorMessage != nullptr) {
                    *errorMessage = QStringLiteral("PVD VTI frames must share dimensions, origin, and spacing; mismatch in: %1")
                                        .arg(frame.path);
                }
                return false;
            }
        }
        for (int index = 0; index < 6; ++index) {
            if (seriesTemplate->extent[index] != extent[index]) {
                if (errorMessage != nullptr) {
                    *errorMessage = QStringLiteral("PVD VTI frames must share extents; mismatch in: %1").arg(frame.path);
                }
                return false;
            }
        }
        if (seriesTemplate->scalarName != resolved.scalarName || seriesTemplate->componentName != resolvedComponent) {
            if (errorMessage != nullptr) {
                *errorMessage = QStringLiteral("PVD VTI frames must expose the same selected point field/component.");
            }
            return false;
        }
    }

    rangeForValues(frameData.scalars, frameData.scalarRange);
    *outFrame = std::move(frameData);
    return true;
}

}  // namespace

bool validateVolumeSeries(const QVector<VolumeFrame>& frames,
                          const VolumeOptions& options,
                          VolumeSeriesData* outSeries,
                          QString* errorMessage) {
    if (outSeries == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Internal error: volume series output pointer is null.");
        }
        return false;
    }
    if (frames.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Ray-tracing volume has no VTI frames to render.");
        }
        return false;
    }
    if (!(options.samplingStep > 0.0)) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Volume sampling step must be greater than zero.");
        }
        return false;
    }

    VolumeSeriesData series;
    series.frames.reserve(frames.size());
    for (const VolumeFrame& frame : frames) {
        VolumeFrameData frameData;
        if (!readVtiFrame(frame, options, &frameData, &series, errorMessage)) {
            return false;
        }
        series.frames.push_back(std::move(frameData));
    }

    double globalRange[2] = {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    for (const VolumeFrameData& frame : std::as_const(series.frames)) {
        globalRange[0] = std::min(globalRange[0], frame.scalarRange[0]);
        globalRange[1] = std::max(globalRange[1], frame.scalarRange[1]);
    }
    if (!std::isfinite(globalRange[0]) || !std::isfinite(globalRange[1]) || !(globalRange[1] > globalRange[0])) {
        globalRange[0] = 0.0;
        globalRange[1] = 1.0;
    }
    if (!options.autoColorRange && options.colorRangeMax > options.colorRangeMin) {
        series.scalarRange[0] = options.colorRangeMin;
        series.scalarRange[1] = options.colorRangeMax;
    } else {
        series.scalarRange[0] = globalRange[0];
        series.scalarRange[1] = globalRange[1];
    }

    *outSeries = std::move(series);
    return true;
}

}  // namespace Streamcenter::Index
