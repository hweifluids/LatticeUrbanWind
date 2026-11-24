#include "index/IndexVolumeBackend.h"

#include "index/IndexOpenGLCanvas.h"
#include "index/IndexRuntime.h"
#include "index/IndexScene.h"

#ifndef STREAMCENTERPLUS_ENABLE_NVIDIA_INDEX
#define STREAMCENTERPLUS_ENABLE_NVIDIA_INDEX 0
#endif

namespace Streamcenter::Index {

RuntimeStatus IndexVolumeBackend::runtimeStatus() const {
    return IndexRuntime::status();
}

bool IndexVolumeBackend::loadOrUpdate(const QString& objectId,
                                      const QVector<VolumeFrame>& frames,
                                      const VolumeOptions& options,
                                      QString* errorMessage) {
    if (objectId.trimmed().isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Internal error: NVIDIA IndeX object id is empty.");
        }
        return false;
    }

    const RuntimeStatus status = runtimeStatus();
    if (!status.available) {
        if (errorMessage != nullptr) {
            *errorMessage = status.message;
        }
        return false;
    }

    VolumeSeriesData series;
    if (!validateVolumeSeries(frames, options, &series, errorMessage)) {
        return false;
    }

#if STREAMCENTERPLUS_ENABLE_NVIDIA_INDEX
    IndexSceneParameters parameters;
    parameters.options = options;
    parameters.scalarRange[0] = series.scalarRange[0];
    parameters.scalarRange[1] = series.scalarRange[1];
    Q_UNUSED(describeSceneParameters(series, parameters));
    if (errorMessage != nullptr) {
        *errorMessage = QStringLiteral("NVIDIA IndeX runtime is available, but the Streamcenter+ native scene binding is not implemented yet. The isolated data, runtime, and OpenGL bridge are ready for the SDK-specific scene adapter.");
    }
    return false;
#else
    Q_UNUSED(series);
    Q_UNUSED(options);
    if (errorMessage != nullptr && errorMessage->isEmpty()) {
        *errorMessage = status.message;
    }
    return false;
#endif
}

void IndexVolumeBackend::remove(const QString& objectId) {
    objects_.remove(objectId);
}

void IndexVolumeBackend::clear() {
    objects_.clear();
}

bool IndexVolumeBackend::render(vtkRenderer* renderer,
                                vtkGenericOpenGLRenderWindow* renderWindow,
                                int frameIndex,
                                QString* errorMessage) {
    Q_UNUSED(frameIndex);
    if (objects_.isEmpty()) {
        return true;
    }
    if (!IndexOpenGLCanvasBridge::validate(renderer, renderWindow, errorMessage)) {
        return false;
    }
    return true;
}

}  // namespace Streamcenter::Index
