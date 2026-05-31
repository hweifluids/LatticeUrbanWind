#pragma once

#include "index/IndexRuntime.h"
#include "index/IndexVolumeData.h"

#include <QMap>
#include <QString>

class vtkGenericOpenGLRenderWindow;
class vtkRenderer;

namespace Streamcenter::Index {

class IndexVolumeBackend {
public:
    RuntimeStatus runtimeStatus() const;

    bool loadOrUpdate(const QString& objectId,
                      const QVector<VolumeFrame>& frames,
                      const VolumeOptions& options,
                      QString* errorMessage);
    void remove(const QString& objectId);
    void clear();
    bool render(vtkRenderer* renderer,
                vtkGenericOpenGLRenderWindow* renderWindow,
                int frameIndex,
                QString* errorMessage);

private:
    struct ObjectState {
        VolumeSeriesData series;
        VolumeOptions options;
    };

    QMap<QString, ObjectState> objects_;
};

}  // namespace Streamcenter::Index
