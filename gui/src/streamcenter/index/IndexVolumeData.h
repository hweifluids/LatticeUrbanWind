#pragma once

#include <QColor>
#include <QString>
#include <QVector>

#include <array>
#include <vector>

namespace Streamcenter::Index {

struct VolumeFrame {
    double timestep = 0.0;
    QString path;
};

struct VolumeOptions {
    QString fieldName;
    QString componentName = QStringLiteral("Magnitude");
    QString colorMap;
    bool autoColorRange = true;
    double colorRangeMin = 0.0;
    double colorRangeMax = 1.0;
    double opacityScale = 2.0;
    double samplingStep = 1.0;
    QString filtering = QStringLiteral("Linear");
    bool preintegration = true;
};

struct VolumeFrameData {
    double timestep = 0.0;
    QString path;
    std::vector<float> scalars;
    double scalarRange[2] = {0.0, 1.0};
};

struct VolumeSeriesData {
    QString scalarName;
    QString componentName;
    int dimensions[3] = {0, 0, 0};
    int extent[6] = {0, -1, 0, -1, 0, -1};
    double origin[3] = {0.0, 0.0, 0.0};
    double spacing[3] = {1.0, 1.0, 1.0};
    double scalarRange[2] = {0.0, 1.0};
    QVector<VolumeFrameData> frames;
};

bool validateVolumeSeries(const QVector<VolumeFrame>& frames,
                          const VolumeOptions& options,
                          VolumeSeriesData* outSeries,
                          QString* errorMessage);

}  // namespace Streamcenter::Index
