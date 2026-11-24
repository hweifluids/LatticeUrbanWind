#pragma once

#include "index/IndexVolumeData.h"

#include <QString>

namespace Streamcenter::Index {

struct IndexSceneParameters {
    VolumeOptions options;
    double scalarRange[2] = {0.0, 1.0};
};

QString describeSceneParameters(const VolumeSeriesData& series, const IndexSceneParameters& parameters);

}  // namespace Streamcenter::Index
