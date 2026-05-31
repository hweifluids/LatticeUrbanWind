#include "index/IndexScene.h"

namespace Streamcenter::Index {

QString describeSceneParameters(const VolumeSeriesData& series, const IndexSceneParameters& parameters) {
    return QStringLiteral("%1 %2, %3x%4x%5, range [%6, %7], step %8")
        .arg(series.scalarName,
             series.componentName,
             QString::number(series.dimensions[0]),
             QString::number(series.dimensions[1]),
             QString::number(series.dimensions[2]),
             QString::number(parameters.scalarRange[0], 'g', 6),
             QString::number(parameters.scalarRange[1], 'g', 6),
             QString::number(parameters.options.samplingStep, 'g', 6));
}

}  // namespace Streamcenter::Index
