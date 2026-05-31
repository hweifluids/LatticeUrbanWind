#pragma once

#include <QImage>
#include <QString>
#include <QVector>

#include <functional>

namespace Streamcenter::Fruc {

struct RuntimeStatus {
    bool available = false;
    QString message;
};

using InterpolationProgressCallback = std::function<bool(double percent, const QString& detail)>;

class NvidiaFrucRuntime {
public:
    static RuntimeStatus status();

    static bool interpolateFrames(const QVector<QImage>& inputFrames,
                                  int multiplier,
                                  QVector<QImage>* outputFrames,
                                  QString* errorMessage = nullptr,
                                  const InterpolationProgressCallback& progressCallback = {});
};

}  // namespace Streamcenter::Fruc
