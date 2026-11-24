#include "luw/progress.hpp"

#include <QJsonDocument>
#include <QJsonObject>
#include <QProcessEnvironment>

#include <algorithm>
#include <cstdio>

namespace luw {

ProgressEmitter::ProgressEmitter(QString defaultLabel)
    : defaultLabel_(std::move(defaultLabel))
    , enabled_(QProcessEnvironment::systemEnvironment()
                   .value(QStringLiteral("LUW_PROGRESS_MODE"))
                   .trimmed()
                   .compare(QStringLiteral("gui"), Qt::CaseInsensitive) == 0) {
    monotonic_.start();
}

bool ProgressEmitter::enabled() const noexcept { return enabled_; }

void ProgressEmitter::stage(const QString& stageName, const QString& detail, const QString& label) {
    emitPayload(stageName, detail, label, std::nullopt, std::nullopt, true, true);
}

void ProgressEmitter::progress(const QString& stageName,
                               std::int64_t current,
                               std::int64_t total,
                               const QString& detail,
                               const QString& label,
                               bool force) {
    emitPayload(stageName, detail, label, current, total, false,
                force || total <= 1 || current <= 1 || current >= total);
}

void ProgressEmitter::complete(const QString& stageName, const QString& detail, const QString& label) {
    emitPayload(stageName, detail, label, 1, 1, false, true);
}

void ProgressEmitter::emitPayload(const QString& stageName,
                                  const QString& detail,
                                  const QString& label,
                                  std::optional<std::int64_t> current,
                                  std::optional<std::int64_t> total,
                                  std::optional<bool> indeterminate,
                                  bool force) {
    if (!enabled_) return;
    const QString summary = label.isEmpty() ? defaultLabel_ : label;
    const bool isIndeterminate = indeterminate.value_or(
        !current || !total || *total <= 0);
    std::optional<std::int64_t> bucket;
    if (!isIndeterminate && current && total && *total > 0) {
        bucket = (std::clamp(*current, std::int64_t{0}, *total) * 200) / *total;
    }
    const qint64 now = monotonic_.elapsed();
    if (!force
        && stageName == lastStage_
        && summary == lastLabel_
        && detail == lastDetail_
        && bucket == lastBucket_
        && now - lastEmitMilliseconds_ < 150) {
        return;
    }

    QJsonObject object;
    object.insert(QStringLiteral("stage"), stageName);
    object.insert(QStringLiteral("label"), summary);
    object.insert(QStringLiteral("detail"), detail);
    object.insert(QStringLiteral("indeterminate"), isIndeterminate);
    if (current) object.insert(QStringLiteral("current"), static_cast<qint64>(*current));
    if (total) object.insert(QStringLiteral("total"), static_cast<qint64>(*total));
    const QByteArray line = QByteArrayLiteral("[[LUW_PROGRESS]]")
        + QJsonDocument(object).toJson(QJsonDocument::Compact) + '\n';
    std::fwrite(line.constData(), 1, static_cast<std::size_t>(line.size()), stdout);
    std::fflush(stdout);

    lastStage_ = stageName;
    lastLabel_ = summary;
    lastDetail_ = detail;
    lastBucket_ = bucket;
    lastEmitMilliseconds_ = now;
}

} // namespace luw
