#pragma once

#include <QElapsedTimer>
#include <QString>

#include <cstdint>
#include <optional>

namespace luw {

class ProgressEmitter final {
public:
    explicit ProgressEmitter(QString defaultLabel);

    bool enabled() const noexcept;
    void stage(const QString& stage, const QString& detail = {}, const QString& label = {});
    void progress(const QString& stage,
                  std::int64_t current,
                  std::int64_t total,
                  const QString& detail = {},
                  const QString& label = {},
                  bool force = false);
    void complete(const QString& stage, const QString& detail = {}, const QString& label = {});

private:
    void emitPayload(const QString& stage,
                     const QString& detail,
                     const QString& label,
                     std::optional<std::int64_t> current,
                     std::optional<std::int64_t> total,
                     std::optional<bool> indeterminate,
                     bool force);

    QString defaultLabel_;
    bool enabled_ = false;
    QString lastStage_;
    QString lastLabel_;
    QString lastDetail_;
    std::optional<std::int64_t> lastBucket_;
    QElapsedTimer monotonic_;
    qint64 lastEmitMilliseconds_ = 0;
};

} // namespace luw
