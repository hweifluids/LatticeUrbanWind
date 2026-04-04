#include "luwgui/ProgressPanel.h"

#include <QAbstractAnimation>
#include <QEvent>
#include <QFontMetrics>
#include <QGraphicsOpacityEffect>
#include <QLabel>
#include <QPalette>
#include <QPropertyAnimation>
#include <QProgressBar>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <limits>

namespace {

QString toGerund(QString text) {
    text = text.trimmed();
    if (text.isEmpty()) {
        return text;
    }

    const auto startsWith = [&](const QString& prefix) {
        return text.startsWith(prefix, Qt::CaseInsensitive);
    };

    if (text.endsWith("ing", Qt::CaseInsensitive)) {
        return text;
    }
    if (text.compare("Solve", Qt::CaseInsensitive) == 0) {
        return QStringLiteral("Solving");
    }
    if (startsWith("Generate ")) {
        return QStringLiteral("Generating ") + text.mid(9);
    }
    if (startsWith("Build ")) {
        return QStringLiteral("Building ") + text.mid(6);
    }
    if (startsWith("Crop ")) {
        return QStringLiteral("Cropping ") + text.mid(5);
    }
    if (startsWith("Inspect ")) {
        return QStringLiteral("Inspecting ") + text.mid(8);
    }
    if (startsWith("Validate ")) {
        return QStringLiteral("Validating ") + text.mid(9);
    }
    if (startsWith("Prepare ")) {
        return QStringLiteral("Preparing ") + text.mid(8);
    }
    if (startsWith("Export ")) {
        return QStringLiteral("Exporting ") + text.mid(7);
    }
    if (startsWith("Clean ")) {
        return QStringLiteral("Cleaning ") + text.mid(6);
    }
    if (startsWith("Load ")) {
        return QStringLiteral("Loading ") + text.mid(5);
    }
    if (startsWith("Save ")) {
        return QStringLiteral("Saving ") + text.mid(5);
    }
    if (startsWith("Run ")) {
        return QStringLiteral("Running ") + text.mid(4);
    }
    return text;
}

QString compactDetail(QString detail, const QString& summary) {
    detail = detail.simplified();
    if (detail.isEmpty()) {
        return detail;
    }

    const QString gerund = toGerund(summary);
    const QStringList prefixes = {
        summary.simplified(),
        gerund.simplified()
    };
    for (const QString& prefix : prefixes) {
        if (prefix.isEmpty()) {
            continue;
        }
        if (detail.startsWith(prefix, Qt::CaseInsensitive)) {
            QString trimmed = detail.mid(prefix.size()).trimmed();
            while (!trimmed.isEmpty() && QStringLiteral(":,-").contains(trimmed.front())) {
                trimmed.remove(0, 1);
                trimmed = trimmed.trimmed();
            }
            if (!trimmed.isEmpty()) {
                return trimmed;
            }
        }
    }
    return detail;
}

QString formatStatusText(const QString& summary, const QString& detail) {
    const QString stage = toGerund(summary).simplified();
    const QString progress = compactDetail(detail, summary);
    if (stage.isEmpty() && progress.isEmpty()) {
        return {};
    }
    if (stage.isEmpty()) {
        return QStringLiteral("(%1)").arg(progress);
    }
    if (progress.isEmpty()) {
        return QStringLiteral("%1...").arg(stage);
    }
    return QStringLiteral("%1... (%2)").arg(stage, progress);
}

QString formatPlainStatusText(const QString& summary, const QString& detail) {
    const QString headline = summary.simplified();
    const QString progress = detail.simplified();
    if (headline.isEmpty() && progress.isEmpty()) {
        return {};
    }
    if (headline.isEmpty()) {
        return QStringLiteral("(%1)").arg(progress);
    }
    if (progress.isEmpty()) {
        return headline;
    }
    return QStringLiteral("%1 (%2)").arg(headline, progress);
}

}

namespace luwgui {

ProgressPanel::ProgressPanel(QWidget* parent)
    : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 2, 0, 3);
    layout->setSpacing(2);

    bar_ = new QProgressBar(this);
    bar_->setFixedHeight(5);
    bar_->setTextVisible(false);
    bar_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    layout->addWidget(bar_);

    statusLabel_ = new QLabel(this);
    statusLabel_->setProperty("muted", true);
    statusLabel_->setTextFormat(Qt::PlainText);
    statusLabel_->setWordWrap(false);
    statusLabel_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    statusLabel_->setIndent(6);
    statusLabel_->setMinimumWidth(0);
    statusLabel_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    opacityEffect_ = new QGraphicsOpacityEffect(statusLabel_);
    opacityEffect_->setOpacity(1.0);
    statusLabel_->setGraphicsEffect(opacityEffect_);
    layout->addWidget(statusLabel_);

    fadeDelayTimer_ = new QTimer(this);
    fadeDelayTimer_->setSingleShot(true);
    fadeDelayTimer_->setInterval(5000);

    fadeAnimation_ = new QPropertyAnimation(opacityEffect_, "opacity", this);
    fadeAnimation_->setDuration(180);
    fadeAnimation_->setStartValue(1.0);
    fadeAnimation_->setEndValue(0.0);

    connect(fadeDelayTimer_, &QTimer::timeout, this, [this] {
        if (!fadeAnimation_) {
            return;
        }
        fadeAnimation_->stop();
        fadeAnimation_->setStartValue(opacityEffect_ ? opacityEffect_->opacity() : 1.0);
        fadeAnimation_->setEndValue(0.0);
        fadeAnimation_->start();
    });
    connect(fadeAnimation_, &QPropertyAnimation::finished, this, &ProgressPanel::clearStatusAfterFade);

    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    refreshStatusLabel();
    setIdle();
}

void ProgressPanel::setIdle(const QString& summary, const QString& detail) {
    resetFadeState();
    fullStatusText_ = formatStatusText(summary, detail);
    refreshStatusLabel();
    bar_->setRange(0, 1);
    bar_->setValue(0);
    updatePresentation(false);
}

void ProgressPanel::setBusy(const QString& summary, const QString& detail) {
    resetFadeState();
    fullStatusText_ = formatStatusText(summary, detail);
    refreshStatusLabel();
    applyProgress(-1, -1, true);
    updatePresentation(true);
}

void ProgressPanel::setProgress(const QString& summary,
                                const QString& detail,
                                qint64 current,
                                qint64 total,
                                bool indeterminate) {
    resetFadeState();
    fullStatusText_ = formatStatusText(summary, detail);
    refreshStatusLabel();
    applyProgress(current, total, indeterminate);
    updatePresentation(true);
}

void ProgressPanel::showTerminalStatus(const QString& summary, const QString& detail) {
    resetFadeState();
    fullStatusText_ = formatPlainStatusText(summary, detail);
    refreshStatusLabel();
    bar_->setRange(0, 1);
    bar_->setValue(0);
    updatePresentation(false);
    if (fadeDelayTimer_) {
        fadeDelayTimer_->start();
    }
}

void ProgressPanel::applyProgress(qint64 current, qint64 total, bool indeterminate) {
    if (indeterminate || current < 0 || total <= 0) {
        bar_->setRange(0, 0);
        return;
    }

    if (total <= std::numeric_limits<int>::max()) {
        bar_->setRange(0, static_cast<int>(total));
        bar_->setValue(static_cast<int>(std::clamp<qint64>(current, 0, total)));
        return;
    }

    constexpr int kScaledMaximum = 1000;
    const double ratio = static_cast<double>(std::clamp<qint64>(current, 0, total))
        / static_cast<double>(total);
    bar_->setRange(0, kScaledMaximum);
    bar_->setValue(static_cast<int>(ratio * static_cast<double>(kScaledMaximum)));
}

void ProgressPanel::updatePresentation(bool active) {
    const QString background = palette().color(QPalette::Window).name();
    const QString border = active
        ? palette().color(QPalette::Mid).name()
        : background;
    const QColor chunkColor = palette().color(QPalette::LinkVisited);
    const QString chunk = active
        ? chunkColor.name()
        : background;

    bar_->setStyleSheet(QStringLiteral(
        "QProgressBar {"
        " border: 1px solid %1;"
        " background-color: %2;"
        " border-radius: 1px;"
        "}"
        "QProgressBar::chunk {"
        " background-color: %3;"
        " margin: 0px;"
        "}"
    ).arg(border, background, chunk));
    updateGeometry();
}

void ProgressPanel::changeEvent(QEvent* event) {
    QWidget::changeEvent(event);
    if (!event) {
        return;
    }
    if (event->type() == QEvent::FontChange || event->type() == QEvent::StyleChange) {
        refreshStatusLabel();
    }
}

void ProgressPanel::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    refreshStatusLabel();
}

void ProgressPanel::refreshStatusLabel() {
    if (!statusLabel_) {
        return;
    }

    const QFontMetrics metrics(statusLabel_->font());
    statusLabel_->setFixedHeight(metrics.height() + 6);

    if (fullStatusText_.isEmpty()) {
        statusLabel_->setText(QStringLiteral(" "));
        statusLabel_->setToolTip(QString());
        return;
    }

    const int availableWidth = std::max(statusLabel_->width() - 12, 0);
    const QString elided = availableWidth > 0
        ? metrics.elidedText(fullStatusText_, Qt::ElideRight, availableWidth)
        : fullStatusText_;
    const bool truncated = elided != fullStatusText_;
    statusLabel_->setText(elided);
    statusLabel_->setToolTip(truncated ? fullStatusText_ : QString());
}

void ProgressPanel::resetFadeState() {
    if (fadeDelayTimer_) {
        fadeDelayTimer_->stop();
    }
    if (fadeAnimation_) {
        fadeAnimation_->stop();
    }
    if (opacityEffect_) {
        opacityEffect_->setOpacity(1.0);
    }
}

void ProgressPanel::clearStatusAfterFade() {
    if (fadeAnimation_ && fadeAnimation_->state() == QAbstractAnimation::Running) {
        return;
    }
    fullStatusText_.clear();
    refreshStatusLabel();
    if (opacityEffect_) {
        opacityEffect_->setOpacity(1.0);
    }
}

}
