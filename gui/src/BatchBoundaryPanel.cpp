#include "luwgui/BatchBoundaryPanel.h"

#include "luwgui/PlotWidgets.h"

#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFocusEvent>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QRegularExpression>
#include <QSignalBlocker>
#include <QTextStream>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace luwgui {

namespace {

QStringList splitLooseVectorTokens(QString text) {
    text.replace('[', ' ');
    text.replace(']', ' ');
    return text.split(QRegularExpression(R"([,\s]+)"), Qt::SkipEmptyParts);
}

QString canonicalizeLooseVectorText(const QString& text) {
    return splitLooseVectorTokens(text).join(", ");
}

QString formatFloatTokenForDisplay(const QString& token) {
    bool ok = false;
    const double value = token.trimmed().toDouble(&ok);
    return ok ? QString::number(value, 'f', 2) : token.trimmed();
}

QString formatFloatList(const QVariantList& values) {
    QStringList parts;
    for (const QVariant& value : values) {
        parts.push_back(QString::number(value.toDouble(), 'f', 2));
    }
    return parts.join(", ");
}

QString formatFloatVectorForDisplay(const QString& text) {
    const QStringList tokens = splitLooseVectorTokens(text);
    if (tokens.isEmpty()) {
        return {};
    }

    QStringList displayTokens;
    displayTokens.reserve(tokens.size());
    for (const QString& token : tokens) {
        displayTokens.push_back(formatFloatTokenForDisplay(token));
    }
    return displayTokens.join(", ");
}

QString renderBracketedListText(const QString& text) {
    const QStringList tokens = splitLooseVectorTokens(text);
    if (tokens.isEmpty()) {
        return {};
    }
    return "[" + tokens.join(", ") + "]";
}

class PrecisionFloatListLineEdit final : public QLineEdit {
public:
    explicit PrecisionFloatListLineEdit(QWidget* parent = nullptr)
        : QLineEdit(parent) {
    }

    void setRawText(const QString& rawText) {
        fullText_ = canonicalizeLooseVectorText(rawText);
        applyPresentation();
    }

    QString canonicalText() const {
        if (hasFocus()) {
            return text().trimmed();
        }
        return fullText_.isEmpty() ? text().trimmed() : fullText_;
    }

    QString commitVisibleText() {
        fullText_ = canonicalizeLooseVectorText(text());
        return fullText_;
    }

protected:
    void focusInEvent(QFocusEvent* event) override {
        QLineEdit::focusInEvent(event);
        const QSignalBlocker blocker(this);
        setText(fullText_);
        selectAll();
    }

    void focusOutEvent(QFocusEvent* event) override {
        fullText_ = canonicalizeLooseVectorText(text());
        QLineEdit::focusOutEvent(event);
        applyPresentation();
    }

private:
    void applyPresentation() {
        const QSignalBlocker blocker(this);
        setText(hasFocus() ? fullText_ : formatFloatVectorForDisplay(fullText_));
        if (!hasFocus()) {
            deselect();
            setCursorPosition(0);
        }
    }

    QString fullText_;
};

QVector<QPointF> readProfileFile(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return {};
    }

    QVector<QPointF> out;
    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine();
        const int slashComment = line.indexOf("//");
        const int hashComment = line.indexOf('#');
        int commentIndex = -1;
        if (slashComment >= 0) {
            commentIndex = slashComment;
        }
        if (hashComment >= 0) {
            commentIndex = (commentIndex < 0) ? hashComment : std::min(commentIndex, hashComment);
        }
        if (commentIndex >= 0) {
            line = line.left(commentIndex);
        }
        line = line.trimmed();
        if (line.isEmpty()) {
            continue;
        }
        line.replace(',', ' ');
        line.replace(';', ' ');
        const QStringList tokens = line.split(QRegularExpression(R"(\s+)"), Qt::SkipEmptyParts);
        if (tokens.size() < 2) {
            continue;
        }
        bool okZ = false;
        bool okU = false;
        const double z = tokens[0].toDouble(&okZ);
        const double u = tokens[1].toDouble(&okU);
        if (okZ && okU) {
            out.push_back(QPointF(z, u));
        }
    }
    std::sort(out.begin(), out.end(), [](const QPointF& a, const QPointF& b) {
        return a.x() < b.x();
    });
    return out;
}

QString selectSavePath(QWidget* parent, const QString& title, const QString& suggestedName) {
    QString path = QFileDialog::getSaveFileName(parent, title, suggestedName, "PNG Image (*.png)");
    if (!path.isEmpty() && QFileInfo(path).suffix().isEmpty()) {
        path += ".png";
    }
    return path;
}

} // namespace

BatchBoundaryPanel::BatchBoundaryPanel(QWidget* parent)
    : QWidget(parent) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);

    modeStack_ = new QStackedWidget(this);
    root->addWidget(modeStack_);

    auto* luwPage = new QWidget(modeStack_);
    auto* luwLayout = new QVBoxLayout(luwPage);
    auto* summaryBox = new QGroupBox("Mode Summary", luwPage);
    auto* summaryLayout = new QVBoxLayout(summaryBox);
    modeSummary_ = new QLabel(summaryBox);
    modeSummary_->setWordWrap(true);
    summaryLayout->addWidget(modeSummary_);
    luwLayout->addWidget(summaryBox);
    luwLayout->addStretch(1);
    modeStack_->addWidget(luwPage);

    auto* dgPage = new QWidget(modeStack_);
    auto* dgLayout = new QVBoxLayout(dgPage);
    auto* dgBox = new QGroupBox("DG Batch Boundary Matrix", dgPage);
    auto* dgBoxLayout = new QVBoxLayout(dgBox);
    auto* dgInputs = new QWidget(dgBox);
    auto* dgInputsLayout = new QGridLayout(dgInputs);
    dgInputsLayout->addWidget(new QLabel("Inflow list (m/s)"), 0, 0);
    dgInflowEdit_ = new PrecisionFloatListLineEdit(dgInputs);
    dgInputsLayout->addWidget(dgInflowEdit_, 0, 1);
    dgInputsLayout->addWidget(new QLabel("Angle list (deg)"), 1, 0);
    dgAngleEdit_ = new PrecisionFloatListLineEdit(dgInputs);
    dgInputsLayout->addWidget(dgAngleEdit_, 1, 1);
    dgBoxLayout->addWidget(dgInputs);
    dgMatrix_ = new QTableWidget(dgBox);
    dgMatrix_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    dgMatrix_->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    dgBoxLayout->addWidget(dgMatrix_, 1);
    dgLayout->addWidget(dgBox);
    modeStack_->addWidget(dgPage);

    auto* pfPage = new QWidget(modeStack_);
    auto* pfLayout = new QVBoxLayout(pfPage);
    auto* pfCasesBox = new QGroupBox("PF Direction Cases", pfPage);
    auto* pfCasesBoxLayout = new QVBoxLayout(pfCasesBox);
    auto* pfInputs = new QWidget(pfCasesBox);
    auto* pfInputsLayout = new QGridLayout(pfInputs);
    pfInputsLayout->addWidget(new QLabel("Angle list (deg)"), 0, 0);
    pfAngleEdit_ = new PrecisionFloatListLineEdit(pfInputs);
    pfInputsLayout->addWidget(pfAngleEdit_, 0, 1);
    pfCasesBoxLayout->addWidget(pfInputs);
    pfCases_ = new QTableWidget(pfCasesBox);
    pfCases_->setColumnCount(4);
    pfCases_->setHorizontalHeaderLabels({"Angle", "dir_x", "dir_y", "Case"});
    pfCases_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    pfCasesBoxLayout->addWidget(pfCases_, 1);
    pfLayout->addWidget(pfCasesBox, 1);

    auto* profileBox = new QGroupBox("Profile Preview", pfPage);
    auto* profileBoxLayout = new QVBoxLayout(profileBox);
    auto* profileHeader = new QWidget(profileBox);
    auto* profileHeaderLayout = new QHBoxLayout(profileHeader);
    profileHeaderLayout->setContentsMargins(0, 0, 0, 0);
    profileHeaderLayout->addWidget(new QLabel("wind_bc/profile.dat", profileHeader));
    profileHeaderLayout->addStretch(1);
    auto* saveProfileButton = new QPushButton("Save Profile Image", profileHeader);
    profileHeaderLayout->addWidget(saveProfileButton);
    profileBoxLayout->addWidget(profileHeader);
    profilePlot_ = new ProfilePlotWidget(profileBox);
    profileBoxLayout->addWidget(profilePlot_);
    profileSamples_ = new QTableWidget(profileBox);
    profileSamples_->setColumnCount(2);
    profileSamples_->setHorizontalHeaderLabels({"z (m)", "U (m/s)"});
    profileSamples_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    profileBoxLayout->addWidget(profileSamples_, 1);
    pfLayout->addWidget(profileBox, 2);
    modeStack_->addWidget(pfPage);

    connect(dgInflowEdit_, &QLineEdit::editingFinished, this, &BatchBoundaryPanel::pushDatasetEdits);
    connect(dgAngleEdit_, &QLineEdit::editingFinished, this, &BatchBoundaryPanel::pushDatasetEdits);
    connect(pfAngleEdit_, &QLineEdit::editingFinished, this, &BatchBoundaryPanel::pushProfileEdits);
    connect(saveProfileButton, &QPushButton::clicked, this, [this] {
        const QString path = selectSavePath(this, "Save Profile Plot", "profile_preview.png");
        if (path.isEmpty()) {
            return;
        }
        QString error;
        if (!profilePlot_->saveImage(path, &error)) {
            QMessageBox::critical(this, "Save profile image", error);
            return;
        }
    });
}

void BatchBoundaryPanel::setDocument(ConfigDocument* document) {
    if (document_ == document) {
        return;
    }

    if (document_) {
        disconnect(document_, nullptr, this, nullptr);
    }
    document_ = document;
    if (document_) {
        connect(document_, &ConfigDocument::changed, this, &BatchBoundaryPanel::refresh);
        connect(document_, &ConfigDocument::modeChanged, this, &BatchBoundaryPanel::refresh);
    }
    refresh();
}

void BatchBoundaryPanel::refresh() {
    const RunMode mode = document_ ? document_->mode() : RunMode::Luw;
    modeStack_->setCurrentIndex(static_cast<int>(mode));
    if (modeSummary_) {
        if (mode == RunMode::Luw) {
            modeSummary_->clear();
        } else {
            modeSummary_->setText(
                QString("Current mode: %1\nThe right column stays available in every mode, but only LUWDG and LUWPF expose batch-specific boundary widgets here.")
                    .arg(runModeDisplayName(mode)));
        }
    }
    updateDatasetTab();
    updateProfileTab();
}

void BatchBoundaryPanel::updateDatasetTab() {
    const QVariantList inflow = document_ ? document_->typedValue("inflow").toList() : QVariantList{};
    const QVariantList angle = document_ ? document_->typedValue("angle").toList() : QVariantList{};

    {
        const QSignalBlocker blockA(dgInflowEdit_);
        const QSignalBlocker blockB(dgAngleEdit_);
        if (auto* edit = dynamic_cast<PrecisionFloatListLineEdit*>(dgInflowEdit_)) {
            edit->setRawText(document_ ? document_->rawValue("inflow") : QString());
        } else {
            dgInflowEdit_->setText(formatFloatList(inflow));
        }
        if (auto* edit = dynamic_cast<PrecisionFloatListLineEdit*>(dgAngleEdit_)) {
            edit->setRawText(document_ ? document_->rawValue("angle") : QString());
        } else {
            dgAngleEdit_->setText(formatFloatList(angle));
        }
    }

    dgMatrix_->clear();
    dgMatrix_->setRowCount(inflow.size());
    dgMatrix_->setColumnCount(angle.size());

    QStringList headers;
    for (const QVariant& angleValue : angle) {
        headers.push_back(QString::number(angleValue.toDouble(), 'f', 1) + " deg");
    }
    dgMatrix_->setHorizontalHeaderLabels(headers);

    QStringList rowHeaders;
    for (const QVariant& inflowValue : inflow) {
        rowHeaders.push_back(QString::number(inflowValue.toDouble(), 'f', 2) + " m/s");
    }
    dgMatrix_->setVerticalHeaderLabels(rowHeaders);

    for (int r = 0; r < inflow.size(); ++r) {
        const double speed = inflow[r].toDouble();
        for (int c = 0; c < angle.size(); ++c) {
            const double deg = angle[c].toDouble();
            const double rad = deg * std::numbers::pi_v<double> / 180.0;
            const double ux = speed * std::cos(rad);
            const double uy = speed * std::sin(rad);
            auto* item = new QTableWidgetItem(
                QString("u=(%1, %2)").arg(ux, 0, 'f', 2).arg(uy, 0, 'f', 2));
            item->setTextAlignment(Qt::AlignCenter);
            item->setBackground(QColor(22, 36 + (c % 2) * 8, 48 + (r % 3) * 10));
            dgMatrix_->setItem(r, c, item);
        }
    }
}

void BatchBoundaryPanel::updateProfileTab() {
    const QVariantList angles = document_ ? document_->typedValue("angle").toList() : QVariantList{};
    {
        const QSignalBlocker blocker(pfAngleEdit_);
        if (auto* edit = dynamic_cast<PrecisionFloatListLineEdit*>(pfAngleEdit_)) {
            edit->setRawText(document_ ? document_->rawValue("angle") : QString());
        } else {
            pfAngleEdit_->setText(formatFloatList(angles));
        }
    }

    pfCases_->setRowCount(angles.size());
    for (int i = 0; i < angles.size(); ++i) {
        const double deg = angles[i].toDouble();
        const double rad = deg * std::numbers::pi_v<double> / 180.0;
        const double dx = std::cos(rad);
        const double dy = std::sin(rad);
        pfCases_->setItem(i, 0, new QTableWidgetItem(QString::number(deg, 'f', 1)));
        pfCases_->setItem(i, 1, new QTableWidgetItem(QString::number(dx, 'f', 3)));
        pfCases_->setItem(i, 2, new QTableWidgetItem(QString::number(dy, 'f', 3)));
        pfCases_->setItem(i, 3, new QTableWidgetItem(QString("PF_%1").arg(i + 1)));
    }

    updateProfileCurve();
}

void BatchBoundaryPanel::updateProfileCurve() {
    QVector<QPointF> samples;
    if (document_) {
        const QString profilePath = QDir(document_->projectDirectory()).filePath("wind_bc/profile.dat");
        samples = readProfileFile(profilePath);
    }

    profilePlot_->setSamples(samples);

    profileSamples_->setRowCount(samples.size());
    for (int i = 0; i < samples.size(); ++i) {
        profileSamples_->setItem(i, 0, new QTableWidgetItem(QString::number(samples[i].x(), 'f', 3)));
        profileSamples_->setItem(i, 1, new QTableWidgetItem(QString::number(samples[i].y(), 'f', 3)));
    }
}

void BatchBoundaryPanel::pushDatasetEdits() {
    if (!document_) {
        return;
    }
    if (auto* edit = dynamic_cast<PrecisionFloatListLineEdit*>(dgInflowEdit_)) {
        document_->setRawValue("inflow", renderBracketedListText(edit->commitVisibleText()));
    } else {
        document_->setTypedValue("inflow", parseFloatListText(dgInflowEdit_->text()));
    }
    if (auto* edit = dynamic_cast<PrecisionFloatListLineEdit*>(dgAngleEdit_)) {
        document_->setRawValue("angle", renderBracketedListText(edit->commitVisibleText()));
    } else {
        document_->setTypedValue("angle", parseFloatListText(dgAngleEdit_->text()));
    }
}

void BatchBoundaryPanel::pushProfileEdits() {
    if (!document_) {
        return;
    }
    if (auto* edit = dynamic_cast<PrecisionFloatListLineEdit*>(pfAngleEdit_)) {
        document_->setRawValue("angle", renderBracketedListText(edit->commitVisibleText()));
    } else {
        document_->setTypedValue("angle", parseFloatListText(pfAngleEdit_->text()));
    }
}

QVariantList BatchBoundaryPanel::parseFloatListText(const QString& text) {
    QVariantList out;
    QString normalized = text;
    normalized.replace('[', ' ');
    normalized.replace(']', ' ');
    const QStringList parts = normalized.split(QRegularExpression(R"([,\s]+)"), Qt::SkipEmptyParts);
    for (const QString& part : parts) {
        bool ok = false;
        const double value = part.toDouble(&ok);
        if (ok) {
            out.push_back(value);
        }
    }
    return out;
}

}
