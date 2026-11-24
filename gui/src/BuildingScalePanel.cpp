#include "luwgui/BuildingScalePanel.h"

#include "luwgui/PlotWidgets.h"

#include <QByteArray>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHash>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPointF>
#include <QPushButton>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>
#include <QtEndian>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <numeric>
#include <vector>

namespace luwgui {

struct BuildingScaleResult {
    QString error;
    QString summary;
    QVector<QPointF> pdfSamples;
    QVector<QPointF> cdfSamples;
    QVector<double> guideLines;
};

namespace {

struct DbfField {
    QString name;
    char type = 'C';
    int length = 0;
};

struct ShapeRecord {
    QVector<QVector<QPointF>> rings;
};

struct DatasetSummary {
    QVector<ShapeRecord> records;
    bool geographic = false;
    QPointF geographicOrigin;
};

quint32 readBigEndian32(const QByteArray& data, int offset) {
    quint32 value = 0;
    std::memcpy(&value, data.constData() + offset, sizeof(value));
    return qFromBigEndian(value);
}

quint32 readLittleEndian32(const QByteArray& data, int offset) {
    quint32 value = 0;
    std::memcpy(&value, data.constData() + offset, sizeof(value));
    return qFromLittleEndian(value);
}

quint16 readLittleEndian16(const QByteArray& data, int offset) {
    quint16 value = 0;
    std::memcpy(&value, data.constData() + offset, sizeof(value));
    return qFromLittleEndian(value);
}

double readLittleEndianDouble(const QByteArray& data, int offset) {
    quint64 bits = 0;
    std::memcpy(&bits, data.constData() + offset, sizeof(bits));
    bits = qFromLittleEndian(bits);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool loadBinaryFile(const QString& path, QByteArray* bytes, QString* errorMessage) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (errorMessage) {
            *errorMessage = file.errorString();
        }
        return false;
    }
    *bytes = file.readAll();
    return true;
}

bool readDbfRows(const QString& path, QVector<DbfField>* fields, QVector<QHash<QString, QString>>* rows, QString* errorMessage) {
    QByteArray bytes;
    if (!loadBinaryFile(path, &bytes, errorMessage)) {
        return false;
    }
    if (bytes.size() < 32) {
        if (errorMessage) {
            *errorMessage = "DBF header is too short.";
        }
        return false;
    }

    const int numRecords = static_cast<int>(readLittleEndian32(bytes, 4));
    const int headerLength = static_cast<int>(readLittleEndian16(bytes, 8));
    const int recordLength = static_cast<int>(readLittleEndian16(bytes, 10));
    if (headerLength <= 32 || recordLength <= 1 || headerLength > bytes.size()) {
        if (errorMessage) {
            *errorMessage = "DBF layout is invalid.";
        }
        return false;
    }

    fields->clear();
    rows->clear();

    int offset = 32;
    while (offset + 32 <= headerLength && static_cast<unsigned char>(bytes[offset]) != 0x0D) {
        const QByteArray rawName = bytes.mid(offset, 11);
        const int nullIndex = rawName.indexOf('\0');
        const QByteArray trimmedName = (nullIndex >= 0) ? rawName.left(nullIndex) : rawName;
        fields->push_back({
            QString::fromLatin1(trimmedName).trimmed().toLower(),
            bytes[offset + 11],
            static_cast<unsigned char>(bytes[offset + 16])
        });
        offset += 32;
    }

    int recordOffset = headerLength;
    for (int rowIndex = 0; rowIndex < numRecords && recordOffset + recordLength <= bytes.size(); ++rowIndex) {
        if (bytes[recordOffset] == '*') {
            recordOffset += recordLength;
            continue;
        }

        QHash<QString, QString> row;
        int fieldOffset = recordOffset + 1;
        for (const DbfField& field : *fields) {
            const QByteArray rawValue = bytes.mid(fieldOffset, field.length);
            row.insert(field.name, QString::fromLocal8Bit(rawValue).trimmed());
            fieldOffset += field.length;
        }
        rows->push_back(row);
        recordOffset += recordLength;
    }
    return true;
}

bool readShapefile(const QString& path, QVector<ShapeRecord>* records, QRectF* bounds, QString* errorMessage) {
    QByteArray bytes;
    if (!loadBinaryFile(path, &bytes, errorMessage)) {
        return false;
    }
    if (bytes.size() < 100) {
        if (errorMessage) {
            *errorMessage = "SHP header is too short.";
        }
        return false;
    }
    if (readBigEndian32(bytes, 0) != 9994u) {
        if (errorMessage) {
            *errorMessage = "Unsupported SHP signature.";
        }
        return false;
    }

    records->clear();
    *bounds = QRectF(
        readLittleEndianDouble(bytes, 36),
        readLittleEndianDouble(bytes, 44),
        readLittleEndianDouble(bytes, 52) - readLittleEndianDouble(bytes, 36),
        readLittleEndianDouble(bytes, 60) - readLittleEndianDouble(bytes, 44));

    int offset = 100;
    while (offset + 8 <= bytes.size()) {
        const int contentBytes = static_cast<int>(readBigEndian32(bytes, offset + 4)) * 2;
        offset += 8;
        if (offset + contentBytes > bytes.size()) {
            break;
        }
        const quint32 shapeType = readLittleEndian32(bytes, offset);
        if (shapeType == 0u) {
            offset += contentBytes;
            continue;
        }
        if (shapeType != 5u && shapeType != 15u && shapeType != 25u) {
            offset += contentBytes;
            continue;
        }
        if (contentBytes < 44) {
            offset += contentBytes;
            continue;
        }

        const int numParts = static_cast<int>(readLittleEndian32(bytes, offset + 36));
        const int numPoints = static_cast<int>(readLittleEndian32(bytes, offset + 40));
        const int partsOffset = offset + 44;
        const int pointsOffset = partsOffset + numParts * 4;
        if (numParts <= 0 || numPoints <= 0 || pointsOffset + numPoints * 16 > offset + contentBytes) {
            offset += contentBytes;
            continue;
        }

        QVector<int> parts;
        parts.reserve(numParts);
        for (int i = 0; i < numParts; ++i) {
            parts.push_back(static_cast<int>(readLittleEndian32(bytes, partsOffset + i * 4)));
        }

        QVector<QPointF> points;
        points.reserve(numPoints);
        for (int i = 0; i < numPoints; ++i) {
            const int pointOffset = pointsOffset + i * 16;
            points.push_back(QPointF(readLittleEndianDouble(bytes, pointOffset), readLittleEndianDouble(bytes, pointOffset + 8)));
        }

        ShapeRecord record;
        for (int part = 0; part < numParts; ++part) {
            const int begin = parts[part];
            const int end = (part + 1 < numParts) ? parts[part + 1] : numPoints;
            if (begin < 0 || begin >= end || end > points.size()) {
                continue;
            }
            QVector<QPointF> ring;
            ring.reserve(end - begin);
            for (int index = begin; index < end; ++index) {
                ring.push_back(points[index]);
            }
            if (ring.size() >= 3) {
                record.rings.push_back(ring);
            }
        }
        if (!record.rings.isEmpty()) {
            records->push_back(record);
        }
        offset += contentBytes;
    }
    return true;
}

QString pickHeightColumn(const QVector<DbfField>& fields, const QVector<QHash<QString, QString>>& rows) {
    const QStringList exactCandidates = {
        "height", "hgt", "building_height", "bldg_height", "bldg_h", "bldgheight", "roof_height", "roof_h", "eave_height", "z"
    };

    auto columnHasNumericData = [&](const QString& name) {
        for (const QHash<QString, QString>& row : rows) {
            bool ok = false;
            row.value(name).toDouble(&ok);
            if (ok) {
                return true;
            }
        }
        return false;
    };

    for (const DbfField& field : fields) {
        if (exactCandidates.contains(field.name) && columnHasNumericData(field.name)) {
            return field.name;
        }
    }
    for (const DbfField& field : fields) {
        if ((field.name.contains("height") || field.name.contains("hgt")) && columnHasNumericData(field.name)) {
            return field.name;
        }
    }
    return {};
}

double signedArea(const QVector<QPointF>& polygon) {
    if (polygon.size() < 3) {
        return 0.0;
    }
    double area = 0.0;
    for (int i = 0; i < polygon.size(); ++i) {
        const QPointF& a = polygon[i];
        const QPointF& b = polygon[(i + 1) % polygon.size()];
        area += a.x() * b.y() - b.x() * a.y();
    }
    return 0.5 * area;
}

QPointF toMetricPoint(const QPointF& point, const QPointF& origin, bool geographic) {
    if (!geographic) {
        return point;
    }
    constexpr double earthRadius = 6378137.0;
    const double lat0 = origin.y() * std::numbers::pi_v<double> / 180.0;
    const double x = earthRadius * (point.x() - origin.x()) * std::numbers::pi_v<double> / 180.0 * std::cos(lat0);
    const double y = earthRadius * (point.y() - origin.y()) * std::numbers::pi_v<double> / 180.0;
    return QPointF(x, y);
}

bool isProbablyGeographic(const QRectF& bounds, const QString& prjText) {
    if (prjText.contains("GEOGCS", Qt::CaseInsensitive) || prjText.contains("GEOGCRS", Qt::CaseInsensitive)) {
        return true;
    }
    return bounds.left() >= -180.0 && bounds.right() <= 180.0 && bounds.top() >= -90.0 && bounds.bottom() <= 90.0;
}

double cross(const QPointF& a, const QPointF& b, const QPointF& c) {
    return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
}

QVector<QPointF> convexHull(QVector<QPointF> points) {
    std::sort(points.begin(), points.end(), [](const QPointF& a, const QPointF& b) {
        return (a.x() < b.x()) || (qFuzzyCompare(a.x(), b.x()) && a.y() < b.y());
    });
    points.erase(std::unique(points.begin(), points.end(), [](const QPointF& a, const QPointF& b) {
        return qFuzzyCompare(a.x(), b.x()) && qFuzzyCompare(a.y(), b.y());
    }), points.end());

    if (points.size() <= 2) {
        return points;
    }

    QVector<QPointF> lower;
    for (const QPointF& point : points) {
        while (lower.size() >= 2 && cross(lower[lower.size() - 2], lower.back(), point) <= 0.0) {
            lower.removeLast();
        }
        lower.push_back(point);
    }

    QVector<QPointF> upper;
    for (int i = points.size() - 1; i >= 0; --i) {
        const QPointF& point = points[i];
        while (upper.size() >= 2 && cross(upper[upper.size() - 2], upper.back(), point) <= 0.0) {
            upper.removeLast();
        }
        upper.push_back(point);
    }

    lower.removeLast();
    upper.removeLast();
    lower += upper;
    return lower;
}

double shortSideLength(const QVector<QPointF>& ring) {
    QVector<QPointF> hull = convexHull(ring);
    if (hull.size() < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (hull.size() == 2) {
        return std::hypot(hull[1].x() - hull[0].x(), hull[1].y() - hull[0].y());
    }

    double bestArea = std::numeric_limits<double>::max();
    double bestShortSide = std::numeric_limits<double>::quiet_NaN();

    for (int i = 0; i < hull.size(); ++i) {
        const QPointF& a = hull[i];
        const QPointF& b = hull[(i + 1) % hull.size()];
        const double angle = std::atan2(b.y() - a.y(), b.x() - a.x());
        const double c = std::cos(-angle);
        const double s = std::sin(-angle);

        double minX = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        double minY = std::numeric_limits<double>::max();
        double maxY = std::numeric_limits<double>::lowest();

        for (const QPointF& point : hull) {
            const double xr = point.x() * c - point.y() * s;
            const double yr = point.x() * s + point.y() * c;
            minX = std::min(minX, xr);
            maxX = std::max(maxX, xr);
            minY = std::min(minY, yr);
            maxY = std::max(maxY, yr);
        }

        const double width = maxX - minX;
        const double height = maxY - minY;
        const double area = width * height;
        if (area < bestArea) {
            bestArea = area;
            bestShortSide = std::min(width, height);
        }
    }

    return bestShortSide;
}

double percentile(const std::vector<double>& sortedValues, double p) {
    if (sortedValues.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double clamped = std::clamp(p, 0.0, 1.0);
    const double position = clamped * static_cast<double>(sortedValues.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(position));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(position));
    if (lo == hi) {
        return sortedValues[lo];
    }
    const double t = position - static_cast<double>(lo);
    return sortedValues[lo] + (sortedValues[hi] - sortedValues[lo]) * t;
}

BuildingScaleResult computeBuildingScale(const QString& shpPath) {
    BuildingScaleResult result;

    QVector<ShapeRecord> records;
    QRectF bounds;
    QString error;
    if (!readShapefile(shpPath, &records, &bounds, &error)) {
        result.error = error;
        return result;
    }
    if (records.isEmpty()) {
        result.error = "No polygon geometry was found in the shapefile.";
        return result;
    }

    const QString dbfPath = QFileInfo(shpPath).absolutePath() + "/" + QFileInfo(shpPath).completeBaseName() + ".dbf";
    QVector<DbfField> fields;
    QVector<QHash<QString, QString>> rows;
    const bool hasDbf = QFileInfo::exists(dbfPath) && readDbfRows(dbfPath, &fields, &rows, nullptr);
    const QString prjPath = QFileInfo(shpPath).absolutePath() + "/" + QFileInfo(shpPath).completeBaseName() + ".prj";
    QString prjText;
    if (QFileInfo::exists(prjPath)) {
        QFile prjFile(prjPath);
        if (prjFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
            prjText = QString::fromUtf8(prjFile.readAll());
        }
    }

    const bool geographic = isProbablyGeographic(bounds, prjText);
    const QPointF origin(bounds.center());
    const QString heightColumn = hasDbf ? pickHeightColumn(fields, rows) : QString();

    std::vector<double> lengths;
    std::vector<double> weights;
    lengths.reserve(records.size());
    weights.reserve(records.size());

    for (int index = 0; index < records.size(); ++index) {
        QVector<QVector<QPointF>> metricRings;
        metricRings.reserve(records[index].rings.size());
        for (const QVector<QPointF>& ring : records[index].rings) {
            QVector<QPointF> transformed;
            transformed.reserve(ring.size());
            for (const QPointF& point : ring) {
                transformed.push_back(toMetricPoint(point, origin, geographic));
            }
            metricRings.push_back(transformed);
        }

        if (metricRings.isEmpty()) {
            continue;
        }

        int largestRingIndex = -1;
        double largestAbsArea = 0.0;
        double referenceSign = 0.0;
        QVector<double> ringAreas;
        ringAreas.reserve(metricRings.size());
        for (int ringIndex = 0; ringIndex < metricRings.size(); ++ringIndex) {
            const double area = signedArea(metricRings[ringIndex]);
            ringAreas.push_back(area);
            if (std::abs(area) > largestAbsArea) {
                largestAbsArea = std::abs(area);
                largestRingIndex = ringIndex;
                referenceSign = (area >= 0.0) ? 1.0 : -1.0;
            }
        }
        if (largestRingIndex < 0 || largestAbsArea <= 0.0) {
            continue;
        }

        double footprintArea = 0.0;
        for (double area : ringAreas) {
            footprintArea += ((area >= 0.0) == (referenceSign >= 0.0)) ? std::abs(area) : -std::abs(area);
        }
        if (footprintArea <= 0.0) {
            footprintArea = 0.0;
            for (double area : ringAreas) {
                footprintArea += std::abs(area);
            }
        }

        const double length = shortSideLength(metricRings[largestRingIndex]);
        double height = 1.0;
        if (!heightColumn.isEmpty() && index < rows.size()) {
            bool ok = false;
            const double value = rows[index].value(heightColumn).toDouble(&ok);
            if (ok && value > 0.0) {
                height = value;
            }
        }

        const double weight = footprintArea * height;
        if (!std::isfinite(length) || !std::isfinite(weight) || length <= 0.0 || weight <= 0.0) {
            continue;
        }

        lengths.push_back(length);
        weights.push_back(weight);
    }

    if (lengths.empty()) {
        result.error = "No valid short-side lengths were computed from the shapefile.";
        return result;
    }

    std::vector<double> sortedLengths = lengths;
    std::sort(sortedLengths.begin(), sortedLengths.end());

    const double minLength = sortedLengths.front();
    const double maxLength = sortedLengths.back();
    const double median = percentile(sortedLengths, 0.5);
    const double q1 = percentile(sortedLengths, 0.25);
    const double q3 = percentile(sortedLengths, 0.75);
    const double iqr = q3 - q1;

    const double totalWeight = std::accumulate(weights.begin(), weights.end(), 0.0);
    double binWidth = 0.0;
    if (iqr > 0.0) {
        binWidth = 2.0 * iqr / std::cbrt(static_cast<double>(sortedLengths.size()));
    }
    if (!(binWidth > 0.0)) {
        binWidth = std::max((maxLength - minLength) / std::sqrt(static_cast<double>(sortedLengths.size())), 0.25);
    }
    const int numBins = std::clamp(static_cast<int>(std::ceil((maxLength - minLength) / binWidth)), 8, 160);
    const double effectiveBinWidth = std::max((maxLength - minLength) / static_cast<double>(numBins), 1e-9);

    std::vector<double> hist(static_cast<std::size_t>(numBins), 0.0);
    for (int i = 0; i < static_cast<int>(lengths.size()); ++i) {
        int bin = static_cast<int>(std::floor((lengths[i] - minLength) / effectiveBinWidth));
        bin = std::clamp(bin, 0, numBins - 1);
        hist[static_cast<std::size_t>(bin)] += weights[i];
    }
    for (int bin = 0; bin < numBins; ++bin) {
        const double center = minLength + (static_cast<double>(bin) + 0.5) * effectiveBinWidth;
        const double pdf = hist[static_cast<std::size_t>(bin)] / (totalWeight * effectiveBinWidth);
        result.pdfSamples.push_back(QPointF(center, pdf));
    }

    std::vector<int> order(static_cast<std::size_t>(lengths.size()));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return lengths[static_cast<std::size_t>(a)] > lengths[static_cast<std::size_t>(b)];
    });
    double cumulative = 0.0;
    for (int orderedIndex : order) {
        cumulative += weights[static_cast<std::size_t>(orderedIndex)] / totalWeight;
        result.cdfSamples.push_back(QPointF(lengths[static_cast<std::size_t>(orderedIndex)], cumulative));
    }

    result.guideLines = {2.0, 5.0, 10.0, 20.0, 50.0};
    result.summary = QString("%1 | count: %2 | min/median/max: %3 / %4 / %5 m | CRS: %6")
                         .arg(heightColumn.isEmpty() ? "Height field: default 1.0" : "Height field: " + heightColumn)
                         .arg(lengths.size())
                         .arg(minLength, 0, 'f', 3)
                         .arg(median, 0, 'f', 3)
                         .arg(maxLength, 0, 'f', 3)
                         .arg(geographic ? "local metric conversion from geographic coordinates" : "projected / native metric coordinates");
    return result;
}

QString selectSavePath(QWidget* parent, const QString& title, const QString& suggestedName) {
    QString path = QFileDialog::getSaveFileName(parent, title, suggestedName, "PNG Image (*.png)");
    if (!path.isEmpty() && QFileInfo(path).suffix().isEmpty()) {
        path += ".png";
    }
    return path;
}

} // namespace

BuildingScalePanel::BuildingScalePanel(QWidget* parent)
    : QWidget(parent)
    , watcher_(new QFutureWatcher<BuildingScaleResult>(this)) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(8, 8, 8, 8);

    auto* header = new QWidget(this);
    auto* headerLayout = new QGridLayout(header);
    headerLayout->setContentsMargins(0, 0, 0, 0);
    headerLayout->addWidget(new QLabel("Building shapefile"), 0, 0);
    fileEdit_ = new QLineEdit(header);
    headerLayout->addWidget(fileEdit_, 0, 1, 1, 3);
    auto* browseButton = new QPushButton("Browse", header);
    headerLayout->addWidget(browseButton, 0, 4);
    analyzeButton_ = new QPushButton("Analyze Building Scale", header);
    headerLayout->addWidget(analyzeButton_, 1, 3, 1, 2);
    auto* saveButton = new QPushButton("Save Plot Image", header);
    headerLayout->addWidget(saveButton, 1, 5);
    root->addWidget(header);

    summaryLabel_ = new QLabel("Select a building shapefile to compute weighted short-side PDF/CDF.", this);
    summaryLabel_->setWordWrap(true);
    summaryLabel_->setProperty("muted", true);
    root->addWidget(summaryLabel_);

    plot_ = new DistributionPlotWidget(this);
    root->addWidget(plot_, 1);

    connect(browseButton, &QPushButton::clicked, this, [this] {
        const QString path = QFileDialog::getOpenFileName(this, "Open Building Shapefile", fileEdit_->text(), "Shapefile (*.shp)");
        if (!path.isEmpty()) {
            setSuggestedFilePath(path);
        }
    });
    connect(analyzeButton_, &QPushButton::clicked, this, &BuildingScalePanel::startAnalysis);
    connect(saveButton, &QPushButton::clicked, this, &BuildingScalePanel::savePlot);
    connect(watcher_, &QFutureWatcher<BuildingScaleResult>::finished, this, [this] {
        analyzeButton_->setEnabled(true);
        const BuildingScaleResult result = watcher_->result();
        if (!result.error.isEmpty()) {
            summaryLabel_->setText(result.error);
            emit statusMessage(result.error);
            return;
        }
        plot_->setCurves(result.pdfSamples, result.cdfSamples, result.guideLines);
        summaryLabel_->setText(result.summary);
        emit statusMessage("Building scale distribution updated.");
    });
}

void BuildingScalePanel::setSuggestedFilePath(const QString& filePath) {
    if (filePath.isEmpty()) {
        return;
    }
    fileEdit_->setText(QFileInfo(filePath).absoluteFilePath());
}

void BuildingScalePanel::startAnalysis() {
    const QString path = fileEdit_->text().trimmed();
    if (path.isEmpty()) {
        QMessageBox::warning(this, "Building scale", "Select a shapefile first.");
        return;
    }
    analyzeButton_->setEnabled(false);
    summaryLabel_->setText("Analyzing building scale distribution...");
    watcher_->setFuture(QtConcurrent::run(computeBuildingScale, path));
}

void BuildingScalePanel::savePlot() {
    const QString path = selectSavePath(this, "Save Building Scale Plot", QFileInfo(fileEdit_->text()).completeBaseName() + "_building_scale.png");
    if (path.isEmpty()) {
        return;
    }
    if (!grab().save(path)) {
        QMessageBox::critical(this, "Save plot image", "Failed to save plot image.");
        return;
    }
    emit statusMessage("Saved building scale plot to " + path);
}

}
