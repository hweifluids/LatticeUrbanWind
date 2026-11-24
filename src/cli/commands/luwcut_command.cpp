#include "luw/luwcut_command.hpp"

#include "luw/deck_document.hpp"
#include "luw/preprocess_compute_config.hpp"
#include "luw/progress.hpp"

#include <QString>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <ostream>
#include <queue>
#include <setjmp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef LUW_HAS_GDAL
#define LUW_HAS_GDAL 0
#endif

#ifndef LUW_HAS_JPEG
#define LUW_HAS_JPEG 0
#endif

#if LUW_HAS_GDAL
#include <cpl_conv.h>
#include <gdal_priv.h>
#include <ogr_api.h>
#include <ogr_spatialref.h>
#include <ogrsf_frmts.h>

#ifdef _WIN32
#include <conio.h>
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#endif

#if LUW_HAS_GDAL && LUW_HAS_JPEG
#include <jpeglib.h>
#endif

namespace luw::cli {

namespace {

namespace fs = std::filesystem;

QString utf8QString(std::string_view value) {
    return QString::fromUtf8(value.data(),
                             static_cast<qsizetype>(value.size()));
}

std::string pathToUtf8(const fs::path &value) {
#ifdef _WIN32
    return QString::fromStdWString(value.wstring()).toUtf8().toStdString();
#else
    return value.string();
#endif
}

fs::path pathFromUtf8(const std::string &value) {
#ifdef _WIN32
    return fs::path(utf8QString(value).toStdWString());
#else
    return fs::path(value);
#endif
}

std::string lowerText(const std::string &value) {
    return utf8QString(value).toLower().toUtf8().toStdString();
}

bool endsWith(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size()) == suffix;
}

std::string pythonDouble(double value) {
    if (std::isnan(value))
        return "nan";
    if (std::isinf(value))
        return value < 0.0 ? "-inf" : "inf";
    std::array<char, 128> buffer{};
    const auto converted =
        std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                      std::chars_format::general);
    if (converted.ec != std::errc{}) {
        std::ostringstream fallback;
        fallback.imbue(std::locale::classic());
        fallback << std::setprecision(std::numeric_limits<double>::max_digits10)
                 << value;
        return fallback.str();
    }
    std::string rendered(buffer.data(), converted.ptr);
    if (rendered.find_first_of(".eE") == std::string::npos)
        rendered += ".0";
    return rendered;
}

std::string fixedDouble(double value, int precision) {
    std::ostringstream rendered;
    rendered.imbue(std::locale::classic());
    rendered << std::fixed << std::setprecision(precision) << value;
    return rendered.str();
}

std::tm localTime(std::time_t value) {
    std::tm result{};
#ifdef _WIN32
    localtime_s(&result, &value);
#else
    localtime_r(&value, &result);
#endif
    return result;
}

std::string dateTimeNow() {
    const std::time_t value = std::time(nullptr);
    const std::tm local = localTime(value);
    std::ostringstream rendered;
    rendered << std::put_time(&local, "%Y-%m-%d %H:%M:%S");
    return rendered.str();
}

#if LUW_HAS_GDAL

constexpr double kMinimumExtrudeArea = 20.0;
constexpr double kMinimumHoleArea = 5.0;

struct DatasetCloser {
    void operator()(GDALDataset *dataset) const noexcept {
        if (dataset != nullptr)
            GDALClose(dataset);
    }
};

struct FeatureCloser {
    void operator()(OGRFeature *feature) const noexcept {
        if (feature != nullptr)
            OGRFeature::DestroyFeature(feature);
    }
};

struct GeometryCloser {
    void operator()(OGRGeometry *geometry) const noexcept {
        if (geometry != nullptr)
            OGRGeometryFactory::destroyGeometry(geometry);
    }
};

struct TransformCloser {
    void operator()(OGRCoordinateTransformation *transform) const noexcept {
        if (transform != nullptr)
            OCTDestroyCoordinateTransformation(transform);
    }
};

using DatasetPtr = std::unique_ptr<GDALDataset, DatasetCloser>;
using FeaturePtr = std::unique_ptr<OGRFeature, FeatureCloser>;
using GeometryPtr = std::unique_ptr<OGRGeometry, GeometryCloser>;
using TransformPtr =
    std::unique_ptr<OGRCoordinateTransformation, TransformCloser>;

struct Record {
    FeaturePtr feature;
    std::optional<double> addedHeight;
    std::optional<int> addedCornerId;
};

Record cloneRecord(const Record &source) {
    Record result;
    result.feature.reset(source.feature != nullptr ? source.feature->Clone()
                                                   : nullptr);
    result.addedHeight = source.addedHeight;
    result.addedCornerId = source.addedCornerId;
    return result;
}

OGRGeometry *geometryOf(Record &record) {
    return record.feature != nullptr ? record.feature->GetGeometryRef()
                                     : nullptr;
}

const OGRGeometry *geometryOf(const Record &record) {
    return record.feature != nullptr ? record.feature->GetGeometryRef()
                                     : nullptr;
}

void setGeometry(Record &record, GeometryPtr geometry) {
    record.feature->SetGeometryDirectly(geometry.release());
}

GeometryPtr cloneGeometry(const OGRGeometry *geometry) {
    return GeometryPtr(geometry != nullptr ? geometry->clone() : nullptr);
}

bool isEmpty(const OGRGeometry *geometry) {
    return geometry == nullptr || geometry->IsEmpty();
}

bool isPolygonal(const OGRGeometry *geometry) {
    if (geometry == nullptr)
        return false;
    const OGRwkbGeometryType type = wkbFlatten(geometry->getGeometryType());
    return type == wkbPolygon || type == wkbMultiPolygon;
}

void appendPolygonClones(const OGRGeometry *geometry,
                         std::vector<GeometryPtr> &polygons) {
    if (geometry == nullptr || geometry->IsEmpty())
        return;
    const OGRwkbGeometryType type = wkbFlatten(geometry->getGeometryType());
    if (type == wkbPolygon) {
        polygons.push_back(cloneGeometry(geometry));
        return;
    }
    if (type != wkbMultiPolygon)
        return;
    const auto *collection =
        dynamic_cast<const OGRGeometryCollection *>(geometry);
    if (collection == nullptr)
        return;
    for (int index = 0; index < collection->getNumGeometries(); ++index) {
        const OGRGeometry *part = collection->getGeometryRef(index);
        if (part != nullptr &&
            wkbFlatten(part->getGeometryType()) == wkbPolygon) {
            polygons.push_back(cloneGeometry(part));
        }
    }
}

GeometryPtr polygonalParts(const OGRGeometry *geometry) {
    if (geometry == nullptr || geometry->IsEmpty())
        return {};
    const OGRwkbGeometryType type = wkbFlatten(geometry->getGeometryType());
    if (type == wkbPolygon || type == wkbMultiPolygon) {
        return cloneGeometry(geometry);
    }

    const auto *collection =
        dynamic_cast<const OGRGeometryCollection *>(geometry);
    if (collection == nullptr)
        return {};
    std::vector<GeometryPtr> polygons;
    for (int index = 0; index < collection->getNumGeometries(); ++index) {
        appendPolygonClones(collection->getGeometryRef(index), polygons);
    }
    if (polygons.empty())
        return {};
    if (polygons.size() == 1U)
        return std::move(polygons.front());

    auto result = GeometryPtr(new OGRMultiPolygon());
    auto *multi = dynamic_cast<OGRMultiPolygon *>(result.get());
    for (const GeometryPtr &polygon : polygons) {
        if (multi->addGeometry(polygon.get()) != OGRERR_NONE)
            return {};
    }
    return result;
}

GeometryPtr makeValidSafe(const OGRGeometry *geometry) {
    if (geometry == nullptr || geometry->IsEmpty())
        return {};
    GeometryPtr valid(geometry->MakeValid());
    if (!valid)
        valid.reset(geometry->Buffer(0.0, 30));
    if (!valid || valid->IsEmpty())
        return {};
    return polygonalParts(valid.get());
}

void printPercentProgress(std::ostream &output, int &nextPercentIndex,
                          std::size_t completed, std::size_t total) {
    const double ratio =
        total > 0U ? static_cast<double>(completed) / static_cast<double>(total)
                   : 1.0;
    while (nextPercentIndex <= 20 &&
           ratio >= static_cast<double>(nextPercentIndex) / 20.0) {
        output << '#';
        if (nextPercentIndex % 4 == 0)
            output << '|' << nextPercentIndex * 5 << "%|";
        output.flush();
        ++nextPercentIndex;
    }
}

std::vector<Record> cleanBuildingGeometries(std::vector<Record> records,
                                            std::ostream &output,
                                            ProgressEmitter &progress) {
    output << "Cleaning step started. Ensuring geometries are valid and "
              "polygonal.\n";
    const std::size_t before = records.size();
    output << "Total geometries to validate: " << before << '\n';
    progress.progress(
        QStringLiteral("geometry_clean"), 0,
        static_cast<std::int64_t>(std::max<std::size_t>(before, 1U)),
        QStringLiteral("Validating geometries 0/%1")
            .arg(static_cast<qulonglong>(before)),
        {}, true);

    if (!progress.enabled()) {
        output << "Validation Progress (# is 5%): |";
        output.flush();
    }
    int nextPercentIndex = 1;
    std::vector<Record> cleaned;
    cleaned.reserve(records.size());
    for (std::size_t index = 0; index < records.size(); ++index) {
        Record &record = records[index];
        GeometryPtr valid = makeValidSafe(geometryOf(record));
        if (valid && !valid->IsEmpty() && isPolygonal(valid.get())) {
            setGeometry(record, std::move(valid));
            cleaned.push_back(std::move(record));
        }
        progress.progress(
            QStringLiteral("geometry_clean"),
            static_cast<std::int64_t>(index + 1U),
            static_cast<std::int64_t>(std::max<std::size_t>(before, 1U)),
            QStringLiteral("Validating geometries %1/%2")
                .arg(static_cast<qulonglong>(index + 1U))
                .arg(static_cast<qulonglong>(before)));
        if (!progress.enabled()) {
            printPercentProgress(output, nextPercentIndex, index + 1U, before);
        }
    }
    if (!progress.enabled()) {
        printPercentProgress(output, nextPercentIndex, before, before);
        output << "|Finished|\n";
        output.flush();
    }

    const std::size_t dropped = before - cleaned.size();
    output << "Cleaning step completed. Dropped features that are "
              "non-polygonal or "
              "unrecoverable. Count dropped: "
           << dropped << ". Remaining: " << cleaned.size() << '\n';
    progress.complete(QStringLiteral("geometry_clean"),
                      QStringLiteral("Validated %1 geometries, kept %2")
                          .arg(static_cast<qulonglong>(before))
                          .arg(static_cast<qulonglong>(cleaned.size())));
    return cleaned;
}

std::vector<Record> finalPolygonalFilter(std::vector<Record> records,
                                         std::ostream &output) {
    const std::size_t before = records.size();
    std::vector<Record> result;
    result.reserve(records.size());
    for (Record &record : records) {
        GeometryPtr valid = makeValidSafe(geometryOf(record));
        if (!valid || valid->IsEmpty() || !isPolygonal(valid.get()))
            continue;
        setGeometry(record, std::move(valid));
        result.push_back(std::move(record));
    }
    const std::size_t dropped = before - result.size();
    if (dropped > 0U) {
        output << "[WARN] Final polygonal filter dropped " << dropped
               << " non-polygonal features before writing.\n";
    }
    return result;
}

std::unique_ptr<OGRSpatialReference> epsgReference(int epsg) {
    auto reference = std::make_unique<OGRSpatialReference>();
    if (reference->importFromEPSG(epsg) != OGRERR_NONE) {
        throw std::runtime_error("Cannot create EPSG:" + std::to_string(epsg) +
                                 " coordinate reference system");
    }
    reference->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    return reference;
}

std::unique_ptr<OGRSpatialReference>
cloneReference(const OGRSpatialReference *source) {
    if (source == nullptr)
        return {};
    std::unique_ptr<OGRSpatialReference> result(source->Clone());
    result->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    return result;
}

TransformPtr coordinateTransform(OGRSpatialReference *source,
                                 OGRSpatialReference *target,
                                 std::string_view operation) {
    TransformPtr transform(OGRCreateCoordinateTransformation(source, target));
    if (!transform) {
        throw std::runtime_error(
            "Cannot create coordinate transformation for " +
            std::string(operation));
    }
    return transform;
}

void transformRecords(std::vector<Record> &records,
                      OGRCoordinateTransformation *transform,
                      std::string_view operation) {
    for (Record &record : records) {
        OGRGeometry *geometry = geometryOf(record);
        if (geometry != nullptr &&
            geometry->transform(transform) != OGRERR_NONE) {
            throw std::runtime_error("Coordinate transformation failed while " +
                                     std::string(operation));
        }
    }
}

GeometryPtr transformedClone(const OGRGeometry *geometry,
                             OGRCoordinateTransformation *transform,
                             std::string_view operation) {
    GeometryPtr result = cloneGeometry(geometry);
    if (result && result->transform(transform) != OGRERR_NONE) {
        throw std::runtime_error("Coordinate transformation failed while " +
                                 std::string(operation));
    }
    return result;
}

std::optional<int> identifiedEpsg(OGRSpatialReference &reference) {
    std::unique_ptr<OGRSpatialReference> candidate(reference.Clone());
    candidate->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    (void)candidate->AutoIdentifyEPSG();
    const char *authority = candidate->GetAuthorityName(nullptr);
    const char *code = candidate->GetAuthorityCode(nullptr);
    if (authority == nullptr || code == nullptr ||
        lowerText(authority) != "epsg") {
        return std::nullopt;
    }
    int value = 0;
    const std::size_t codeLength = std::char_traits<char>::length(code);
    const auto converted = std::from_chars(code, code + codeLength, value);
    if (converted.ec != std::errc{} || converted.ptr != code + codeLength)
        return std::nullopt;
    return value;
}

std::string referenceText(OGRSpatialReference &reference) {
    if (const std::optional<int> epsg = identifiedEpsg(reference)) {
        return "EPSG:" + std::to_string(*epsg);
    }
    char *text = nullptr;
    if (reference.exportToWkt(&text) == OGRERR_NONE && text != nullptr) {
        std::string result(text);
        CPLFree(text);
        return result;
    }
    CPLFree(text);
    return "None";
}

int utmZone(double longitude) {
    int zone = static_cast<int>(std::floor((longitude + 180.0) / 6.0)) + 1;
    return std::clamp(zone, 1, 60);
}

int utmEpsg(double longitude, double latitude) {
    return (latitude >= 0.0 ? 32600 : 32700) + utmZone(longitude);
}

bool envelopesIntersect(const OGREnvelope &left, const OGREnvelope &right) {
    return left.MinX <= right.MaxX && left.MaxX >= right.MinX &&
           left.MinY <= right.MaxY && left.MaxY >= right.MinY;
}

OGREnvelope envelopeOf(const OGRGeometry *geometry) {
    OGREnvelope envelope{};
    if (geometry != nullptr)
        geometry->getEnvelope(&envelope);
    return envelope;
}

std::optional<OGREnvelope> totalEnvelope(const std::vector<Record> &records) {
    bool initialized = false;
    OGREnvelope result{};
    for (const Record &record : records) {
        const OGRGeometry *geometry = geometryOf(record);
        if (geometry == nullptr || geometry->IsEmpty())
            continue;
        const OGREnvelope envelope = envelopeOf(geometry);
        if (!initialized) {
            result = envelope;
            initialized = true;
        } else {
            result.Merge(envelope);
        }
    }
    return initialized ? std::optional<OGREnvelope>(result) : std::nullopt;
}

struct LonLatBounds {
    double lonMin = 0.0;
    double lonMax = 0.0;
    double latMin = 0.0;
    double latMax = 0.0;
};

std::string formatLonLatBounds(const LonLatBounds &bounds) {
    return "lon[" + fixedDouble(bounds.lonMin, 6) + ", " +
           fixedDouble(bounds.lonMax, 6) + "] lat[" +
           fixedDouble(bounds.latMin, 6) + ", " +
           fixedDouble(bounds.latMax, 6) + ']';
}

bool containsBounds(const LonLatBounds &target, const LonLatBounds &input) {
    return input.lonMin <= target.lonMin && input.lonMax >= target.lonMax &&
           input.latMin <= target.latMin && input.latMax >= target.latMax;
}

double maximumMissPercent(const LonLatBounds &target,
                          const LonLatBounds &input) {
    const double longitudeSpan = std::abs(target.lonMax - target.lonMin);
    const double latitudeSpan = std::abs(target.latMax - target.latMin);
    if (longitudeSpan <= 0.0 || latitudeSpan <= 0.0)
        return 100.0;
    const double missingLonMin = std::max(0.0, input.lonMin - target.lonMin);
    const double missingLonMax = std::max(0.0, target.lonMax - input.lonMax);
    const double missingLatMin = std::max(0.0, input.latMin - target.latMin);
    const double missingLatMax = std::max(0.0, target.latMax - input.latMax);
    const double longitudePercent =
        std::max(missingLonMin, missingLonMax) / longitudeSpan * 100.0;
    const double latitudePercent =
        std::max(missingLatMin, missingLatMax) / latitudeSpan * 100.0;
    return std::max(longitudePercent, latitudePercent);
}

std::optional<std::string> timedInputLine(std::ostream &output,
                                          std::string_view prompt,
                                          double timeoutSeconds) {
    output << prompt;
    output.flush();
#ifdef _WIN32
    std::string buffer;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(timeoutSeconds);
    while (std::chrono::steady_clock::now() < deadline) {
        if (_kbhit() != 0) {
            const int character = _getch();
            if (character == '\r' || character == '\n') {
                output << '\n';
                output.flush();
                return buffer;
            }
            if (character == 3)
                throw std::runtime_error("Interrupted");
            if (character == '\b') {
                if (!buffer.empty()) {
                    buffer.pop_back();
                    output << "\b \b";
                    output.flush();
                }
                continue;
            }
            if (character == 0 || character == 0xE0) {
                if (_kbhit() != 0)
                    (void)_getch();
                continue;
            }
            buffer.push_back(static_cast<char>(character));
            output << static_cast<char>(character);
            output.flush();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    output << '\n';
    output.flush();
    return std::nullopt;
#else
    fd_set readSet;
    FD_ZERO(&readSet);
    FD_SET(STDIN_FILENO, &readSet);
    timeval timeout{};
    timeout.tv_sec = static_cast<long>(timeoutSeconds);
    timeout.tv_usec =
        static_cast<long>((timeoutSeconds - timeout.tv_sec) * 1000000.0);
    const int status =
        select(STDIN_FILENO + 1, &readSet, nullptr, nullptr, &timeout);
    if (status <= 0) {
        output << '\n';
        output.flush();
        return std::nullopt;
    }
    std::string line;
    if (!std::getline(std::cin, line))
        return std::nullopt;
    if (!line.empty() && line.back() == '\r')
        line.pop_back();
    return line;
#endif
}

enum class CoverageDecision { Continue, Cancel };

CoverageDecision confirmCoverage(std::string_view kind,
                                 const LonLatBounds &target,
                                 const LonLatBounds &input,
                                 std::ostream &output) {
    if (containsBounds(target, input))
        return CoverageDecision::Continue;
    const double miss = maximumMissPercent(target, input);
    if (miss < 0.1) {
        output << "[WARN] " << kind
               << " bounds are slightly smaller than target (max miss "
               << fixedDouble(miss, 4)
               << "% < 0.1%). Continue without interruption.\n";
        return CoverageDecision::Continue;
    }

    output << "[WARN] " << kind
           << " bounds do not fully cover the target area.\n"
           << "[WARN] Target lon/lat bounds: " << formatLonLatBounds(target)
           << '\n'
           << "[WARN] Input  lon/lat bounds: " << formatLonLatBounds(input)
           << '\n';
    constexpr int timeoutSeconds = 5;
    const std::optional<std::string> answer = timedInputLine(
        output,
        "Continue anyway? (Y/N) [auto-continue in 5s]: ", timeoutSeconds);
    if (!answer) {
        output << "[WARN] No input received (timeout 5s). Continuing by "
                  "default.\n";
        return CoverageDecision::Continue;
    }
    std::string normalized = lowerText(*answer);
    const auto first = normalized.find_first_not_of(" \t\r\n");
    const auto last = normalized.find_last_not_of(" \t\r\n");
    normalized = first == std::string::npos
                     ? std::string{}
                     : normalized.substr(first, last - first + 1U);
    if (normalized == "n" || normalized == "no") {
        output << "[INFO] User canceled. Exiting.\n";
        return CoverageDecision::Cancel;
    }
    if (normalized.empty() || normalized == "y" || normalized == "yes") {
        output << "[WARN] Continuing despite bounds mismatch.\n";
        return CoverageDecision::Continue;
    }
    output << "[WARN] Invalid input '" << *answer
           << "'. Continuing by default.\n";
    return CoverageDecision::Continue;
}

struct RotatedDomain {
    GeometryPtr polygon;
    double rotationDegrees = 0.0;
    double pivotX = 0.0;
    double pivotY = 0.0;
};

std::pair<double, double> rotatePoint(double x, double y, double degrees,
                                      double centerX, double centerY) {
    const double theta = degrees * std::acos(-1.0) / 180.0;
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);
    return {
        cosine * (x - centerX) - sine * (y - centerY) + centerX,
        sine * (x - centerX) + cosine * (y - centerY) + centerY,
    };
}

GeometryPtr
rectanglePolygon(const std::array<std::pair<double, double>, 4> &points) {
    auto polygon = GeometryPtr(new OGRPolygon());
    OGRLinearRing ring;
    for (const auto &[x, y] : points)
        ring.addPoint(x, y);
    ring.addPoint(points.front().first, points.front().second);
    if (dynamic_cast<OGRPolygon *>(polygon.get())->addRing(&ring) !=
        OGRERR_NONE)
        return {};
    return polygon;
}

RotatedDomain makeRotatedDomain(double minLongitude, double maxLongitude,
                                double minLatitude, double maxLatitude,
                                OGRCoordinateTransformation *toUtm) {
    std::array<double, 4> x{minLongitude, maxLongitude, maxLongitude,
                            minLongitude};
    std::array<double, 4> y{minLatitude, minLatitude, maxLatitude, maxLatitude};
    if (!toUtm->Transform(4, x.data(), y.data())) {
        throw std::runtime_error("Coordinate transformation failed while "
                                 "projecting clipping corners");
    }
    const double centerX = (x[0] + x[1] + x[2] + x[3]) / 4.0;
    const double centerY = (y[0] + y[1] + y[2] + y[3]) / 4.0;
    const double angleRadians = std::atan2(y[1] - y[0], x[1] - x[0]);
    const double rotationDegrees = -angleRadians * 180.0 / std::acos(-1.0);

    std::array<std::pair<double, double>, 4> rotated{};
    for (std::size_t index = 0; index < rotated.size(); ++index) {
        rotated[index] =
            rotatePoint(x[index], y[index], rotationDegrees, centerX, centerY);
    }
    double minimumX = rotated.front().first;
    double maximumX = rotated.front().first;
    double minimumY = rotated.front().second;
    double maximumY = rotated.front().second;
    for (const auto &[rotatedX, rotatedY] : rotated) {
        minimumX = std::min(minimumX, rotatedX);
        maximumX = std::max(maximumX, rotatedX);
        minimumY = std::min(minimumY, rotatedY);
        maximumY = std::max(maximumY, rotatedY);
    }

    const std::array<std::pair<double, double>, 4> aligned{{
        {maximumX, minimumY},
        {maximumX, maximumY},
        {minimumX, maximumY},
        {minimumX, minimumY},
    }};
    std::array<std::pair<double, double>, 4> unrotated{};
    for (std::size_t index = 0; index < aligned.size(); ++index) {
        unrotated[index] =
            rotatePoint(aligned[index].first, aligned[index].second,
                        -rotationDegrees, centerX, centerY);
    }
    RotatedDomain result;
    result.polygon = rectanglePolygon(unrotated);
    if (!result.polygon)
        throw std::runtime_error("Cannot construct rotated clipping rectangle");
    result.rotationDegrees = rotationDegrees;
    result.pivotX = centerX;
    result.pivotY = centerY;
    return result;
}

class EnvelopeIndex final {
  public:
    explicit EnvelopeIndex(const std::vector<Record> &records) {
        envelopes_.reserve(records.size());
        std::vector<std::size_t> indices;
        indices.reserve(records.size());
        for (std::size_t index = 0; index < records.size(); ++index) {
            envelopes_.push_back(envelopeOf(geometryOf(records[index])));
            indices.push_back(index);
        }
        root_ = build(indices, 0U, indices.size());
    }

    std::vector<std::size_t> query(const OGREnvelope &envelope) const {
        std::vector<std::size_t> result;
        queryNode(root_, envelope, result);
        std::sort(result.begin(), result.end());
        return result;
    }

  private:
    struct Node {
        OGREnvelope envelope{};
        int left = -1;
        int right = -1;
        std::vector<std::size_t> entries;
    };

    static double center(const OGREnvelope &envelope, bool xAxis) {
        return xAxis ? (envelope.MinX + envelope.MaxX) * 0.5
                     : (envelope.MinY + envelope.MaxY) * 0.5;
    }

    int build(std::vector<std::size_t> &indices, std::size_t begin,
              std::size_t end) {
        if (begin == end)
            return -1;
        Node node;
        node.envelope = envelopes_[indices[begin]];
        for (std::size_t index = begin + 1U; index < end; ++index) {
            node.envelope.Merge(envelopes_[indices[index]]);
        }
        const int nodeIndex = static_cast<int>(nodes_.size());
        nodes_.push_back(node);
        if (end - begin <= 8U) {
            nodes_[nodeIndex].entries.assign(
                indices.begin() + static_cast<std::ptrdiff_t>(begin),
                indices.begin() + static_cast<std::ptrdiff_t>(end));
            return nodeIndex;
        }
        const bool xAxis = (node.envelope.MaxX - node.envelope.MinX) >=
                           (node.envelope.MaxY - node.envelope.MinY);
        const std::size_t middle = begin + (end - begin) / 2U;
        std::nth_element(indices.begin() + static_cast<std::ptrdiff_t>(begin),
                         indices.begin() + static_cast<std::ptrdiff_t>(middle),
                         indices.begin() + static_cast<std::ptrdiff_t>(end),
                         [&](std::size_t left, std::size_t right) {
                             return center(envelopes_[left], xAxis) <
                                    center(envelopes_[right], xAxis);
                         });
        const int left = build(indices, begin, middle);
        const int right = build(indices, middle, end);
        nodes_[nodeIndex].left = left;
        nodes_[nodeIndex].right = right;
        return nodeIndex;
    }

    void queryNode(int nodeIndex, const OGREnvelope &envelope,
                   std::vector<std::size_t> &result) const {
        if (nodeIndex < 0)
            return;
        const Node &node = nodes_[static_cast<std::size_t>(nodeIndex)];
        if (!envelopesIntersect(node.envelope, envelope))
            return;
        if (!node.entries.empty()) {
            for (std::size_t index : node.entries) {
                if (envelopesIntersect(envelopes_[index], envelope))
                    result.push_back(index);
            }
            return;
        }
        queryNode(node.left, envelope, result);
        queryNode(node.right, envelope, result);
    }

    std::vector<OGREnvelope> envelopes_;
    std::vector<Node> nodes_;
    int root_ = -1;
};

std::optional<int> exactFieldIndex(const OGRFeatureDefn &definition,
                                   std::string_view name) {
    for (int index = 0; index < definition.GetFieldCount(); ++index) {
        const OGRFieldDefn *field = definition.GetFieldDefn(index);
        if (field != nullptr && name == field->GetNameRef())
            return index;
    }
    return std::nullopt;
}

struct HeightField {
    std::string name;
    std::optional<int> sourceIndex;
};

HeightField selectHeightField(const OGRFeatureDefn &definition) {
    std::vector<std::string> columns;
    columns.reserve(static_cast<std::size_t>(definition.GetFieldCount()) + 1U);
    std::unordered_map<std::string, std::string> lowerMap;
    for (int index = 0; index < definition.GetFieldCount(); ++index) {
        const OGRFieldDefn *field = definition.GetFieldDefn(index);
        const std::string name =
            field != nullptr ? field->GetNameRef() : std::string{};
        columns.push_back(name);
        lowerMap[lowerText(name)] = name;
    }
    columns.emplace_back("geometry");
    lowerMap["geometry"] = "geometry";

    constexpr std::array<std::string_view, 12> exactPriority{{
        "height",
        "height_m",
        "heightm",
        "measuredheight",
        "buildingheight",
        "bldgheight",
        "bldg_h",
        "bldghgt",
        "roof_height",
        "eave_height",
        "max_height",
        "elevation",
    }};
    for (std::string_view candidate : exactPriority) {
        const auto found = lowerMap.find(std::string(candidate));
        if (found != lowerMap.end()) {
            return {found->second, exactFieldIndex(definition, found->second)};
        }
    }
    for (const std::string &column : columns) {
        const std::string lower = lowerText(column);
        if (lower.find("height") != std::string::npos ||
            endsWith(lower, "hgt") || endsWith(lower, "_h")) {
            return {column, exactFieldIndex(definition, column)};
        }
    }
    constexpr std::array<std::string_view, 6> levelNames{{
        "levels",
        "level",
        "storeys",
        "stories",
        "floors",
        "floor",
    }};
    for (std::string_view candidate : levelNames) {
        const auto found = lowerMap.find(std::string(candidate));
        if (found != lowerMap.end()) {
            return {found->second, exactFieldIndex(definition, found->second)};
        }
    }
    return {"elevation", std::nullopt};
}

std::optional<double> numericField(const OGRFeature &feature, int index) {
    if (!feature.IsFieldSetAndNotNull(index))
        return std::nullopt;
    const OGRFieldDefn *definition = feature.GetDefnRef()->GetFieldDefn(index);
    if (definition == nullptr)
        return std::nullopt;
    const OGRFieldType type = definition->GetType();
    if (type == OFTInteger || type == OFTInteger64 || type == OFTReal) {
        const double value = feature.GetFieldAsDouble(index);
        return std::isfinite(value) ? std::optional<double>(value)
                                    : std::nullopt;
    }
    const char *raw = feature.GetFieldAsString(index);
    if (raw == nullptr)
        return std::nullopt;
    char *end = nullptr;
    errno = 0;
    const double value = std::strtod(raw, &end);
    if (end == raw || errno == ERANGE || !std::isfinite(value))
        return std::nullopt;
    while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end)) != 0)
        ++end;
    return *end == '\0' ? std::optional<double>(value) : std::nullopt;
}

double geometryArea(const OGRGeometry *geometry) {
    if (geometry == nullptr)
        return 0.0;
    return OGR_G_Area(
        reinterpret_cast<OGRGeometryH>(const_cast<OGRGeometry *>(geometry)));
}

GeometryPtr unionComponent(const std::vector<Record> &records,
                           const std::vector<std::size_t> &indices) {
    if (indices.empty())
        return {};
    if (indices.size() == 1U)
        return cloneGeometry(geometryOf(records[indices.front()]));

    OGRGeometryCollection collection;
    for (std::size_t index : indices) {
        if (collection.addGeometry(geometryOf(records[index])) != OGRERR_NONE) {
            break;
        }
    }
    // Shapely's unary_union accepts a heterogeneous polygon collection.
    // OGR's deprecated UnionCascaded instead requires a MultiPolygon and
    // emits a GEOS error for this collection on current GDAL releases.
    GeometryPtr result(collection.UnaryUnion());
    if (result && !result->IsEmpty())
        return result;

    result = cloneGeometry(geometryOf(records[indices.front()]));
    for (std::size_t offset = 1; result && offset < indices.size(); ++offset) {
        GeometryPtr next(result->Union(geometryOf(records[indices[offset]])));
        if (next && !next->IsEmpty())
            result = std::move(next);
    }
    return result;
}

std::vector<Record> unionOverlappingBuildings(std::vector<Record> records,
                                              const HeightField &height,
                                              const OGRFeatureDefn &definition,
                                              std::ostream &output,
                                              ProgressEmitter &progress) {
    output << "Overlap union step started. Building overlap graph.\n";
    if (records.empty()) {
        output << "Overlap union skipped because there are no features.\n";
        progress.complete(QStringLiteral("overlap_union"),
                          QStringLiteral("No footprints need overlap merging"));
        return records;
    }

    const std::size_t count = records.size();
    EnvelopeIndex index(records);
    std::vector<bool> visited(count, false);
    std::vector<std::vector<std::size_t>> components;
    output << "Total seeds to check for overlap components: " << count << '\n';
    progress.progress(QStringLiteral("overlap_union"), 0,
                      static_cast<std::int64_t>(count),
                      QStringLiteral("Finding overlap components 0/%1")
                          .arg(static_cast<qulonglong>(count)),
                      {}, true);
    if (!progress.enabled()) {
        output << "Component discovery progress (# is 5%): |";
        output.flush();
    }
    int nextPercentIndex = 1;

    for (std::size_t seed = 0; seed < count; ++seed) {
        if (!visited[seed]) {
            std::vector<std::size_t> component{seed};
            std::vector<std::size_t> stack{seed};
            visited[seed] = true;
            while (!stack.empty()) {
                const std::size_t current = stack.back();
                stack.pop_back();
                const OGRGeometry *currentGeometry =
                    geometryOf(records[current]);
                for (std::size_t candidate :
                     index.query(envelopeOf(currentGeometry))) {
                    if (candidate == current || visited[candidate])
                        continue;
                    const OGRGeometry *candidateGeometry =
                        geometryOf(records[candidate]);
                    if (!currentGeometry->Intersects(candidateGeometry))
                        continue;
                    GeometryPtr intersection(
                        currentGeometry->Intersection(candidateGeometry));
                    const double area = geometryArea(intersection.get());
                    if (std::isfinite(area) && area > 0.0) {
                        visited[candidate] = true;
                        stack.push_back(candidate);
                        component.push_back(candidate);
                    }
                }
            }
            components.push_back(std::move(component));
        }

        progress.progress(QStringLiteral("overlap_union"),
                          static_cast<std::int64_t>(seed + 1U),
                          static_cast<std::int64_t>(count),
                          QStringLiteral("Finding overlap components %1/%2")
                              .arg(static_cast<qulonglong>(seed + 1U))
                              .arg(static_cast<qulonglong>(count)));
        if (!progress.enabled()) {
            printPercentProgress(output, nextPercentIndex, seed + 1U, count);
        }
    }
    if (!progress.enabled()) {
        printPercentProgress(output, nextPercentIndex, count, count);
        output << "|Finished|\n";
        output.flush();
    }

    const std::optional<int> cornerSource =
        exactFieldIndex(definition, "corner_id");
    std::vector<Record> merged;
    merged.reserve(components.size());
    std::size_t mergedComponentCount = 0;
    for (const std::vector<std::size_t> &component : components) {
        if (component.size() == 1U) {
            merged.push_back(std::move(records[component.front()]));
            continue;
        }
        Record base = cloneRecord(records[component.front()]);
        GeometryPtr unioned = unionComponent(records, component);
        if (unioned)
            setGeometry(base, std::move(unioned));

        if (height.sourceIndex) {
            std::optional<double> maximum;
            for (std::size_t member : component) {
                const std::optional<double> value =
                    numericField(*records[member].feature, *height.sourceIndex);
                if (value && (!maximum || *value > *maximum))
                    maximum = value;
            }
            if (maximum) {
                const OGRFieldDefn *heightDefinition =
                    definition.GetFieldDefn(*height.sourceIndex);
                if (heightDefinition != nullptr &&
                    heightDefinition->GetType() == OFTString) {
                    base.feature->SetField(*height.sourceIndex,
                                           pythonDouble(*maximum).c_str());
                } else {
                    base.feature->SetField(*height.sourceIndex, *maximum);
                }
            } else {
                base.feature->SetFieldNull(*height.sourceIndex);
            }
        }
        if (cornerSource)
            base.feature->SetFieldNull(*cornerSource);
        merged.push_back(std::move(base));
        ++mergedComponentCount;
    }

    output << "Overlap union step completed. Components found: "
           << components.size()
           << ". Merged components with size greater than one: "
           << mergedComponentCount << ". Output features: " << merged.size()
           << '\n';
    progress.complete(QStringLiteral("overlap_union"),
                      QStringLiteral("Found %1 overlap groups, merged %2")
                          .arg(static_cast<qulonglong>(components.size()))
                          .arg(static_cast<qulonglong>(mergedComponentCount)));
    return merged;
}

std::vector<Record> clipRecords(std::vector<Record> records,
                                const OGRGeometry *clippingPolygon) {
    const OGREnvelope clippingEnvelope = envelopeOf(clippingPolygon);
    std::vector<Record> clipped;
    clipped.reserve(records.size());
    for (Record &record : records) {
        OGRGeometry *geometry = geometryOf(record);
        if (geometry == nullptr || geometry->IsEmpty() ||
            !envelopesIntersect(envelopeOf(geometry), clippingEnvelope) ||
            !geometry->Intersects(clippingPolygon)) {
            continue;
        }
        GeometryPtr intersection(geometry->Intersection(clippingPolygon));
        if (!intersection || intersection->IsEmpty())
            continue;
        setGeometry(record, std::move(intersection));
        clipped.push_back(std::move(record));
    }
    return clipped;
}

std::vector<Record> filterByArea(std::vector<Record> records,
                                 std::ostream &output,
                                 ProgressEmitter &progress) {
    const std::size_t count = records.size();
    if (count == 0U)
        return {};
    progress.progress(QStringLiteral("area_filter"), 0,
                      static_cast<std::int64_t>(count),
                      QStringLiteral("Filtering footprints by area 0/%1")
                          .arg(static_cast<qulonglong>(count)),
                      {}, true);

    const OGRGeometry *first = nullptr;
    for (const Record &record : records) {
        if (!isEmpty(geometryOf(record))) {
            first = geometryOf(record);
            break;
        }
    }
    if (first == nullptr)
        return {};
    // GDAL 3.12 requires the destination point passed to PointOnSurface to
    // be non-empty.  Its coordinates are overwritten on success.
    OGRPoint representative(0.0, 0.0);
    OGRErr representativeStatus = OGRERR_FAILURE;
    if (const auto *polygon = dynamic_cast<const OGRPolygon *>(first)) {
        representativeStatus = polygon->PointOnSurface(&representative);
    } else if (const auto *multi =
                   dynamic_cast<const OGRMultiPolygon *>(first)) {
        representativeStatus = multi->PointOnSurface(&representative);
    } else {
        representativeStatus = first->Centroid(&representative);
    }
    if (representativeStatus != OGRERR_NONE)
        return {};
    auto wgs84 = epsgReference(4326);
    auto areaReference =
        epsgReference(utmEpsg(representative.getX(), representative.getY()));
    TransformPtr toArea = coordinateTransform(wgs84.get(), areaReference.get(),
                                              "footprint area filtering");

    std::vector<Record> kept;
    kept.reserve(records.size());
    for (std::size_t index = 0; index < records.size(); ++index) {
        Record &record = records[index];
        bool retain = false;
        if (!isEmpty(geometryOf(record))) {
            GeometryPtr projected = cloneGeometry(geometryOf(record));
            if (projected &&
                projected->transform(toArea.get()) == OGRERR_NONE) {
                const double area = geometryArea(projected.get());
                retain = std::isfinite(area) && area >= kMinimumExtrudeArea;
            }
        }
        if (retain)
            kept.push_back(std::move(record));
        progress.progress(QStringLiteral("area_filter"),
                          static_cast<std::int64_t>(index + 1U),
                          static_cast<std::int64_t>(count),
                          QStringLiteral("Filtering footprints by area %1/%2")
                              .arg(static_cast<qulonglong>(index + 1U))
                              .arg(static_cast<qulonglong>(count)));
    }
    output << "Area filter step completed. Kept features: " << kept.size()
           << '\n';
    return kept;
}

GeometryPtr polygonWithoutSmallHoles(const OGRPolygon &source,
                                     double minimumArea, std::size_t &removed) {
    auto result = GeometryPtr(new OGRPolygon());
    auto *polygon = dynamic_cast<OGRPolygon *>(result.get());
    const OGRLinearRing *exterior = source.getExteriorRing();
    if (exterior == nullptr || polygon->addRing(exterior) != OGRERR_NONE) {
        return cloneGeometry(&source);
    }
    for (int index = 0; index < source.getNumInteriorRings(); ++index) {
        const OGRLinearRing *ring = source.getInteriorRing(index);
        if (ring == nullptr)
            continue;
        OGRPolygon hole;
        if (hole.addRing(ring) != OGRERR_NONE) {
            (void)polygon->addRing(ring);
            continue;
        }
        const double area = geometryArea(&hole);
        if (std::isfinite(area) && area < minimumArea) {
            ++removed;
            continue;
        }
        (void)polygon->addRing(ring);
    }
    return result;
}

GeometryPtr geometryWithoutSmallHoles(const OGRGeometry *geometry,
                                      double minimumArea,
                                      std::size_t &removed) {
    if (geometry == nullptr || geometry->IsEmpty())
        return cloneGeometry(geometry);
    const OGRwkbGeometryType type = wkbFlatten(geometry->getGeometryType());
    if (type == wkbPolygon) {
        const auto *polygon = dynamic_cast<const OGRPolygon *>(geometry);
        return polygon != nullptr
                   ? polygonWithoutSmallHoles(*polygon, minimumArea, removed)
                   : cloneGeometry(geometry);
    }
    if (type != wkbMultiPolygon)
        return cloneGeometry(geometry);
    const auto *source = dynamic_cast<const OGRGeometryCollection *>(geometry);
    if (source == nullptr)
        return cloneGeometry(geometry);
    std::vector<GeometryPtr> polygons;
    polygons.reserve(static_cast<std::size_t>(source->getNumGeometries()));
    for (int index = 0; index < source->getNumGeometries(); ++index) {
        const auto *polygon =
            dynamic_cast<const OGRPolygon *>(source->getGeometryRef(index));
        if (polygon != nullptr && !polygon->IsEmpty()) {
            polygons.push_back(
                polygonWithoutSmallHoles(*polygon, minimumArea, removed));
        }
    }
    if (polygons.empty())
        return cloneGeometry(geometry);
    if (polygons.size() == 1U)
        return std::move(polygons.front());
    auto result = GeometryPtr(new OGRMultiPolygon());
    auto *multi = dynamic_cast<OGRMultiPolygon *>(result.get());
    for (const GeometryPtr &polygon : polygons)
        (void)multi->addGeometry(polygon.get());
    return result;
}

std::size_t removeSmallHoles(std::vector<Record> &records,
                             OGRCoordinateTransformation *toUtm,
                             OGRCoordinateTransformation *toWgs84,
                             std::ostream &, ProgressEmitter &progress) {
    transformRecords(records, toUtm,
                     "projecting footprints for hole filtering");
    std::size_t removed = 0;
    const std::size_t count = records.size();
    progress.progress(
        QStringLiteral("remove_holes"), 0,
        static_cast<std::int64_t>(std::max<std::size_t>(count, 1U)),
        QStringLiteral("Removing small holes 0/%1")
            .arg(static_cast<qulonglong>(count)),
        {}, true);
    for (std::size_t index = 0; index < records.size(); ++index) {
        Record &record = records[index];
        const OGRGeometry *geometry = geometryOf(record);
        const OGRwkbGeometryType type =
            geometry != nullptr ? wkbFlatten(geometry->getGeometryType())
                                : wkbUnknown;
        if (geometry != nullptr && !geometry->IsEmpty() &&
            (type == wkbPolygon || type == wkbMultiPolygon)) {
            setGeometry(record, geometryWithoutSmallHoles(
                                    geometry, kMinimumHoleArea, removed));
            progress.progress(
                QStringLiteral("remove_holes"),
                static_cast<std::int64_t>(index + 1U),
                static_cast<std::int64_t>(std::max<std::size_t>(count, 1U)),
                QStringLiteral("Removing small holes %1/%2")
                    .arg(static_cast<qulonglong>(index + 1U))
                    .arg(static_cast<qulonglong>(count)));
        } else if (geometry == nullptr || geometry->IsEmpty()) {
            progress.progress(
                QStringLiteral("remove_holes"),
                static_cast<std::int64_t>(index + 1U),
                static_cast<std::int64_t>(std::max<std::size_t>(count, 1U)),
                QStringLiteral("Removing small holes %1/%2")
                    .arg(static_cast<qulonglong>(index + 1U))
                    .arg(static_cast<qulonglong>(count)));
        }
    }
    progress.complete(QStringLiteral("remove_holes"),
                      QStringLiteral("Removed %1 interior holes")
                          .arg(static_cast<qulonglong>(removed)));
    transformRecords(records, toWgs84,
                     "returning hole-filtered footprints to EPSG 4326");
    return removed;
}

GeometryPtr square(double centerX, double centerY, double halfSize) {
    const std::array<std::pair<double, double>, 4> corners{{
        {centerX + halfSize, centerY - halfSize},
        {centerX + halfSize, centerY + halfSize},
        {centerX - halfSize, centerY + halfSize},
        {centerX - halfSize, centerY - halfSize},
    }};
    return rectanglePolygon(corners);
}

std::vector<Record> makeCornerRecords(const OGRGeometry *domain,
                                      OGRFeatureDefn &sourceDefinition,
                                      const HeightField &height,
                                      OGRCoordinateTransformation *toWgs84) {
    const auto *polygon = dynamic_cast<const OGRPolygon *>(domain);
    if (polygon == nullptr || polygon->getExteriorRing() == nullptr) {
        throw std::runtime_error("Rotated clipping rectangle is not polygonal");
    }
    OGRPoint centroid;
    if (polygon->Centroid(&centroid) != OGRERR_NONE) {
        throw std::runtime_error(
            "Cannot calculate clipping rectangle centroid");
    }

    const OGRLinearRing *ring = polygon->getExteriorRing();
    std::array<std::pair<double, double>, 4> corners{};
    if (ring->getNumPoints() >= 4) {
        for (int index = 0; index < 4; ++index) {
            corners[static_cast<std::size_t>(index)] = {ring->getX(index),
                                                        ring->getY(index)};
        }
    } else {
        const OGREnvelope bounds = envelopeOf(domain);
        corners = {{{bounds.MinX, bounds.MinY},
                    {bounds.MaxX, bounds.MinY},
                    {bounds.MaxX, bounds.MaxY},
                    {bounds.MinX, bounds.MaxY}}};
    }

    const std::optional<int> cornerIndex =
        exactFieldIndex(sourceDefinition, "corner_id");
    std::vector<Record> result;
    result.reserve(4U);
    for (std::size_t index = 0; index < corners.size(); ++index) {
        const auto [x, y] = corners[index];
        const double vectorX = centroid.getX() - x;
        const double vectorY = centroid.getY() - y;
        const double length = std::hypot(vectorX, vectorY);
        const double centerX = length == 0.0 ? x : x + vectorX / length;
        const double centerY = length == 0.0 ? y : y + vectorY / length;
        GeometryPtr geometry = square(centerX, centerY, 0.5);
        if (geometry->transform(toWgs84) != OGRERR_NONE) {
            throw std::runtime_error("Coordinate transformation failed while "
                                     "creating corner squares");
        }

        Record record;
        record.feature.reset(OGRFeature::CreateFeature(&sourceDefinition));
        setGeometry(record, std::move(geometry));
        if (height.sourceIndex) {
            const OGRFieldDefn *heightDefinition =
                sourceDefinition.GetFieldDefn(*height.sourceIndex);
            if (heightDefinition != nullptr &&
                heightDefinition->GetType() == OFTString) {
                record.feature->SetField(*height.sourceIndex, "1.0");
            } else {
                record.feature->SetField(*height.sourceIndex, 1.0);
            }
        } else {
            record.addedHeight = 1.0;
        }
        if (cornerIndex) {
            record.feature->SetField(*cornerIndex,
                                     static_cast<int>(index + 1U));
        } else {
            record.addedCornerId = static_cast<int>(index + 1U);
        }
        result.push_back(std::move(record));
    }
    return result;
}

void removeExistingShapefile(const fs::path &path) {
    constexpr std::array<std::string_view, 7> extensions{{
        ".shp",
        ".shx",
        ".dbf",
        ".prj",
        ".cpg",
        ".qix",
        ".fix",
    }};
    for (std::string_view extension : extensions) {
        fs::path sidecar = path;
        sidecar.replace_extension(extension);
        std::error_code existsError;
        if (!fs::exists(sidecar, existsError) || existsError)
            continue;
        std::error_code removeError;
        if (!fs::remove(sidecar, removeError) || removeError) {
            throw std::runtime_error(
                "Cannot replace existing shapefile component: " +
                pathToUtf8(sidecar));
        }
    }
}

OGRFieldType outputFieldType(const OGRFieldDefn &source,
                             bool receivesMissingCornerRows,
                             bool receivesFloatingCornerHeight) {
    const OGRFieldType type = source.GetType();
    if ((receivesMissingCornerRows || receivesFloatingCornerHeight) &&
        (type == OFTInteger || type == OFTInteger64)) {
        return OFTReal;
    }
    return type;
}

void copyFieldValue(const OGRFeature &source, int sourceIndex,
                    OGRFeature &target, int targetIndex) {
    if (!source.IsFieldSet(sourceIndex))
        return;
    if (source.IsFieldNull(sourceIndex)) {
        target.SetFieldNull(targetIndex);
        return;
    }
    const OGRFieldDefn *sourceDefinition =
        source.GetDefnRef()->GetFieldDefn(sourceIndex);
    const OGRFieldDefn *targetDefinition =
        target.GetDefnRef()->GetFieldDefn(targetIndex);
    if (sourceDefinition == nullptr || targetDefinition == nullptr)
        return;
    const OGRFieldType sourceType = sourceDefinition->GetType();
    const OGRFieldType targetType = targetDefinition->GetType();
    if (targetType == OFTReal &&
        (sourceType == OFTInteger || sourceType == OFTInteger64 ||
         sourceType == OFTReal)) {
        target.SetField(targetIndex, source.GetFieldAsDouble(sourceIndex));
        return;
    }
    if (targetType == sourceType) {
        target.SetField(targetIndex, source.GetRawFieldRef(sourceIndex));
    } else {
        target.SetField(targetIndex, source.GetFieldAsString(sourceIndex));
    }
}

struct OutputSchema {
    std::vector<int> sourceToOutput;
    int heightIndex = -1;
    int cornerIndex = -1;
    std::vector<std::string> displayColumns;
};

OutputSchema createOutputFields(OGRLayer &layer, const OGRFeatureDefn &source,
                                const HeightField &height) {
    OutputSchema result;
    result.sourceToOutput.resize(
        static_cast<std::size_t>(source.GetFieldCount()), -1);
    result.displayColumns.reserve(
        static_cast<std::size_t>(source.GetFieldCount()) + 3U);
    const std::optional<int> existingCorner =
        exactFieldIndex(source, "corner_id");
    for (int index = 0; index < source.GetFieldCount(); ++index) {
        const OGRFieldDefn *definition = source.GetFieldDefn(index);
        if (definition == nullptr)
            continue;
        const OGRFieldType copiedType = outputFieldType(
            *definition, !existingCorner || index != *existingCorner,
            height.sourceIndex && index == *height.sourceIndex);
        OGRFieldDefn copied(definition->GetNameRef(), copiedType);
        if (copiedType == definition->GetType()) {
            copied.SetSubType(definition->GetSubType());
        }
        copied.SetWidth(definition->GetWidth());
        copied.SetPrecision(definition->GetPrecision());
        copied.SetNullable(TRUE);
        if (layer.CreateField(&copied, TRUE) != OGRERR_NONE) {
            throw std::runtime_error("Cannot create output field: " +
                                     std::string(definition->GetNameRef()));
        }
        const int outputIndex = layer.GetLayerDefn()->GetFieldCount() - 1;
        result.sourceToOutput[static_cast<std::size_t>(index)] = outputIndex;
        result.displayColumns.emplace_back(definition->GetNameRef());
        if (height.sourceIndex && index == *height.sourceIndex) {
            result.heightIndex = outputIndex;
        }
        if (existingCorner && index == *existingCorner) {
            result.cornerIndex = outputIndex;
        }
    }
    result.displayColumns.emplace_back("geometry");

    if (!height.sourceIndex) {
        OGRFieldDefn heightField(height.name.c_str(), OFTReal);
        heightField.SetWidth(24);
        heightField.SetPrecision(15);
        if (layer.CreateField(&heightField, TRUE) != OGRERR_NONE) {
            throw std::runtime_error("Cannot create output height field: " +
                                     height.name);
        }
        result.heightIndex = layer.GetLayerDefn()->GetFieldCount() - 1;
        result.displayColumns.push_back(height.name);
    }
    if (!existingCorner) {
        OGRFieldDefn cornerField("corner_id", OFTReal);
        cornerField.SetWidth(24);
        cornerField.SetPrecision(15);
        if (layer.CreateField(&cornerField, TRUE) != OGRERR_NONE) {
            throw std::runtime_error("Cannot create output field: corner_id");
        }
        result.cornerIndex = layer.GetLayerDefn()->GetFieldCount() - 1;
        result.displayColumns.emplace_back("corner_id");
    }
    return result;
}

OutputSchema writeShapefile(const fs::path &outputPath,
                            const std::string &caseName,
                            OGRSpatialReference &outputReference,
                            OGRFeatureDefn &sourceDefinition,
                            const HeightField &height,
                            const std::vector<Record> &records) {
    removeExistingShapefile(outputPath);
    GDALDriver *driver =
        GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
    if (driver == nullptr) {
        throw std::runtime_error("ESRI Shapefile GDAL driver is unavailable");
    }
    DatasetPtr dataset(driver->Create(pathToUtf8(outputPath).c_str(), 0, 0, 0,
                                      GDT_Unknown, nullptr));
    if (!dataset) {
        throw std::runtime_error("Cannot create output shapefile: " +
                                 pathToUtf8(outputPath));
    }
    char **options = nullptr;
    options = CSLSetNameValue(options, "ENCODING", "UTF-8");
    OGRLayer *layer = dataset->CreateLayer(caseName.c_str(), &outputReference,
                                           wkbPolygon, options);
    CSLDestroy(options);
    if (layer == nullptr) {
        throw std::runtime_error("Cannot create output shapefile layer: " +
                                 pathToUtf8(outputPath));
    }
    OutputSchema schema = createOutputFields(*layer, sourceDefinition, height);
    for (const Record &record : records) {
        FeaturePtr outputFeature(
            OGRFeature::CreateFeature(layer->GetLayerDefn()));
        for (int sourceIndex = 0;
             sourceIndex < sourceDefinition.GetFieldCount(); ++sourceIndex) {
            const int outputIndex =
                schema.sourceToOutput[static_cast<std::size_t>(sourceIndex)];
            if (outputIndex >= 0) {
                copyFieldValue(*record.feature, sourceIndex, *outputFeature,
                               outputIndex);
            }
        }
        if (record.addedHeight && schema.heightIndex >= 0) {
            outputFeature->SetField(schema.heightIndex, *record.addedHeight);
        }
        if (record.addedCornerId && schema.cornerIndex >= 0) {
            outputFeature->SetField(schema.cornerIndex,
                                    static_cast<double>(*record.addedCornerId));
        }
        if (outputFeature->SetGeometry(geometryOf(record)) != OGRERR_NONE) {
            throw std::runtime_error("Cannot assign output footprint geometry");
        }
        if (layer->CreateFeature(outputFeature.get()) != OGRERR_NONE) {
            throw std::runtime_error("Cannot write output footprint feature");
        }
    }
    if (layer->SyncToDisk() != OGRERR_NONE) {
        throw std::runtime_error("Cannot flush output shapefile: " +
                                 pathToUtf8(outputPath));
    }
    return schema;
}

std::string formatFieldList(const std::vector<std::string> &fields) {
    std::string result = "[";
    for (std::size_t index = 0; index < fields.size(); ++index) {
        if (index != 0U)
            result += ", ";
        result += '\'' + fields[index] + '\'';
    }
    result += ']';
    return result;
}

struct PreviewCanvas {
    static constexpr int width = 2400;
    static constexpr int height = 2400;
    static constexpr int mapBottom = 470;
    static constexpr int margin = 55;

    std::vector<std::uint8_t> red = std::vector<std::uint8_t>(
        static_cast<std::size_t>(width * height), 255U);
    std::vector<std::uint8_t> green = red;
    std::vector<std::uint8_t> blue = red;
    OGREnvelope bounds{};

    std::pair<int, int> pixel(double x, double y) const {
        const double spanX = std::max(bounds.MaxX - bounds.MinX, 1e-12);
        const double spanY = std::max(bounds.MaxY - bounds.MinY, 1e-12);
        const double availableWidth = width - 2.0 * margin;
        const double availableHeight = height - mapBottom - 2.0 * margin;
        const double scale =
            std::min(availableWidth / spanX, availableHeight / spanY);
        const double drawnWidth = spanX * scale;
        const double drawnHeight = spanY * scale;
        const double offsetX = (width - drawnWidth) * 0.5;
        const double offsetY =
            mapBottom + margin + (availableHeight - drawnHeight) * 0.5;
        const int column =
            static_cast<int>(std::lround(offsetX + (x - bounds.MinX) * scale));
        const int row = static_cast<int>(
            std::lround(height - 1.0 - (offsetY + (y - bounds.MinY) * scale)));
        return {column, row};
    }

    void setPixel(int x, int y, std::array<std::uint8_t, 3> color) {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return;
        const std::size_t offset = static_cast<std::size_t>(y * width + x);
        red[offset] = color[0];
        green[offset] = color[1];
        blue[offset] = color[2];
    }

    void line(double x0, double y0, double x1, double y1,
              std::array<std::uint8_t, 3> color, int thickness = 1) {
        auto [startX, startY] = pixel(x0, y0);
        const auto [endX, endY] = pixel(x1, y1);
        const int deltaX = std::abs(endX - startX);
        const int stepX = startX < endX ? 1 : -1;
        const int deltaY = -std::abs(endY - startY);
        const int stepY = startY < endY ? 1 : -1;
        int error = deltaX + deltaY;
        for (;;) {
            for (int offsetY = -thickness; offsetY <= thickness; ++offsetY) {
                for (int offsetX = -thickness; offsetX <= thickness;
                     ++offsetX) {
                    setPixel(startX + offsetX, startY + offsetY, color);
                }
            }
            if (startX == endX && startY == endY)
                break;
            const int twiceError = 2 * error;
            if (twiceError >= deltaY) {
                error += deltaY;
                startX += stepX;
            }
            if (twiceError <= deltaX) {
                error += deltaX;
                startY += stepY;
            }
        }
    }
};

#if LUW_HAS_JPEG

constexpr int previewOutputScale = 16;
constexpr int previewOutputWidth = PreviewCanvas::width * previewOutputScale;
constexpr int previewOutputHeight = PreviewCanvas::height * previewOutputScale;
constexpr int previewOutputDpi = 4800;

struct JpegErrorManager {
    jpeg_error_mgr standard{};
    jmp_buf *jumpBuffer = nullptr;
    char *message = nullptr;
};

void jpegErrorExit(j_common_ptr compressor) {
    auto *error = reinterpret_cast<JpegErrorManager *>(compressor->err);
    (*compressor->err->format_message)(compressor, error->message);
    longjmp(*error->jumpBuffer, 1);
}

std::FILE *openExclusiveBinary(const fs::path &path, bool &alreadyExists,
                               int &errorNumber) {
    alreadyExists = false;
    errorNumber = 0;
#ifdef _WIN32
    int descriptor = -1;
    const errno_t status = _wsopen_s(
        &descriptor, path.c_str(),
        _O_BINARY | _O_WRONLY | _O_CREAT | _O_EXCL, _SH_DENYRW,
        _S_IREAD | _S_IWRITE);
    if (status != 0) {
        errorNumber = static_cast<int>(status);
        alreadyExists = status == EEXIST;
        return nullptr;
    }
    std::FILE *stream = _fdopen(descriptor, "wb");
    if (stream == nullptr) {
        errorNumber = errno;
        (void)_close(descriptor);
    }
    return stream;
#else
    const int descriptor =
        ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (descriptor < 0) {
        errorNumber = errno;
        alreadyExists = errno == EEXIST;
        return nullptr;
    }
    std::FILE *stream = fdopen(descriptor, "wb");
    if (stream == nullptr) {
        errorNumber = errno;
        (void)::close(descriptor);
    }
    return stream;
#endif
}

enum class PreviewWriteResult { Created, PreservedExisting };

PreviewWriteResult writeScaledPreviewJpeg(const PreviewCanvas &canvas,
                                          const fs::path &outputImage) {
    bool alreadyExists = false;
    int openError = 0;
    std::FILE *rawFile =
        openExclusiveBinary(outputImage, alreadyExists, openError);
    if (rawFile == nullptr) {
        if (alreadyExists) return PreviewWriteResult::PreservedExisting;
        throw std::runtime_error(
            "Cannot create native preview image exclusively: " +
            pathToUtf8(outputImage) + ": " +
            std::error_code(openError, std::generic_category()).message());
    }

    std::unique_ptr<std::FILE, decltype(&std::fclose)> outputFile(
        rawFile, &std::fclose);
    std::vector<JSAMPLE> scanline(
        static_cast<std::size_t>(previewOutputWidth) * 3U);
    jpeg_compress_struct compressor{};
    JpegErrorManager error{};
    jmp_buf jumpBuffer{};
    std::array<char, JMSG_LENGTH_MAX> errorMessage{};
    compressor.err = jpeg_std_error(&error.standard);
    error.standard.error_exit = jpegErrorExit;
    error.jumpBuffer = &jumpBuffer;
    error.message = errorMessage.data();

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4611)
#endif
    const int jumpStatus = setjmp(jumpBuffer);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    if (jumpStatus != 0) {
        jpeg_destroy_compress(&compressor);
        throw std::runtime_error("Cannot save native preview image: " +
                                 pathToUtf8(outputImage) + ": " +
                                 std::string(error.message));
    }

    jpeg_create_compress(&compressor);
    jpeg_stdio_dest(&compressor, outputFile.get());
    compressor.image_width = previewOutputWidth;
    compressor.image_height = previewOutputHeight;
    compressor.input_components = 3;
    compressor.in_color_space = JCS_RGB;
    jpeg_set_defaults(&compressor);
    jpeg_set_quality(&compressor, 95, TRUE);
    // GDAL's JPEG CreateCopy unconditionally enables optimized Huffman coding,
    // which makes libjpeg retain full-image coefficient arrays. Disabling it
    // keeps this 1.47-gigapixel compatibility image strictly scanline-bounded.
    compressor.optimize_coding = FALSE;
    compressor.density_unit = 1; // dots per inch
    compressor.X_density = previewOutputDpi;
    compressor.Y_density = previewOutputDpi;
    jpeg_start_compress(&compressor, TRUE);

    while (compressor.next_scanline < compressor.image_height) {
        const std::size_t sourceY =
            static_cast<std::size_t>(compressor.next_scanline) /
            static_cast<std::size_t>(previewOutputScale);
        const std::size_t sourceRow =
            sourceY * static_cast<std::size_t>(PreviewCanvas::width);
        std::size_t destination = 0U;
        for (int sourceX = 0; sourceX < PreviewCanvas::width; ++sourceX) {
            const std::size_t source =
                sourceRow + static_cast<std::size_t>(sourceX);
            for (int repeat = 0; repeat < previewOutputScale; ++repeat) {
                scanline[destination++] = canvas.red[source];
                scanline[destination++] = canvas.green[source];
                scanline[destination++] = canvas.blue[source];
            }
        }
        JSAMPROW row = scanline.data();
        if (jpeg_write_scanlines(&compressor, &row, 1) != 1U) {
            jpeg_abort_compress(&compressor);
            jpeg_destroy_compress(&compressor);
            throw std::runtime_error(
                "Cannot save native preview image scanline: " +
                pathToUtf8(outputImage));
        }
    }
    jpeg_finish_compress(&compressor);
    jpeg_destroy_compress(&compressor);
    return PreviewWriteResult::Created;
}

#endif // LUW_HAS_JPEG

void drawRing(PreviewCanvas &canvas, const OGRLineString &ring,
              std::array<std::uint8_t, 3> color, int thickness) {
    for (int index = 1; index < ring.getNumPoints(); ++index) {
        canvas.line(ring.getX(index - 1), ring.getY(index - 1),
                    ring.getX(index), ring.getY(index), color, thickness);
    }
}

void drawGeometry(PreviewCanvas &canvas, const OGRGeometry *geometry,
                  std::array<std::uint8_t, 3> outline,
                  std::array<std::uint8_t, 3> holeColor, int thickness) {
    if (geometry == nullptr || geometry->IsEmpty())
        return;
    const OGRwkbGeometryType type = wkbFlatten(geometry->getGeometryType());
    if (type == wkbPolygon) {
        const auto *polygon = dynamic_cast<const OGRPolygon *>(geometry);
        if (polygon == nullptr)
            return;
        if (const OGRLinearRing *exterior = polygon->getExteriorRing()) {
            drawRing(canvas, *exterior, outline, thickness);
        }
        for (int index = 0; index < polygon->getNumInteriorRings(); ++index) {
            if (const OGRLinearRing *ring = polygon->getInteriorRing(index)) {
                drawRing(canvas, *ring, holeColor, thickness);
            }
        }
        return;
    }
    if (type == wkbMultiPolygon) {
        const auto *collection =
            dynamic_cast<const OGRGeometryCollection *>(geometry);
        if (collection == nullptr)
            return;
        for (int index = 0; index < collection->getNumGeometries(); ++index) {
            drawGeometry(canvas, collection->getGeometryRef(index), outline,
                         holeColor, thickness);
        }
    }
}

std::pair<std::size_t, double>
holeStatistics(const std::vector<GeometryPtr> &geometries) {
    std::size_t count = 0;
    double area = 0.0;
    for (const GeometryPtr &geometry : geometries) {
        std::vector<const OGRPolygon *> polygons;
        const OGRwkbGeometryType type =
            geometry ? wkbFlatten(geometry->getGeometryType()) : wkbUnknown;
        if (type == wkbPolygon) {
            polygons.push_back(
                dynamic_cast<const OGRPolygon *>(geometry.get()));
        } else if (type == wkbMultiPolygon) {
            const auto *collection =
                dynamic_cast<const OGRGeometryCollection *>(geometry.get());
            for (int index = 0; collection != nullptr &&
                                index < collection->getNumGeometries();
                 ++index) {
                polygons.push_back(dynamic_cast<const OGRPolygon *>(
                    collection->getGeometryRef(index)));
            }
        }
        for (const OGRPolygon *polygon : polygons) {
            if (polygon == nullptr)
                continue;
            for (int index = 0; index < polygon->getNumInteriorRings();
                 ++index) {
                const OGRLinearRing *ring = polygon->getInteriorRing(index);
                if (ring == nullptr)
                    continue;
                OGRPolygon hole;
                if (hole.addRing(ring) == OGRERR_NONE) {
                    const double holeArea = geometryArea(&hole);
                    if (std::isfinite(holeArea)) {
                        ++count;
                        area += holeArea;
                    }
                }
            }
        }
    }
    return {count, area};
}

void saveOutlinePreview(const std::vector<Record> &records,
                        OGRSpatialReference &sourceReference,
                        OGRSpatialReference &previewReference,
                        const OGRGeometry *domainPreview,
                        const HeightField &height, const std::string &caseName,
                        const fs::path &sourceShapefile,
                        const fs::path &outputDirectory, bool autoDetectedUtm,
                        std::ostream &output, ProgressEmitter &progress) {
    output << "Preview step started. Preparing map outline and info panel.\n";
    progress.stage(QStringLiteral("preview"),
                   QStringLiteral("Generating preview image"));
    TransformPtr toPreview = coordinateTransform(
        &sourceReference, &previewReference, "preview projection");
    std::vector<GeometryPtr> projected;
    projected.reserve(records.size());
    for (const Record &record : records) {
        projected.push_back(transformedClone(geometryOf(record),
                                             toPreview.get(),
                                             "projecting preview footprints"));
    }
    if (autoDetectedUtm) {
        output << "Preview step uses auto detected UTM CRS for plotting: "
               << referenceText(previewReference) << ".\n";
    } else {
        output << "Preview step uses projected CRS: "
               << referenceText(previewReference) << '\n';
    }
    output << "Preview step detected height column: " << height.name << '\n';

    PreviewCanvas canvas;
    bool initialized = false;
    for (const GeometryPtr &geometry : projected) {
        if (!geometry || geometry->IsEmpty())
            continue;
        const OGREnvelope envelope = envelopeOf(geometry.get());
        if (!initialized) {
            canvas.bounds = envelope;
            initialized = true;
        } else {
            canvas.bounds.Merge(envelope);
        }
    }
    if (!initialized)
        canvas.bounds = envelopeOf(domainPreview);
    const double paddingX =
        std::max((canvas.bounds.MaxX - canvas.bounds.MinX) * 0.02, 1.0);
    const double paddingY =
        std::max((canvas.bounds.MaxY - canvas.bounds.MinY) * 0.02, 1.0);
    canvas.bounds.MinX -= paddingX;
    canvas.bounds.MaxX += paddingX;
    canvas.bounds.MinY -= paddingY;
    canvas.bounds.MaxY += paddingY;
    for (const GeometryPtr &geometry : projected) {
        drawGeometry(canvas, geometry.get(), {0U, 0U, 0U}, {0U, 70U, 220U}, 0);
    }
    drawGeometry(canvas, domainPreview, {240U, 140U, 0U}, {240U, 140U, 0U}, 1);

    const auto [holeCount, holeArea] = holeStatistics(projected);
    if (holeCount > 0U) {
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = 0.0;
        for (const GeometryPtr &geometry : projected) {
            std::vector<const OGRPolygon *> polygons;
            if (geometry &&
                wkbFlatten(geometry->getGeometryType()) == wkbPolygon) {
                polygons.push_back(
                    dynamic_cast<const OGRPolygon *>(geometry.get()));
            } else if (geometry && wkbFlatten(geometry->getGeometryType()) ==
                                       wkbMultiPolygon) {
                const auto *collection =
                    dynamic_cast<const OGRGeometryCollection *>(geometry.get());
                for (int index = 0; collection != nullptr &&
                                    index < collection->getNumGeometries();
                     ++index) {
                    polygons.push_back(dynamic_cast<const OGRPolygon *>(
                        collection->getGeometryRef(index)));
                }
            }
            for (const OGRPolygon *polygon : polygons) {
                for (int index = 0; polygon != nullptr &&
                                    index < polygon->getNumInteriorRings();
                     ++index) {
                    OGRPolygon hole;
                    (void)hole.addRing(polygon->getInteriorRing(index));
                    const double area = geometryArea(&hole);
                    if (std::isfinite(area)) {
                        minimum = std::min(minimum, area);
                        maximum = std::max(maximum, area);
                    }
                }
            }
        }
        output << "Preview step hole validation: holes=" << holeCount
               << ", area_sum=" << fixedDouble(holeArea, 3)
               << ", area_min=" << fixedDouble(minimum, 3)
               << ", area_max=" << fixedDouble(maximum, 3)
               << " (square meters)\n";
    } else {
        output << "Preview step hole validation: no interior rings detected.\n";
    }

    const fs::path outputImage = outputDirectory / (caseName + "_preview.jpg");
    (void)sourceShapefile;
#if LUW_HAS_JPEG
    output << "Preview raster size: " << previewOutputWidth << 'x'
           << previewOutputHeight
           << " pixels (8x8 inches at 4800 DPI).\n"
           << "Encoding preview JPEG scanline-by-scanline with bounded memory.\n";
    const PreviewWriteResult writeResult =
        writeScaledPreviewJpeg(canvas, outputImage);
    if (writeResult == PreviewWriteResult::PreservedExisting) {
        output << "Preview image already exists; preserving it without overwrite: "
               << pathToUtf8(outputImage) << '\n';
        progress.complete(
            QStringLiteral("preview"),
            QStringLiteral("Preserved existing preview image: %1")
                .arg(utf8QString(pathToUtf8(outputImage.filename()))));
        return;
    }
#else
    throw std::runtime_error(
        "Native 4800-DPI preview generation requires libjpeg-turbo; rebuild "
        "with JPEG::JPEG available");
#endif
    output << "Preview step completed. Saved preview image to "
           << pathToUtf8(outputImage) << '\n';
    progress.complete(
        QStringLiteral("preview"),
        QStringLiteral("Saved preview image: %1")
            .arg(utf8QString(pathToUtf8(outputImage.filename()))));
}

std::vector<Record> readFeatures(OGRLayer &layer) {
    std::vector<Record> result;
    const GIntBig count = layer.GetFeatureCount(FALSE);
    if (count > 0 && static_cast<unsigned long long>(count) <=
                         static_cast<unsigned long long>(
                             std::numeric_limits<std::size_t>::max())) {
        result.reserve(static_cast<std::size_t>(count));
    }
    layer.ResetReading();
    while (OGRFeature *feature = layer.GetNextFeature()) {
        Record record;
        record.feature.reset(feature);
        result.push_back(std::move(record));
    }
    return result;
}

std::vector<fs::path> shapefilesIn(const fs::path &directory) {
    std::vector<fs::path> result;
    std::error_code directoryError;
    if (!fs::is_directory(directory, directoryError))
        return result;
    std::error_code iteratorError;
    for (fs::directory_iterator iterator(directory, iteratorError), end;
         !iteratorError && iterator != end; iterator.increment(iteratorError)) {
        std::error_code typeError;
        if (!iterator->is_regular_file(typeError) || typeError)
            continue;
        const std::string extension = pathToUtf8(iterator->path().extension());
#ifdef _WIN32
        const bool isShapefile = lowerText(extension) == ".shp";
#else
        const bool isShapefile = extension == ".shp";
#endif
        if (isShapefile) {
            result.push_back(iterator->path());
        }
    }
    if (iteratorError &&
        iteratorError != std::errc::no_such_file_or_directory) {
        throw std::runtime_error(
            "Cannot inspect building database directory: " +
            pathToUtf8(directory) + ": " + iteratorError.message());
    }
    std::sort(result.begin(), result.end(),
              [](const fs::path &left, const fs::path &right) {
                  return pathToUtf8(left.filename()) <
                         pathToUtf8(right.filename());
              });
    return result;
}

bool previewRequested(const CommandArguments &arguments) {
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string value = lowerText(arguments[index]);
        if (value == "--preview" || value == "preview" || value == "1" ||
            value == "true" || value == "yes") {
            return true;
        }
    }
    return false;
}

bool validationRequested(const CommandArguments &arguments) {
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string value = lowerText(arguments[index]);
        if (value == "--validate" || value == "validate")
            return true;
    }
    return false;
}

int runLuwcutWithGdal(const CommandArguments &arguments, std::ostream &output,
                      std::ostream &) {
    output << "Program started.\n";
    ProgressEmitter progress(QStringLiteral("Cropping Geometry Domain"));
    progress.stage(QStringLiteral("load_shapefile"),
                   QStringLiteral("Reading source building shapefile"));
    output << "Start time: " << dateTimeNow() << '\n';
    if (arguments.empty()) {
        output
            << "Usage: python 2_shpCutter.py <path-to-conf-file> [--preview] "
               "[--validate]\n";
        return 2;
    }
    const fs::path configuration =
        fs::absolute(pathFromUtf8(arguments.front())).lexically_normal();
    if (!fs::exists(configuration)) {
        throw std::runtime_error("conf file not found: " +
                                 pathToUtf8(configuration));
    }
    const bool savePreview = previewRequested(arguments);
    const bool doubleValidation = validationRequested(arguments);
    const fs::path projectHome = configuration.parent_path();
    const DeckDocument deck =
        DeckDocument::load(utf8QString(pathToUtf8(configuration)));
    const PreprocessComputeConfig computePolicy =
        resolvePreprocessComputeConfig(&deck).first;
    output << "[INFO] Preprocessing compute policy: "
           << preprocessComputePolicyText(computePolicy).toStdString() << '\n'
           << "[INFO] Geometry cropping has no GPU compute kernel; this stage "
              "runs on CPU.\n";
    const std::string caseName =
        deck.textOr(QStringLiteral("casename")).toUtf8().toStdString();
    if (caseName.empty())
        throw std::runtime_error("casename not found in conf");
    const auto longitudePair = deck.pair(QStringLiteral("cut_lon_manual"));
    const auto latitudePair = deck.pair(QStringLiteral("cut_lat_manual"));
    if (!longitudePair || !latitudePair) {
        throw std::runtime_error(
            "Deck does not provide clipping range, missing "
            "cut_lon_manual/cut_lat_manual");
    }
    const double longitudeMinimum = longitudePair->first;
    const double longitudeMaximum = longitudePair->second;
    const double latitudeMinimum = latitudePair->first;
    const double latitudeMaximum = latitudePair->second;

    const fs::path buildingDirectory = projectHome / "building_db";
    const std::vector<fs::path> candidates = shapefilesIn(buildingDirectory);
    if (candidates.size() != 1U) {
        throw std::runtime_error("Expected exactly one .shp under " +
                                 pathToUtf8(buildingDirectory) + ", found " +
                                 std::to_string(candidates.size()));
    }
    const fs::path inputPath = candidates.front();
    const fs::path outputDirectory = projectHome / "proj_temp" / "cutted_shp";
    std::error_code createError;
    fs::create_directories(outputDirectory, createError);
    if (createError) {
        throw std::runtime_error(
            "Cannot create output directory: " + pathToUtf8(outputDirectory) +
            ": " + createError.message());
    }
    const fs::path outputPath = outputDirectory / (caseName + ".shp");
    output << "Input path: " << pathToUtf8(inputPath) << '\n'
           << "Output directory: " << pathToUtf8(outputDirectory) << '\n'
           << "Output shapefile: " << pathToUtf8(outputPath) << '\n'
           << "Case name: " << caseName << '\n'
           << "Bbox lon range: (" << pythonDouble(longitudeMinimum) << ", "
           << pythonDouble(longitudeMaximum) << ")\n"
           << "Bbox lat range: (" << pythonDouble(latitudeMinimum) << ", "
           << pythonDouble(latitudeMaximum) << ")\n"
           << "Reading input shapefile.\n";

    GDALAllRegister();
    DatasetPtr input(static_cast<GDALDataset *>(GDALOpenEx(
        pathToUtf8(inputPath).c_str(), GDAL_OF_VECTOR | GDAL_OF_READONLY,
        nullptr, nullptr, nullptr)));
    if (!input)
        throw std::runtime_error("Cannot read input shapefile: " +
                                 pathToUtf8(inputPath));
    OGRLayer *inputLayer = input->GetLayer(0);
    if (inputLayer == nullptr)
        throw std::runtime_error("Input shapefile has no readable layer");
    OGRFeatureDefn *sourceDefinition = inputLayer->GetLayerDefn();
    if (sourceDefinition == nullptr)
        throw std::runtime_error("Input shapefile has no feature schema");
    std::vector<Record> records = readFeatures(*inputLayer);
    output << "Read completed. Feature count: " << records.size() << '\n';

    std::unique_ptr<OGRSpatialReference> originalReference =
        cloneReference(inputLayer->GetSpatialRef());
    if (!originalReference) {
        output << "Notice: source SHP has no CRS. Assuming EPSG 4326 WGS84 "
                  "geographic.\n";
        originalReference = epsgReference(4326);
    }
    output << "Original CRS: " << referenceText(*originalReference) << '\n';
    auto wgs84 = epsgReference(4326);
    const bool originalIsWgs84 =
        identifiedEpsg(*originalReference) == std::optional<int>(4326);
    if (!originalIsWgs84) {
        output << "Reprojecting to EPSG 4326 for bbox based clipping.\n";
        TransformPtr toWgs84 =
            coordinateTransform(originalReference.get(), wgs84.get(),
                                "source reprojection to EPSG 4326");
        transformRecords(records, toWgs84.get(),
                         "reprojecting source footprints to EPSG 4326");
        output << "Reprojection to EPSG 4326 completed.\n";
    } else {
        output << "Data already in EPSG 4326. No reprojection needed.\n";
    }

    if (const std::optional<OGREnvelope> envelope = totalEnvelope(records)) {
        const LonLatBounds target{longitudeMinimum, longitudeMaximum,
                                  latitudeMinimum, latitudeMaximum};
        const LonLatBounds source{envelope->MinX, envelope->MaxX,
                                  envelope->MinY, envelope->MaxY};
        if (confirmCoverage("Building SHP", target, source, output) ==
            CoverageDecision::Cancel) {
            return 1;
        }
    } else {
        output << "[WARN] Building SHP bounds are slightly smaller than target "
                  "(max miss 0.0000% < 0.1%). Continue without interruption.\n";
    }

    output << "Bbox constructed. MinLon: " << pythonDouble(longitudeMinimum)
           << ", MinLat: " << pythonDouble(latitudeMinimum)
           << ", MaxLon: " << pythonDouble(longitudeMaximum)
           << ", MaxLat: " << pythonDouble(latitudeMaximum) << '\n';
    const double longitudeCenter = (longitudeMinimum + longitudeMaximum) * 0.5;
    const double latitudeCenter = (latitudeMinimum + latitudeMaximum) * 0.5;
    auto utmReference = epsgReference(utmEpsg(longitudeCenter, latitudeCenter));
    TransformPtr wgsToUtm = coordinateTransform(wgs84.get(), utmReference.get(),
                                                "clipping-domain projection");
    TransformPtr utmToWgs = coordinateTransform(
        utmReference.get(), wgs84.get(), "clipping-domain inverse projection");
    RotatedDomain domain =
        makeRotatedDomain(longitudeMinimum, longitudeMaximum, latitudeMinimum,
                          latitudeMaximum, wgsToUtm.get());

    progress.stage(QStringLiteral("clip_domain"),
                   QStringLiteral("Cropping buildings to rotated CFD domain"));
    output << "Clipping to rotated-rectangle bbox (four-corner projected). "
              "rotate_deg="
           << fixedDouble(domain.rotationDegrees, 6) << ", pivot=("
           << fixedDouble(domain.pivotX, 3) << ", "
           << fixedDouble(domain.pivotY, 3) << ")\n";
    records = cleanBuildingGeometries(std::move(records), output, progress);
    transformRecords(records, wgsToUtm.get(),
                     "projecting footprints for clipping");
    records = clipRecords(std::move(records), domain.polygon.get());
    transformRecords(records, utmToWgs.get(),
                     "returning clipped footprints to EPSG 4326");
    output << "Clipping completed. Features after clip: " << records.size()
           << '\n';

    if (doubleValidation) {
        output << "Starting geometry cleaning before area filtering.\n";
        records = cleanBuildingGeometries(std::move(records), output, progress);
        output << "Geometry cleaning finished. Features after cleaning: "
               << records.size() << '\n';
    } else {
        output
            << "Geometry cleaning skipped because --validate not provided.\n";
    }

    const HeightField height = selectHeightField(*sourceDefinition);
    output << "Height column for processing: " << height.name << '\n'
           << "Starting overlap union on cleaned geometries.\n";
    records = unionOverlappingBuildings(std::move(records), height,
                                        *sourceDefinition, output, progress);
    output << "Overlap union finished. Features after union: " << records.size()
           << '\n'
           << "Area filter step started. Minimum area: 20.0 square meters.\n";
    if (!records.empty()) {
        records = filterByArea(std::move(records), output, progress);
    } else {
        output << "Area filter skipped because there are no geometries after "
                  "union.\n";
    }
    output << "Features after area filter: " << records.size() << '\n';
    progress.stage(QStringLiteral("remove_holes"),
                   QStringLiteral("Removing small interior holes"));
    output << "Small interior rings removal started. Threshold: 5.0 m^2.\n";
    std::size_t removedHoles = 0;
    if (!records.empty()) {
        removedHoles = removeSmallHoles(records, wgsToUtm.get(), utmToWgs.get(),
                                        output, progress);
    }
    output << "Small interior rings removed: " << removedHoles << '\n'
           << "Features after hole removal: " << records.size() << '\n'
           << "Adding four 1m by 1m corner squares inside the bbox.\n";

    std::vector<Record> cornerRecords = makeCornerRecords(
        domain.polygon.get(), *sourceDefinition, height, utmToWgs.get());
    output << "Corner squares prepared and reprojected to EPSG 4326.\n";
    records.reserve(records.size() + cornerRecords.size());
    for (Record &record : cornerRecords)
        records.push_back(std::move(record));
    output << "Corner squares appended. Total features now: " << records.size()
           << '\n';

    if (!originalIsWgs84) {
        output << "Reprojecting result back to original CRS: "
               << referenceText(*originalReference) << '\n';
        TransformPtr toOriginal =
            coordinateTransform(wgs84.get(), originalReference.get(),
                                "final source-CRS reprojection");
        transformRecords(records, toOriginal.get(),
                         "reprojecting results to the original CRS");
        output << "Reprojection to original CRS completed.\n";
    } else {
        output << "Original CRS is EPSG 4326. No final reprojection needed.\n";
    }
    records = finalPolygonalFilter(std::move(records), output);
    progress.stage(QStringLiteral("save_crop"),
                   QStringLiteral("Saving cropped shapefile: %1")
                       .arg(utf8QString(pathToUtf8(outputPath.filename()))));
    output << "Writing output shapefile.\n";
    const OutputSchema outputSchema =
        writeShapefile(outputPath, caseName, *originalReference,
                       *sourceDefinition, height, records);
    output << "Write completed.\n";
    progress.complete(QStringLiteral("save_crop"),
                      QStringLiteral("Saved cropped shapefile: %1")
                          .arg(utf8QString(pathToUtf8(outputPath.filename()))));
    output << "Processing finished. Feature count written: " << records.size()
           << '\n'
           << "Output file path: "
           << pathToUtf8(fs::absolute(outputPath).lexically_normal()) << '\n';
    if (!outputSchema.displayColumns.empty()) {
        output << "Output fields example:\n"
               << formatFieldList(outputSchema.displayColumns) << '\n';
    }

    if (savePreview) {
        output << "Generating preview image.\n";
        const bool autoDetectedUtm = !originalReference->IsProjected();
        std::unique_ptr<OGRSpatialReference> previewReference =
            autoDetectedUtm ? cloneReference(utmReference.get())
                            : cloneReference(originalReference.get());
        const std::array<std::pair<double, double>, 4> normalBounds{{
            {longitudeMaximum, latitudeMinimum},
            {longitudeMaximum, latitudeMaximum},
            {longitudeMinimum, latitudeMaximum},
            {longitudeMinimum, latitudeMinimum},
        }};
        GeometryPtr previewBounds = rectanglePolygon(normalBounds);
        TransformPtr boundsToPreview = coordinateTransform(
            wgs84.get(), previewReference.get(), "preview-bounds projection");
        if (previewBounds->transform(boundsToPreview.get()) != OGRERR_NONE) {
            throw std::runtime_error("Coordinate transformation failed while "
                                     "projecting preview bounds");
        }
        saveOutlinePreview(records, *originalReference, *previewReference,
                           previewBounds.get(), height, caseName, inputPath,
                           outputDirectory, autoDetectedUtm, output, progress);
        output << "Preview generation finished.\n";
    } else {
        output << "Preview generation skipped. Use --preview to enable.\n";
    }
    output << "Program finished. End time: " << dateTimeNow()
           << ". Exiting now.\n";
    return 0;
}

#endif // LUW_HAS_GDAL

} // namespace

int run_luwcut(const CommandArguments &arguments, std::ostream &output,
               std::ostream &error) {
#if LUW_HAS_GDAL
    return runLuwcutWithGdal(arguments, output, error);
#else
    (void)arguments;
    (void)output;
    error
        << "ERROR: Native luwcut requires GDAL/OGR with GEOS support; rebuild "
           "with "
           "LUW_ENABLE_GEOSPATIAL=ON and a discoverable GDAL installation.\n";
    return 1;
#endif
}

} // namespace luw::cli
