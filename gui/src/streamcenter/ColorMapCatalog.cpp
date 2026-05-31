#include "ColorMapCatalog.h"

#include <vtkColorTransferFunction.h>
#include <vtkLookupTable.h>
#include <vtkSmartPointer.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QObject>
#include <QSaveFile>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Streamcenter {
namespace {

constexpr auto kColorMapsFileName = "ColorMaps.json";

QString normalizedPath(const QString& path) {
    return QDir::fromNativeSeparators(QDir::cleanPath(path));
}

QVector<ColorMapDefinition>& activeDefinitionsStorage() {
    static QVector<ColorMapDefinition> definitions;
    return definitions;
}

QString& activeFileStorage() {
    static QString path;
    return path;
}

bool isFinite(double value) {
    return std::isfinite(value);
}

double normalizedColorComponent(double value) {
    if (!isFinite(value)) {
        return 0.0;
    }
    if (value > 1.0 && value <= 255.0) {
        value /= 255.0;
    }
    return std::clamp(value, 0.0, 1.0);
}

QColor colorFromJsonArray(const QJsonArray& array, const QColor& fallback) {
    if (array.size() < 3) {
        return fallback;
    }
    const double red = normalizedColorComponent(array.at(0).toDouble(fallback.redF()));
    const double green = normalizedColorComponent(array.at(1).toDouble(fallback.greenF()));
    const double blue = normalizedColorComponent(array.at(2).toDouble(fallback.blueF()));
    const double alpha = array.size() >= 4
        ? normalizedColorComponent(array.at(3).toDouble(fallback.alphaF()))
        : fallback.alphaF();
    return QColor::fromRgbF(red, green, blue, alpha);
}

QJsonArray colorToJsonArray(const QColor& color) {
    QJsonArray array;
    array.append(color.redF());
    array.append(color.greenF());
    array.append(color.blueF());
    if (color.alphaF() < 1.0) {
        array.append(color.alphaF());
    }
    return array;
}

QJsonArray pointsToJsonArray(const QVector<ColorMapPoint>& points) {
    QJsonArray array;
    for (const ColorMapPoint& point : points) {
        array.append(point.x);
        array.append(point.color.redF());
        array.append(point.color.greenF());
        array.append(point.color.blueF());
    }
    return array;
}

QJsonArray colorMapArrayFromDocument(const QJsonDocument& document) {
    if (document.isArray()) {
        return document.array();
    }
    if (!document.isObject()) {
        return {};
    }

    const QJsonObject object = document.object();
    for (const QString& key : {QStringLiteral("ColorMaps"),
                               QStringLiteral("colorMaps"),
                               QStringLiteral("colormaps"),
                               QStringLiteral("Maps"),
                               QStringLiteral("maps")}) {
        if (object.value(key).isArray()) {
            return object.value(key).toArray();
        }
    }
    if (object.value(QStringLiteral("Name")).isString()
        && object.value(QStringLiteral("RGBPoints")).isArray()) {
        QJsonArray array;
        array.append(object);
        return array;
    }
    return {};
}

QByteArray stripJsonTrailingCommas(const QByteArray& input) {
    QByteArray output;
    output.reserve(input.size());

    bool inString = false;
    bool escaped = false;
    for (qsizetype index = 0; index < input.size(); ++index) {
        const char ch = input.at(index);
        if (inString) {
            output.append(ch);
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                inString = false;
            }
            continue;
        }

        if (ch == '"') {
            inString = true;
            output.append(ch);
            continue;
        }

        if (ch == ',') {
            qsizetype lookahead = index + 1;
            while (lookahead < input.size()
                   && (input.at(lookahead) == ' '
                       || input.at(lookahead) == '\t'
                       || input.at(lookahead) == '\r'
                       || input.at(lookahead) == '\n')) {
                ++lookahead;
            }
            if (lookahead < input.size()
                && (input.at(lookahead) == ']' || input.at(lookahead) == '}')) {
                continue;
            }
        }

        output.append(ch);
    }
    return output;
}

bool parseColorMapObject(const QJsonObject& object,
                         ColorMapDefinition* definition,
                         QString* errorMessage) {
    if (definition == nullptr) {
        return false;
    }

    ColorMapDefinition parsed;
    parsed.source = object;
    parsed.name = object.value(QStringLiteral("Name")).toString().trimmed();
    if (parsed.name.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Color map entry is missing a non-empty Name field.");
        }
        return false;
    }

    parsed.colorSpace = object.value(QStringLiteral("ColorSpace")).toString(QStringLiteral("RGB")).trimmed();
    if (parsed.colorSpace.isEmpty()) {
        parsed.colorSpace = QStringLiteral("RGB");
    }
    parsed.nanColor = colorFromJsonArray(object.value(QStringLiteral("NanColor")).toArray(), parsed.nanColor);

    const QJsonArray rgbPoints = object.value(QStringLiteral("RGBPoints")).toArray();
    if (rgbPoints.size() < 8 || rgbPoints.size() % 4 != 0) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Color map \"%1\" has an invalid RGBPoints array.").arg(parsed.name);
        }
        return false;
    }

    for (int index = 0; index + 3 < rgbPoints.size(); index += 4) {
        const double x = rgbPoints.at(index).toDouble(std::numeric_limits<double>::quiet_NaN());
        const double r = rgbPoints.at(index + 1).toDouble(std::numeric_limits<double>::quiet_NaN());
        const double g = rgbPoints.at(index + 2).toDouble(std::numeric_limits<double>::quiet_NaN());
        const double b = rgbPoints.at(index + 3).toDouble(std::numeric_limits<double>::quiet_NaN());
        if (!isFinite(x) || !isFinite(r) || !isFinite(g) || !isFinite(b)) {
            if (errorMessage != nullptr) {
                *errorMessage = QObject::tr("Color map \"%1\" contains a non-numeric RGBPoints value.").arg(parsed.name);
            }
            return false;
        }
        ColorMapPoint point;
        point.x = x;
        point.color = QColor::fromRgbF(normalizedColorComponent(r),
                                       normalizedColorComponent(g),
                                       normalizedColorComponent(b));
        parsed.points.push_back(point);
    }

    std::stable_sort(parsed.points.begin(), parsed.points.end(), [](const ColorMapPoint& lhs, const ColorMapPoint& rhs) {
        return lhs.x < rhs.x;
    });
    if (parsed.points.size() < 2 || !(parsed.points.back().x > parsed.points.front().x)) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Color map \"%1\" needs at least two distinct control point positions.").arg(parsed.name);
        }
        return false;
    }

    *definition = parsed;
    return true;
}

vtkSmartPointer<vtkColorTransferFunction> makeTransferFunction(const ColorMapDefinition& definition) {
    vtkSmartPointer<vtkColorTransferFunction> transfer = vtkSmartPointer<vtkColorTransferFunction>::New();
    const QString colorSpace = definition.colorSpace.trimmed().toLower();
    if (colorSpace.contains(QStringLiteral("ciede2000"))) {
        transfer->SetColorSpaceToLabCIEDE2000();
    } else if (colorSpace.contains(QStringLiteral("diverging"))) {
        transfer->SetColorSpaceToDiverging();
    } else if (colorSpace.contains(QStringLiteral("lab")) || colorSpace.contains(QStringLiteral("cielab"))) {
        transfer->SetColorSpaceToLab();
    } else if (colorSpace.contains(QStringLiteral("hsv"))) {
        transfer->SetColorSpaceToHSV();
    } else if (colorSpace.contains(QStringLiteral("step"))) {
        transfer->SetColorSpaceToStep();
    } else {
        transfer->SetColorSpaceToRGB();
    }
    transfer->ClampingOn();
    for (const ColorMapPoint& point : definition.points) {
        transfer->AddRGBPoint(point.x, point.color.redF(), point.color.greenF(), point.color.blueF());
    }
    return transfer;
}

QColor sampleTransfer(vtkColorTransferFunction* transfer,
                      const ColorMapDefinition& definition,
                      double t) {
    if (transfer == nullptr || definition.points.isEmpty()) {
        return QColor::fromRgbF(0.0, 0.0, 0.0);
    }
    if (definition.points.size() == 1) {
        return definition.points.front().color;
    }
    const double boundedT = std::clamp(t, 0.0, 1.0);
    const double minX = definition.points.front().x;
    const double maxX = definition.points.back().x;
    const double x = minX + boundedT * (maxX - minX);
    double rgb[3] = {0.0, 0.0, 0.0};
    transfer->GetColor(x, rgb);
    return QColor::fromRgbF(normalizedColorComponent(rgb[0]),
                            normalizedColorComponent(rgb[1]),
                            normalizedColorComponent(rgb[2]));
}

} // namespace

QString defaultColorMapsFilePath() {
    return normalizedPath(QDir(QString::fromUtf8(STREAMCENTERPLUS_REPO_ROOT))
                              .absoluteFilePath(QStringLiteral("config/") + QString::fromLatin1(kColorMapsFileName)));
}

QString projectColorMapsFilePath(const QString& visualizationDirectory) {
    return normalizedPath(QDir(visualizationDirectory).absoluteFilePath(QString::fromLatin1(kColorMapsFileName)));
}

bool ensureProjectColorMapsFile(const QString& visualizationDirectory, QString* errorMessage, bool* changed) {
    if (changed != nullptr) {
        *changed = false;
    }
    if (visualizationDirectory.trimmed().isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Project visualization directory is empty.");
        }
        return false;
    }

    if (!QDir().mkpath(visualizationDirectory)) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Cannot create visualization directory: %1").arg(visualizationDirectory);
        }
        return false;
    }

    const QString path = projectColorMapsFilePath(visualizationDirectory);
    const QString defaultPath = defaultColorMapsFilePath();
    if (!QFileInfo::exists(defaultPath)) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Default color map file was not found: %1").arg(defaultPath);
        }
        return false;
    }

    if (QFileInfo::exists(path)) {
        QVector<ColorMapDefinition> projectDefinitions;
        if (!loadColorMapsFile(path, &projectDefinitions, errorMessage)) {
            return false;
        }

        QVector<ColorMapDefinition> defaultDefinitions;
        if (!loadColorMapsFile(defaultPath, &defaultDefinitions, nullptr)) {
            return true;
        }

        bool appended = false;
        for (const ColorMapDefinition& definition : std::as_const(defaultDefinitions)) {
            if (findColorMap(projectDefinitions, definition.name) != nullptr) {
                continue;
            }
            projectDefinitions.push_back(definition);
            appended = true;
        }

        if (!appended) {
            return true;
        }
        if (!writeColorMapsFile(path, projectDefinitions, errorMessage)) {
            return false;
        }
        if (changed != nullptr) {
            *changed = true;
        }
        return true;
    }

    QVector<ColorMapDefinition> defaultDefinitions;
    if (!loadColorMapsFile(defaultPath, &defaultDefinitions, errorMessage)) {
        return false;
    }

    if (!QFile::copy(defaultPath, path)) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Cannot copy default color maps to project: %1").arg(path);
        }
        return false;
    }
    if (changed != nullptr) {
        *changed = true;
    }
    return true;
}

bool loadColorMapsFile(const QString& path,
                       QVector<ColorMapDefinition>* definitions,
                       QString* errorMessage) {
    if (definitions != nullptr) {
        definitions->clear();
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Cannot open color map file %1: %2").arg(path, file.errorString());
        }
        return false;
    }

    const QByteArray contents = file.readAll();
    QJsonParseError parseError{};
    QJsonDocument document = QJsonDocument::fromJson(contents, &parseError);
    if (parseError.error != QJsonParseError::NoError) {
        QJsonParseError tolerantParseError{};
        const QByteArray repairedContents = stripJsonTrailingCommas(contents);
        const QJsonDocument repairedDocument = QJsonDocument::fromJson(repairedContents, &tolerantParseError);
        if (tolerantParseError.error == QJsonParseError::NoError) {
            document = repairedDocument;
            parseError = tolerantParseError;
        }
    }
    if (parseError.error != QJsonParseError::NoError) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Cannot parse color map file %1: %2").arg(path, parseError.errorString());
        }
        return false;
    }

    const QJsonArray array = colorMapArrayFromDocument(document);
    if (array.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Color map file %1 does not contain any color map entries.").arg(path);
        }
        return false;
    }

    QVector<ColorMapDefinition> parsed;
    QStringList errors;
    for (const QJsonValue& value : array) {
        if (!value.isObject()) {
            continue;
        }
        ColorMapDefinition definition;
        QString entryError;
        if (parseColorMapObject(value.toObject(), &definition, &entryError)) {
            parsed.push_back(definition);
        } else if (!entryError.isEmpty()) {
            errors.push_back(entryError);
        }
    }

    if (parsed.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = errors.isEmpty()
                ? QObject::tr("Color map file %1 did not contain a valid color map.").arg(path)
                : errors.join(QStringLiteral("\n"));
        }
        return false;
    }

    if (definitions != nullptr) {
        *definitions = parsed;
    }
    return true;
}

bool writeColorMapsFile(const QString& path,
                        const QVector<ColorMapDefinition>& definitions,
                        QString* errorMessage) {
    if (definitions.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Cannot save an empty color map list.");
        }
        return false;
    }

    const QFileInfo info(path);
    if (!QDir().mkpath(info.absolutePath())) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Cannot create color map directory: %1").arg(info.absolutePath());
        }
        return false;
    }

    QJsonArray array;
    for (const ColorMapDefinition& definition : definitions) {
        array.append(colorMapToJsonObject(definition));
    }

    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Cannot write color map file %1: %2").arg(path, file.errorString());
        }
        return false;
    }
    file.write(QJsonDocument(array).toJson(QJsonDocument::Indented));
    if (!file.commit()) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Cannot commit color map file %1: %2").arg(path, file.errorString());
        }
        return false;
    }
    return true;
}

bool appendImportedColorMaps(const QString& projectColorMapsPath,
                             const QString& importPath,
                             QStringList* importedNames,
                             QString* errorMessage) {
    if (importedNames != nullptr) {
        importedNames->clear();
    }

    QVector<ColorMapDefinition> existing;
    if (QFileInfo::exists(projectColorMapsPath)
        && !loadColorMapsFile(projectColorMapsPath, &existing, errorMessage)) {
        return false;
    }

    QVector<ColorMapDefinition> imported;
    if (!loadColorMapsFile(importPath, &imported, errorMessage)) {
        return false;
    }

    for (ColorMapDefinition definition : imported) {
        definition.name = uniqueColorMapName(existing, definition.name);
        definition.source.insert(QStringLiteral("Name"), definition.name);
        existing.push_back(definition);
        if (importedNames != nullptr) {
            importedNames->push_back(definition.name);
        }
    }

    return writeColorMapsFile(projectColorMapsPath, existing, errorMessage);
}

QString firstColorMapName(const QVector<ColorMapDefinition>& definitions) {
    return definitions.isEmpty() ? QString() : definitions.front().name;
}

const ColorMapDefinition* findColorMap(const QVector<ColorMapDefinition>& definitions, const QString& name) {
    const QString trimmed = name.trimmed();
    if (trimmed.isEmpty()) {
        return nullptr;
    }
    for (const ColorMapDefinition& definition : definitions) {
        if (definition.name.compare(trimmed, Qt::CaseInsensitive) == 0) {
            return &definition;
        }
    }
    return nullptr;
}

QString uniqueColorMapName(const QVector<ColorMapDefinition>& definitions,
                           const QString& preferredName,
                           const QString& ignoredName) {
    const QString base = preferredName.trimmed().isEmpty()
        ? QObject::tr("Color map")
        : preferredName.trimmed();
    auto exists = [&](const QString& candidate) {
        if (!ignoredName.trimmed().isEmpty()
            && candidate.compare(ignoredName.trimmed(), Qt::CaseInsensitive) == 0) {
            return false;
        }
        return findColorMap(definitions, candidate) != nullptr;
    };

    if (!exists(base)) {
        return base;
    }

    int suffix = 2;
    while (true) {
        const QString candidate = QStringLiteral("%1 (%2)").arg(base).arg(suffix++);
        if (!exists(candidate)) {
            return candidate;
        }
    }
}

QJsonObject colorMapToJsonObject(const ColorMapDefinition& definition) {
    QJsonObject object = definition.source;
    object.insert(QStringLiteral("Name"), definition.name);
    object.insert(QStringLiteral("ColorSpace"), definition.colorSpace.trimmed().isEmpty()
                      ? QStringLiteral("RGB")
                      : definition.colorSpace.trimmed());
    object.insert(QStringLiteral("NanColor"), colorToJsonArray(definition.nanColor));
    object.insert(QStringLiteral("RGBPoints"), pointsToJsonArray(definition.points));
    return object;
}

QColor sampleColorMap(const ColorMapDefinition& definition, double t) {
    if (definition.points.isEmpty()) {
        return QColor::fromRgbF(0.0, 0.0, 0.0);
    }
    vtkSmartPointer<vtkColorTransferFunction> transfer = makeTransferFunction(definition);
    return sampleTransfer(transfer.GetPointer(), definition, t);
}

QImage renderColorMapPreview(const ColorMapDefinition& definition, const QSize& size) {
    const QSize boundedSize(std::max(1, size.width()), std::max(1, size.height()));
    QImage image(boundedSize, QImage::Format_ARGB32_Premultiplied);
    if (definition.points.isEmpty()) {
        image.fill(QColor(128, 128, 128).rgba());
        return image;
    }

    vtkSmartPointer<vtkColorTransferFunction> transfer = makeTransferFunction(definition);
    for (int x = 0; x < image.width(); ++x) {
        const double t = image.width() <= 1 ? 0.0 : static_cast<double>(x) / static_cast<double>(image.width() - 1);
        const QColor color = sampleTransfer(transfer.GetPointer(), definition, t);
        for (int y = 0; y < image.height(); ++y) {
            image.setPixelColor(x, y, color);
        }
    }
    return image;
}

void fillLookupTable(vtkLookupTable* table, const ColorMapDefinition& definition, int tableSize) {
    if (table == nullptr) {
        return;
    }
    const int boundedSize = std::max(2, tableSize);
    table->SetNumberOfTableValues(boundedSize);
    table->SetTableRange(0.0, 1.0);
    vtkSmartPointer<vtkColorTransferFunction> transfer = makeTransferFunction(definition);
    for (int index = 0; index < boundedSize; ++index) {
        const double t = boundedSize <= 1 ? 0.0 : static_cast<double>(index) / static_cast<double>(boundedSize - 1);
        const QColor color = sampleTransfer(transfer.GetPointer(), definition, t);
        table->SetTableValue(index, color.redF(), color.greenF(), color.blueF(), 1.0);
    }
    table->SetNanColor(definition.nanColor.redF(),
                       definition.nanColor.greenF(),
                       definition.nanColor.blueF(),
                       definition.nanColor.alphaF());
    table->Build();
}

void setActiveColorMapsFile(const QString& path) {
    activeFileStorage() = normalizedPath(path);
    reloadActiveColorMaps(nullptr);
}

QString activeColorMapsFile() {
    return activeFileStorage();
}

bool reloadActiveColorMaps(QString* errorMessage) {
    activeDefinitionsStorage().clear();
    const QString path = activeFileStorage();
    if (path.trimmed().isEmpty()) {
        return true;
    }
    return loadColorMapsFile(path, &activeDefinitionsStorage(), errorMessage);
}

QVector<ColorMapDefinition> activeColorMaps() {
    return activeDefinitionsStorage();
}

const ColorMapDefinition* activeColorMapByName(const QString& name) {
    return findColorMap(activeDefinitionsStorage(), name);
}

QString activeDefaultColorMapName() {
    return firstColorMapName(activeDefinitionsStorage());
}

} // namespace Streamcenter
