#pragma once

#include <QColor>
#include <QImage>
#include <QJsonObject>
#include <QSize>
#include <QString>
#include <QStringList>
#include <QVector>

class vtkLookupTable;

namespace Streamcenter {

struct ColorMapPoint {
    double x = 0.0;
    QColor color = QColor::fromRgbF(0.0, 0.0, 0.0);
};

struct ColorMapDefinition {
    QString name;
    QString colorSpace = QStringLiteral("RGB");
    QColor nanColor = QColor::fromRgbF(0.5, 0.5, 0.5);
    QVector<ColorMapPoint> points;
    QJsonObject source;
};

QString defaultColorMapsFilePath();
QString projectColorMapsFilePath(const QString& visualizationDirectory);
bool ensureProjectColorMapsFile(const QString& visualizationDirectory,
                                QString* errorMessage = nullptr,
                                bool* changed = nullptr);

bool loadColorMapsFile(const QString& path,
                       QVector<ColorMapDefinition>* definitions,
                       QString* errorMessage = nullptr);
bool writeColorMapsFile(const QString& path,
                        const QVector<ColorMapDefinition>& definitions,
                        QString* errorMessage = nullptr);
bool appendImportedColorMaps(const QString& projectColorMapsPath,
                             const QString& importPath,
                             QStringList* importedNames = nullptr,
                             QString* errorMessage = nullptr);

QString firstColorMapName(const QVector<ColorMapDefinition>& definitions);
const ColorMapDefinition* findColorMap(const QVector<ColorMapDefinition>& definitions, const QString& name);
QString uniqueColorMapName(const QVector<ColorMapDefinition>& definitions,
                           const QString& preferredName,
                           const QString& ignoredName = QString());
QJsonObject colorMapToJsonObject(const ColorMapDefinition& definition);

QColor sampleColorMap(const ColorMapDefinition& definition, double t);
QImage renderColorMapPreview(const ColorMapDefinition& definition, const QSize& size);
void fillLookupTable(vtkLookupTable* table, const ColorMapDefinition& definition, int tableSize = 256);

void setActiveColorMapsFile(const QString& path);
QString activeColorMapsFile();
bool reloadActiveColorMaps(QString* errorMessage = nullptr);
QVector<ColorMapDefinition> activeColorMaps();
const ColorMapDefinition* activeColorMapByName(const QString& name);
QString activeDefaultColorMapName();

} // namespace Streamcenter
