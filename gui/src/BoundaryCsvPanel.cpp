#include "luwgui/BoundaryCsvPanel.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QRegularExpression>
#include <QSignalBlocker>
#include <QTextStream>
#include <QVBoxLayout>

#include <QVTKOpenGLNativeWidget.h>

#include <vtkActor.h>
#include <vtkCellArray.h>
#include <vtkFloatArray.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkLookupTable.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkScalarBarActor.h>
#include <vtkTextProperty.h>
#include <vtkVertexGlyphFilter.h>

#include <algorithm>
#include <cmath>

namespace luwgui {

namespace {

QStringList splitCsvRow(const QString& line) {
    QStringList out;
    QString current;
    bool inQuotes = false;
    for (const QChar ch : line) {
        if (ch == '"') {
            inQuotes = !inQuotes;
            continue;
        }
        if (ch == ',' && !inQuotes) {
            out.push_back(current.trimmed());
            current.clear();
            continue;
        }
        current += ch;
    }
    out.push_back(current.trimmed());
    return out;
}

int findHeaderIndex(const QStringList& headers, const QString& key) {
    for (int i = 0; i < headers.size(); ++i) {
        if (headers[i].trimmed().compare(key, Qt::CaseInsensitive) == 0) {
            return i;
        }
    }
    return -1;
}

bool readDoubleAt(const QStringList& columns, int index, double* value) {
    if (!value || index < 0 || index >= columns.size()) {
        return false;
    }
    bool ok = false;
    const double parsed = columns[index].trimmed().toDouble(&ok);
    if (!ok) {
        return false;
    }
    *value = parsed;
    return true;
}

vtkSmartPointer<vtkLookupTable> buildLookupTable() {
    auto lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetNumberOfTableValues(256);
    lut->Build();
    for (int i = 0; i < 256; ++i) {
        const double t = static_cast<double>(i) / 255.0;
        const double r = std::clamp(1.4 * t, 0.0, 1.0);
        const double g = std::clamp(1.2 * (1.0 - std::abs(2.0 * t - 1.0)), 0.0, 1.0);
        const double b = std::clamp(1.25 * (1.0 - t), 0.0, 1.0);
        lut->SetTableValue(i, r, g, b, 1.0);
    }
    return lut;
}

void setLegendTextStyle(vtkTextProperty* property, int fontSize) {
    if (!property) {
        return;
    }
    property->SetColor(0.0, 0.0, 0.0);
    property->SetBold(false);
    property->SetItalic(false);
    property->SetShadow(false);
    property->SetFontSize(fontSize);
}

} // namespace

BoundaryCsvPanel::BoundaryCsvPanel(QWidget* parent)
    : QWidget(parent) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(6);

    auto* header = new QWidget(this);
    auto* headerLayout = new QGridLayout(header);
    headerLayout->setContentsMargins(0, 0, 0, 0);
    headerLayout->addWidget(new QLabel("Boundary CSV"), 0, 0);
    fileLabel_ = new QLabel("No CSV loaded", header);
    fileLabel_->setWordWrap(true);
    headerLayout->addWidget(fileLabel_, 0, 1, 1, 3);
    auto* reloadButton = new QPushButton("Reload", header);
    headerLayout->addWidget(reloadButton, 0, 4);
    headerLayout->addWidget(new QLabel("Color"), 1, 0);
    colorCombo_ = new QComboBox(header);
    headerLayout->addWidget(colorCombo_, 1, 1, 1, 2);
    root->addWidget(header);

    vtkWidget_ = new QVTKOpenGLNativeWidget(this);
    root->addWidget(vtkWidget_, 1);

    renderWindow_ = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    renderer_ = vtkSmartPointer<vtkRenderer>::New();
    renderWindow_->AddRenderer(renderer_);
    vtkWidget_->setRenderWindow(renderWindow_);
    renderer_->SetBackground(1.0, 1.0, 1.0);

    polyData_ = vtkSmartPointer<vtkPolyData>::New();
    glyphFilter_ = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    glyphFilter_->SetInputData(polyData_);
    mapper_ = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper_->SetInputConnection(glyphFilter_->GetOutputPort());
    actor_ = vtkSmartPointer<vtkActor>::New();
    actor_->SetMapper(mapper_);
    actor_->GetProperty()->SetRepresentationToPoints();
    actor_->GetProperty()->SetPointSize(4.0);
    actor_->GetProperty()->RenderPointsAsSpheresOn();
    renderer_->AddActor(actor_);

    lookupTable_ = buildLookupTable();
    scalarBar_ = vtkSmartPointer<vtkScalarBarActor>::New();
    scalarBar_->SetLookupTable(lookupTable_);
    scalarBar_->SetTitle("");
    scalarBar_->SetNumberOfLabels(5);
    scalarBar_->SetMaximumWidthInPixels(90);
    scalarBar_->SetMaximumHeightInPixels(200);
    setLegendTextStyle(scalarBar_->GetTitleTextProperty(), 11);
    setLegendTextStyle(scalarBar_->GetLabelTextProperty(), 10);
    renderer_->AddActor2D(scalarBar_);
    scalarBar_->SetVisibility(false);

    connect(reloadButton, &QPushButton::clicked, this, &BoundaryCsvPanel::reloadLatestCsv);
    connect(colorCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) {
        updateColorField();
    });
}

void BoundaryCsvPanel::setProjectDirectory(const QString& projectDirectory) {
    const QString normalized = QFileInfo(projectDirectory).absoluteFilePath();
    if (projectDirectory_ == normalized) {
        return;
    }
    projectDirectory_ = normalized;
    reloadLatestCsv();
}

void BoundaryCsvPanel::reloadLatestCsv() {
    const QString csvPath = latestBoundaryCsv();
    if (csvPath.isEmpty()) {
        currentFile_.clear();
        fileLabel_->setText("No boundary CSV found in proj_temp");
        polyData_->Initialize();
        glyphFilter_->Update();
        scalarBar_->SetVisibility(false);
        renderWindow_->Render();
        return;
    }

    QFile file(csvPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        emit statusMessage("Failed to open boundary CSV " + csvPath);
        return;
    }

    QTextStream in(&file);
    const QString headerLine = in.readLine();
    const QStringList headers = splitCsvRow(headerLine);
    const int xIndex = std::max(findHeaderIndex(headers, "X"), findHeaderIndex(headers, "x"));
    const int yIndex = std::max(findHeaderIndex(headers, "Y"), findHeaderIndex(headers, "y"));
    const int zIndex = std::max(findHeaderIndex(headers, "Z"), findHeaderIndex(headers, "z"));
    const int uIndex = std::max(findHeaderIndex(headers, "u"), findHeaderIndex(headers, "U"));
    const int vIndex = std::max(findHeaderIndex(headers, "v"), findHeaderIndex(headers, "V"));
    const int wIndex = std::max(findHeaderIndex(headers, "w"), findHeaderIndex(headers, "W"));
    const int tIndex = std::max(findHeaderIndex(headers, "T"), findHeaderIndex(headers, "t"));
    const int patchIndex = findHeaderIndex(headers, "patch");

    if (xIndex < 0 || yIndex < 0 || zIndex < 0) {
        emit statusMessage("Boundary CSV does not contain X/Y/Z columns.");
        return;
    }

    auto points = vtkSmartPointer<vtkPoints>::New();
    auto speedArray = vtkSmartPointer<vtkFloatArray>::New();
    speedArray->SetName("speed");
    auto uArray = vtkSmartPointer<vtkFloatArray>::New();
    uArray->SetName("u");
    auto vArray = vtkSmartPointer<vtkFloatArray>::New();
    vArray->SetName("v");
    auto wArray = vtkSmartPointer<vtkFloatArray>::New();
    wArray->SetName("w");
    auto temperatureArray = vtkSmartPointer<vtkFloatArray>::New();
    temperatureArray->SetName("temperature");
    auto patchArray = vtkSmartPointer<vtkFloatArray>::New();
    patchArray->SetName("patch");

    while (!in.atEnd()) {
        const QString line = in.readLine().trimmed();
        if (line.isEmpty()) {
            continue;
        }
        const QStringList columns = splitCsvRow(line);
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        if (!readDoubleAt(columns, xIndex, &x) || !readDoubleAt(columns, yIndex, &y) || !readDoubleAt(columns, zIndex, &z)) {
            continue;
        }
        points->InsertNextPoint(x, y, z);

        double u = 0.0;
        double v = 0.0;
        double w = 0.0;
        const bool hasU = readDoubleAt(columns, uIndex, &u);
        const bool hasV = readDoubleAt(columns, vIndex, &v);
        const bool hasW = readDoubleAt(columns, wIndex, &w);
        speedArray->InsertNextValue(static_cast<float>((hasU || hasV || hasW) ? std::sqrt(u * u + v * v + w * w) : z));
        uArray->InsertNextValue(static_cast<float>(u));
        vArray->InsertNextValue(static_cast<float>(v));
        wArray->InsertNextValue(static_cast<float>(w));

        double temperature = 0.0;
        temperatureArray->InsertNextValue(static_cast<float>(readDoubleAt(columns, tIndex, &temperature) ? temperature : 0.0));

        double patch = 0.0;
        patchArray->InsertNextValue(static_cast<float>(readDoubleAt(columns, patchIndex, &patch) ? patch : 0.0));
    }

    polyData_->Initialize();
    polyData_->SetPoints(points);
    polyData_->GetPointData()->Initialize();
    polyData_->GetPointData()->AddArray(speedArray);
    if (uIndex >= 0) {
        polyData_->GetPointData()->AddArray(uArray);
    }
    if (vIndex >= 0) {
        polyData_->GetPointData()->AddArray(vArray);
    }
    if (wIndex >= 0) {
        polyData_->GetPointData()->AddArray(wArray);
    }
    if (tIndex >= 0) {
        polyData_->GetPointData()->AddArray(temperatureArray);
    }
    if (patchIndex >= 0) {
        polyData_->GetPointData()->AddArray(patchArray);
    }
    polyData_->GetPointData()->SetScalars(speedArray);
    glyphFilter_->Update();

    {
        const QSignalBlocker blocker(colorCombo_);
        colorCombo_->clear();
        colorCombo_->addItem("Speed", "speed");
        if (uIndex >= 0) {
            colorCombo_->addItem("u", "u");
        }
        if (vIndex >= 0) {
            colorCombo_->addItem("v", "v");
        }
        if (wIndex >= 0) {
            colorCombo_->addItem("w", "w");
        }
        if (tIndex >= 0) {
            colorCombo_->addItem("Temperature", "temperature");
        }
        if (patchIndex >= 0) {
            colorCombo_->addItem("Patch", "patch");
        }
        colorCombo_->setCurrentIndex(0);
    }

    currentFile_ = csvPath;
    fileLabel_->setText(QFileInfo(csvPath).fileName());
    updateColorField();
    renderer_->ResetCamera();
    renderWindow_->Render();
    emit statusMessage("Loaded boundary CSV " + QFileInfo(csvPath).fileName());
}

QString BoundaryCsvPanel::latestBoundaryCsv() const {
    if (projectDirectory_.isEmpty()) {
        return {};
    }
    const QDir csvDir(QDir(projectDirectory_).filePath("proj_temp"));
    if (!csvDir.exists()) {
        return {};
    }
    QFileInfoList files = csvDir.entryInfoList({"SurfData_*.csv"}, QDir::Files, QDir::Time);
    if (files.isEmpty()) {
        files = csvDir.entryInfoList({"*.csv"}, QDir::Files, QDir::Time);
    }
    return files.isEmpty() ? QString() : files.front().absoluteFilePath();
}

void BoundaryCsvPanel::updateColorField() {
    const QString arrayName = colorCombo_->currentData().toString();
    if (!polyData_ || arrayName.isEmpty()) {
        mapper_->ScalarVisibilityOff();
        scalarBar_->SetVisibility(false);
        renderWindow_->Render();
        return;
    }

    auto* array = polyData_->GetPointData()->GetArray(arrayName.toLocal8Bit().constData());
    if (!array) {
        mapper_->ScalarVisibilityOff();
        scalarBar_->SetVisibility(false);
        renderWindow_->Render();
        return;
    }

    mapper_->SetLookupTable(lookupTable_);
    mapper_->SetScalarModeToUsePointFieldData();
    mapper_->SelectColorArray(arrayName.toLocal8Bit().constData());
    mapper_->ScalarVisibilityOn();
    double range[2] = {0.0, 0.0};
    array->GetRange(range);
    if (!(range[1] > range[0])) {
        range[1] = range[0] + 1.0;
    }
    mapper_->SetScalarRange(range);
    scalarBar_->SetLookupTable(lookupTable_);
    scalarBar_->SetVisibility(true);
    renderWindow_->Render();
}

} // namespace luwgui
