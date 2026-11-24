#include "luwgui/WavenumberPanel.h"

#include "luwgui/PlotWidgets.h"

#include <QComboBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSignalBlocker>
#include <QTabBar>
#include <QTabWidget>
#include <QTimer>
#include <QVariantList>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkDataSet.h>
#include <vtkDoubleArray.h>
#include <vtkFieldData.h>
#include <vtkGenericDataObjectReader.h>
#include <vtkImageData.h>
#include <vtkImageFFT.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

namespace luwgui {

struct WavenumberAnalysisResult {
    QString error;
    QString summary;
    QVector<QPointF> energySamples;
    QVector<QPointF> compensatedSamples;
    QVector<LesLayerSpectrumResult> lesLayers;
    double kNyquist = 0.0;
    double kTrust = 0.0;
};

namespace {

struct ArrayChoice {
    QString association;
    QString name;
};

struct LayerTarget {
    double targetHeight = 0.0;
    double actualHeight = 0.0;
    int zIndex = 0;
};

struct HorizontalSpectrumResult {
    QVector<double> energy;
    double validFraction = 0.0;
};

vtkSmartPointer<vtkImageData> readImageData(const QString& filePath, QString* errorMessage) {
    auto reader = vtkSmartPointer<vtkGenericDataObjectReader>::New();
    reader->SetFileName(filePath.toLocal8Bit().constData());
    reader->Update();

    auto* dataSet = vtkDataSet::SafeDownCast(reader->GetOutput());
    auto* imageData = vtkImageData::SafeDownCast(dataSet);
    if (!imageData) {
        if (errorMessage) {
            *errorMessage = "Wavenumber analysis requires legacy VTK image data or structured points.";
        }
        return nullptr;
    }

    auto copy = vtkSmartPointer<vtkImageData>::New();
    copy->ShallowCopy(imageData);
    return copy;
}

QVector<ArrayChoice> vectorArrayChoices(const QString& filePath, QString* errorMessage) {
    QVector<ArrayChoice> choices;
    vtkSmartPointer<vtkImageData> imageData = readImageData(filePath, errorMessage);
    if (!imageData) {
        return choices;
    }

    auto appendChoices = [&](vtkFieldData* fieldData, const QString& association) {
        if (!fieldData) {
            return;
        }
        for (int i = 0; i < fieldData->GetNumberOfArrays(); ++i) {
            vtkDataArray* array = fieldData->GetArray(i);
            if (!array || !array->GetName() || array->GetNumberOfComponents() < 3) {
                continue;
            }
            choices.push_back({association, QString::fromLocal8Bit(array->GetName())});
        }
    };

    appendChoices(imageData->GetPointData(), "point");
    appendChoices(imageData->GetCellData(), "cell");
    return choices;
}

vtkDataArray* resolveVectorArray(
    vtkImageData* imageData,
    const ArrayChoice& choice,
    int dims[3],
    double spacing[3],
    double origin[3],
    QString* errorMessage) {
    imageData->GetDimensions(dims);
    imageData->GetSpacing(spacing);
    imageData->GetOrigin(origin);

    vtkDataArray* vectors = nullptr;
    if (choice.association == "cell") {
        vectors = imageData->GetCellData()->GetArray(choice.name.toLocal8Bit().constData());
        dims[0] = std::max(0, dims[0] - 1);
        dims[1] = std::max(0, dims[1] - 1);
        dims[2] = std::max(0, dims[2] - 1);
    } else {
        vectors = imageData->GetPointData()->GetArray(choice.name.toLocal8Bit().constData());
    }

    if (!vectors || vectors->GetNumberOfComponents() < 3) {
        if (errorMessage) {
            *errorMessage = "Selected vector field is unavailable or does not contain 3 components.";
        }
        return nullptr;
    }
    if (dims[0] <= 1 || dims[1] <= 1 || dims[2] <= 1) {
        if (errorMessage) {
            *errorMessage = "Grid dimensions are too small for spectral analysis.";
        }
        return nullptr;
    }

    const vtkIdType totalTuples = static_cast<vtkIdType>(dims[0]) * dims[1] * dims[2];
    if (vectors->GetNumberOfTuples() < totalTuples) {
        if (errorMessage) {
            *errorMessage = "Vector field tuple count is smaller than the grid sample count.";
        }
        return nullptr;
    }
    return vectors;
}

vtkIdType tupleIndex(int x, int y, int z, int nx, int ny) {
    return static_cast<vtkIdType>(x) + static_cast<vtkIdType>(y) * nx + static_cast<vtkIdType>(z) * nx * ny;
}

bool tupleIsValid(vtkDataArray* vectors, vtkIdType tuple) {
    return vectors->GetComponent(tuple, 0) != 0.0
        || vectors->GetComponent(tuple, 1) != 0.0
        || vectors->GetComponent(tuple, 2) != 0.0;
}

QVector<double> computeLayerCoverages(vtkDataArray* vectors, const int dims[3]) {
    QVector<double> coverages;
    coverages.reserve(dims[2]);
    const int nx = dims[0];
    const int ny = dims[1];
    const double layerPoints = static_cast<double>(nx * ny);
    for (int z = 0; z < dims[2]; ++z) {
        int validCount = 0;
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (tupleIsValid(vectors, tupleIndex(x, y, z, nx, ny))) {
                    ++validCount;
                }
            }
        }
        coverages.push_back(validCount / std::max(1.0, layerPoints));
    }
    return coverages;
}

QVector<LayerTarget> buildTargetLayers(double originZ, double dz, int nz) {
    QVector<LayerTarget> targets;
    if (nz <= 0 || !(dz > 0.0)) {
        return targets;
    }

    const double topHeight = originZ + dz * (nz - 1);
    int previousIndex = -1;
    for (double targetHeight = 50.0; targetHeight <= topHeight + 1.0e-9; targetHeight += 50.0) {
        const int zIndex = std::clamp(static_cast<int>(std::llround((targetHeight - originZ) / dz)), 0, nz - 1);
        if (zIndex == previousIndex) {
            continue;
        }
        previousIndex = zIndex;
        targets.push_back({targetHeight, originZ + zIndex * dz, zIndex});
    }

    if (targets.isEmpty()) {
        targets.push_back({originZ + dz, originZ + dz, std::min(1, nz - 1)});
    }
    return targets;
}

HorizontalSpectrumResult computeHorizontalSpectrum(
    vtkDataArray* vectors,
    const int dims[3],
    const double spacing[3],
    int zIndex,
    QString* errorMessage) {
    HorizontalSpectrumResult result;
    const int nx = dims[0];
    const int ny = dims[1];
    const vtkIdType planeSize = static_cast<vtkIdType>(nx) * ny;
    std::vector<unsigned char> validMask(static_cast<std::size_t>(planeSize), 0u);
    int validCount = 0;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const vtkIdType idx = tupleIndex(x, y, zIndex, nx, ny);
            const bool valid = tupleIsValid(vectors, idx);
            validMask[static_cast<std::size_t>(y * nx + x)] = valid ? 1u : 0u;
            if (valid) {
                ++validCount;
            }
        }
    }

    result.validFraction = static_cast<double>(validCount) / std::max(1.0, static_cast<double>(planeSize));
    result.energy = QVector<double>(nx * ny, 0.0);
    if (validCount == 0) {
        return result;
    }

    const double normalization = static_cast<double>(planeSize) * validCount;
    for (int component = 0; component < 3; ++component) {
        double mean = 0.0;
        for (vtkIdType idx = 0; idx < planeSize; ++idx) {
            if (validMask[static_cast<std::size_t>(idx)] != 0u) {
                mean += vectors->GetComponent(tupleIndex(static_cast<int>(idx % nx), static_cast<int>(idx / nx), zIndex, nx, ny), component);
            }
        }
        mean /= static_cast<double>(validCount);

        auto componentImage = vtkSmartPointer<vtkImageData>::New();
        componentImage->SetDimensions(nx, ny, 1);
        componentImage->SetSpacing(spacing[0], spacing[1], 1.0);
        componentImage->SetOrigin(0.0, 0.0, 0.0);

        auto scalars = vtkSmartPointer<vtkDoubleArray>::New();
        scalars->SetNumberOfComponents(1);
        scalars->SetNumberOfTuples(planeSize);
        scalars->SetName("component");
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const vtkIdType flat = static_cast<vtkIdType>(y) * nx + x;
                const vtkIdType idx = tupleIndex(x, y, zIndex, nx, ny);
                const double value = validMask[static_cast<std::size_t>(flat)] != 0u
                    ? (vectors->GetComponent(idx, component) - mean)
                    : 0.0;
                scalars->SetValue(flat, value);
            }
        }
        componentImage->GetPointData()->SetScalars(scalars);

        auto fft = vtkSmartPointer<vtkImageFFT>::New();
        fft->SetInputData(componentImage);
        fft->Update();

        vtkImageData* fftImage = fft->GetOutput();
        vtkDataArray* fftScalars = fftImage ? fftImage->GetPointData()->GetScalars() : nullptr;
        if (!fftScalars || fftScalars->GetNumberOfComponents() < 2) {
            if (errorMessage) {
                *errorMessage = "FFT failed to produce complex layer output.";
            }
            return {};
        }

        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const vtkIdType flat = static_cast<vtkIdType>(y) * nx + x;
                const int shiftedX = (x + nx / 2) % nx;
                const int shiftedY = (y + ny / 2) % ny;
                const int shiftedIndex = shiftedY * nx + shiftedX;
                const double re = fftScalars->GetComponent(flat, 0);
                const double im = fftScalars->GetComponent(flat, 1);
                result.energy[shiftedIndex] += 0.5 * (re * re + im * im) / normalization;
            }
        }
    }

    return result;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double rank = std::clamp(p, 0.0, 100.0) * (values.size() - 1) / 100.0;
    const std::size_t low = static_cast<std::size_t>(std::floor(rank));
    const std::size_t high = static_cast<std::size_t>(std::ceil(rank));
    const double t = rank - low;
    return values[low] * (1.0 - t) + values[high] * t;
}

std::pair<double, double> robustLogColorLimits(const std::vector<double>& positiveLogValues) {
    if (positiveLogValues.empty()) {
        return {-12.0, 0.0};
    }
    std::vector<double> values = positiveLogValues;
    double minValue = percentile(values, 5.0);
    double maxValue = percentile(std::move(values), 99.5);
    if (!std::isfinite(minValue) || !std::isfinite(maxValue) || minValue >= maxValue) {
        minValue = -12.0;
        maxValue = 0.0;
    }
    return {minValue, maxValue};
}

double shiftedFrequencyAt(int shiftedIndex, int n, double spacing) {
    const int centeredIndex = shiftedIndex - n / 2;
    return 2.0 * std::numbers::pi_v<double> * static_cast<double>(centeredIndex) / (static_cast<double>(n) * spacing);
}

WavenumberAnalysisResult computeAnalysis(const QString& filePath, const ArrayChoice& choice) {
    WavenumberAnalysisResult result;

    QString error;
    vtkSmartPointer<vtkImageData> imageData = readImageData(filePath, &error);
    if (!imageData) {
        result.error = error;
        return result;
    }

    int dims[3] = {0, 0, 0};
    double spacing[3] = {1.0, 1.0, 1.0};
    double origin[3] = {0.0, 0.0, 0.0};
    vtkDataArray* vectors = resolveVectorArray(imageData, choice, dims, spacing, origin, &error);
    if (!vectors) {
        result.error = error;
        return result;
    }

    const QVector<double> coverages = computeLayerCoverages(vectors, dims);
    const QVector<LayerTarget> targets = buildTargetLayers(origin[2], spacing[2], dims[2]);

    std::vector<HorizontalSpectrumResult> rawLayers;
    rawLayers.reserve(static_cast<std::size_t>(targets.size()));
    std::vector<double> positiveLogValues;
    for (const LayerTarget& target : targets) {
        HorizontalSpectrumResult layer = computeHorizontalSpectrum(vectors, dims, spacing, target.zIndex, &error);
        if (!error.isEmpty()) {
            result.error = error;
            return result;
        }
        for (double value : layer.energy) {
            if (value > 0.0 && std::isfinite(value)) {
                positiveLogValues.push_back(std::log10(value));
            }
        }
        rawLayers.push_back(std::move(layer));
    }

    const auto [logMin, logMax] = robustLogColorLimits(positiveLogValues);
    const double epsilon = std::pow(10.0, logMin);
    const double kxMin = shiftedFrequencyAt(0, dims[0], spacing[0]);
    const double kxMax = shiftedFrequencyAt(dims[0] - 1, dims[0], spacing[0]);
    const double kyMin = shiftedFrequencyAt(0, dims[1], spacing[1]);
    const double kyMax = shiftedFrequencyAt(dims[1] - 1, dims[1], spacing[1]);

    for (int i = 0; i < targets.size(); ++i) {
        const LayerTarget& target = targets[i];
        const HorizontalSpectrumResult& layer = rawLayers[static_cast<std::size_t>(i)];

        LesLayerSpectrumResult view;
        view.label = QString("%1 m").arg(target.actualHeight, 0, 'f', 1);
        view.title = QString("LES spectra | z = %1 m | valid = %2%")
            .arg(target.actualHeight, 0, 'f', 1)
            .arg(layer.validFraction * 100.0, 0, 'f', 1);
        view.columns = dims[0];
        view.rows = dims[1];
        view.samples.resize(layer.energy.size());
        for (int idx = 0; idx < layer.energy.size(); ++idx) {
            view.samples[idx] = std::log10(std::max(layer.energy[idx], epsilon));
        }
        view.xMin = kxMin;
        view.xMax = kxMax;
        view.yMin = kyMin;
        view.yMax = kyMax;
        view.valueMin = logMin;
        view.valueMax = logMax;
        result.lesLayers.push_back(std::move(view));
    }

    int zStart = -1;
    for (int z = 0; z < coverages.size(); ++z) {
        if (coverages[z] >= 0.999999) {
            zStart = z;
            break;
        }
    }
    if (zStart < 0) {
        for (int z = 0; z < coverages.size(); ++z) {
            if (coverages[z] > 0.0) {
                zStart = z;
                break;
            }
        }
    }
    if (zStart < 0) {
        result.error = "No valid horizontal layer is available for isotropic spectrum analysis.";
        return result;
    }

    const int nx = dims[0];
    const int ny = dims[1];
    const int nzSub = dims[2] - zStart;
    const vtkIdType nTot = static_cast<vtkIdType>(nx) * ny * nzSub;
    std::vector<double> modeEnergy(static_cast<std::size_t>(nTot), 0.0);
    for (int component = 0; component < 3; ++component) {
        double mean = 0.0;
        for (int z = zStart; z < dims[2]; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    mean += vectors->GetComponent(tupleIndex(x, y, z, nx, ny), component);
                }
            }
        }
        mean /= static_cast<double>(nTot);

        auto componentImage = vtkSmartPointer<vtkImageData>::New();
        componentImage->SetDimensions(nx, ny, nzSub);
        componentImage->SetSpacing(spacing);
        componentImage->SetOrigin(0.0, 0.0, 0.0);

        auto scalars = vtkSmartPointer<vtkDoubleArray>::New();
        scalars->SetNumberOfComponents(1);
        scalars->SetNumberOfTuples(nTot);
        scalars->SetName("component");
        vtkIdType flat = 0;
        for (int z = zStart; z < dims[2]; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x, ++flat) {
                    scalars->SetValue(flat, vectors->GetComponent(tupleIndex(x, y, z, nx, ny), component) - mean);
                }
            }
        }
        componentImage->GetPointData()->SetScalars(scalars);

        auto fft = vtkSmartPointer<vtkImageFFT>::New();
        fft->SetInputData(componentImage);
        fft->Update();

        vtkImageData* fftImage = fft->GetOutput();
        vtkDataArray* fftScalars = fftImage ? fftImage->GetPointData()->GetScalars() : nullptr;
        if (!fftScalars || fftScalars->GetNumberOfComponents() < 2) {
            result.error = "FFT failed to produce 3D complex output.";
            return result;
        }

        for (vtkIdType i = 0; i < nTot; ++i) {
            const double re = fftScalars->GetComponent(i, 0);
            const double im = fftScalars->GetComponent(i, 1);
            modeEnergy[static_cast<std::size_t>(i)] += 0.5 * (re * re + im * im)
                / (static_cast<double>(nTot) * static_cast<double>(nTot));
        }
    }

    const auto waveNumber = [](int index, int n, double spacingValue) {
        const int wrapped = (index <= n / 2) ? index : (index - n);
        return 2.0 * std::numbers::pi_v<double> * static_cast<double>(wrapped) / (static_cast<double>(n) * spacingValue);
    };
    const double dk = std::max({
        2.0 * std::numbers::pi_v<double> / (nx * spacing[0]),
        2.0 * std::numbers::pi_v<double> / (ny * spacing[1]),
        2.0 * std::numbers::pi_v<double> / (nzSub * spacing[2])
    });
    double kMax = 0.0;
    for (vtkIdType tuple = 0; tuple < nTot; ++tuple) {
        const int x = static_cast<int>(tuple % nx);
        const int y = static_cast<int>((tuple / nx) % ny);
        const int z = static_cast<int>(tuple / (static_cast<vtkIdType>(nx) * ny));
        const double kx = waveNumber(x, nx, spacing[0]);
        const double ky = waveNumber(y, ny, spacing[1]);
        const double kz = waveNumber(z, nzSub, spacing[2]);
        kMax = std::max(kMax, std::sqrt(kx * kx + ky * ky + kz * kz));
    }

    const int binCount = std::max(2, static_cast<int>(std::floor(kMax / dk)) + 1);
    std::vector<double> shellEnergy(static_cast<std::size_t>(binCount), 0.0);
    std::vector<int> shellCount(static_cast<std::size_t>(binCount), 0);
    for (vtkIdType tuple = 0; tuple < nTot; ++tuple) {
        const int x = static_cast<int>(tuple % nx);
        const int y = static_cast<int>((tuple / nx) % ny);
        const int z = static_cast<int>(tuple / (static_cast<vtkIdType>(nx) * ny));
        const double kx = waveNumber(x, nx, spacing[0]);
        const double ky = waveNumber(y, ny, spacing[1]);
        const double kz = waveNumber(z, nzSub, spacing[2]);
        const double k = std::sqrt(kx * kx + ky * ky + kz * kz);
        int bin = static_cast<int>(std::floor(k / dk));
        bin = std::clamp(bin, 0, binCount - 1);
        shellEnergy[static_cast<std::size_t>(bin)] += modeEnergy[static_cast<std::size_t>(tuple)];
        shellCount[static_cast<std::size_t>(bin)] += 1;
    }

    for (int bin = 0; bin < binCount; ++bin) {
        if (shellCount[static_cast<std::size_t>(bin)] == 0) {
            continue;
        }
        const double kCenter = (static_cast<double>(bin) + 0.5) * dk;
        if (!(kCenter > 0.0)) {
            continue;
        }
        const double eK = shellEnergy[static_cast<std::size_t>(bin)] / dk;
        if (!std::isfinite(eK) || eK <= 0.0) {
            continue;
        }
        result.energySamples.push_back(QPointF(kCenter, eK));
        result.compensatedSamples.push_back(QPointF(kCenter, std::pow(kCenter, 5.0 / 3.0) * eK));
    }

    result.kNyquist = std::numbers::pi_v<double> / std::max({spacing[0], spacing[1], spacing[2]});
    result.kTrust = 0.5 * result.kNyquist;
    result.summary = QString(
        "Field: %1:%2 | dims: %3 x %4 x %5 | isotropic z-start: %6 m | LES layers: %7 | Nyquist: %8")
        .arg(choice.association)
        .arg(choice.name)
        .arg(dims[0])
        .arg(dims[1])
        .arg(dims[2])
        .arg(origin[2] + zStart * spacing[2], 0, 'f', 2)
        .arg(result.lesLayers.size())
        .arg(result.kNyquist, 0, 'g', 6);
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

WavenumberPanel::WavenumberPanel(QWidget* parent)
    : QWidget(parent)
    , watcher_(new QFutureWatcher<WavenumberAnalysisResult>(this)) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(8, 8, 8, 8);

    auto* header = new QWidget(this);
    auto* headerLayout = new QGridLayout(header);
    headerLayout->setContentsMargins(0, 0, 0, 0);
    headerLayout->addWidget(new QLabel("VTK field"), 0, 0);
    fileEdit_ = new QLineEdit(header);
    headerLayout->addWidget(fileEdit_, 0, 1, 1, 3);
    auto* browseButton = new QPushButton("Browse", header);
    headerLayout->addWidget(browseButton, 0, 4);
    auto* refreshButton = new QPushButton("Refresh Arrays", header);
    headerLayout->addWidget(refreshButton, 0, 5);

    headerLayout->addWidget(new QLabel("Vector array"), 1, 0);
    arrayCombo_ = new QComboBox(header);
    headerLayout->addWidget(arrayCombo_, 1, 1, 1, 2);
    analyzeButton_ = new QPushButton("Compute Spectra", header);
    headerLayout->addWidget(analyzeButton_, 1, 3, 1, 2);
    auto* saveButton = new QPushButton("Save Plot Image", header);
    headerLayout->addWidget(saveButton, 1, 5);
    root->addWidget(header);

    summaryLabel_ = new QLabel("Select a legacy VTK image dataset to compute isotropic and LES spectra.", this);
    summaryLabel_->setWordWrap(true);
    summaryLabel_->setProperty("muted", true);
    root->addWidget(summaryLabel_);

    plotTabs_ = new QTabWidget(this);
    plotTabs_->tabBar()->setUsesScrollButtons(false);
    plotTabs_->tabBar()->setExpanding(true);
    plotTabs_->tabBar()->setElideMode(Qt::ElideNone);
    energyPlot_ = new SpectrumPlotWidget(plotTabs_);
    energyPlot_->setTitle("3D Isotropic Energy Spectrum");
    energyPlot_->setXAxisTitle("Wavenumber k (rad / m)");
    energyPlot_->setYAxisTitle("E(k)");
    plotTabs_->addTab(energyPlot_, "Ek");

    compensatedPlot_ = new SpectrumPlotWidget(plotTabs_);
    compensatedPlot_->setTitle("Compensated Spectrum");
    compensatedPlot_->setXAxisTitle("Wavenumber k (rad / m)");
    compensatedPlot_->setYAxisTitle("k^(5/3) E(k)");
    plotTabs_->addTab(compensatedPlot_, "Ek Compensated");

    lesPlot_ = new HeatmapPlotWidget(plotTabs_);
    lesPlot_->setXAxisTitle("k_x (rad / m)");
    lesPlot_->setYAxisTitle("k_y (rad / m)");
    lesPlot_->setColorBarTitle("log10(E_2D)");
    plotTabs_->addTab(lesPlot_, "LES spectra");

    lesTabCorner_ = new QWidget(plotTabs_);
    auto* lesCornerLayout = new QHBoxLayout(lesTabCorner_);
    lesCornerLayout->setContentsMargins(0, 0, 0, 0);
    lesCornerLayout->setSpacing(6);
    lesCornerLayout->addWidget(new QLabel("Height", lesTabCorner_));
    lesHeightCombo_ = new QComboBox(lesTabCorner_);
    lesCornerLayout->addWidget(lesHeightCombo_);

    root->addWidget(plotTabs_, 1);
    updateTabUi();

    connect(fileEdit_, &QLineEdit::editingFinished, this, &WavenumberPanel::updateArrayChoices);
    connect(browseButton, &QPushButton::clicked, this, [this] {
        const QString path = QFileDialog::getOpenFileName(this, "Open VTK Field", fileEdit_->text(), "VTK (*.vtk)");
        if (!path.isEmpty()) {
            setSuggestedFilePath(path);
        }
    });
    connect(refreshButton, &QPushButton::clicked, this, &WavenumberPanel::updateArrayChoices);
    connect(analyzeButton_, &QPushButton::clicked, this, &WavenumberPanel::startAnalysis);
    connect(saveButton, &QPushButton::clicked, this, &WavenumberPanel::savePlots);
    connect(plotTabs_, &QTabWidget::currentChanged, this, &WavenumberPanel::updateTabUi);
    connect(lesHeightCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, &WavenumberPanel::updateLesLayerView);
    connect(watcher_, &QFutureWatcher<WavenumberAnalysisResult>::finished, this, [this] {
        analyzeButton_->setEnabled(true);
        const WavenumberAnalysisResult result = watcher_->result();
        if (!result.error.isEmpty()) {
            summaryLabel_->setText(result.error);
            emit statusMessage(result.error);
            return;
        }

        lesLayers_ = result.lesLayers;
        energyPlot_->setSpectrum(result.energySamples, result.kNyquist, result.kTrust);
        compensatedPlot_->setSpectrum(result.compensatedSamples, result.kNyquist, result.kTrust);

        {
            const QSignalBlocker blocker(lesHeightCombo_);
            lesHeightCombo_->clear();
            for (const LesLayerSpectrumResult& layer : lesLayers_) {
                lesHeightCombo_->addItem(layer.label);
            }
            if (lesHeightCombo_->count() > 0) {
                lesHeightCombo_->setCurrentIndex(0);
            }
        }
        updateLesLayerView();

        summaryLabel_->setText(result.summary);
        emit statusMessage("Wavenumber and LES spectra updated.");
    });
}

void WavenumberPanel::setSuggestedFilePath(const QString& filePath, bool autoAnalyze) {
    if (filePath.isEmpty()) {
        return;
    }
    fileEdit_->setText(QFileInfo(filePath).absoluteFilePath());
    updateArrayChoices();
    if (autoAnalyze && watcher_ && !watcher_->isRunning()) {
        QTimer::singleShot(0, this, [this] {
            startAnalysis();
        });
    }
}

void WavenumberPanel::updateTabUi() {
    const bool lesTabActive = plotTabs_ && plotTabs_->currentWidget() == lesPlot_;
    if (plotTabs_ && lesTabCorner_) {
        plotTabs_->setCornerWidget(lesTabActive ? lesTabCorner_ : nullptr, Qt::TopRightCorner);
    }
    if (lesTabCorner_) {
        lesTabCorner_->setVisible(lesTabActive);
    }
}

void WavenumberPanel::updateArrayChoices() {
    const QString path = fileEdit_->text().trimmed();
    const QString previous = arrayCombo_->currentText();
    arrayCombo_->clear();

    if (path.isEmpty()) {
        summaryLabel_->setText("Select a legacy VTK image dataset to compute isotropic and LES spectra.");
        return;
    }

    QString error;
    const QVector<ArrayChoice> choices = vectorArrayChoices(path, &error);
    if (!error.isEmpty()) {
        summaryLabel_->setText(error);
        return;
    }
    for (const ArrayChoice& choice : choices) {
        arrayCombo_->addItem(choice.association + ":" + choice.name, QVariantList{choice.association, choice.name});
    }

    int desiredIndex = arrayCombo_->findText(previous);
    if (desiredIndex < 0) desiredIndex = arrayCombo_->findText("point:u_avg");
    if (desiredIndex < 0) desiredIndex = arrayCombo_->findText("point:data");
    if (desiredIndex < 0) desiredIndex = arrayCombo_->findText("point:u");
    if (desiredIndex < 0) desiredIndex = arrayCombo_->findText("point:velocity");
    if (desiredIndex < 0 && arrayCombo_->count() > 0) {
        desiredIndex = 0;
    }
    if (desiredIndex >= 0) {
        arrayCombo_->setCurrentIndex(desiredIndex);
    }

    if (arrayCombo_->count() == 0) {
        summaryLabel_->setText("No 3-component vector arrays were found in the selected file.");
    } else {
        summaryLabel_->setText("Ready to compute isotropic and LES spectra for " + QFileInfo(path).fileName() + ".");
    }
}

void WavenumberPanel::startAnalysis() {
    const QString path = fileEdit_->text().trimmed();
    if (path.isEmpty()) {
        QMessageBox::warning(this, "Wavenumber spectrum", "Select a VTK file first.");
        return;
    }
    if (arrayCombo_->currentIndex() < 0) {
        updateArrayChoices();
    }
    if (arrayCombo_->currentIndex() < 0) {
        QMessageBox::warning(this, "Wavenumber spectrum", "No valid vector array is available.");
        return;
    }

    const QVariantList payload = arrayCombo_->currentData().toList();
    const ArrayChoice choice{
        payload.value(0).toString(),
        payload.value(1).toString()
    };

    emit guiActionRequested("Computing wavenumber spectra for " + QFileInfo(path).fileName());
    analyzeButton_->setEnabled(false);
    summaryLabel_->setText("Computing isotropic and LES spectra...");
    watcher_->setFuture(QtConcurrent::run(computeAnalysis, path, choice));
}

void WavenumberPanel::updateLesLayerView() {
    if (!lesPlot_) {
        return;
    }
    const int index = lesHeightCombo_ ? lesHeightCombo_->currentIndex() : -1;
    if (index < 0 || index >= lesLayers_.size()) {
        lesPlot_->setTitle("LES spectra");
        lesPlot_->clear();
        return;
    }

    const LesLayerSpectrumResult& layer = lesLayers_[index];
    lesPlot_->setTitle(layer.title);
    lesPlot_->setHeatmap(
        layer.columns,
        layer.rows,
        layer.samples,
        layer.xMin,
        layer.xMax,
        layer.yMin,
        layer.yMax,
        layer.valueMin,
        layer.valueMax);
}

void WavenumberPanel::savePlots() {
    const QString tabName = plotTabs_ ? plotTabs_->tabText(plotTabs_->currentIndex()).replace(' ', '_') : QString("wavenumber");
    const QString suggested = QFileInfo(fileEdit_->text()).completeBaseName() + "_" + tabName.toLower() + ".png";
    const QString path = selectSavePath(this, "Save Wavenumber Plot", suggested);
    if (path.isEmpty()) {
        return;
    }

    auto* exportable = qobject_cast<ExportablePlotWidget*>(plotTabs_->currentWidget());
    QString error;
    if (!exportable || !exportable->saveImage(path, &error)) {
        QMessageBox::critical(this, "Save plot image", error.isEmpty() ? "Failed to save plot image." : error);
        return;
    }
    emit guiActionRequested("Saved wavenumber plot " + QFileInfo(path).fileName());
    emit statusMessage("Saved plot to " + path);
}

} // namespace luwgui
