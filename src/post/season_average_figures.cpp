#include "season_average_internal.hpp"

#include "luw/legacy_vtk.hpp"
#include "post_internal.hpp"

#include <vtkErrorCode.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <ostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace luw::post::detail {

namespace {

namespace fs = std::filesystem;
constexpr std::array<int, 9> targetHeights{50, 100, 150, 200, 300, 400, 500, 600, 800};
constexpr double baseHeight = -50.0;
constexpr double layerStep = 10.0;

std::string upperAscii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char value) {
        return static_cast<char>(std::toupper(value));
    });
    return text;
}

struct Color final {
    unsigned char red = 255;
    unsigned char green = 255;
    unsigned char blue = 255;
};

class Canvas final {
public:
    Canvas(int width, int height)
        : width_(width), height_(height) {
        if (width <= 0 || height <= 0) throw std::runtime_error("Invalid figure dimensions");
        const std::uint64_t values = static_cast<std::uint64_t>(width)
            * static_cast<std::uint64_t>(height) * 3ULL;
        if (values > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("Seasonal-average figure is too large");
        }
        pixels_.assign(static_cast<std::size_t>(values), 255U);
    }

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] std::vector<unsigned char>& pixels() noexcept { return pixels_; }

    void pixel(int x, int y, Color color) noexcept {
        if (x < 0 || y < 0 || x >= width_ || y >= height_) return;
        // VTK image row zero is the lower row; canvas coordinates start at the top.
        const std::size_t storageY = static_cast<std::size_t>(height_ - 1 - y);
        const std::size_t index = (storageY * static_cast<std::size_t>(width_)
                                   + static_cast<std::size_t>(x)) * 3U;
        pixels_[index] = color.red;
        pixels_[index + 1U] = color.green;
        pixels_[index + 2U] = color.blue;
    }

    void fill(int x0, int y0, int x1, int y1, Color color) noexcept {
        x0 = std::clamp(x0, 0, width_);
        x1 = std::clamp(x1, 0, width_);
        y0 = std::clamp(y0, 0, height_);
        y1 = std::clamp(y1, 0, height_);
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) pixel(x, y, color);
        }
    }

    void line(int x0, int y0, int x1, int y1, Color color) noexcept {
        const int dx = std::abs(x1 - x0);
        const int sx = x0 < x1 ? 1 : -1;
        const int dy = -std::abs(y1 - y0);
        const int sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;
        while (true) {
            pixel(x0, y0, color);
            if (x0 == x1 && y0 == y1) break;
            const int doubled = 2 * error;
            if (doubled >= dy) { error += dy; x0 += sx; }
            if (doubled <= dx) { error += dx; y0 += sy; }
        }
    }

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<unsigned char> pixels_;
};

std::array<unsigned char, 7> glyph(char character) {
    switch (static_cast<char>(std::toupper(static_cast<unsigned char>(character)))) {
    case 'A': return {14,17,17,31,17,17,17}; case 'B': return {30,17,17,30,17,17,30};
    case 'C': return {14,17,16,16,16,17,14}; case 'D': return {30,17,17,17,17,17,30};
    case 'E': return {31,16,16,30,16,16,31}; case 'F': return {31,16,16,30,16,16,16};
    case 'G': return {14,17,16,23,17,17,15}; case 'H': return {17,17,17,31,17,17,17};
    case 'I': return {14,4,4,4,4,4,14}; case 'J': return {1,1,1,1,17,17,14};
    case 'K': return {17,18,20,24,20,18,17}; case 'L': return {16,16,16,16,16,16,31};
    case 'M': return {17,27,21,21,17,17,17}; case 'N': return {17,25,21,19,17,17,17};
    case 'O': return {14,17,17,17,17,17,14}; case 'P': return {30,17,17,30,16,16,16};
    case 'Q': return {14,17,17,17,21,18,13}; case 'R': return {30,17,17,30,20,18,17};
    case 'S': return {15,16,16,14,1,1,30}; case 'T': return {31,4,4,4,4,4,4};
    case 'U': return {17,17,17,17,17,17,14}; case 'V': return {17,17,17,17,17,10,4};
    case 'W': return {17,17,17,21,21,21,10}; case 'X': return {17,17,10,4,10,17,17};
    case 'Y': return {17,17,10,4,4,4,4}; case 'Z': return {31,1,2,4,8,16,31};
    case '0': return {14,17,19,21,25,17,14}; case '1': return {4,12,4,4,4,4,14};
    case '2': return {14,17,1,2,4,8,31}; case '3': return {30,1,1,14,1,1,30};
    case '4': return {2,6,10,18,31,2,2}; case '5': return {31,16,16,30,1,1,30};
    case '6': return {14,16,16,30,17,17,14}; case '7': return {31,1,2,4,8,8,8};
    case '8': return {14,17,17,14,17,17,14}; case '9': return {14,17,17,15,1,1,14};
    case '.': return {0,0,0,0,0,6,6}; case ',': return {0,0,0,0,6,6,4};
    case ':': return {0,6,6,0,6,6,0}; case '-': return {0,0,0,31,0,0,0};
    case '+': return {0,4,4,31,4,4,0}; case '/': return {1,2,2,4,8,8,16};
    case '(': return {2,4,8,8,8,4,2}; case ')': return {8,4,2,2,2,4,8};
    case '[': return {14,8,8,8,8,8,14}; case ']': return {14,2,2,2,2,2,14};
    case '_': return {0,0,0,0,0,0,31}; case '=': return {0,31,0,31,0,0,0};
    default: return {};
    }
}

int textWidth(std::string_view text, int scale) {
    return static_cast<int>(text.size()) * 6 * scale;
}

void drawText(Canvas& canvas, int x, int y, std::string_view text,
              int scale, Color color) {
    for (const char character : text) {
        const auto rows = glyph(character);
        for (int row = 0; row < 7; ++row) {
            for (int column = 0; column < 5; ++column) {
                if ((rows[static_cast<std::size_t>(row)] & (1U << (4 - column))) == 0) continue;
                canvas.fill(x + column * scale, y + row * scale,
                            x + (column + 1) * scale, y + (row + 1) * scale, color);
            }
        }
        x += 6 * scale;
    }
}

std::tm localTimeNow() {
    const std::time_t raw = std::time(nullptr);
    std::tm result{};
#ifdef _WIN32
    localtime_s(&result, &raw);
#else
    localtime_r(&raw, &result);
#endif
    return result;
}

void log(std::ostream& output, const std::string& message) {
    const std::tm local = localTimeNow();
    output << '[' << std::put_time(&local, "%H:%M:%S") << "] " << message << std::endl;
}

void logRule(std::ostream& output, const std::string& title) {
    log(output, std::string(24, '=') + " " + title + " " + std::string(24, '='));
}

Color interpolateColor(Color left, Color right, double weight) {
    const auto channel = [weight](unsigned char a, unsigned char b) {
        return static_cast<unsigned char>(std::clamp(
            std::lround(static_cast<double>(a) * (1.0 - weight) + static_cast<double>(b) * weight),
            0L, 255L));
    };
    return {channel(left.red, right.red), channel(left.green, right.green),
            channel(left.blue, right.blue)};
}

Color anchored(double normalized, std::span<const Color> colors) {
    normalized = std::clamp(normalized, 0.0, 1.0);
    const double position = normalized * static_cast<double>(colors.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = std::min(lower + 1U, colors.size() - 1U);
    return interpolateColor(colors[lower], colors[upper], position - static_cast<double>(lower));
}

Color coolwarm(double normalized) {
    constexpr std::array<Color, 5> colors{{
        {59,76,192}, {141,176,254}, {221,221,221}, {244,154,123}, {180,4,38}}};
    return anchored(normalized, colors);
}

Color turbo(double normalized) {
    const double x = std::clamp(normalized, 0.0, 1.0);
    const auto polynomial = [x](const std::array<double, 6>& coefficient) {
        return coefficient[0] + x * (coefficient[1] + x * (coefficient[2]
            + x * (coefficient[3] + x * (coefficient[4] + x * coefficient[5]))));
    };
    constexpr std::array<double, 6> red{0.13572138,4.61539260,-42.66032258,132.13108234,-152.94239396,59.28637943};
    constexpr std::array<double, 6> green{0.09140261,2.19418839,4.84296658,-14.18503333,4.27729857,2.82956604};
    constexpr std::array<double, 6> blue{0.10667330,12.64194608,-60.58204836,110.36276771,-89.90310912,27.34824973};
    const auto channel = [&](const std::array<double, 6>& coefficient) {
        return static_cast<unsigned char>(std::clamp(std::lround(polynomial(coefficient) * 255.0), 0L, 255L));
    };
    return {channel(red), channel(green), channel(blue)};
}

Color magma(double normalized) {
    constexpr std::array<Color, 6> colors{{
        {0,0,4}, {45,17,95}, {126,37,130}, {205,64,113}, {250,142,8}, {252,253,191}}};
    return anchored(normalized, colors);
}

enum class Palette { Coolwarm, Turbo, Magma };

Color colorFor(Palette palette, double normalized) {
    switch (palette) {
    case Palette::Coolwarm: return coolwarm(normalized);
    case Palette::Turbo: return turbo(normalized);
    case Palette::Magma: return magma(normalized);
    }
    return {};
}

double percentile(std::vector<float> values, double quantile) {
    values.erase(std::remove_if(values.begin(), values.end(), [](float value) {
        return !std::isfinite(value);
    }), values.end());
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double position = static_cast<double>(values.size() - 1U) * quantile;
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = std::min(lower + 1U, values.size() - 1U);
    return static_cast<double>(values[lower]) * (1.0 - (position - lower))
        + static_cast<double>(values[upper]) * (position - lower);
}

std::string fixed(double value, int precision) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

void writePng(const fs::path& path, Canvas& canvas) {
    fs::create_directories(path.parent_path());
    vtkNew<vtkUnsignedCharArray> pixels;
    pixels->SetNumberOfComponents(3);
    const std::size_t valueCount = canvas.pixels().size();
    if (valueCount > static_cast<std::size_t>(std::numeric_limits<vtkIdType>::max())) {
        throw std::runtime_error("PNG pixel buffer exceeds VTK array limits");
    }
    pixels->SetArray(canvas.pixels().data(), static_cast<vtkIdType>(valueCount), 1);
    vtkNew<vtkImageData> image;
    image->SetDimensions(canvas.width(), canvas.height(), 1);
    image->GetPointData()->SetScalars(pixels);
    vtkNew<vtkPNGWriter> writer;
    const std::string utf8 = pathUtf8(path);
    writer->SetFileName(utf8.c_str());
    writer->SetInputData(image);
    writer->Write();
    if (writer->GetErrorCode() != vtkErrorCode::NoError || !fs::is_regular_file(path)) {
        throw std::runtime_error("Failed to write PNG: " + utf8);
    }
}

struct FigureSpec final {
    std::string field;
    std::string title;
    std::string colorbar;
    std::string filename;
    Palette palette = Palette::Turbo;
    bool symmetric = false;
};

void renderFigure(const fs::path& vtkPath,
                  const LegacyStructuredPoints& vtk,
                  const FigureSpec& spec,
                  const fs::path& outputPath,
                  int dpi,
                  std::ostream& output) {
    logRule(output, "Plot " + spec.field);
    log(output, "Generate figure for field " + spec.field + " -> " + spec.filename);
    const LegacyVtkField* field = vtk.field(spec.field);
    if (field == nullptr || field->components != 1
        || field->association != LegacyVtkAssociation::PointData) {
        throw std::runtime_error("Scalar field '" + spec.field + "' expected in " + pathUtf8(vtkPath));
    }
    LegacyVtkPlaneReader reader(vtk);
    std::map<int, std::vector<float>> slices;
    std::vector<double> lows;
    std::vector<double> highs;
    const int nz = vtk.dimensions[2];
    for (const int height : targetHeights) {
        const int z = static_cast<int>(std::nearbyint((height - baseHeight) / layerStep));
        if (z < 0 || z >= nz) continue;
        std::vector<float> values = reader.readPointPlane(*field, static_cast<std::size_t>(z));
        lows.push_back(percentile(values, 0.02));
        highs.push_back(percentile(values, 0.98));
        slices.emplace(z, std::move(values));
        log(output, "  Loaded z=" + std::to_string(z) + " (" + std::to_string(height)
                        + " m) for figure " + spec.field);
    }
    double minimum = 0.0;
    double maximum = 1.0;
    lows.erase(std::remove_if(lows.begin(), lows.end(), [](double value) { return !std::isfinite(value); }), lows.end());
    highs.erase(std::remove_if(highs.begin(), highs.end(), [](double value) { return !std::isfinite(value); }), highs.end());
    if (!lows.empty() && !highs.empty()) {
        minimum = *std::min_element(lows.begin(), lows.end());
        maximum = *std::max_element(highs.begin(), highs.end());
        if (spec.symmetric) {
            const double absolute = std::max(std::abs(minimum), std::abs(maximum));
            minimum = -(std::isfinite(absolute) && absolute > 0.0 ? absolute : 1.0);
            maximum = -minimum;
        } else if (!std::isfinite(minimum) || !std::isfinite(maximum) || maximum <= minimum) {
            maximum = minimum + 1.0;
        }
    } else if (spec.symmetric) {
        minimum = -1.0;
        maximum = 1.0;
    }

    if (dpi > std::numeric_limits<int>::max() / 18) {
        throw std::runtime_error("Figure DPI is too large");
    }
    const int width = 18 * dpi;
    const int heightPixels = 16 * dpi;
    Canvas canvas(width, heightPixels);
    const Color black{0,0,0};
    const Color border{80,80,80};
    const Color gray{225,225,225};
    const int titleScale = std::max(1, dpi / 48);
    const int labelScale = std::max(1, dpi / 65);
    const int topMargin = std::max(35, heightPixels * 7 / 100);
    const int bottomMargin = std::max(20, heightPixels * 3 / 100);
    const int leftMargin = std::max(20, width * 2 / 100);
    const int colorbarArea = std::max(70, width * 7 / 100);
    const int gridRight = width - colorbarArea;
    const int usableWidth = gridRight - leftMargin;
    const int usableHeight = heightPixels - topMargin - bottomMargin;
    const int nx = vtk.dimensions[0];
    const int ny = vtk.dimensions[1];
    const double physicalWidth = std::max(std::abs(vtk.spacing[0] * static_cast<double>(nx - 1)), 1.0e-12);
    const double physicalHeight = std::max(std::abs(vtk.spacing[1] * static_cast<double>(ny - 1)), 1.0e-12);

    for (std::size_t panel = 0; panel < targetHeights.size(); ++panel) {
        const int row = static_cast<int>(panel / 3U);
        const int column = static_cast<int>(panel % 3U);
        const int cellX0 = leftMargin + usableWidth * column / 3;
        const int cellX1 = leftMargin + usableWidth * (column + 1) / 3;
        const int cellY0 = topMargin + usableHeight * row / 3;
        const int cellY1 = topMargin + usableHeight * (row + 1) / 3;
        const int panelTitleHeight = std::max(15, 10 * labelScale);
        int imageX0 = cellX0 + std::max(4, dpi / 20);
        int imageX1 = cellX1 - std::max(4, dpi / 20);
        int imageY0 = cellY0 + panelTitleHeight;
        int imageY1 = cellY1 - std::max(4, dpi / 20);
        const double targetAspect = physicalWidth / physicalHeight;
        const double availableAspect = static_cast<double>(imageX1 - imageX0)
            / static_cast<double>(std::max(1, imageY1 - imageY0));
        if (availableAspect > targetAspect) {
            const int desired = std::max(1, static_cast<int>((imageY1 - imageY0) * targetAspect));
            imageX0 += ((imageX1 - imageX0) - desired) / 2;
            imageX1 = imageX0 + desired;
        } else {
            const int desired = std::max(1, static_cast<int>((imageX1 - imageX0) / targetAspect));
            imageY0 += ((imageY1 - imageY0) - desired) / 2;
            imageY1 = imageY0 + desired;
        }
        const int targetHeight = targetHeights[panel];
        const int z = static_cast<int>(std::nearbyint((targetHeight - baseHeight) / layerStep));
        const std::string panelTitle = std::to_string(targetHeight) + "M (Z=" + std::to_string(z) + ")";
        drawText(canvas, (cellX0 + cellX1 - textWidth(panelTitle, labelScale)) / 2,
                 cellY0 + 2, panelTitle, labelScale, black);
        const auto slice = slices.find(z);
        if (slice == slices.end()) {
            canvas.fill(imageX0, imageY0, imageX1, imageY1, gray);
            const std::string text = "OUT_OF_RANGE";
            drawText(canvas, (imageX0 + imageX1 - textWidth(text, labelScale)) / 2,
                     (imageY0 + imageY1 - 7 * labelScale) / 2, text, labelScale, border);
        } else {
            const std::vector<float>& values = slice->second;
            const int drawWidth = std::max(1, imageX1 - imageX0);
            const int drawHeight = std::max(1, imageY1 - imageY0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int py = 0; py < drawHeight; ++py) {
                const double sourceY = drawHeight == 1 ? 0.0
                    : static_cast<double>(drawHeight - 1 - py) * static_cast<double>(ny - 1)
                        / static_cast<double>(drawHeight - 1);
                const std::size_t y0 = static_cast<std::size_t>(std::floor(sourceY));
                const std::size_t y1 = std::min(y0 + 1U, static_cast<std::size_t>(ny - 1));
                const double fy = sourceY - static_cast<double>(y0);
                for (int px = 0; px < drawWidth; ++px) {
                    const double sourceX = drawWidth == 1 ? 0.0
                        : static_cast<double>(px) * static_cast<double>(nx - 1)
                            / static_cast<double>(drawWidth - 1);
                    const std::size_t x0 = static_cast<std::size_t>(std::floor(sourceX));
                    const std::size_t x1 = std::min(x0 + 1U, static_cast<std::size_t>(nx - 1));
                    const double fx = sourceX - static_cast<double>(x0);
                    const double lower = values[y0 * static_cast<std::size_t>(nx) + x0] * (1.0 - fx)
                        + values[y0 * static_cast<std::size_t>(nx) + x1] * fx;
                    const double upper = values[y1 * static_cast<std::size_t>(nx) + x0] * (1.0 - fx)
                        + values[y1 * static_cast<std::size_t>(nx) + x1] * fx;
                    const double value = lower * (1.0 - fy) + upper * fy;
                    const double normalized = std::isfinite(value)
                        ? (value - minimum) / (maximum - minimum) : 0.0;
                    canvas.pixel(imageX0 + px, imageY0 + py,
                                 std::isfinite(value) ? colorFor(spec.palette, normalized)
                                                      : Color{245,245,245});
                }
            }
        }
        canvas.line(imageX0, imageY0, imageX1, imageY0, border);
        canvas.line(imageX0, imageY1, imageX1, imageY1, border);
        canvas.line(imageX0, imageY0, imageX0, imageY1, border);
        canvas.line(imageX1, imageY0, imageX1, imageY1, border);
    }

    const int barX0 = gridRight + colorbarArea / 5;
    const int barX1 = barX0 + std::max(10, colorbarArea / 5);
    const int barY0 = topMargin + usableHeight / 12;
    const int barY1 = topMargin + usableHeight * 11 / 12;
    for (int y = barY0; y < barY1; ++y) {
        const double normalized = 1.0 - static_cast<double>(y - barY0)
            / static_cast<double>(std::max(1, barY1 - barY0 - 1));
        canvas.fill(barX0, y, barX1, y + 1, colorFor(spec.palette, normalized));
    }
    canvas.line(barX0, barY0, barX1, barY0, black);
    canvas.line(barX0, barY1, barX1, barY1, black);
    for (int tick = 0; tick <= 4; ++tick) {
        const int y = barY1 - (barY1 - barY0) * tick / 4;
        canvas.line(barX1, y, barX1 + std::max(3, dpi / 25), y, black);
        const std::string value = fixed(minimum + (maximum - minimum) * tick / 4.0, 2);
        drawText(canvas, barX1 + std::max(6, dpi / 18), y - 3 * labelScale,
                 value, labelScale, black);
    }
    drawText(canvas, barX0, std::min(heightPixels - 8 * labelScale, barY1 + 4 * labelScale),
             spec.colorbar, labelScale, black);
    const std::string title = upperAscii(spec.title) + " (9 LAYERS) | "
        + upperAscii(pathUtf8(vtkPath.filename()));
    drawText(canvas, (width - textWidth(title, titleScale)) / 2,
             std::max(2, dpi / 10), title, titleScale, black);
    const std::string subtitle = "HEIGHT=-50+10*Z, LOCAL GRID=" + std::to_string(nx)
        + "X" + std::to_string(ny);
    drawText(canvas, (width - textWidth(subtitle, labelScale)) / 2,
             std::max(12, 9 * titleScale + dpi / 10), subtitle, labelScale, black);

    writePng(outputPath, canvas);
    log(output, "Saved figure: " + pathUtf8(outputPath));
}

} // namespace

void renderSeasonAverageFigures(const fs::path& vtkPath,
                                const fs::path& figureDirectory,
                                int dpi,
                                std::ostream& output) {
    const LegacyStructuredPoints vtk = LegacyStructuredPoints::readHeader(vtkPath);
    const std::array<FigureSpec, 5> figures{{
        {"u", "Season Average U Component", "U (M/S)", "season_average_u_9layers.png", Palette::Coolwarm, true},
        {"v", "Season Average V Component", "V (M/S)", "season_average_v_9layers.png", Palette::Coolwarm, true},
        {"w", "Season Average W Component", "W (M/S)", "season_average_w_9layers.png", Palette::Coolwarm, true},
        {"vm", "Season Average Velocity Magnitude", "VM (M/S)", "season_average_vm_9layers.png", Palette::Turbo, false},
        {"tke", "Season Average TKE", "TKE (M^2/S^2)", "season_average_tke_9layers.png", Palette::Magma, false},
    }};
    for (const FigureSpec& figure : figures) {
        renderFigure(vtkPath, vtk, figure, figureDirectory / figure.filename,
                     std::max(80, dpi), output);
    }
}

} // namespace luw::post::detail
