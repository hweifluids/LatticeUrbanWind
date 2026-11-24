#include "luw/vtk_post_commands.hpp"

#include "post_internal.hpp"

#include <vtkFFT.h>
#include <vtk_zlib.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cctype>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <numeric>
#include <optional>
#include <ostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#ifndef LUW_HAS_POCKETFFT
#define LUW_HAS_POCKETFFT 0
#endif

#if LUW_HAS_POCKETFFT
#include <pocketfft_hdronly.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace luw::cli {
namespace {

namespace fs = std::filesystem;
using luw::post::detail::pathUtf8;

constexpr double pi = 3.141592653589793238462643383279502884;

class ArgumentError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct Arguments final {
    fs::path vtkPath = "uvwrho-guangzhou4kmC3_20250910091000.vtk";
    std::optional<fs::path> outputDirectory;
    double heightInterval = 50.0;
    double heightStart = 50.0;
    int workers = -1;
    bool quickTest = false;
    int testHeightCount = 3;
    int test3dLayers = 24;
    bool help = false;
};

struct StructuredPointsMeta final {
    fs::path path;
    std::string datasetType;
    std::array<int, 3> dimensions{};
    std::array<double, 3> spacing{};
    std::array<double, 3> origin{};
    std::uint64_t pointData = 0;
    std::string dataName = "data";
    std::string dtypeName;
    int components = 1;
    std::uint64_t dataOffset = 0;
};

struct TargetLayer final {
    double targetHeight = 0.0;
    int zIndex = 0;
    double actualHeight = 0.0;
};

struct LayerSpectrum final {
    TargetLayer target;
    std::vector<float> energy;
    std::vector<float> logEnergy;
    double validFraction = 0.0;
    std::uint64_t validPoints = 0;
    fs::path outputPng;
};

struct IsotropicSpectrum final {
    int zStart = 0;
    int zStop = 0;
    double zStartHeight = 0.0;
    double zStopHeight = 0.0;
    double dk = 0.0;
    double kNyquist = 0.0;
    std::vector<double> kCenters;
    std::vector<double> shellEnergy;
    std::vector<double> shellModeWeight;
    std::vector<double> energy;
    std::vector<double> compensated;
};

[[nodiscard]] fs::path pathFromUtf8(std::string_view text) {
#ifdef _WIN32
    if (text.empty()) return {};
    const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                                         static_cast<int>(text.size()), nullptr, 0);
    if (size <= 0) return fs::path(std::string(text));
    std::wstring wide(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                        static_cast<int>(text.size()), wide.data(), size);
    return fs::path(wide);
#else
    return fs::path(text);
#endif
}

[[nodiscard]] fs::path absoluteNormalized(const fs::path& path) {
    std::error_code error;
    fs::path absolute = fs::absolute(path, error);
    if (error) absolute = path;
    return absolute.lexically_normal();
}

[[nodiscard]] double parseDouble(std::string_view option, std::string_view text) {
    std::string copy(text);
    std::size_t used = 0;
    try {
        const double value = std::stod(copy, &used);
        if (used != copy.size()) throw ArgumentError("argument " + std::string(option)
                                                     + ": invalid float value: '" + copy + "'");
        return value;
    } catch (const ArgumentError&) {
        throw;
    } catch (...) {
        throw ArgumentError("argument " + std::string(option)
                            + ": invalid float value: '" + copy + "'");
    }
}

[[nodiscard]] int parseInt(std::string_view option, std::string_view text) {
    const std::string copy(text);
    std::size_t used = 0;
    try {
        const long long parsed = std::stoll(copy, &used, 10);
        if (used != copy.size() || parsed < std::numeric_limits<int>::min()
            || parsed > std::numeric_limits<int>::max()) {
            throw std::out_of_range("int");
        }
        return static_cast<int>(parsed);
    } catch (...) {
        throw ArgumentError("argument " + std::string(option)
                            + ": invalid int value: '" + copy + "'");
    }
}

void printShortUsage(std::ostream& stream) {
    stream
        << "usage: les_spectra.py [-h] [--output-dir OUTPUT_DIR]\n"
        << "                      [--height-interval HEIGHT_INTERVAL]\n"
        << "                      [--height-start HEIGHT_START] [--workers WORKERS]\n"
        << "                      [--quick-test] [--test-height-count TEST_HEIGHT_COUNT]\n"
        << "                      [--test-3d-layers TEST_3D_LAYERS]\n"
        << "                      [vtk_path]\n";
}

void printUsage(std::ostream& stream) {
    printShortUsage(stream);
    stream
        << "\nCompute horizontal kx-ky spectra every 50 m and a 3D isotropic energy spectrum\n"
        << "for LES resolution checks from a legacy VTK file.\n\n"
        << "positional arguments:\n"
        << "  vtk_path              Path to the legacy VTK file.\n\n"
        << "options:\n"
        << "  -h, --help            show this help message and exit\n"
        << "  --output-dir OUTPUT_DIR\n"
        << "                        Directory where figures, CSVs and metadata are\n"
        << "                        written.\n"
        << "  --height-interval HEIGHT_INTERVAL\n"
        << "                        Target height interval in meters for horizontal\n"
        << "                        spectra.\n"
        << "  --height-start HEIGHT_START\n"
        << "                        First target height in meters for horizontal spectra.\n"
        << "  --workers WORKERS     Worker count passed to scipy.fft routines (-1 uses all\n"
        << "                        cores).\n"
        << "  --quick-test          Run a smaller, faster validation pass before the full\n"
        << "                        run.\n"
        << "  --test-height-count TEST_HEIGHT_COUNT\n"
        << "                        Number of heights used in quick-test mode.\n"
        << "  --test-3d-layers TEST_3D_LAYERS\n"
        << "                        Number of vertical layers used for the 3D FFT in\n"
        << "                        quick-test mode.\n";
}

[[nodiscard]] Arguments parseArguments(const std::vector<std::string>& values) {
    Arguments result;
    bool positionalSeen = false;
    bool optionsEnabled = true;
    static constexpr std::array<std::string_view, 8> longOptions{
        "--help", "--output-dir", "--height-interval", "--height-start",
        "--workers", "--quick-test", "--test-height-count", "--test-3d-layers"};
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (optionsEnabled && values[index] == "--") {
            optionsEnabled = false;
            continue;
        }
        std::string canonical = values[index];
        if (optionsEnabled && canonical.starts_with("--")) {
            const std::size_t equal = canonical.find('=');
            const std::string_view name(canonical.data(),
                                        equal == std::string::npos ? canonical.size() : equal);
            const bool exact = std::find(longOptions.begin(), longOptions.end(), name)
                             != longOptions.end();
            if (!exact) {
                std::vector<std::string_view> matches;
                for (const std::string_view option : longOptions) {
                    if (option.starts_with(name)) matches.push_back(option);
                }
                if (matches.size() > 1U) {
                    std::string message = "ambiguous option: " + std::string(name) + " could match ";
                    for (std::size_t match = 0; match < matches.size(); ++match) {
                        if (match != 0U) message += ", ";
                        message += matches[match];
                    }
                    throw ArgumentError(message);
                }
                if (matches.size() == 1U) {
                    canonical = std::string(matches.front())
                              + (equal == std::string::npos ? "" : values[index].substr(equal));
                }
            }
        }
        const std::string_view token = canonical;
        if (optionsEnabled && (token == "-h" || token == "--help")) {
            result.help = true;
            return result;
        }
        if (optionsEnabled && token == "--quick-test") {
            result.quickTest = true;
            continue;
        }
        auto optionValue = [&](std::string_view option) -> std::string_view {
            const std::string prefix = std::string(option) + '=';
            if (token.starts_with(prefix)) return token.substr(prefix.size());
            if (token == option) {
                if (++index >= values.size()) {
                    throw ArgumentError("argument " + std::string(option) + ": expected one argument");
                }
                return values[index];
            }
            return {};
        };
        if (optionsEnabled && (token == "--output-dir" || token.starts_with("--output-dir="))) {
            result.outputDirectory = pathFromUtf8(optionValue("--output-dir"));
        } else if (optionsEnabled && (token == "--height-interval" || token.starts_with("--height-interval="))) {
            result.heightInterval = parseDouble("--height-interval", optionValue("--height-interval"));
        } else if (optionsEnabled && (token == "--height-start" || token.starts_with("--height-start="))) {
            result.heightStart = parseDouble("--height-start", optionValue("--height-start"));
        } else if (optionsEnabled && (token == "--workers" || token.starts_with("--workers="))) {
            result.workers = parseInt("--workers", optionValue("--workers"));
        } else if (optionsEnabled && (token == "--test-height-count" || token.starts_with("--test-height-count="))) {
            result.testHeightCount = parseInt("--test-height-count", optionValue("--test-height-count"));
        } else if (optionsEnabled && (token == "--test-3d-layers" || token.starts_with("--test-3d-layers="))) {
            result.test3dLayers = parseInt("--test-3d-layers", optionValue("--test-3d-layers"));
        } else if (optionsEnabled && !token.empty() && token.front() == '-') {
            throw ArgumentError("unrecognized arguments: " + std::string(token));
        } else if (!positionalSeen) {
            result.vtkPath = pathFromUtf8(token);
            positionalSeen = true;
        } else {
            throw ArgumentError("unrecognized arguments: " + std::string(token));
        }
    }
    return result;
}

[[nodiscard]] std::vector<std::string> splitWhitespace(std::string_view line) {
    std::istringstream stream{std::string(line)};
    std::vector<std::string> result;
    for (std::string item; stream >> item;) result.push_back(std::move(item));
    return result;
}

[[nodiscard]] std::string trim(std::string value) {
    const auto notSpace = [](unsigned char character) { return !std::isspace(character); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(), value.end());
    return value;
}

[[nodiscard]] StructuredPointsMeta parseLegacyStructuredPoints(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("VTK file not found: " + pathUtf8(path));

    StructuredPointsMeta meta;
    meta.path = absoluteNormalized(path);
    bool hasDimensions = false;
    bool hasSpacing = false;
    bool hasOrigin = false;
    bool hasPointData = false;
    bool binary = false;
    bool dataOffsetFound = false;

    for (std::string line; std::getline(input, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string text = trim(std::move(line));
        const std::vector<std::string> parts = splitWhitespace(text);
        if (text == "BINARY") binary = true;
        if (parts.empty()) continue;
        try {
            if (parts[0] == "DATASET" && parts.size() >= 2) {
                meta.datasetType = text.substr(text.find_first_of(" \t") + 1);
            } else if (parts[0] == "DIMENSIONS" && parts.size() == 4) {
                meta.dimensions = {std::stoi(parts[1]), std::stoi(parts[2]), std::stoi(parts[3])};
                hasDimensions = true;
            } else if (parts[0] == "SPACING" && parts.size() == 4) {
                meta.spacing = {std::stod(parts[1]), std::stod(parts[2]), std::stod(parts[3])};
                hasSpacing = true;
            } else if (parts[0] == "ORIGIN" && parts.size() == 4) {
                meta.origin = {std::stod(parts[1]), std::stod(parts[2]), std::stod(parts[3])};
                hasOrigin = true;
            } else if (parts[0] == "POINT_DATA" && parts.size() == 2) {
                meta.pointData = std::stoull(parts[1]);
                hasPointData = true;
            } else if (parts[0] == "SCALARS" && parts.size() >= 3) {
                meta.dataName = parts[1];
                meta.dtypeName = parts[2];
                if (parts.size() >= 4) meta.components = std::stoi(parts[3]);
            } else if (text == "LOOKUP_TABLE default") {
                const std::streampos position = input.tellg();
                if (position < 0) throw std::runtime_error("Invalid VTK data offset.");
                meta.dataOffset = static_cast<std::uint64_t>(position);
                dataOffsetFound = true;
                break;
            }
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid legacy VTK header field: " + text);
        }
    }

    if (!binary) throw std::runtime_error("Only BINARY legacy VTK data is supported.");
    if (meta.datasetType != "STRUCTURED_POINTS") {
        throw std::runtime_error("Unsupported DATASET type: '" + meta.datasetType + "'");
    }
    if (!hasDimensions || !hasSpacing || !hasOrigin || !hasPointData) {
        throw std::runtime_error("Incomplete VTK header; missing required grid metadata.");
    }
    if (meta.dtypeName != "float") {
        throw std::runtime_error("Only float data is supported, got '" + meta.dtypeName + "'.");
    }
    if (!dataOffsetFound) {
        throw std::runtime_error("Could not locate LOOKUP_TABLE/data offset in VTK file.");
    }
    if (std::any_of(meta.dimensions.begin(), meta.dimensions.end(),
                    [](int value) { return value <= 0; })
        || meta.components <= 0) {
        throw std::runtime_error("VTK dimensions and scalar component count must be positive.");
    }
    return meta;
}

class ReadOnlyMapping final {
public:
    explicit ReadOnlyMapping(const fs::path& path) {
#ifdef _WIN32
        file_ = CreateFileW(path.c_str(), GENERIC_READ,
                            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Cannot open VTK file: " + pathUtf8(path));
        }
        LARGE_INTEGER length{};
        if (!GetFileSizeEx(file_, &length) || length.QuadPart < 0) {
            close();
            throw std::runtime_error("Cannot determine VTK file size: " + pathUtf8(path));
        }
        size_ = static_cast<std::uint64_t>(length.QuadPart);
        mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) {
            close();
            throw std::runtime_error("Cannot map VTK file: " + pathUtf8(path));
        }
        data_ = static_cast<const unsigned char*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (!data_) {
            close();
            throw std::runtime_error("Cannot map VTK file: " + pathUtf8(path));
        }
#else
        file_ = ::open(path.c_str(), O_RDONLY);
        if (file_ < 0) throw std::runtime_error("Cannot open VTK file: " + pathUtf8(path));
        struct stat status {};
        if (::fstat(file_, &status) != 0 || status.st_size < 0) {
            close();
            throw std::runtime_error("Cannot determine VTK file size: " + pathUtf8(path));
        }
        size_ = static_cast<std::uint64_t>(status.st_size);
        void* mapped = ::mmap(nullptr, static_cast<std::size_t>(size_), PROT_READ, MAP_SHARED, file_, 0);
        if (mapped == MAP_FAILED) {
            data_ = nullptr;
            close();
            throw std::runtime_error("Cannot map VTK file: " + pathUtf8(path));
        }
        data_ = static_cast<const unsigned char*>(mapped);
#endif
    }

    ~ReadOnlyMapping() { close(); }
    ReadOnlyMapping(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping& operator=(const ReadOnlyMapping&) = delete;

    [[nodiscard]] const unsigned char* data() const noexcept { return data_; }
    [[nodiscard]] std::uint64_t size() const noexcept { return size_; }

private:
    void close() noexcept {
#ifdef _WIN32
        if (data_) UnmapViewOfFile(data_);
        if (mapping_) CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        data_ = nullptr;
        mapping_ = nullptr;
        file_ = INVALID_HANDLE_VALUE;
#else
        if (data_) ::munmap(const_cast<unsigned char*>(data_), static_cast<std::size_t>(size_));
        if (file_ >= 0) ::close(file_);
        data_ = nullptr;
        file_ = -1;
#endif
    }

    const unsigned char* data_ = nullptr;
    std::uint64_t size_ = 0;
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int file_ = -1;
#endif
};

class VtkFloatView final {
public:
    VtkFloatView(const StructuredPointsMeta& meta, const ReadOnlyMapping& mapping)
        : meta_(meta), bytes_(mapping.data() + meta.dataOffset) {
        const std::uint64_t nx = static_cast<std::uint64_t>(meta.dimensions[0]);
        const std::uint64_t ny = static_cast<std::uint64_t>(meta.dimensions[1]);
        const std::uint64_t nz = static_cast<std::uint64_t>(meta.dimensions[2]);
        if (nx > std::numeric_limits<std::uint64_t>::max() / ny
            || nx * ny > std::numeric_limits<std::uint64_t>::max() / nz) {
            throw std::runtime_error("DIMENSIONS product overflows the supported size.");
        }
        points_ = nx * ny * nz;
        if (points_ != meta.pointData) {
            throw std::runtime_error("POINT_DATA=" + std::to_string(meta.pointData)
                                     + " does not match DIMENSIONS product="
                                     + std::to_string(points_) + ".");
        }
        const std::uint64_t components = static_cast<std::uint64_t>(meta.components);
        if (points_ > std::numeric_limits<std::uint64_t>::max() / components / 4U) {
            throw std::runtime_error("VTK scalar payload size overflows the supported size.");
        }
        payloadBytes_ = points_ * components * 4U;
        if (meta.dataOffset > mapping.size() || payloadBytes_ > mapping.size() - meta.dataOffset) {
            throw std::runtime_error("VTK scalar payload is shorter than declared by the header.");
        }
    }

    [[nodiscard]] float value(int z, int y, int x, int component) const noexcept {
        const std::uint64_t nx = static_cast<std::uint64_t>(meta_.dimensions[0]);
        const std::uint64_t ny = static_cast<std::uint64_t>(meta_.dimensions[1]);
        const std::uint64_t index = (((static_cast<std::uint64_t>(z) * ny
                                     + static_cast<std::uint64_t>(y)) * nx
                                     + static_cast<std::uint64_t>(x))
                                     * static_cast<std::uint64_t>(meta_.components)
                                     + static_cast<std::uint64_t>(component));
        const unsigned char* source = bytes_ + index * 4U;
        const std::uint32_t bits = (static_cast<std::uint32_t>(source[0]) << 24U)
                                 | (static_cast<std::uint32_t>(source[1]) << 16U)
                                 | (static_cast<std::uint32_t>(source[2]) << 8U)
                                 | static_cast<std::uint32_t>(source[3]);
        return std::bit_cast<float>(bits);
    }

private:
    const StructuredPointsMeta& meta_;
    const unsigned char* bytes_ = nullptr;
    std::uint64_t points_ = 0;
    std::uint64_t payloadBytes_ = 0;
};

[[nodiscard]] int availableWorkers() noexcept {
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

[[nodiscard]] int resolveWorkers(int requested) {
    if (requested == 0) {
        throw std::runtime_error("workers must not be zero");
    }
    const int maximum = availableWorkers();
    if (requested > 0) return std::max(1, std::min(requested, maximum));
    const int resolved = maximum + requested + 1;
    if (resolved < 1) {
        throw std::runtime_error("workers value out of range; got " + std::to_string(requested)
                                 + ", must not be less than -" + std::to_string(maximum));
    }
    return resolved;
}

[[nodiscard]] std::vector<TargetLayer> buildTargets(const StructuredPointsMeta& meta,
                                                     double start, double interval) {
    if (interval == 0.0 || !std::isfinite(interval) || !std::isfinite(start)) {
        throw std::runtime_error("arange: cannot compute length");
    }
    const double origin = meta.origin[2];
    const double top = origin + meta.spacing[2] * static_cast<double>(meta.dimensions[2] - 1);
    const double stop = top + 1.0e-9;
    std::vector<TargetLayer> result;
    std::vector<unsigned char> seen(static_cast<std::size_t>(meta.dimensions[2]), 0U);
    constexpr std::size_t maximumTargets = 100000000U;
    for (std::size_t index = 0; index < maximumTargets; ++index) {
        const double target = start + static_cast<double>(index) * interval;
        if ((interval > 0.0 && !(target < stop)) || (interval < 0.0 && !(target > stop))) break;
        const double indexValue = (target - origin) / meta.spacing[2];
        if (!std::isfinite(indexValue)
            || indexValue < static_cast<double>(std::numeric_limits<int>::min())
            || indexValue > static_cast<double>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("cannot convert target height to a finite z-index");
        }
        const int z = static_cast<int>(std::nearbyint(indexValue));
        if (z >= 0 && z < meta.dimensions[2] && seen[static_cast<std::size_t>(z)] == 0U) {
            seen[static_cast<std::size_t>(z)] = 1U;
            result.push_back({target, z, origin + static_cast<double>(z) * meta.spacing[2]});
        }
    }
    if (result.size() == maximumTargets) throw std::runtime_error("Maximum allowed size exceeded");
    return result;
}

[[nodiscard]] bool pointValid(const VtkFloatView& data,
                              const StructuredPointsMeta& meta,
                              int z, int y, int x) noexcept {
    for (int component = 0; component < meta.components; ++component) {
        if (data.value(z, y, x, component) != 0.0F) return true;
    }
    return false;
}

[[nodiscard]] std::vector<double> computeCoverages(const VtkFloatView& data,
                                                   const StructuredPointsMeta& meta) {
    const int nx = meta.dimensions[0];
    const int ny = meta.dimensions[1];
    const std::uint64_t total = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny);
    std::vector<double> result(static_cast<std::size_t>(meta.dimensions[2]));
    for (int z = 0; z < meta.dimensions[2]; ++z) {
        std::uint64_t valid = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : valid) schedule(static)
#endif
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) valid += pointValid(data, meta, z, y, x) ? 1U : 0U;
        }
        result[static_cast<std::size_t>(z)] = static_cast<double>(valid) / static_cast<double>(total);
    }
    return result;
}

// Keep multidimensional coefficients at complex64, matching scipy.fft for a
// float32 input.  Production builds use pocketfft (the same algorithm family
// as SciPy); VTK's bundled kissfft is the dependency-light fallback.  VTK can
// be configured with double scalars, so the workspace converts at its edge
// and quantizes every transformed axis back to complex64.
using Complex = std::complex<float>;

class FftWorkspace final {
public:
    void realToComplex(const float* input, std::size_t size, Complex* output) {
#if LUW_HAS_POCKETFFT
        pocketInput_.resize(size);
        pocketOutput_.resize(size);
        for (std::size_t index = 0; index < size; ++index) pocketInput_[index] = {input[index], 0.0F};
        transformPocket(size);
        copyPocket(output, size);
#else
        vtkReal_.resize(size);
        vtkOutput_.resize(size);
        for (std::size_t index = 0; index < size; ++index) {
            vtkReal_[index] = static_cast<vtkFFT::ScalarNumber>(input[index]);
        }
        vtkFFT::Fft(vtkReal_.data(), size, vtkOutput_.data());
        copyVtk(output, size);
#endif
    }

    void realToHalfComplex(const float* input, std::size_t size, Complex* output) {
        const std::size_t outputSize = size / 2U + 1U;
#if LUW_HAS_POCKETFFT
        pocketReal_.assign(input, input + size);
        pocketOutput_.resize(outputSize);
        const pocketfft::shape_t shape{size};
        const pocketfft::stride_t inputStride{static_cast<std::ptrdiff_t>(sizeof(float))};
        const pocketfft::stride_t outputStride{
            static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
        const pocketfft::shape_t axes{0};
        pocketfft::r2c(shape, inputStride, outputStride, axes, pocketfft::FORWARD,
                       pocketReal_.data(), pocketOutput_.data(), 1.0F, 1U);
        copyPocket(output, outputSize);
#else
        vtkReal_.resize(size);
        vtkOutput_.resize(outputSize);
        for (std::size_t index = 0; index < size; ++index) {
            vtkReal_[index] = static_cast<vtkFFT::ScalarNumber>(input[index]);
        }
        vtkFFT::RFft(vtkReal_.data(), size, vtkOutput_.data());
        copyVtk(output, outputSize);
#endif
    }

    void complexToComplex(const Complex* input, std::size_t size, Complex* output) {
#if LUW_HAS_POCKETFFT
        pocketInput_.resize(size);
        pocketOutput_.resize(size);
        for (std::size_t index = 0; index < size; ++index) {
            pocketInput_[index] = input[index];
        }
        transformPocket(size);
        copyPocket(output, size);
#else
        vtkInput_.resize(size);
        vtkOutput_.resize(size);
        for (std::size_t index = 0; index < size; ++index) {
            vtkInput_[index] = {static_cast<vtkFFT::ScalarNumber>(input[index].real()),
                                static_cast<vtkFFT::ScalarNumber>(input[index].imag())};
        }
        vtkFFT::Fft(vtkInput_.data(), size, vtkOutput_.data());
        copyVtk(output, size);
#endif
    }

private:
#if LUW_HAS_POCKETFFT
    void transformPocket(std::size_t size) {
        const pocketfft::shape_t shape{size};
        const pocketfft::stride_t stride{
            static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
        const pocketfft::shape_t axes{0};
        pocketfft::c2c(shape, stride, stride, axes, pocketfft::FORWARD,
                       pocketInput_.data(), pocketOutput_.data(), 1.0F, 1U);
    }

    void copyPocket(Complex* output, std::size_t size) const noexcept {
        for (std::size_t index = 0; index < size; ++index) {
            output[index] = {pocketOutput_[index].real(), pocketOutput_[index].imag()};
        }
    }

    std::vector<float> pocketReal_;
    std::vector<std::complex<float>> pocketInput_;
    std::vector<std::complex<float>> pocketOutput_;
#else
    void copyVtk(Complex* output, std::size_t size) const noexcept {
        for (std::size_t index = 0; index < size; ++index) {
            output[index] = {static_cast<float>(vtkOutput_[index].r),
                             static_cast<float>(vtkOutput_[index].i)};
        }
    }

    std::vector<vtkFFT::ScalarNumber> vtkReal_;
    std::vector<vtkFFT::ComplexNumber> vtkInput_;
    std::vector<vtkFFT::ComplexNumber> vtkOutput_;
#endif
};

void fft2(std::vector<float>& real, int ny, int nx, int workers,
          std::vector<Complex>& output) {
    const std::size_t total = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    output.resize(total);
#if LUW_HAS_POCKETFFT
    for (std::size_t index = 0; index < total; ++index) output[index] = {real[index], 0.0F};
    const pocketfft::shape_t shape{static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)};
    const pocketfft::stride_t stride{
        static_cast<std::ptrdiff_t>(static_cast<std::size_t>(nx) * sizeof(Complex)),
        static_cast<std::ptrdiff_t>(sizeof(Complex))};
    const pocketfft::shape_t axes{0, 1};
    pocketfft::c2c(shape, stride, stride, axes, pocketfft::FORWARD,
                   output.data(), output.data(), 1.0F, static_cast<std::size_t>(workers));
#else
#ifdef _OPENMP
#pragma omp parallel num_threads(workers) if(ny > 1)
#endif
    {
        FftWorkspace workspace;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int y = 0; y < ny; ++y) {
            const std::size_t offset = static_cast<std::size_t>(y) * static_cast<std::size_t>(nx);
            workspace.realToComplex(real.data() + offset, static_cast<std::size_t>(nx),
                                    output.data() + offset);
        }
    }
#ifdef _OPENMP
#pragma omp parallel num_threads(workers) if(nx > 1)
#endif
    {
        FftWorkspace workspace;
        std::vector<Complex> input(static_cast<std::size_t>(ny));
        std::vector<Complex> transformed(static_cast<std::size_t>(ny));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                const Complex value = output[static_cast<std::size_t>(y)
                                             * static_cast<std::size_t>(nx)
                                             + static_cast<std::size_t>(x)];
                input[static_cast<std::size_t>(y)] = value;
            }
            workspace.complexToComplex(input.data(), input.size(), transformed.data());
            for (int y = 0; y < ny; ++y) {
                output[static_cast<std::size_t>(y) * static_cast<std::size_t>(nx)
                       + static_cast<std::size_t>(x)] = transformed[static_cast<std::size_t>(y)];
            }
        }
    }
#endif
}

[[nodiscard]] LayerSpectrum computeHorizontal(const VtkFloatView& data,
                                              const StructuredPointsMeta& meta,
                                              TargetLayer target,
                                              int workers) {
    const int nx = meta.dimensions[0];
    const int ny = meta.dimensions[1];
    const std::size_t total = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    std::vector<unsigned char> valid(total);
    std::uint64_t validCount = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(workers) reduction(+ : validCount) schedule(static)
#endif
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(nx)
                                    + static_cast<std::size_t>(x);
            valid[index] = pointValid(data, meta, target.zIndex, y, x) ? 1U : 0U;
            validCount += valid[index];
        }
    }

    LayerSpectrum result;
    result.target = target;
    result.validPoints = validCount;
    result.validFraction = static_cast<double>(validCount) / static_cast<double>(total);
    result.energy.assign(total, 0.0F);
    if (validCount == 0) return result;

    std::vector<double> accumulated(total, 0.0);
    std::vector<float> field(total);
    std::vector<Complex> coefficients;
    const float normalization = static_cast<float>(static_cast<double>(total)
                                                   * static_cast<double>(validCount));
    for (int component = 0; component < std::min(meta.components, 3); ++component) {
        double sum = 0.0;
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(nx)
                                        + static_cast<std::size_t>(x);
                const float value = data.value(target.zIndex, y, x, component);
                field[index] = value;
                if (valid[index] != 0U) sum += static_cast<double>(value);
            }
        }
        const double mean = sum / static_cast<double>(validCount);
        for (std::size_t index = 0; index < total; ++index) {
            field[index] = valid[index] != 0U
                ? static_cast<float>(static_cast<double>(field[index]) - mean) : 0.0F;
        }
        fft2(field, ny, nx, workers, coefficients);
        for (std::size_t index = 0; index < total; ++index) {
            const Complex coefficient = coefficients[index];
            const float magnitude = std::abs(coefficient);
            const float squared = magnitude * magnitude;
            const float contribution = 0.5F * squared / normalization;
            accumulated[index] += static_cast<double>(contribution);
        }
    }

    // np.fft.fftshift moves the zero-frequency sample to the center.  For an
    // odd axis this is a ceil(n/2) source rotation; for an even axis it is n/2.
    const int shiftX = (nx + 1) / 2;
    const int shiftY = (ny + 1) / 2;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const int sourceX = (x + shiftX) % nx;
            const int sourceY = (y + shiftY) % ny;
            result.energy[static_cast<std::size_t>(y) * static_cast<std::size_t>(nx)
                          + static_cast<std::size_t>(x)] =
                static_cast<float>(accumulated[static_cast<std::size_t>(sourceY)
                                               * static_cast<std::size_t>(nx)
                                               + static_cast<std::size_t>(sourceX)]);
        }
    }
    return result;
}

[[nodiscard]] std::vector<double> fftFrequencies(int count, double spacing, bool shifted) {
    std::vector<double> result(static_cast<std::size_t>(count));
    const int positiveStop = (count + 1) / 2;
    const double frequencyStep = 1.0 / (static_cast<double>(count) * spacing);
    for (int index = 0; index < count; ++index) {
        const int wave = index < positiveStop ? index : index - count;
        result[static_cast<std::size_t>(index)] = (2.0 * pi)
            * (static_cast<double>(wave) * frequencyStep);
    }
    if (shifted) std::rotate(result.begin(), result.begin() + (count + 1) / 2, result.end());
    return result;
}

[[nodiscard]] std::vector<double> realFftFrequencies(int count, double spacing) {
    std::vector<double> result(static_cast<std::size_t>(count / 2 + 1));
    const double frequencyStep = 1.0 / (static_cast<double>(count) * spacing);
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = (2.0 * pi)
            * (static_cast<double>(index) * frequencyStep);
    }
    return result;
}

[[nodiscard]] double percentile(std::vector<float> values, double percent) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double position = (static_cast<double>(values.size()) - 1.0) * percent / 100.0;
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return static_cast<double>(values[lower]) * (1.0 - fraction)
         + static_cast<double>(values[upper]) * fraction;
}

[[nodiscard]] std::pair<double, double> logColorLimits(std::vector<LayerSpectrum>& layers) {
    std::vector<float> positiveLogs;
    for (const LayerSpectrum& layer : layers) {
        for (const float value : layer.energy) {
            if (value > 0.0F) positiveLogs.push_back(std::log10(value));
        }
    }
    double minimum = -12.0;
    double maximum = 0.0;
    if (!positiveLogs.empty()) {
        minimum = percentile(positiveLogs, 5.0);
        maximum = percentile(std::move(positiveLogs), 99.5);
        if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
            minimum = -12.0;
            maximum = 0.0;
        }
        if (std::abs(maximum - minimum) <= 1.0e-9
            * std::max(std::abs(minimum), std::abs(maximum))) {
            maximum = minimum + 1.0;
        }
    }
    const float epsilon = static_cast<float>(std::pow(10.0, minimum));
    for (LayerSpectrum& layer : layers) {
        layer.logEnergy.resize(layer.energy.size());
        for (std::size_t index = 0; index < layer.energy.size(); ++index) {
            layer.logEnergy[index] = std::log10(std::max(layer.energy[index], epsilon));
        }
    }
    return {minimum, maximum};
}

void realFft3(std::vector<float>& field, int nz, int ny, int nx, int workers,
              std::vector<Complex>& coefficients) {
    const int nxHalf = nx / 2 + 1;
    const std::size_t complexCount = static_cast<std::size_t>(nz)
                                   * static_cast<std::size_t>(ny)
                                   * static_cast<std::size_t>(nxHalf);
    coefficients.resize(complexCount);
#if LUW_HAS_POCKETFFT
    const pocketfft::shape_t shape{static_cast<std::size_t>(nz),
                                   static_cast<std::size_t>(ny),
                                   static_cast<std::size_t>(nx)};
    const pocketfft::stride_t inputStride{
        static_cast<std::ptrdiff_t>(static_cast<std::size_t>(ny)
                                    * static_cast<std::size_t>(nx) * sizeof(float)),
        static_cast<std::ptrdiff_t>(static_cast<std::size_t>(nx) * sizeof(float)),
        static_cast<std::ptrdiff_t>(sizeof(float))};
    const pocketfft::stride_t outputStride{
        static_cast<std::ptrdiff_t>(static_cast<std::size_t>(ny)
                                    * static_cast<std::size_t>(nxHalf) * sizeof(Complex)),
        static_cast<std::ptrdiff_t>(static_cast<std::size_t>(nxHalf) * sizeof(Complex)),
        static_cast<std::ptrdiff_t>(sizeof(Complex))};
    const pocketfft::shape_t axes{0, 1, 2};
    pocketfft::r2c(shape, inputStride, outputStride, axes, pocketfft::FORWARD,
                   field.data(), coefficients.data(), 1.0F,
                   static_cast<std::size_t>(workers));
#else
#ifdef _OPENMP
#pragma omp parallel num_threads(workers) if(nz * ny > 1)
#endif
    {
        FftWorkspace workspace;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int line = 0; line < nz * ny; ++line) {
            const std::size_t realOffset = static_cast<std::size_t>(line) * static_cast<std::size_t>(nx);
            const std::size_t complexOffset = static_cast<std::size_t>(line)
                                            * static_cast<std::size_t>(nxHalf);
            workspace.realToHalfComplex(field.data() + realOffset, static_cast<std::size_t>(nx),
                                        coefficients.data() + complexOffset);
        }
    }
#ifdef _OPENMP
#pragma omp parallel num_threads(workers) if(nz * nxHalf > 1)
#endif
    {
        FftWorkspace workspace;
        std::vector<Complex> input(static_cast<std::size_t>(ny));
        std::vector<Complex> transformed(static_cast<std::size_t>(ny));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int line = 0; line < nz * nxHalf; ++line) {
            const int z = line / nxHalf;
            const int x = line % nxHalf;
            for (int y = 0; y < ny; ++y) {
                const Complex value =
                    coefficients[(static_cast<std::size_t>(z) * static_cast<std::size_t>(ny)
                                  + static_cast<std::size_t>(y)) * static_cast<std::size_t>(nxHalf)
                                  + static_cast<std::size_t>(x)];
                input[static_cast<std::size_t>(y)] = value;
            }
            workspace.complexToComplex(input.data(), input.size(), transformed.data());
            for (int y = 0; y < ny; ++y) {
                coefficients[(static_cast<std::size_t>(z) * static_cast<std::size_t>(ny)
                              + static_cast<std::size_t>(y)) * static_cast<std::size_t>(nxHalf)
                              + static_cast<std::size_t>(x)] = transformed[static_cast<std::size_t>(y)];
            }
        }
    }
#ifdef _OPENMP
#pragma omp parallel num_threads(workers) if(ny * nxHalf > 1)
#endif
    {
        FftWorkspace workspace;
        std::vector<Complex> input(static_cast<std::size_t>(nz));
        std::vector<Complex> transformed(static_cast<std::size_t>(nz));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int line = 0; line < ny * nxHalf; ++line) {
            const int y = line / nxHalf;
            const int x = line % nxHalf;
            for (int z = 0; z < nz; ++z) {
                const Complex value =
                    coefficients[(static_cast<std::size_t>(z) * static_cast<std::size_t>(ny)
                                  + static_cast<std::size_t>(y)) * static_cast<std::size_t>(nxHalf)
                                  + static_cast<std::size_t>(x)];
                input[static_cast<std::size_t>(z)] = value;
            }
            workspace.complexToComplex(input.data(), input.size(), transformed.data());
            for (int z = 0; z < nz; ++z) {
                coefficients[(static_cast<std::size_t>(z) * static_cast<std::size_t>(ny)
                              + static_cast<std::size_t>(y)) * static_cast<std::size_t>(nxHalf)
                              + static_cast<std::size_t>(x)] = transformed[static_cast<std::size_t>(z)];
            }
        }
    }
#endif
}

[[nodiscard]] bool numpyIsCloseToOne(double value) noexcept {
    return std::abs(value - 1.0) <= 1.0e-8 + 1.0e-5;
}

[[nodiscard]] IsotropicSpectrum computeIsotropic(const VtkFloatView& data,
                                                 const StructuredPointsMeta& meta,
                                                 const std::vector<double>& coverages,
                                                 int workers,
                                                 std::optional<int> quickLayers,
                                                 std::ostream& output) {
    int zStart = -1;
    for (int z = 0; z < meta.dimensions[2]; ++z) {
        if (numpyIsCloseToOne(coverages[static_cast<std::size_t>(z)])) {
            zStart = z;
            break;
        }
    }
    if (zStart < 0) throw std::runtime_error("No fully valid horizontal layer exists for the 3D spectrum.");
    int zStop = meta.dimensions[2];
    if (quickLayers) {
        zStop = std::min(meta.dimensions[2], zStart + *quickLayers);
        if (zStop - zStart < 4) {
            throw std::runtime_error("Quick-test 3D subvolume is too small to be meaningful.");
        }
    }
    const int nx = meta.dimensions[0];
    const int ny = meta.dimensions[1];
    const int nz = zStop - zStart;
    const int nxHalf = nx / 2 + 1;
    output << "3D spectrum subvolume: z-index " << zStart << ':' << zStop << " ("
           << std::fixed << std::setprecision(2)
           << meta.origin[2] + static_cast<double>(zStart) * meta.spacing[2] << " to "
           << meta.origin[2] + static_cast<double>(zStop - 1) * meta.spacing[2]
           << " m)\n" << std::defaultfloat;

    const std::vector<double> kx = realFftFrequencies(nx, meta.spacing[0]);
    const std::vector<double> ky = fftFrequencies(ny, meta.spacing[1], false);
    const std::vector<double> kz = fftFrequencies(nz, meta.spacing[2], false);
    const double dk = std::max({2.0 * pi / (static_cast<double>(nx) * meta.spacing[0]),
                                2.0 * pi / (static_cast<double>(ny) * meta.spacing[1]),
                                2.0 * pi / (static_cast<double>(nz) * meta.spacing[2])});
    const double maximumKx = *std::max_element(kx.begin(), kx.end());
    double maximumKy = 0.0;
    double maximumKz = 0.0;
    for (const double value : ky) maximumKy = std::max(maximumKy, std::abs(value));
    for (const double value : kz) maximumKz = std::max(maximumKz, std::abs(value));
    const double maximumK = std::sqrt(maximumKx * maximumKx
                                    + maximumKy * maximumKy + maximumKz * maximumKz);
    const int binCount = static_cast<int>(std::floor(maximumK / dk)) + 1;

    IsotropicSpectrum result;
    result.zStart = zStart;
    result.zStop = zStop;
    result.zStartHeight = meta.origin[2] + static_cast<double>(zStart) * meta.spacing[2];
    result.zStopHeight = meta.origin[2] + static_cast<double>(zStop - 1) * meta.spacing[2];
    result.dk = dk;
    result.kNyquist = pi / std::max({meta.spacing[0], meta.spacing[1], meta.spacing[2]});
    result.kCenters.resize(static_cast<std::size_t>(binCount));
    result.shellEnergy.assign(static_cast<std::size_t>(binCount), 0.0);
    result.shellModeWeight.assign(static_cast<std::size_t>(binCount), 0.0);
    for (int bin = 0; bin < binCount; ++bin) {
        result.kCenters[static_cast<std::size_t>(bin)] = (static_cast<double>(bin) + 0.5) * dk;
    }

    std::vector<double> kxWeights(static_cast<std::size_t>(nxHalf), 1.0);
    if (nxHalf > 1) {
        const int stop = nx % 2 == 0 ? nxHalf - 1 : nxHalf;
        for (int x = 1; x < stop; ++x) kxWeights[static_cast<std::size_t>(x)] = 2.0;
    }
    std::vector<int> shellIndex(static_cast<std::size_t>(nz) * static_cast<std::size_t>(ny)
                               * static_cast<std::size_t>(nxHalf));
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nxHalf; ++x) {
                int shell = static_cast<int>(std::floor(std::sqrt(
                    kz[static_cast<std::size_t>(z)] * kz[static_cast<std::size_t>(z)]
                  + ky[static_cast<std::size_t>(y)] * ky[static_cast<std::size_t>(y)]
                  + kx[static_cast<std::size_t>(x)] * kx[static_cast<std::size_t>(x)]) / dk));
                shell = std::clamp(shell, 0, binCount - 1);
                const std::size_t index = (static_cast<std::size_t>(z) * static_cast<std::size_t>(ny)
                                          + static_cast<std::size_t>(y)) * static_cast<std::size_t>(nxHalf)
                                          + static_cast<std::size_t>(x);
                shellIndex[index] = shell;
                result.shellModeWeight[static_cast<std::size_t>(shell)]
                    += kxWeights[static_cast<std::size_t>(x)];
            }
        }
    }

    const std::size_t total = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny)
                            * static_cast<std::size_t>(nz);
    const double normalization = 0.5 / (static_cast<double>(total) * static_cast<double>(total));
    std::vector<float> field(total);
    std::vector<Complex> coefficients;
    for (int component = 0; component < std::min(meta.components, 3); ++component) {
        output << "3D FFT component " << component + 1 << "/3\n";
        double sum = 0.0;
        std::size_t index = 0;
        for (int z = zStart; z < zStop; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x, ++index) {
                    field[index] = data.value(z, y, x, component);
                    sum += static_cast<double>(field[index]);
                }
            }
        }
        const double mean = sum / static_cast<double>(total);
        for (float& value : field) value = static_cast<float>(static_cast<double>(value) - mean);
        realFft3(field, nz, ny, nx, workers, coefficients);
        const float normalization32 = static_cast<float>(normalization);
        for (std::size_t coefficientIndex = 0; coefficientIndex < coefficients.size(); ++coefficientIndex) {
            const Complex coefficient = coefficients[coefficientIndex];
            const float magnitude = std::abs(coefficient);
            const float squared = magnitude * magnitude;
            const float normalized = normalization32 * squared;
            const int x = static_cast<int>(coefficientIndex % static_cast<std::size_t>(nxHalf));
            result.shellEnergy[static_cast<std::size_t>(shellIndex[coefficientIndex])]
                += static_cast<double>(normalized) * kxWeights[static_cast<std::size_t>(x)];
        }
        coefficients.clear();
        coefficients.shrink_to_fit();
    }

    result.energy.assign(static_cast<std::size_t>(binCount), 0.0);
    result.compensated.assign(static_cast<std::size_t>(binCount), 0.0);
    for (int bin = 0; bin < binCount; ++bin) {
        const std::size_t index = static_cast<std::size_t>(bin);
        if (result.shellModeWeight[index] > 0.0) result.energy[index] = result.shellEnergy[index] / dk;
        if (result.kCenters[index] > 0.0) {
            result.compensated[index] = result.energy[index]
                                      * std::pow(result.kCenters[index], 5.0 / 3.0);
        }
    }
    return result;
}

struct Color final {
    unsigned char red = 255;
    unsigned char green = 255;
    unsigned char blue = 255;
};

class Canvas final {
public:
    Canvas(int width, int height, Color background = {255, 255, 255})
        : width_(width), height_(height),
          pixels_(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), background) {
        if (width <= 0 || height <= 0) throw std::runtime_error("Canvas dimensions must be positive.");
    }

    void pixel(int x, int y, Color color) noexcept {
        if (x < 0 || y < 0 || x >= width_ || y >= height_) return;
        pixels_[static_cast<std::size_t>(y) * static_cast<std::size_t>(width_)
                + static_cast<std::size_t>(x)] = color;
    }

    [[nodiscard]] Color pixel(int x, int y) const noexcept {
        return pixels_[static_cast<std::size_t>(y) * static_cast<std::size_t>(width_)
                       + static_cast<std::size_t>(x)];
    }

    void fill(int left, int top, int right, int bottom, Color color) noexcept {
        left = std::clamp(left, 0, width_);
        right = std::clamp(right, 0, width_);
        top = std::clamp(top, 0, height_);
        bottom = std::clamp(bottom, 0, height_);
        if (right <= left || bottom <= top) return;
        for (int y = top; y < bottom; ++y) {
            auto begin = pixels_.begin() + static_cast<std::ptrdiff_t>(y) * width_ + left;
            std::fill(begin, begin + (right - left), color);
        }
    }

    void line(int x0, int y0, int x1, int y1, Color color, int thickness = 1) noexcept {
        const int dx = std::abs(x1 - x0);
        const int sx = x0 < x1 ? 1 : -1;
        const int dy = -std::abs(y1 - y0);
        const int sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;
        for (;;) {
            fill(x0 - thickness / 2, y0 - thickness / 2,
                 x0 + (thickness + 1) / 2, y0 + (thickness + 1) / 2, color);
            if (x0 == x1 && y0 == y1) break;
            const int twice = 2 * error;
            if (twice >= dy) { error += dy; x0 += sx; }
            if (twice <= dx) { error += dx; y0 += sy; }
        }
    }

    void circle(int centerX, int centerY, int radius, Color color) noexcept {
        int x = radius;
        int y = 0;
        int error = 0;
        while (x >= y) {
            pixel(centerX + x, centerY + y, color); pixel(centerX + y, centerY + x, color);
            pixel(centerX - y, centerY + x, color); pixel(centerX - x, centerY + y, color);
            pixel(centerX - x, centerY - y, color); pixel(centerX - y, centerY - x, color);
            pixel(centerX + y, centerY - x, color); pixel(centerX + x, centerY - y, color);
            ++y;
            if (error <= 0) error += 2 * y + 1;
            if (error > 0) { --x; error -= 2 * x + 1; }
        }
    }

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<Color> pixels_;
};

[[nodiscard]] std::array<unsigned char, 7> glyph(char character) noexcept {
    const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
    switch (c) {
    case 'A': return {14,17,17,31,17,17,17}; case 'B': return {30,17,17,30,17,17,30};
    case 'C': return {14,17,16,16,16,17,14}; case 'D': return {30,17,17,17,17,17,30};
    case 'E': return {31,16,16,30,16,16,31}; case 'F': return {31,16,16,30,16,16,16};
    case 'G': return {14,17,16,23,17,17,15}; case 'H': return {17,17,17,31,17,17,17};
    case 'I': return {31,4,4,4,4,4,31}; case 'J': return {7,2,2,2,18,18,12};
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
    case '^': return {4,10,17,0,0,0,0}; case '|': return {4,4,4,4,4,4,4};
    case '%': return {17,2,4,8,17,0,0}; case '<': return {2,4,8,4,2,0,0};
    case '>': return {8,4,2,4,8,0,0}; case '*': return {0,21,14,31,14,21,0};
    default: return {};
    }
}

[[nodiscard]] int textWidth(std::string_view text, int scale) noexcept {
    return static_cast<int>(text.size()) * 6 * scale;
}

void drawText(Canvas& canvas, int x, int y, std::string_view text, int scale,
              Color color = {25, 25, 25}) noexcept {
    for (const char character : text) {
        const auto rows = glyph(character);
        for (int row = 0; row < 7; ++row) {
            for (int column = 0; column < 5; ++column) {
                if ((rows[static_cast<std::size_t>(row)] & (1U << (4 - column))) == 0U) continue;
                canvas.fill(x + column * scale, y + row * scale,
                            x + (column + 1) * scale, y + (row + 1) * scale, color);
            }
        }
        x += 6 * scale;
    }
}

void drawCenteredText(Canvas& canvas, int centerX, int y, std::string_view text, int scale,
                      Color color = {25, 25, 25}) noexcept {
    drawText(canvas, centerX - textWidth(text, scale) / 2, y, text, scale, color);
}

[[nodiscard]] Color magma(double normalized) noexcept {
    struct Stop final { double position; Color color; };
    static constexpr std::array<Stop, 9> stops{{
        {0.000, {0,0,4}}, {0.125, {27,12,65}}, {0.250, {79,12,107}},
        {0.375, {120,28,109}}, {0.500, {165,44,96}}, {0.625, {207,68,70}},
        {0.750, {237,105,37}}, {0.875, {251,155,6}}, {1.000, {252,253,191}}}};
    const double value = std::clamp(normalized, 0.0, 1.0);
    std::size_t upper = 1;
    while (upper + 1 < stops.size() && value > stops[upper].position) ++upper;
    const Stop& first = stops[upper - 1];
    const Stop& second = stops[upper];
    const double fraction = (value - first.position) / (second.position - first.position);
    const auto blend = [fraction](unsigned char left, unsigned char right) {
        return static_cast<unsigned char>(std::lround(static_cast<double>(left) * (1.0 - fraction)
                                                     + static_cast<double>(right) * fraction));
    };
    return {blend(first.color.red, second.color.red),
            blend(first.color.green, second.color.green),
            blend(first.color.blue, second.color.blue)};
}

void writeU32(std::ostream& stream, std::uint32_t value) {
    const std::array<unsigned char, 4> bytes{
        static_cast<unsigned char>((value >> 24U) & 0xffU),
        static_cast<unsigned char>((value >> 16U) & 0xffU),
        static_cast<unsigned char>((value >> 8U) & 0xffU),
        static_cast<unsigned char>(value & 0xffU)};
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void writeChunk(std::ostream& stream, std::string_view type, std::span<const unsigned char> data) {
    writeU32(stream, static_cast<std::uint32_t>(data.size()));
    stream.write(type.data(), static_cast<std::streamsize>(type.size()));
    if (!data.empty()) {
        stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    uLong checksum = crc32(0L, Z_NULL, 0);
    checksum = crc32(checksum, reinterpret_cast<const Bytef*>(type.data()), static_cast<uInt>(type.size()));
    if (!data.empty()) checksum = crc32(checksum, data.data(), static_cast<uInt>(data.size()));
    writeU32(stream, static_cast<std::uint32_t>(checksum));
}

void writeScaledPng(const fs::path& path, const Canvas& canvas,
                    int outputWidth, int outputHeight) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Cannot create PNG: " + pathUtf8(path));
    static constexpr std::array<unsigned char, 8> signature{137,80,78,71,13,10,26,10};
    stream.write(reinterpret_cast<const char*>(signature.data()),
                 static_cast<std::streamsize>(signature.size()));
    std::array<unsigned char, 13> ihdr{};
    for (int byte = 0; byte < 4; ++byte) {
        ihdr[static_cast<std::size_t>(byte)] = static_cast<unsigned char>(
            (static_cast<std::uint32_t>(outputWidth) >> (24 - byte * 8)) & 0xffU);
        ihdr[static_cast<std::size_t>(byte + 4)] = static_cast<unsigned char>(
            (static_cast<std::uint32_t>(outputHeight) >> (24 - byte * 8)) & 0xffU);
    }
    ihdr[8] = 8;
    ihdr[9] = 2;
    writeChunk(stream, "IHDR", ihdr);

    z_stream compressor{};
    if (deflateInit(&compressor, Z_DEFAULT_COMPRESSION) != Z_OK) {
        throw std::runtime_error("Cannot initialize PNG compressor.");
    }
    try {
        std::vector<unsigned char> row(static_cast<std::size_t>(outputWidth) * 3U + 1U);
        std::vector<unsigned char> compressed(256U * 1024U);
        auto drain = [&](int flush) {
            int status = Z_OK;
            do {
                compressor.next_out = compressed.data();
                compressor.avail_out = static_cast<uInt>(compressed.size());
                status = deflate(&compressor, flush);
                if (status != Z_OK && status != Z_STREAM_END) {
                    throw std::runtime_error("PNG compression failed.");
                }
                const std::size_t produced = compressed.size() - compressor.avail_out;
                if (produced != 0U) writeChunk(stream, "IDAT", {compressed.data(), produced});
            } while (compressor.avail_out == 0U || (flush == Z_FINISH && status != Z_STREAM_END));
        };
        for (int y = 0; y < outputHeight; ++y) {
            row[0] = 0;
            const int sourceY = std::min(canvas.height() - 1,
                static_cast<int>(static_cast<std::int64_t>(y) * canvas.height() / outputHeight));
            for (int x = 0; x < outputWidth; ++x) {
                const int sourceX = std::min(canvas.width() - 1,
                    static_cast<int>(static_cast<std::int64_t>(x) * canvas.width() / outputWidth));
                const Color color = canvas.pixel(sourceX, sourceY);
                row[1U + static_cast<std::size_t>(x) * 3U] = color.red;
                row[2U + static_cast<std::size_t>(x) * 3U] = color.green;
                row[3U + static_cast<std::size_t>(x) * 3U] = color.blue;
            }
            compressor.next_in = row.data();
            compressor.avail_in = static_cast<uInt>(row.size());
            while (compressor.avail_in != 0U) drain(Z_NO_FLUSH);
        }
        drain(Z_FINISH);
        (void)deflateEnd(&compressor);
    } catch (...) {
        (void)deflateEnd(&compressor);
        throw;
    }
    writeChunk(stream, "IEND", {});
    if (!stream) throw std::runtime_error("Failed while writing PNG: " + pathUtf8(path));
}

struct Rectangle final {
    int left = 0;
    int top = 0;
    int width = 0;
    int height = 0;
};

void drawHeatmap(Canvas& canvas, Rectangle rectangle, const std::vector<float>& values,
                 int nx, int ny, double minimum, double maximum) noexcept {
    const double span = maximum > minimum ? maximum - minimum : 1.0;
    for (int py = 0; py < rectangle.height; ++py) {
        const int sourceY = std::clamp(
            static_cast<int>(static_cast<std::int64_t>(rectangle.height - 1 - py) * ny
                             / rectangle.height), 0, ny - 1);
        for (int px = 0; px < rectangle.width; ++px) {
            const int sourceX = std::clamp(
                static_cast<int>(static_cast<std::int64_t>(px) * nx / rectangle.width), 0, nx - 1);
            const float value = values[static_cast<std::size_t>(sourceY) * static_cast<std::size_t>(nx)
                                       + static_cast<std::size_t>(sourceX)];
            canvas.pixel(rectangle.left + px, rectangle.top + py,
                         magma((static_cast<double>(value) - minimum) / span));
        }
    }
    const Color black{20, 20, 20};
    canvas.line(rectangle.left, rectangle.top, rectangle.left + rectangle.width, rectangle.top, black);
    canvas.line(rectangle.left, rectangle.top + rectangle.height,
                rectangle.left + rectangle.width, rectangle.top + rectangle.height, black);
    canvas.line(rectangle.left, rectangle.top, rectangle.left, rectangle.top + rectangle.height, black);
    canvas.line(rectangle.left + rectangle.width, rectangle.top,
                rectangle.left + rectangle.width, rectangle.top + rectangle.height, black);
}

void drawColorbar(Canvas& canvas, Rectangle rectangle, double minimum, double maximum,
                  int textScale = 1) noexcept {
    for (int y = 0; y < rectangle.height; ++y) {
        const double normalized = 1.0 - static_cast<double>(y) / std::max(1, rectangle.height - 1);
        canvas.fill(rectangle.left, rectangle.top + y, rectangle.left + rectangle.width,
                    rectangle.top + y + 1, magma(normalized));
    }
    const Color black{20, 20, 20};
    canvas.line(rectangle.left, rectangle.top, rectangle.left + rectangle.width, rectangle.top, black);
    canvas.line(rectangle.left, rectangle.top + rectangle.height,
                rectangle.left + rectangle.width, rectangle.top + rectangle.height, black);
    canvas.line(rectangle.left, rectangle.top, rectangle.left, rectangle.top + rectangle.height, black);
    canvas.line(rectangle.left + rectangle.width, rectangle.top,
                rectangle.left + rectangle.width, rectangle.top + rectangle.height, black);
    std::ostringstream upper;
    upper << std::fixed << std::setprecision(2) << maximum;
    std::ostringstream lower;
    lower << std::fixed << std::setprecision(2) << minimum;
    drawText(canvas, rectangle.left + rectangle.width + 5, rectangle.top - 2,
             upper.str(), textScale, black);
    drawText(canvas, rectangle.left + rectangle.width + 5,
             rectangle.top + rectangle.height - 7 * textScale, lower.str(), textScale, black);
}

[[nodiscard]] std::string percentage(double value, int precision = 1) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value * 100.0 << '%';
    return stream.str();
}

[[nodiscard]] std::string fixedNumber(double value, int precision) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

void plotHorizontalLayer(const LayerSpectrum& layer, const StructuredPointsMeta& meta,
                         double minimum, double maximum) {
    Canvas canvas(620, 540);
    const Rectangle plot{72, 55, 430, 400};
    drawHeatmap(canvas, plot, layer.logEnergy, meta.dimensions[0], meta.dimensions[1], minimum, maximum);
    drawColorbar(canvas, {520, 68, 20, 365}, minimum, maximum);
    drawCenteredText(canvas, plot.left + plot.width / 2, 18,
                     "Z = " + fixedNumber(layer.target.actualHeight, 2)
                     + " M, VALID = " + percentage(layer.validFraction), 2);
    drawCenteredText(canvas, plot.left + plot.width / 2, 488, "KX [RAD M^-1]", 2);
    drawText(canvas, 8, 250, "KY", 2);
    drawText(canvas, 552, 250, "LOG10(E2D)", 1);
    writeScaledPng(layer.outputPng, canvas, 1364, 1188);
}

void plotHorizontalOverview(const std::vector<const LayerSpectrum*>& layers,
                            const StructuredPointsMeta& meta,
                            const fs::path& outputPath,
                            double minimum, double maximum,
                            int columns, int baseWidth, int baseHeight,
                            int outputWidth, int outputHeight,
                            bool validInside, bool integerHeight) {
    if (layers.empty()) throw std::runtime_error("Number of rows must be a positive integer, not 0");
    const int rows = static_cast<int>((layers.size() + static_cast<std::size_t>(columns) - 1U)
                                     / static_cast<std::size_t>(columns));
    Canvas canvas(baseWidth, baseHeight);
    const int leftMargin = std::max(30, baseWidth / 45);
    const int rightMargin = std::max(65, baseWidth / 18);
    const int topMargin = std::max(35, baseHeight / 22);
    const int bottomMargin = std::max(42, baseHeight / 18);
    const int gapX = std::max(14, baseWidth / 100);
    const int gapY = std::max(28, baseHeight / 30);
    const int cellWidth = std::max(1, (baseWidth - leftMargin - rightMargin
                                     - gapX * (columns - 1)) / columns);
    const int cellHeight = std::max(1, (baseHeight - topMargin - bottomMargin
                                      - gapY * (rows - 1)) / rows);
    const int titleScale = validInside ? 3 : 2;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const int row = static_cast<int>(index / static_cast<std::size_t>(columns));
        const int column = static_cast<int>(index % static_cast<std::size_t>(columns));
        const int cellLeft = leftMargin + column * (cellWidth + gapX);
        const int cellTop = topMargin + row * (cellHeight + gapY);
        const int titleHeight = validInside ? 32 : 43;
        const Rectangle plot{cellLeft + 8, cellTop + titleHeight,
                             std::max(1, cellWidth - 16), std::max(1, cellHeight - titleHeight - 8)};
        drawHeatmap(canvas, plot, layers[index]->logEnergy,
                    meta.dimensions[0], meta.dimensions[1], minimum, maximum);
        std::string heightText;
        if (integerHeight) {
            heightText = std::to_string(static_cast<long long>(std::nearbyint(
                layers[index]->target.actualHeight))) + " M";
        } else {
            heightText = fixedNumber(layers[index]->target.actualHeight, 1) + " M";
        }
        drawCenteredText(canvas, cellLeft + cellWidth / 2, cellTop + 2,
                         heightText, titleScale);
        if (validInside) {
            const std::string valid = "VALID=" + percentage(layers[index]->validFraction);
            const int scale = 2;
            const int x = plot.left + 6;
            const int y = plot.top + 7;
            canvas.fill(x - 3, y - 3, x + textWidth(valid, scale) + 3, y + 7 * scale + 3,
                        {30, 30, 30});
            drawText(canvas, x, y, valid, scale, {255, 255, 255});
        } else {
            drawCenteredText(canvas, cellLeft + cellWidth / 2, cellTop + 21,
                             "VALID=" + percentage(layers[index]->validFraction), 1);
        }
    }
    drawColorbar(canvas, {baseWidth - rightMargin + 15, topMargin + 25,
                          std::max(14, rightMargin / 5),
                          std::max(40, baseHeight - topMargin - bottomMargin - 50)},
                 minimum, maximum, validInside ? 2 : 1);
    drawCenteredText(canvas, baseWidth / 2, baseHeight - 24, "KX [RAD M^-1]", validInside ? 3 : 2);
    drawText(canvas, 5, baseHeight / 2, "KY [RAD M^-1]", validInside ? 2 : 1);
    writeScaledPng(outputPath, canvas, outputWidth, outputHeight);
}

[[nodiscard]] std::pair<double, double> positiveLogRange(const std::vector<double>& values) {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (const double value : values) {
        if (value > 0.0 && std::isfinite(value)) {
            const double log = std::log10(value);
            minimum = std::min(minimum, log);
            maximum = std::max(maximum, log);
        }
    }
    if (!std::isfinite(minimum) || !std::isfinite(maximum)) return {-12.0, 0.0};
    if (!(maximum > minimum)) maximum = minimum + 1.0;
    const double padding = (maximum - minimum) * 0.08;
    return {minimum - padding, maximum + padding};
}

void drawLogSeries(Canvas& canvas, Rectangle plot,
                   const std::vector<double>& xValues,
                   const std::vector<double>& yValues,
                   std::pair<double, double> xRange,
                   std::pair<double, double> yRange,
                   Color color, int thickness, bool markers) noexcept {
    bool previous = false;
    int previousX = 0;
    int previousY = 0;
    for (std::size_t index = 0; index < std::min(xValues.size(), yValues.size()); ++index) {
        if (!(xValues[index] > 0.0) || !(yValues[index] > 0.0)
            || !std::isfinite(xValues[index]) || !std::isfinite(yValues[index])) {
            previous = false;
            continue;
        }
        const double x = std::log10(xValues[index]);
        const double y = std::log10(yValues[index]);
        const int px = plot.left + static_cast<int>(std::lround(
            (x - xRange.first) / (xRange.second - xRange.first) * plot.width));
        const int py = plot.top + plot.height - static_cast<int>(std::lround(
            (y - yRange.first) / (yRange.second - yRange.first) * plot.height));
        if (previous) canvas.line(previousX, previousY, px, py, color, thickness);
        if (markers) canvas.circle(px, py, 3, color);
        previous = true;
        previousX = px;
        previousY = py;
    }
}

void drawLogGrid(Canvas& canvas, Rectangle plot,
                 std::pair<double, double> xRange,
                 std::pair<double, double> yRange) noexcept {
    const Color grid{215, 215, 215};
    const Color black{25, 25, 25};
    for (int exponent = static_cast<int>(std::floor(xRange.first));
         exponent <= static_cast<int>(std::ceil(xRange.second)); ++exponent) {
        const int x = plot.left + static_cast<int>(std::lround(
            (static_cast<double>(exponent) - xRange.first)
            / (xRange.second - xRange.first) * plot.width));
        canvas.line(x, plot.top, x, plot.top + plot.height, grid);
        drawText(canvas, x - 12, plot.top + plot.height + 10,
                 "1E" + std::to_string(exponent), 1, black);
    }
    for (int exponent = static_cast<int>(std::floor(yRange.first));
         exponent <= static_cast<int>(std::ceil(yRange.second)); ++exponent) {
        const int y = plot.top + plot.height - static_cast<int>(std::lround(
            (static_cast<double>(exponent) - yRange.first)
            / (yRange.second - yRange.first) * plot.height));
        canvas.line(plot.left, y, plot.left + plot.width, y, grid);
        drawText(canvas, plot.left - 42, y - 3, "1E" + std::to_string(exponent), 1, black);
    }
    canvas.line(plot.left, plot.top, plot.left + plot.width, plot.top, black);
    canvas.line(plot.left, plot.top + plot.height, plot.left + plot.width, plot.top + plot.height, black);
    canvas.line(plot.left, plot.top, plot.left, plot.top + plot.height, black);
    canvas.line(plot.left + plot.width, plot.top, plot.left + plot.width, plot.top + plot.height, black);
}

void drawVerticalWaveNumber(Canvas& canvas, Rectangle plot, double value,
                            std::pair<double, double> xRange, Color color,
                            bool dotted) noexcept {
    if (!(value > 0.0)) return;
    const double log = std::log10(value);
    const int x = plot.left + static_cast<int>(std::lround(
        (log - xRange.first) / (xRange.second - xRange.first) * plot.width));
    if (dotted) {
        for (int y = plot.top; y < plot.top + plot.height; y += 8) {
            canvas.line(x, y, x, std::min(y + 3, plot.top + plot.height), color, 2);
        }
    } else {
        for (int y = plot.top; y < plot.top + plot.height; y += 12) {
            canvas.line(x, y, x, std::min(y + 7, plot.top + plot.height), color, 2);
        }
    }
}

[[nodiscard]] std::pair<double, double> xLogRange(const IsotropicSpectrum& result) {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = 0.0;
    for (std::size_t index = 0; index < result.kCenters.size(); ++index) {
        if (result.kCenters[index] > 0.0 && result.energy[index] > 0.0) {
            minimum = std::min(minimum, result.kCenters[index]);
            maximum = std::max(maximum, result.kCenters[index]);
        }
    }
    minimum = std::min(minimum, result.kNyquist * 0.5);
    maximum = std::max(maximum, result.kNyquist);
    if (!(minimum > 0.0) || !(maximum > minimum)) return {-2.0, 0.0};
    const double low = std::log10(minimum);
    const double high = std::log10(maximum);
    const double pad = (high - low) * 0.06;
    return {low - pad, high + pad};
}

void plotIsotropic(const IsotropicSpectrum& result, const fs::path& outputPath, bool plain) {
    const int width = plain ? 800 : 860;
    const int height = plain ? 540 : 580;
    Canvas canvas(width, height);
    const Rectangle plot{88, plain ? 38 : 86, plain ? 655 : 665, plain ? 425 : 395};
    const auto xRange = xLogRange(result);
    const auto yRange = positiveLogRange(result.energy);
    drawLogGrid(canvas, plot, xRange, yRange);
    drawVerticalWaveNumber(canvas, plot, result.kNyquist, xRange, {220, 20, 60}, false);
    drawVerticalWaveNumber(canvas, plot, result.kNyquist * 0.5, xRange, {90, 90, 90}, true);
    drawLogSeries(canvas, plot, result.kCenters, result.energy, xRange, yRange,
                  {31, 119, 180}, plain ? 1 : 2, plain);
    if (!plain) {
        const auto compensatedRange = positiveLogRange(result.compensated);
        drawLogSeries(canvas, plot, result.kCenters, result.compensated,
                      xRange, compensatedRange, {255, 127, 14}, 2, false);
        drawText(canvas, plot.left + 18, plot.top + 16, "E(K)", 2, {31,119,180});
        drawText(canvas, plot.left + 18, plot.top + 40, "K^5/3 E(K)", 2, {255,127,14});
        drawCenteredText(canvas, width / 2, 18, "3D ISOTROPIC KINETIC ENERGY SPECTRUM", 2);
        drawCenteredText(canvas, width / 2, 43,
                         "Z = " + fixedNumber(result.zStartHeight, 2) + " TO "
                         + fixedNumber(result.zStopHeight, 2) + " M", 1);
        drawText(canvas, width - 76, height / 2, "K^5/3 E(K)", 1);
    } else {
        // Match the plain plot's explicit -5/3 reference segment when the
        // requested [0.12, 0.22] rad/m interval intersects valid samples.
        std::vector<std::pair<double, double>> positive;
        for (std::size_t index = 0; index < result.kCenters.size(); ++index) {
            if (result.kCenters[index] > 0.0 && result.energy[index] > 0.0) {
                positive.emplace_back(result.kCenters[index], result.energy[index]);
            }
        }
        if (positive.size() >= 2) {
            const double k1 = std::max(0.12, positive.front().first);
            const double k2 = std::min(0.22, positive.back().first);
            if (k2 > k1) {
                const auto interpolateLog = [&](double k) {
                    auto upper = std::lower_bound(positive.begin(), positive.end(), k,
                        [](const auto& pair, double target) { return pair.first < target; });
                    if (upper == positive.begin()) return std::log10(upper->second);
                    if (upper == positive.end()) return std::log10(positive.back().second);
                    const auto lower = std::prev(upper);
                    const double fraction = (std::log10(k) - std::log10(lower->first))
                                          / (std::log10(upper->first) - std::log10(lower->first));
                    return std::log10(lower->second) * (1.0 - fraction)
                         + std::log10(upper->second) * fraction;
                };
                const double y1 = std::pow(10.0, interpolateLog(k1)) * 0.75;
                const double y2 = y1 * std::pow(k2 / k1, -5.0 / 3.0);
                drawLogSeries(canvas, plot, {k1, k2}, {y1, y2}, xRange, yRange,
                              {0, 0, 0}, 2, false);
                drawText(canvas, plot.left + plot.width / 2, plot.top + plot.height / 2,
                         "-5/3", 2);
            }
        }
    }
    drawCenteredText(canvas, plot.left + plot.width / 2, height - 35,
                     "WAVENUMBER K [RAD M^-1]", 2);
    drawText(canvas, 8, height / 2, "ENERGY E(K)", 1);
    drawText(canvas, plot.left + plot.width - 150, plot.top + 16, "NYQUIST", 1, {220,20,60});
    writeScaledPng(outputPath, canvas, plain ? 1760 : 1892, plain ? 1188 : 1276);
}

[[nodiscard]] std::string pythonFloat(double value) {
    if (std::isnan(value)) return "nan";
    if (std::isinf(value)) return value < 0.0 ? "-inf" : "inf";
    std::array<char, 64> buffer{};
    const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                         std::chars_format::general);
    if (converted.ec != std::errc{}) throw std::runtime_error("Cannot format floating-point value.");
    std::string result(buffer.data(), converted.ptr);
    if (result.find_first_of(".eE") == std::string::npos) result += ".0";
    return result;
}

[[nodiscard]] std::string jsonFloat(double value) {
    if (std::isnan(value)) return "NaN";
    if (std::isinf(value)) return value < 0.0 ? "-Infinity" : "Infinity";
    return pythonFloat(value);
}

[[nodiscard]] std::string jsonString(std::string_view text) {
    std::string result;
    result.reserve(text.size() + 2U);
    result.push_back('"');
    static constexpr char hexadecimal[] = "0123456789abcdef";
    for (const unsigned char character : text) {
        switch (character) {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\b': result += "\\b"; break;
        case '\f': result += "\\f"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (character < 0x20U) {
                result += "\\u00";
                result.push_back(hexadecimal[character >> 4U]);
                result.push_back(hexadecimal[character & 0x0fU]);
            } else {
                result.push_back(static_cast<char>(character));
            }
            break;
        }
    }
    result.push_back('"');
    return result;
}

[[nodiscard]] std::string csvField(std::string_view text) {
    if (text.find_first_of(",\"\r\n") == std::string_view::npos) return std::string(text);
    std::string result = "\"";
    for (const char character : text) {
        if (character == '"') result += "\"\"";
        else result.push_back(character);
    }
    result.push_back('"');
    return result;
}

void saveLayerCsv(const std::vector<LayerSpectrum>& layers, const fs::path& path) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Cannot create CSV: " + pathUtf8(path));
    stream << "target_height_m,actual_height_m,z_index,valid_fraction,valid_points,output_png\r\n";
    for (const LayerSpectrum& layer : layers) {
        stream << pythonFloat(layer.target.targetHeight) << ','
               << pythonFloat(layer.target.actualHeight) << ','
               << layer.target.zIndex << ',' << pythonFloat(layer.validFraction) << ','
               << layer.validPoints << ',' << csvField(pathUtf8(layer.outputPng)) << "\r\n";
    }
    if (!stream) throw std::runtime_error("Failed while writing CSV: " + pathUtf8(path));
}

void saveIsotropicCsv(const IsotropicSpectrum& result, const fs::path& path) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Cannot create CSV: " + pathUtf8(path));
    stream << "k_rad_per_m,E_k,k_5_3_E_k,shell_energy,shell_mode_weight\r\n";
    for (std::size_t index = 0; index < result.kCenters.size(); ++index) {
        stream << pythonFloat(result.kCenters[index]) << ','
               << pythonFloat(result.energy[index]) << ','
               << pythonFloat(result.compensated[index]) << ','
               << pythonFloat(result.shellEnergy[index]) << ','
               << pythonFloat(result.shellModeWeight[index]) << "\r\n";
    }
    if (!stream) throw std::runtime_error("Failed while writing CSV: " + pathUtf8(path));
}

template <typename Range, typename Formatter>
void writeJsonArray(std::ostream& stream, const Range& values, Formatter formatter) {
    stream << '[';
    bool first = true;
    for (const auto& value : values) {
        if (!first) stream << ", ";
        first = false;
        stream << formatter(value);
    }
    stream << ']';
}

void saveSummary(const fs::path& path,
                 const StructuredPointsMeta& meta,
                 const std::vector<double>& coverages,
                 const std::vector<LayerSpectrum>& layers,
                 std::pair<double, double> kxLimits,
                 std::pair<double, double> kyLimits,
                 std::pair<double, double> colorLimits,
                 const IsotropicSpectrum& isotropic,
                 const fs::path& isotropicPng,
                 const fs::path& isotropicPlainPng,
                 const fs::path& isotropicCsv,
                 const fs::path& overviewPng,
                 const std::optional<fs::path>& selectedOverview,
                 bool quickTest,
                 double elapsedSeconds) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Cannot create JSON summary: " + pathUtf8(path));
    stream << "{\n"
           << "  \"source_vtk\": " << jsonString(pathUtf8(meta.path)) << ",\n"
           << "  \"quick_test\": " << (quickTest ? "true" : "false") << ",\n"
           << "  \"grid\": {\n"
           << "    \"path\": " << jsonString(pathUtf8(meta.path)) << ",\n"
           << "    \"dataset_type\": " << jsonString(meta.datasetType) << ",\n"
           << "    \"dimensions_xyz\": [" << meta.dimensions[0] << ", "
           << meta.dimensions[1] << ", " << meta.dimensions[2] << "],\n"
           << "    \"spacing_xyz\": [" << jsonFloat(meta.spacing[0]) << ", "
           << jsonFloat(meta.spacing[1]) << ", " << jsonFloat(meta.spacing[2]) << "],\n"
           << "    \"origin_xyz\": [" << jsonFloat(meta.origin[0]) << ", "
           << jsonFloat(meta.origin[1]) << ", " << jsonFloat(meta.origin[2]) << "],\n"
           << "    \"point_data\": " << meta.pointData << ",\n"
           << "    \"data_name\": " << jsonString(meta.dataName) << ",\n"
           << "    \"dtype_name\": " << jsonString(meta.dtypeName) << ",\n"
           << "    \"n_components\": " << meta.components << ",\n"
           << "    \"data_offset\": " << meta.dataOffset << "\n"
           << "  },\n"
           << "  \"coverages_by_layer\": ";
    writeJsonArray(stream, coverages, [](double value) { return jsonFloat(value); });
    stream << ",\n  \"target_layers\": [";
    if (!layers.empty()) stream << '\n';
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const LayerSpectrum& layer = layers[index];
        stream << "    {\n"
               << "      \"target_height_m\": " << jsonFloat(layer.target.targetHeight) << ",\n"
               << "      \"actual_height_m\": " << jsonFloat(layer.target.actualHeight) << ",\n"
               << "      \"z_index\": " << layer.target.zIndex << ",\n"
               << "      \"valid_fraction\": " << jsonFloat(layer.validFraction) << ",\n"
               << "      \"valid_points\": " << layer.validPoints << ",\n"
               << "      \"output_png\": " << jsonString(pathUtf8(layer.outputPng)) << "\n"
               << "    }" << (index + 1U == layers.size() ? "\n" : ",\n");
    }
    stream << "  ],\n"
           << "  \"kx_limits_rad_per_m\": [" << jsonFloat(kxLimits.first) << ", "
           << jsonFloat(kxLimits.second) << "],\n"
           << "  \"ky_limits_rad_per_m\": [" << jsonFloat(kyLimits.first) << ", "
           << jsonFloat(kyLimits.second) << "],\n"
           << "  \"color_limits_log10\": [" << jsonFloat(colorLimits.first) << ", "
           << jsonFloat(colorLimits.second) << "],\n"
           << "  \"isotropic_spectrum\": {\n"
           << "    \"z_start_index\": " << isotropic.zStart << ",\n"
           << "    \"z_stop_index\": " << isotropic.zStop << ",\n"
           << "    \"z_start_height_m\": " << jsonFloat(isotropic.zStartHeight) << ",\n"
           << "    \"z_stop_height_m\": " << jsonFloat(isotropic.zStopHeight) << ",\n"
           << "    \"dk_rad_per_m\": " << jsonFloat(isotropic.dk) << ",\n"
           << "    \"k_nyquist_rad_per_m\": " << jsonFloat(isotropic.kNyquist) << ",\n"
           << "    \"plot_png\": " << jsonString(pathUtf8(isotropicPng)) << ",\n"
           << "    \"plain_plot_png\": " << jsonString(pathUtf8(isotropicPlainPng)) << ",\n"
           << "    \"csv\": " << jsonString(pathUtf8(isotropicCsv)) << "\n"
           << "  },\n"
           << "  \"overview_png\": " << jsonString(pathUtf8(overviewPng)) << ",\n"
           << "  \"selected_overview_png\": ";
    if (selectedOverview) stream << jsonString(pathUtf8(*selectedOverview));
    else stream << "null";
    stream << ",\n  \"elapsed_seconds\": " << jsonFloat(elapsedSeconds) << "\n}\n";
    if (!stream) throw std::runtime_error("Failed while writing JSON summary: " + pathUtf8(path));
}

[[nodiscard]] std::string fourDigitHeight(long long value) {
    const bool negative = value < 0;
    const unsigned long long magnitude = negative
        ? static_cast<unsigned long long>(-(value + 1LL)) + 1ULL
        : static_cast<unsigned long long>(value);
    std::string digits = std::to_string(magnitude);
    const std::size_t width = negative ? 3U : 4U;
    if (digits.size() < width) digits.insert(digits.begin(), width - digits.size(), '0');
    return negative ? '-' + digits : digits;
}

[[nodiscard]] std::pair<double, double> limits(const std::vector<double>& values) {
    const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
    return {*minimum, *maximum};
}

int execute(const Arguments& arguments, std::ostream& output) {
    const auto started = std::chrono::steady_clock::now();
    const fs::path vtkPath = absoluteNormalized(arguments.vtkPath);
    if (!fs::exists(vtkPath)) throw std::runtime_error("VTK file not found: " + pathUtf8(vtkPath));
    fs::path outputDirectory;
    if (arguments.outputDirectory) outputDirectory = absoluteNormalized(*arguments.outputDirectory);
    else outputDirectory = vtkPath.parent_path() / (pathUtf8(vtkPath.stem()) + "_les_spectra");
    fs::create_directories(outputDirectory);
    const fs::path layerDirectory = outputDirectory / "layers_kxky";
    fs::create_directories(layerDirectory);

    const StructuredPointsMeta meta = parseLegacyStructuredPoints(vtkPath);
    const ReadOnlyMapping mapping(vtkPath);
    const VtkFloatView data(meta, mapping);
    std::vector<TargetLayer> targets = buildTargets(meta, arguments.heightStart,
                                                    arguments.heightInterval);
    if (arguments.quickTest) {
        std::size_t keep = 0;
        if (arguments.testHeightCount >= 0) {
            keep = std::min(targets.size(), static_cast<std::size_t>(arguments.testHeightCount));
        } else {
            const std::size_t remove = static_cast<std::size_t>(
                -static_cast<long long>(arguments.testHeightCount));
            keep = remove >= targets.size() ? 0U : targets.size() - remove;
        }
        targets.resize(keep);
    }

    output << "Parsed grid: nx=" << meta.dimensions[0] << ", ny=" << meta.dimensions[1]
           << ", nz=" << meta.dimensions[2] << ", dx=" << pythonFloat(meta.spacing[0])
           << ", dy=" << pythonFloat(meta.spacing[1]) << ", dz="
           << pythonFloat(meta.spacing[2]) << '\n'
           << "Selected " << targets.size() << " target height layers\n";
    const std::vector<double> coverages = computeCoverages(data, meta);
    const int workers = resolveWorkers(arguments.workers);
    std::vector<LayerSpectrum> layers;
    layers.reserve(targets.size());
    for (const TargetLayer& target : targets) {
        output << "Computing horizontal spectrum at z-index " << target.zIndex << " ("
               << std::fixed << std::setprecision(2) << target.actualHeight << " m)\n"
               << std::defaultfloat;
        layers.push_back(computeHorizontal(data, meta, target, workers));
    }
    const auto colorLimits = logColorLimits(layers);
    for (LayerSpectrum& layer : layers) {
        const long long rounded = static_cast<long long>(std::nearbyint(layer.target.actualHeight));
        layer.outputPng = absoluteNormalized(layerDirectory
            / ("kxky_z" + fourDigitHeight(rounded) + "m.png"));
        plotHorizontalLayer(layer, meta, colorLimits.first, colorLimits.second);
    }

    const fs::path overviewPng = absoluteNormalized(outputDirectory / "kxky_overview.png");
    std::vector<const LayerSpectrum*> allLayers;
    allLayers.reserve(layers.size());
    for (const LayerSpectrum& layer : layers) allLayers.push_back(&layer);
    const int overviewRows = static_cast<int>((layers.size() + 3U) / 4U);
    plotHorizontalOverview(allLayers, meta, overviewPng,
                           colorLimits.first, colorLimits.second,
                           4, 1880, std::max(400, overviewRows * 400),
                           4136, overviewRows * 880, false, false);

    std::vector<const LayerSpectrum*> selected;
    for (const LayerSpectrum& layer : layers) {
        const long long target = static_cast<long long>(std::nearbyint(layer.target.targetHeight));
        if (target == 100 || target == 200 || target == 300
            || target == 400 || target == 500 || target == 600) {
            selected.push_back(&layer);
        }
    }
    const fs::path selectedPath = absoluteNormalized(outputDirectory
        / "kxky_overview_100_200_300_400_500_600_largefont.png");
    std::optional<fs::path> selectedOverview;
    if (!selected.empty()) {
        plotHorizontalOverview(selected, meta, selectedPath,
                               colorLimits.first, colorLimits.second,
                               3, 1950, 1200, 4290, 2640, true, true);
        selectedOverview = selectedPath;
    }
    saveLayerCsv(layers, outputDirectory / "kxky_layer_metadata.csv");

    output << "Computing 3D isotropic spectrum\n";
    const std::optional<int> quickLayers = arguments.quickTest
        ? std::optional<int>(arguments.test3dLayers) : std::nullopt;
    const IsotropicSpectrum isotropic = computeIsotropic(data, meta, coverages,
                                                         workers, quickLayers, output);
    const fs::path isotropicPng = absoluteNormalized(outputDirectory / "isotropic_spectrum.png");
    const fs::path plainPng = absoluteNormalized(outputDirectory / "isotropic_spectrum_plain.png");
    const fs::path isotropicCsv = absoluteNormalized(outputDirectory / "isotropic_spectrum.csv");
    plotIsotropic(isotropic, isotropicPng, false);
    plotIsotropic(isotropic, plainPng, true);
    saveIsotropicCsv(isotropic, isotropicCsv);

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const std::vector<double> kx = fftFrequencies(meta.dimensions[0], meta.spacing[0], true);
    const std::vector<double> ky = fftFrequencies(meta.dimensions[1], meta.spacing[1], true);
    saveSummary(outputDirectory / "run_summary.json", meta, coverages, layers,
                limits(kx), limits(ky), colorLimits, isotropic,
                isotropicPng, plainPng, isotropicCsv, overviewPng,
                selectedOverview, arguments.quickTest, elapsed);
    output << "Outputs written to: " << pathUtf8(outputDirectory) << '\n'
           << "Elapsed: " << std::fixed << std::setprecision(1) << elapsed << " s\n"
           << std::defaultfloat;
    return 0;
}

} // namespace

int run_les_spectra(const std::vector<std::string>& arguments,
                    std::ostream& output,
                    std::ostream& error) {
    try {
        const Arguments parsed = parseArguments(arguments);
        if (parsed.help) {
            printUsage(output);
            return 0;
        }
        return execute(parsed, output);
    } catch (const ArgumentError& exception) {
        printShortUsage(error);
        error << "les_spectra.py: error: " << exception.what() << '\n';
        return 2;
    } catch (const std::exception& exception) {
        error << "ERROR: " << exception.what() << '\n';
        return 1;
    }
}

} // namespace luw::cli
