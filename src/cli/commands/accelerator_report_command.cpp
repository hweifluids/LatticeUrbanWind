#include "luw/accelerator_report_command.hpp"

#include "luw/detect_opencl_command.hpp"

#include <QCoreApplication>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLibrary>
#include <QSysInfo>
#include <QStringList>

#include <array>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef LUW_HAS_OPENCL
#define LUW_HAS_OPENCL 0
#endif

#if LUW_HAS_OPENCL
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

namespace luw::cli {

namespace {

struct Options {
    QString requirementsPath;
    bool prepareCudaRuntime = false;
    bool json = false;
    bool help = false;
};

QString utf8(const std::string& value) {
    return QString::fromUtf8(value.data(), static_cast<qsizetype>(value.size()));
}

constexpr std::string_view kLegacyUsage =
    "usage: accelerator_runtime.py [-h] [--prepare-cuda-runtime]\n"
    "                              [--startup-report STARTUP_REPORT] [--json]\n";

constexpr std::string_view kLegacyHelp =
    "usage: accelerator_runtime.py [-h] [--prepare-cuda-runtime]\n"
    "                              [--startup-report STARTUP_REPORT] [--json]\n"
    "\n"
    "Prepare accelerator runtimes and print startup diagnostics.\n"
    "\n"
    "options:\n"
    "  -h, --help            show this help message and exit\n"
    "  --prepare-cuda-runtime\n"
    "                        Prepare the local CUDA runtime layout and print its\n"
    "                        summary.\n"
    "  --startup-report STARTUP_REPORT\n"
    "                        Requirements file used to build a startup diagnostics\n"
    "                        JSON report.\n"
    "  --json                Print the result as JSON.\n";

void print_usage(std::ostream& output) {
    output << kLegacyHelp;
}

void print_parse_error(std::ostream& error, std::string_view message) {
    error << kLegacyUsage
          << "accelerator_runtime.py: error: " << message << '\n';
}

bool parse_options(const CommandArguments& arguments,
                   Options& options,
                   std::ostream& error) {
    std::vector<std::string> unrecognized;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];
        if (argument == "--help" || argument == "-h") {
            options.help = true;
            // argparse's help action exits immediately, so even unknown
            // arguments seen before or after it are not reported.
            return true;
        } else if (argument == "--json") {
            options.json = true;
        } else if (argument == "--prepare-cuda-runtime") {
            options.prepareCudaRuntime = true;
        } else if (argument == "--startup-report") {
            if (index + 1 >= arguments.size() ||
                (arguments[index + 1].size() > 1 &&
                 arguments[index + 1].front() == '-')) {
                print_parse_error(
                    error,
                    "argument --startup-report: expected one argument");
                return false;
            }
            options.requirementsPath = utf8(arguments[++index]);
        } else if (argument.starts_with("--startup-report=")) {
            options.requirementsPath = utf8(
                argument.substr(std::string("--startup-report=").size()));
        } else {
            unrecognized.push_back(argument);
        }
    }

    if (!unrecognized.empty()) {
        std::string message = "unrecognized arguments:";
        for (const std::string& argument : unrecognized) {
            message += ' ';
            message += argument;
        }
        print_parse_error(error, message);
        return false;
    }
    return true;
}

QString cuda_driver_version(int encodedVersion) {
    if (encodedVersion <= 0) return {};
    const int major = encodedVersion / 1000;
    const int minor = (encodedVersion % 1000) / 10;
    return QStringLiteral("%1.%2").arg(major).arg(minor);
}

QJsonObject cuda_device(const QString& name,
                        const QString& driverVersion,
                        const QString& computeCapability) {
    QJsonObject device;
    device.insert(QStringLiteral("name"), name);
    device.insert(QStringLiteral("vendor"), QStringLiteral("NVIDIA"));
    device.insert(QStringLiteral("version"), QStringLiteral("CUDA"));
    device.insert(QStringLiteral("driver_version"), driverVersion);
    device.insert(QStringLiteral("compute_capability"), computeCapability);
    device.insert(QStringLiteral("compute_units"), QJsonValue(QJsonValue::Null));
    return device;
}

QJsonObject probe_cuda_environment() {
    QJsonObject result;
    result.insert(QStringLiteral("configured"), false);
    result.insert(QStringLiteral("prepared"), false);
    result.insert(QStringLiteral("available"), false);
    // The old probe never promoted missing CUDA to its warning flag; retain
    // that public JSON behavior while keeping a summary/error string.
    result.insert(QStringLiteral("warning"), false);
    result.insert(QStringLiteral("summary"), QString());
    result.insert(QStringLiteral("error"), QString());
    result.insert(QStringLiteral("provider"), QString());
    result.insert(QStringLiteral("source_root"), QString());
    result.insert(QStringLiteral("runtime_root"), QString());
    result.insert(QStringLiteral("cuda_home"), QString());
    result.insert(QStringLiteral("numba_version"), QString());
    result.insert(QStringLiteral("runtime_version"), QString());
    result.insert(QStringLiteral("driver_version"), QString());
    result.insert(QStringLiteral("devices"), QJsonArray());

#if defined(Q_OS_WIN)
    QLibrary library(QStringLiteral("nvcuda"));
#elif defined(Q_OS_LINUX)
    QLibrary library(QStringLiteral("cuda"));
#else
    result.insert(QStringLiteral("summary"),
                  QStringLiteral("CUDA probe unavailable on this platform."));
    return result;
#endif

    if (!library.load()) {
        result.insert(QStringLiteral("summary"),
                      QStringLiteral("CUDA driver is not available: %1")
                          .arg(library.errorString()));
        result.insert(QStringLiteral("error"), library.errorString());
        return result;
    }
    result.insert(QStringLiteral("configured"), true);
    result.insert(QStringLiteral("provider"), QStringLiteral("native-cuda-driver-api"));

    using CuResult = int;
    using CuDevice = int;
    using CuInit = CuResult (*)(unsigned int);
    using CuDriverGetVersion = CuResult (*)(int*);
    using CuDeviceGetCount = CuResult (*)(int*);
    using CuDeviceGet = CuResult (*)(CuDevice*, int);
    using CuDeviceGetName = CuResult (*)(char*, int, CuDevice);
    using CuDeviceComputeCapability = CuResult (*)(int*, int*, CuDevice);

    const auto cuInit = reinterpret_cast<CuInit>(library.resolve("cuInit"));
    const auto cuDriverGetVersion =
        reinterpret_cast<CuDriverGetVersion>(library.resolve("cuDriverGetVersion"));
    const auto cuDeviceGetCount =
        reinterpret_cast<CuDeviceGetCount>(library.resolve("cuDeviceGetCount"));
    const auto cuDeviceGet =
        reinterpret_cast<CuDeviceGet>(library.resolve("cuDeviceGet"));
    const auto cuDeviceGetName =
        reinterpret_cast<CuDeviceGetName>(library.resolve("cuDeviceGetName"));
    const auto cuDeviceComputeCapability = reinterpret_cast<CuDeviceComputeCapability>(
        library.resolve("cuDeviceComputeCapability"));

    if (!cuInit || !cuDriverGetVersion || !cuDeviceGetCount || !cuDeviceGet
        || !cuDeviceGetName || !cuDeviceComputeCapability) {
        const QString message =
            QStringLiteral("CUDA probe unavailable: the driver API is incomplete.");
        result.insert(QStringLiteral("summary"), message);
        result.insert(QStringLiteral("error"), message);
        return result;
    }

    const CuResult initStatus = cuInit(0);
    if (initStatus != 0) {
        const QString message =
            QStringLiteral("CUDA driver is not available (error %1).").arg(initStatus);
        result.insert(QStringLiteral("summary"), message);
        result.insert(QStringLiteral("error"), message);
        return result;
    }

    int encodedDriverVersion = 0;
    if (cuDriverGetVersion(&encodedDriverVersion) != 0) encodedDriverVersion = 0;
    const QString driverVersion = cuda_driver_version(encodedDriverVersion);
    result.insert(QStringLiteral("driver_version"), driverVersion);

    int deviceCount = 0;
    const CuResult countStatus = cuDeviceGetCount(&deviceCount);
    if (countStatus != 0) {
        const QString message =
            QStringLiteral("CUDA device discovery failed (error %1).").arg(countStatus);
        result.insert(QStringLiteral("summary"), message);
        result.insert(QStringLiteral("error"), message);
        return result;
    }
    if (deviceCount <= 0) {
        result.insert(QStringLiteral("summary"),
                      QStringLiteral("CUDA ready, but no supported NVIDIA GPU was detected."));
        return result;
    }

    QJsonArray devices;
    for (int index = 0; index < deviceCount; ++index) {
        CuDevice device = 0;
        if (cuDeviceGet(&device, index) != 0) continue;

        std::array<char, 256> name{};
        if (cuDeviceGetName(name.data(), static_cast<int>(name.size()), device) != 0) continue;

        int computeMajor = 0;
        int computeMinor = 0;
        QString computeCapability;
        if (cuDeviceComputeCapability(&computeMajor, &computeMinor, device) == 0) {
            computeCapability =
                QStringLiteral("%1.%2").arg(computeMajor).arg(computeMinor);
        }
        devices.append(cuda_device(QString::fromUtf8(name.data()).trimmed(),
                                   driverVersion, computeCapability));
    }
    result.insert(QStringLiteral("devices"), devices);
    result.insert(QStringLiteral("available"), !devices.isEmpty());

    if (devices.isEmpty()) {
        result.insert(QStringLiteral("summary"),
                      QStringLiteral("CUDA devices were reported, but none could be enumerated."));
        return result;
    }

    const QJsonObject primary = devices.first().toObject();
    QStringList details;
    if (!driverVersion.isEmpty()) {
        details << QStringLiteral("driver %1").arg(driverVersion);
    }
    const QString computeCapability =
        primary.value(QStringLiteral("compute_capability")).toString();
    if (!computeCapability.isEmpty()) {
        details << QStringLiteral("cc %1").arg(computeCapability);
    }
    const QString suffix = details.isEmpty()
        ? QString()
        : QStringLiteral(" (%1)").arg(details.join(QStringLiteral(", ")));
    result.insert(
        QStringLiteral("summary"),
        QStringLiteral("CUDA available on %1 device(s); primary device: %2%3")
            .arg(devices.size())
            .arg(primary.value(QStringLiteral("name")).toString(), suffix));
    return result;
}

#if LUW_HAS_OPENCL

QString opencl_platform_string(cl_platform_id platform, cl_platform_info field) {
    std::size_t size = 0;
    if (clGetPlatformInfo(platform, field, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return {};
    }
    std::vector<char> buffer(size, '\0');
    if (clGetPlatformInfo(platform, field, buffer.size(), buffer.data(), nullptr)
        != CL_SUCCESS) {
        return {};
    }
    return QString::fromUtf8(buffer.data()).trimmed();
}

QString opencl_device_string(cl_device_id device, cl_device_info field) {
    std::size_t size = 0;
    if (clGetDeviceInfo(device, field, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return {};
    }
    std::vector<char> buffer(size, '\0');
    if (clGetDeviceInfo(device, field, buffer.size(), buffer.data(), nullptr)
        != CL_SUCCESS) {
        return {};
    }
    return QString::fromUtf8(buffer.data()).trimmed();
}

template <typename Value>
Value opencl_device_value(cl_device_id device, cl_device_info field) {
    Value value{};
    if (clGetDeviceInfo(device, field, sizeof(value), &value, nullptr) != CL_SUCCESS) {
        return Value{};
    }
    return value;
}

QJsonObject opencl_device(cl_device_id device) {
    const cl_device_type type =
        opencl_device_value<cl_device_type>(device, CL_DEVICE_TYPE);
    const cl_uint computeUnits =
        opencl_device_value<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
    const cl_uint clockMHz =
        opencl_device_value<cl_uint>(device, CL_DEVICE_MAX_CLOCK_FREQUENCY);
    const cl_ulong globalMemory =
        opencl_device_value<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);

    QJsonObject info;
    info.insert(QStringLiteral("name"), opencl_device_string(device, CL_DEVICE_NAME));
    info.insert(QStringLiteral("vendor"), opencl_device_string(device, CL_DEVICE_VENDOR));
    info.insert(QStringLiteral("version"), opencl_device_string(device, CL_DEVICE_VERSION));
    info.insert(QStringLiteral("driver_version"),
                opencl_device_string(device, CL_DRIVER_VERSION));
    info.insert(QStringLiteral("compute_capability"), QString());
    info.insert(QStringLiteral("compute_units"), static_cast<int>(computeUnits));
    info.insert(QStringLiteral("type"),
                QString::fromStdString(opencl_diagnostics_detail::device_type_string(
                    static_cast<std::uint64_t>(type))));
    info.insert(QStringLiteral("max_clock_mhz"), static_cast<int>(clockMHz));
    info.insert(QStringLiteral("global_memory_bytes"),
                static_cast<double>(globalMemory));
    return info;
}

#endif

QJsonObject probe_opencl_environment() {
    QJsonObject result;
    result.insert(QStringLiteral("available"), false);
    result.insert(QStringLiteral("warning"), false);
    result.insert(QStringLiteral("summary"), QString());
    result.insert(QStringLiteral("error"), QString());
    result.insert(QStringLiteral("version"), QString());
    result.insert(QStringLiteral("platforms"), QJsonArray());
    result.insert(QStringLiteral("devices"), QJsonArray());

#if !LUW_HAS_OPENCL
    result.insert(QStringLiteral("warning"), true);
    const QString message = QStringLiteral(
        "OpenCL probe unavailable: this native build has no OpenCL support.");
    result.insert(QStringLiteral("summary"), message);
    result.insert(QStringLiteral("error"), message);
    return result;
#else
    cl_uint platformCount = 0;
    const cl_int countStatus = clGetPlatformIDs(0, nullptr, &platformCount);
    if (countStatus != CL_SUCCESS) {
        result.insert(QStringLiteral("warning"), true);
        QString detail;
        if (countStatus == -1001) {
            detail = QStringLiteral("clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR");
        } else {
            detail = QStringLiteral("OpenCL error %1").arg(countStatus);
        }
        result.insert(QStringLiteral("error"), detail);
        result.insert(QStringLiteral("summary"),
                      QStringLiteral("OpenCL platform discovery failed: %1").arg(detail));
        return result;
    }
    if (platformCount == 0) {
        result.insert(QStringLiteral("warning"), true);
        result.insert(QStringLiteral("summary"), QStringLiteral(
            "OpenCL is installed, but no OpenCL platform was detected."));
        return result;
    }

    std::vector<cl_platform_id> platforms(platformCount);
    const cl_int platformStatus =
        clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    if (platformStatus != CL_SUCCESS) {
        result.insert(QStringLiteral("warning"), true);
        const QString detail = QStringLiteral("OpenCL error %1").arg(platformStatus);
        result.insert(QStringLiteral("error"), detail);
        result.insert(QStringLiteral("summary"),
                      QStringLiteral("OpenCL platform discovery failed: %1").arg(detail));
        return result;
    }

    QJsonArray platformPayloads;
    QJsonArray devicePayloads;
    QStringList uniqueVersions;
    for (cl_platform_id platform : platforms) {
        QJsonObject platformInfo;
        platformInfo.insert(QStringLiteral("name"),
                            opencl_platform_string(platform, CL_PLATFORM_NAME));
        platformInfo.insert(QStringLiteral("vendor"),
                            opencl_platform_string(platform, CL_PLATFORM_VENDOR));
        const QString version =
            opencl_platform_string(platform, CL_PLATFORM_VERSION);
        platformInfo.insert(QStringLiteral("version"), version);
        platformInfo.insert(QStringLiteral("device_count"), 0);
        if (!version.isEmpty() && !uniqueVersions.contains(version)) {
            uniqueVersions << version;
        }

        cl_uint deviceCount = 0;
        const cl_int countDevicesStatus =
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
        if (countDevicesStatus == CL_DEVICE_NOT_FOUND ||
            (countDevicesStatus == CL_SUCCESS && deviceCount == 0)) {
            platformPayloads.append(platformInfo);
            continue;
        }
        if (countDevicesStatus != CL_SUCCESS) {
            platformInfo.insert(
                QStringLiteral("error"),
                QStringLiteral("OpenCL error %1 while enumerating devices")
                    .arg(countDevicesStatus));
            platformPayloads.append(platformInfo);
            continue;
        }

        std::vector<cl_device_id> devices(deviceCount);
        const cl_int devicesStatus = clGetDeviceIDs(
            platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);
        if (devicesStatus != CL_SUCCESS) {
            platformInfo.insert(
                QStringLiteral("error"),
                QStringLiteral("OpenCL error %1 while enumerating devices")
                    .arg(devicesStatus));
            platformPayloads.append(platformInfo);
            continue;
        }

        platformInfo.insert(QStringLiteral("device_count"),
                            static_cast<int>(devices.size()));
        for (cl_device_id device : devices) {
            devicePayloads.append(opencl_device(device));
        }
        platformPayloads.append(platformInfo);
    }

    result.insert(QStringLiteral("platforms"), platformPayloads);
    result.insert(QStringLiteral("devices"), devicePayloads);
    result.insert(QStringLiteral("version"),
                  uniqueVersions.join(QStringLiteral(" | ")));
    result.insert(QStringLiteral("available"), !devicePayloads.isEmpty());
    result.insert(QStringLiteral("warning"), devicePayloads.isEmpty());
    if (devicePayloads.isEmpty()) {
        result.insert(QStringLiteral("summary"), QStringLiteral(
            "OpenCL platforms were detected, but no device could be enumerated."));
    } else {
        const QString primary =
            devicePayloads.first().toObject().value(QStringLiteral("name")).toString();
        result.insert(
            QStringLiteral("summary"),
            QStringLiteral(
                "OpenCL available on %1 platform(s) and %2 device(s); primary device: %3")
                .arg(platformCount)
                .arg(devicePayloads.size())
                .arg(primary));
    }
    return result;
#endif
}

QJsonObject native_contract() {
    QJsonObject contract;
    contract.insert(QStringLiteral("python_free"), true);
    contract.insert(QStringLiteral("python_requirements_check_applicable"), false);
    contract.insert(QStringLiteral("numba_runtime_applicable"), false);
    contract.insert(QStringLiteral("cuda_wheel_preparation_applicable"), false);
    contract.insert(QStringLiteral("diagnostics_backend"),
                    QStringLiteral("CUDA Driver API + OpenCL C API"));
    return contract;
}

QJsonObject side_effects() {
    QJsonObject effects;
    effects.insert(QStringLiteral("files_written"), false);
    effects.insert(QStringLiteral("environment_modified"), false);
    effects.insert(QStringLiteral("driver_queries_only"), true);
    return effects;
}

QJsonObject startup_report(const Options& options) {
    QJsonObject requirementsSummary;
    requirementsSummary.insert(QStringLiteral("total"), 0);
    requirementsSummary.insert(QStringLiteral("passed"), 0);
    requirementsSummary.insert(QStringLiteral("failed"), 0);
    requirementsSummary.insert(
        QStringLiteral("text"),
        QStringLiteral("Python package import checks are not applicable to the native build."));

    QString requirementsPath;
    if (!options.requirementsPath.isEmpty()) {
        requirementsPath = QFileInfo(options.requirementsPath).absoluteFilePath();
    }

    QJsonObject report;
    report.insert(QStringLiteral("schema_version"), 1);
    report.insert(QStringLiteral("host_platform"),
                  QStringLiteral("%1 (%2)")
                      .arg(QSysInfo::prettyProductName(),
                           QSysInfo::currentCpuArchitecture()));
    // Retain the old top-level keys so JSON consumers can distinguish an
    // intentionally absent Python runtime from a missing field.
    report.insert(QStringLiteral("python_executable"), QString());
    report.insert(QStringLiteral("python_version"), QString());
    report.insert(QStringLiteral("requirements_path"), requirementsPath);
    report.insert(QStringLiteral("checks"), QJsonArray());
    report.insert(QStringLiteral("requirements_summary"), requirementsSummary);
    report.insert(QStringLiteral("native_executable"),
                  QCoreApplication::applicationFilePath());
    report.insert(QStringLiteral("native_version"),
                  QCoreApplication::applicationVersion());
    report.insert(QStringLiteral("native_runtime"), native_contract());
    report.insert(QStringLiteral("side_effects"), side_effects());
    report.insert(QStringLiteral("cuda"), probe_cuda_environment());
    report.insert(QStringLiteral("opencl"), probe_opencl_environment());
    return report;
}

QJsonObject prepare_cuda_compatibility_report() {
    const QJsonObject cuda = probe_cuda_environment();
    QJsonObject report;
    report.insert(QStringLiteral("prepared"), false);
    report.insert(QStringLiteral("configured"),
                  cuda.value(QStringLiteral("configured")).toBool(false));
    report.insert(QStringLiteral("applicable"), false);
    report.insert(QStringLiteral("provider"),
                  cuda.value(QStringLiteral("provider")).toString());
    report.insert(QStringLiteral("source_root"), QString());
    report.insert(QStringLiteral("runtime_root"), QString());
    report.insert(QStringLiteral("cuda_home"), QString());
    report.insert(QStringLiteral("message"), QStringLiteral(
        "CUDA runtime wheel preparation is not applicable to the Python-free native build; "
        "the native runtime uses the installed CUDA driver directly."));
    report.insert(QStringLiteral("files_written"), false);
    report.insert(QStringLiteral("environment_modified"), false);
    report.insert(QStringLiteral("cuda"), cuda);
    return report;
}

void print_json(const QJsonObject& object, std::ostream& output) {
    const QByteArray json = QJsonDocument(object).toJson(QJsonDocument::Compact);
    output.write(json.constData(), static_cast<std::streamsize>(json.size()));
    output.put('\n');
}

} // namespace

int run_accelerator_report(const CommandArguments& arguments,
                           std::ostream& output,
                           std::ostream& error) {
    Options options;
    if (!parse_options(arguments, options, error)) {
        return 2;
    }
    if (options.help) {
        print_usage(output);
        return 0;
    }

    // The compatibility preparation mode is deliberately query-only: unlike
    // the old Python/Numba path it never copies wheels or changes CUDA_HOME,
    // CUDA_PATH, PATH, LD_LIBRARY_PATH, or DYLD_LIBRARY_PATH.
    if (options.prepareCudaRuntime) {
        const QJsonObject report = prepare_cuda_compatibility_report();
        if (options.json) print_json(report, output);
        return 0;
    }

    if (!options.requirementsPath.isEmpty()) {
        const QJsonObject report = startup_report(options);
        if (options.json) print_json(report, output);
        return 0;
    }

    // Match argparse's legacy branch: no selected operation (including a lone
    // --json) prints help instead of synthesising a startup report.
    print_usage(output);
    return 0;
}

} // namespace luw::cli
