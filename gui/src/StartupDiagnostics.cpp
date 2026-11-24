#include "luwgui/StartupDiagnostics.h"

#include "luwgui/RuntimePaths.h"

#include <QDir>
#include <QFileInfo>
#include <QLibrary>
#include <QProcess>
#include <QProcessEnvironment>
#include <QStringList>
#include <QSysInfo>

#include <array>
#include <vector>

#ifndef LUW_GUI_HAS_OPENCL
#define LUW_GUI_HAS_OPENCL 0
#endif

#if LUW_GUI_HAS_OPENCL
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>
#endif

namespace luwgui {

namespace {

QString buildProcessErrorMessage(const StartupCheckResult& result) {
    QStringList lines;
    if (!result.processError.trimmed().isEmpty()) {
        lines << result.processError.trimmed();
    }
    if (!result.rawStderr.trimmed().isEmpty()) {
        lines << result.rawStderr.trimmed();
    }
    if (!result.rawStdout.trimmed().isEmpty()) {
        lines << result.rawStdout.trimmed();
    }
    return lines.join('\n').trimmed();
}

QString cudaDriverVersion(int encodedVersion) {
    if (encodedVersion <= 0) {
        return {};
    }
    const int major = encodedVersion / 1000;
    const int minor = (encodedVersion % 1000) / 10;
    return QStringLiteral("%1.%2").arg(major).arg(minor);
}

AcceleratorCheck probeCudaEnvironment() {
    AcceleratorCheck result;

#if defined(Q_OS_WIN)
    QLibrary library(QStringLiteral("nvcuda"));
#elif defined(Q_OS_LINUX)
    QLibrary library(QStringLiteral("cuda"));
#else
    result.summary = QStringLiteral("CUDA probe unavailable on this platform.");
    return result;
#endif

    if (!library.load()) {
        result.summary = QStringLiteral("CUDA driver is not available: %1").arg(library.errorString());
        result.errorText = library.errorString();
        return result;
    }

    using CuResult = int;
    using CuDevice = int;
    using CuInit = CuResult (*)(unsigned int);
    using CuDriverGetVersion = CuResult (*)(int*);
    using CuDeviceGetCount = CuResult (*)(int*);
    using CuDeviceGet = CuResult (*)(CuDevice*, int);
    using CuDeviceGetName = CuResult (*)(char*, int, CuDevice);
    using CuDeviceComputeCapability = CuResult (*)(int*, int*, CuDevice);

    const auto cuInit = reinterpret_cast<CuInit>(library.resolve("cuInit"));
    const auto cuDriverGetVersion = reinterpret_cast<CuDriverGetVersion>(library.resolve("cuDriverGetVersion"));
    const auto cuDeviceGetCount = reinterpret_cast<CuDeviceGetCount>(library.resolve("cuDeviceGetCount"));
    const auto cuDeviceGet = reinterpret_cast<CuDeviceGet>(library.resolve("cuDeviceGet"));
    const auto cuDeviceGetName = reinterpret_cast<CuDeviceGetName>(library.resolve("cuDeviceGetName"));
    const auto cuDeviceComputeCapability =
        reinterpret_cast<CuDeviceComputeCapability>(library.resolve("cuDeviceComputeCapability"));

    if (!cuInit || !cuDriverGetVersion || !cuDeviceGetCount || !cuDeviceGet
        || !cuDeviceGetName || !cuDeviceComputeCapability) {
        result.summary = QStringLiteral("CUDA probe unavailable: the driver API is incomplete.");
        result.errorText = result.summary;
        return result;
    }

    const CuResult initStatus = cuInit(0);
    if (initStatus != 0) {
        result.summary = QStringLiteral("CUDA driver is not available (error %1).").arg(initStatus);
        result.errorText = result.summary;
        return result;
    }

    int encodedDriverVersion = 0;
    if (cuDriverGetVersion(&encodedDriverVersion) != 0) {
        encodedDriverVersion = 0;
    }
    const QString driverVersion = cudaDriverVersion(encodedDriverVersion);

    int deviceCount = 0;
    const CuResult countStatus = cuDeviceGetCount(&deviceCount);
    if (countStatus != 0) {
        result.summary = QStringLiteral("CUDA device discovery failed (error %1).").arg(countStatus);
        result.errorText = result.summary;
        return result;
    }
    if (deviceCount <= 0) {
        result.summary = QStringLiteral("CUDA ready, but no supported NVIDIA GPU was detected.");
        return result;
    }

    for (int index = 0; index < deviceCount; ++index) {
        CuDevice device = 0;
        if (cuDeviceGet(&device, index) != 0) {
            continue;
        }

        std::array<char, 256> name{};
        if (cuDeviceGetName(name.data(), static_cast<int>(name.size()), device) != 0) {
            continue;
        }

        int computeMajor = 0;
        int computeMinor = 0;
        const bool hasComputeCapability =
            cuDeviceComputeCapability(&computeMajor, &computeMinor, device) == 0;

        AcceleratorDeviceCheck info;
        info.name = QString::fromUtf8(name.data()).trimmed();
        info.vendor = QStringLiteral("NVIDIA");
        info.version = QStringLiteral("CUDA");
        info.driverVersion = driverVersion;
        if (hasComputeCapability) {
            info.computeCapability = QStringLiteral("%1.%2").arg(computeMajor).arg(computeMinor);
        }
        result.devices.push_back(info);
    }

    result.available = !result.devices.isEmpty();
    if (!result.available) {
        result.summary = QStringLiteral("CUDA devices were reported, but none could be enumerated.");
        return result;
    }

    const AcceleratorDeviceCheck& primary = result.devices.front();
    QStringList details;
    if (!driverVersion.isEmpty()) {
        details << QStringLiteral("driver %1").arg(driverVersion);
    }
    if (!primary.computeCapability.isEmpty()) {
        details << QStringLiteral("cc %1").arg(primary.computeCapability);
    }
    const QString suffix = details.isEmpty()
        ? QString()
        : QStringLiteral(" (%1)").arg(details.join(QStringLiteral(", ")));
    result.summary = QStringLiteral("CUDA available on %1 device(s); primary device: %2%3")
                         .arg(result.devices.size())
                         .arg(primary.name, suffix);
    return result;
}

#if LUW_GUI_HAS_OPENCL
QString openClPlatformString(cl_platform_id platform, cl_platform_info key) {
    size_t byteCount = 0;
    if (clGetPlatformInfo(platform, key, 0, nullptr, &byteCount) != CL_SUCCESS || byteCount == 0) {
        return {};
    }
    std::vector<char> buffer(byteCount, '\0');
    if (clGetPlatformInfo(platform, key, byteCount, buffer.data(), nullptr) != CL_SUCCESS) {
        return {};
    }
    return QString::fromUtf8(buffer.data()).trimmed();
}

QString openClDeviceString(cl_device_id device, cl_device_info key) {
    size_t byteCount = 0;
    if (clGetDeviceInfo(device, key, 0, nullptr, &byteCount) != CL_SUCCESS || byteCount == 0) {
        return {};
    }
    std::vector<char> buffer(byteCount, '\0');
    if (clGetDeviceInfo(device, key, byteCount, buffer.data(), nullptr) != CL_SUCCESS) {
        return {};
    }
    return QString::fromUtf8(buffer.data()).trimmed();
}

QString openClDeviceTypeName(cl_device_type type) {
    QStringList names;
    if ((type & CL_DEVICE_TYPE_DEFAULT) != 0) names << QStringLiteral("DEFAULT");
    if ((type & CL_DEVICE_TYPE_CPU) != 0) names << QStringLiteral("CPU");
    if ((type & CL_DEVICE_TYPE_GPU) != 0) names << QStringLiteral("GPU");
    if ((type & CL_DEVICE_TYPE_ACCELERATOR) != 0) names << QStringLiteral("ACCELERATOR");
#ifdef CL_DEVICE_TYPE_CUSTOM
    if ((type & CL_DEVICE_TYPE_CUSTOM) != 0) names << QStringLiteral("CUSTOM");
#endif
    return names.isEmpty() ? QString::number(static_cast<qulonglong>(type))
                           : names.join(QStringLiteral(" | "));
}
#endif

AcceleratorCheck probeOpenClEnvironment() {
    AcceleratorCheck result;

#if !LUW_GUI_HAS_OPENCL
    result.warning = true;
    result.summary = QStringLiteral("OpenCL probe unavailable: this build has no OpenCL support.");
    result.errorText = result.summary;
    return result;
#else
    cl_uint platformCount = 0;
    const cl_int countStatus = clGetPlatformIDs(0, nullptr, &platformCount);
    if (countStatus != CL_SUCCESS) {
        result.warning = true;
        result.summary = QStringLiteral("OpenCL platform discovery failed (error %1).").arg(countStatus);
        result.errorText = result.summary;
        return result;
    }
    if (platformCount == 0) {
        result.warning = true;
        result.summary = QStringLiteral("OpenCL is installed, but no OpenCL platform was detected.");
        return result;
    }

    std::vector<cl_platform_id> platforms(platformCount);
    const cl_int platformStatus = clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    if (platformStatus != CL_SUCCESS) {
        result.warning = true;
        result.summary = QStringLiteral("OpenCL platform discovery failed (error %1).").arg(platformStatus);
        result.errorText = result.summary;
        return result;
    }

    QStringList platformVersions;
    for (cl_platform_id platform : platforms) {
        const QString platformVersion = openClPlatformString(platform, CL_PLATFORM_VERSION);
        if (!platformVersion.isEmpty() && !platformVersions.contains(platformVersion)) {
            platformVersions << platformVersion;
        }

        cl_uint deviceCount = 0;
        const cl_int deviceCountStatus =
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
        if (deviceCountStatus == CL_DEVICE_NOT_FOUND || deviceCount == 0) {
            continue;
        }
        if (deviceCountStatus != CL_SUCCESS) {
            continue;
        }

        std::vector<cl_device_id> devices(deviceCount);
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr)
            != CL_SUCCESS) {
            continue;
        }

        for (cl_device_id device : devices) {
            AcceleratorDeviceCheck info;
            info.name = openClDeviceString(device, CL_DEVICE_NAME);
            info.vendor = openClDeviceString(device, CL_DEVICE_VENDOR);
            info.version = openClDeviceString(device, CL_DEVICE_VERSION);
            info.driverVersion = openClDeviceString(device, CL_DRIVER_VERSION);
            cl_device_type deviceType = 0;
            if (clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType),
                                &deviceType, nullptr) == CL_SUCCESS) {
                info.type = openClDeviceTypeName(deviceType);
            }
            cl_uint computeUnits = 0;
            if (clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                                sizeof(computeUnits), &computeUnits, nullptr) == CL_SUCCESS) {
                info.computeUnits = static_cast<int>(computeUnits);
            }
            cl_uint clockMHz = 0;
            if (clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                sizeof(clockMHz), &clockMHz, nullptr) == CL_SUCCESS) {
                info.maxClockMHz = static_cast<int>(clockMHz);
            }
            cl_ulong globalMemory = 0;
            if (clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                                sizeof(globalMemory), &globalMemory, nullptr) == CL_SUCCESS) {
                info.globalMemoryBytes = static_cast<quint64>(globalMemory);
            }
            result.devices.push_back(info);
        }
    }

    result.version = platformVersions.join(QStringLiteral(" | "));
    result.available = !result.devices.isEmpty();
    result.warning = !result.available;
    if (result.available) {
        result.summary =
            QStringLiteral("OpenCL available on %1 platform(s) and %2 device(s); primary device: %3")
                .arg(platformCount)
                .arg(result.devices.size())
                .arg(result.devices.front().name);
    } else {
        result.summary = QStringLiteral("OpenCL platforms were detected, but no device could be enumerated.");
    }
    return result;
#endif
}

} // namespace

bool StartupCheckResult::hasWarnings() const {
    return !nativeResolved || !culbmResolved || !schemaResolved
        || !processError.trimmed().isEmpty() || opencl.warning || cuda.warning;
}

QString StartupCheckResult::warningText() const {
    QStringList lines;
    lines << QStringLiteral("Startup finished, but the native runtime self-check reported problems.");
    lines << QStringLiteral("The GUI continued to load, but preprocessing, solver orchestration, or post-processing commands may fail.");
    lines << QString();
    lines << QStringLiteral("Repository root: ") + QDir::toNativeSeparators(repoRoot);
    lines << QStringLiteral("Native command runner: ")
        + (nativeExecutable.isEmpty()
               ? QStringLiteral("Unresolved.")
               : QDir::toNativeSeparators(nativeExecutable));
    if (!nativeVersion.isEmpty()) {
        lines << QStringLiteral("Native runtime version: ") + nativeVersion;
    }
    lines << QStringLiteral("CULBM solver (default): ")
        + (culbmExecutable.isEmpty()
               ? QStringLiteral("Unresolved.")
               : QDir::toNativeSeparators(culbmExecutable));
    lines << QStringLiteral("FluidX3D solver (optional): ")
        + (fluidX3DExecutable.isEmpty()
               ? QStringLiteral("Unresolved.")
               : QDir::toNativeSeparators(fluidX3DExecutable));
    lines << QStringLiteral("Deck schema: ")
        + (schemaPath.isEmpty()
               ? QStringLiteral("Unresolved.")
               : QDir::toNativeSeparators(schemaPath));
    if (!hostPlatform.isEmpty()) {
        lines << QStringLiteral("Host platform: ") + hostPlatform;
    }

    if (opencl.warning) {
        lines << QString();
        lines << QStringLiteral("OpenCL:");
        lines << (opencl.summary.isEmpty()
                      ? QStringLiteral("OpenCL is unavailable.")
                      : opencl.summary);
        if (!opencl.errorText.isEmpty() && opencl.errorText != opencl.summary) {
            lines << opencl.errorText;
        }
    }
    if (cuda.warning) {
        lines << QString();
        lines << QStringLiteral("CUDA:");
        lines << (cuda.summary.isEmpty()
                      ? QStringLiteral("CUDA is unavailable.")
                      : cuda.summary);
        if (!cuda.errorText.isEmpty() && cuda.errorText != cuda.summary) {
            lines << cuda.errorText;
        }
    }

    const QString processMessage = buildProcessErrorMessage(*this);
    if (!processMessage.isEmpty()) {
        lines << QString();
        lines << QStringLiteral("Self-check process details:");
        lines << processMessage;
    }
    return lines.join('\n');
}

StartupCheckResult runNativeEnvironmentSelfCheck(const QString& repoRoot, int timeoutMs) {
    StartupCheckResult result;
    result.repoRoot = repoRoot;
    result.hostPlatform = QStringLiteral("%1 (%2)")
                              .arg(QSysInfo::prettyProductName(),
                                   QSysInfo::currentCpuArchitecture());
    result.nativeExecutable = findNativeExecutable(repoRoot);
    result.nativeResolved = !result.nativeExecutable.trimmed().isEmpty();
    result.culbmExecutable = findCulbmExecutable(repoRoot);
    result.culbmResolved = !result.culbmExecutable.trimmed().isEmpty();
    result.fluidX3DExecutable = findFluidX3DExecutable(repoRoot);
    result.fluidX3DResolved = !result.fluidX3DExecutable.trimmed().isEmpty();
    result.schemaPath = resolveRepoFilePath(repoRoot, QStringLiteral("core/deck_schema.json"));
    result.schemaResolved = QFileInfo::exists(result.schemaPath);
    result.cuda = probeCudaEnvironment();
    result.opencl = probeOpenClEnvironment();

    if (!result.nativeResolved) {
        result.processError = QStringLiteral("Could not locate the native luw command runner.");
        return result;
    }

    QProcess process;
    process.setWorkingDirectory(repoRoot);

    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert(QStringLiteral("LUW_HOME"), repoRoot);
    QString path = environment.value(QStringLiteral("PATH"));
    const QString binPath = resolveRepoFilePath(repoRoot, QStringLiteral("bin"));
    if (!path.contains(binPath, Qt::CaseInsensitive)) {
        path.prepend(binPath + QDir::listSeparator());
    }
    environment.insert(QStringLiteral("PATH"), path);
    process.setProcessEnvironment(environment);
    process.setProcessChannelMode(QProcess::SeparateChannels);

    process.start(result.nativeExecutable, {QStringLiteral("--version")});
    if (!process.waitForStarted(5000)) {
        result.processError = QStringLiteral("Failed to start the native luw command runner.");
        result.rawStderr = process.errorString();
        return result;
    }

    if (!process.waitForFinished(timeoutMs)) {
        process.kill();
        process.waitForFinished(3000);
        result.processError = QStringLiteral("The native runtime self-check timed out.");
        result.rawStdout = QString::fromUtf8(process.readAllStandardOutput());
        result.rawStderr = QString::fromUtf8(process.readAllStandardError());
        return result;
    }

    result.rawStdout = QString::fromUtf8(process.readAllStandardOutput());
    result.rawStderr = QString::fromUtf8(process.readAllStandardError());

    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        result.processError = QStringLiteral("The native runtime self-check exited with a non-zero status.");
        return result;
    }
    result.nativeVersion = result.rawStdout.trimmed();
    return result;
}

} // namespace luwgui
