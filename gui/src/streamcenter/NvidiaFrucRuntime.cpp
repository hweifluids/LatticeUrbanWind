#include "NvidiaFrucRuntime.h"

#include <QDir>
#include <QFileInfo>
#include <QLibrary>
#include <QObject>
#include <QProcessEnvironment>
#include <QStringList>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#if defined(Q_OS_WIN)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifndef STREAMCENTERPLUS_ENABLE_NVOF_FRUC
#define STREAMCENTERPLUS_ENABLE_NVOF_FRUC 0
#endif

#ifndef STREAMCENTERPLUS_NVOF_FRUC_SDK_ROOT
#define STREAMCENTERPLUS_NVOF_FRUC_SDK_ROOT ""
#endif

#ifndef STREAMCENTERPLUS_NVOF_FRUC_RUNTIME_PATH
#define STREAMCENTERPLUS_NVOF_FRUC_RUNTIME_PATH ""
#endif

#if STREAMCENTERPLUS_ENABLE_NVOF_FRUC
#include <cuda.h>

#if __has_include(<NvOFFRUC.h>)
#include <NvOFFRUC.h>
#define STREAMCENTERPLUS_NVOF_FRUC_OFF_NAMES 1
#elif __has_include(<NvFRUC.h>)
#include <NvFRUC.h>
#define STREAMCENTERPLUS_NVOF_FRUC_OFF_NAMES 0
#else
#error "STREAMCENTERPLUS_ENABLE_NVOF_FRUC requires NvOFFRUC.h or NvFRUC.h"
#endif
#endif

namespace Streamcenter::Fruc {
namespace {

#if STREAMCENTERPLUS_ENABLE_NVOF_FRUC

#if STREAMCENTERPLUS_NVOF_FRUC_OFF_NAMES
using FrucCreateParam = NvOFFRUC_CREATE_PARAM;
using FrucHandle = NvOFFRUCHandle;
using FrucRegisterParam = NvOFFRUC_REGISTER_RESOURCE_PARAM;
using FrucStatus = NvOFFRUC_STATUS;
using FrucProcessInParam = NvOFFRUC_PROCESS_IN_PARAMS;
using FrucProcessOutParam = NvOFFRUC_PROCESS_OUT_PARAMS;
using FrucUnregisterParam = NvOFFRUC_UNREGISTER_RESOURCE_PARAM;
using CreateFn = PtrToFuncNvOFFRUCCreate;
using DestroyFn = PtrToFuncNvOFFRUCDestroy;
using ProcessFn = PtrToFuncNvOFFRUCProcess;
using RegisterFn = PtrToFuncNvOFFRUCRegisterResource;
using UnregisterFn = PtrToFuncNvOFFRUCUnregisterResource;
constexpr FrucStatus kFrucSuccess = NvOFFRUC_SUCCESS;
constexpr auto kCudaResource = CudaResource;
constexpr auto kArgbSurface = ARGBSurface;
constexpr auto kCudaResourceCuArray = CudaResourceCuArray;
#else
using FrucCreateParam = NvFRUC_CREATE_PARAM;
using FrucHandle = NvFRUCHandle;
using FrucRegisterParam = NvFRUC_REGISTER_RESOURCE_PARAM;
using FrucStatus = NvFRUC_STATUS;
using FrucProcessInParam = NvFRUC_PROCESS_IN_PARAMS;
using FrucProcessOutParam = NvFRUC_PROCESS_OUT_PARAMS;
using FrucUnregisterParam = NvFRUC_UNREGISTER_RESOURCE_PARAM;
using CreateFn = PtrToFuncNvFRUCCreate;
using DestroyFn = PtrToFuncNvFRUCDestroy;
using ProcessFn = PtrToFuncNvFRUCProcess;
using RegisterFn = PtrToFuncNvFRUCRegisterResource;
using UnregisterFn = PtrToFuncNvFRUCUnregisterResource;
constexpr FrucStatus kFrucSuccess = NvFRUC_SUCCESS;
constexpr auto kCudaResource = static_cast<NvFRUCResourceType>(0);
constexpr auto kArgbSurface = static_cast<NvFRUCSurfaceFormat>(1);
constexpr auto kCudaResourceCuArray = static_cast<NvFRUCCUDAResourceType>(1);
#endif

QString cudaResultText(CUresult result) {
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    if (name != nullptr && text != nullptr) {
        return QStringLiteral("%1: %2").arg(QString::fromLatin1(name), QString::fromLatin1(text));
    }
    if (name != nullptr) {
        return QString::fromLatin1(name);
    }
    return QStringLiteral("CUDA error %1").arg(static_cast<int>(result));
}

#if defined(Q_OS_WIN)
QString windowsErrorText(DWORD error) {
    wchar_t* buffer = nullptr;
    const DWORD length = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        error,
        0,
        reinterpret_cast<LPWSTR>(&buffer),
        0,
        nullptr);
    QString message;
    if (length > 0 && buffer != nullptr) {
        message = QString::fromWCharArray(buffer, static_cast<int>(length)).trimmed();
    }
    if (buffer != nullptr) {
        LocalFree(buffer);
    }
    if (message.isEmpty()) {
        return QStringLiteral("Windows error %1").arg(static_cast<unsigned long>(error));
    }
    return QStringLiteral("Windows error %1: %2").arg(static_cast<unsigned long>(error)).arg(message);
}
#endif

bool checkCuda(CUresult result, const QString& action, QString* errorMessage) {
    if (result == CUDA_SUCCESS) {
        return true;
    }
    if (errorMessage != nullptr) {
        *errorMessage = QObject::tr("%1 failed: %2").arg(action, cudaResultText(result));
    }
    return false;
}

QStringList frucRootCandidates() {
    QStringList roots;
    const QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    roots << QString::fromUtf8(STREAMCENTERPLUS_NVOF_FRUC_SDK_ROOT)
          << env.value(QStringLiteral("STREAMCENTERPLUS_NVOF_SDK_ROOT"))
          << env.value(QStringLiteral("STREAMCENTERPLUS_NVOF_FRUC_ROOT"))
          << env.value(QStringLiteral("NVIDIA_OPTICAL_FLOW_SDK_ROOT"))
          << env.value(QStringLiteral("NV_OF_SDK_ROOT"))
          << env.value(QStringLiteral("NVOF_SDK_ROOT"));
    roots.removeAll(QString());
    roots.removeDuplicates();
    return roots;
}

QStringList runtimeCandidates() {
    QStringList candidates;
    const QString configuredRuntime = QString::fromUtf8(STREAMCENTERPLUS_NVOF_FRUC_RUNTIME_PATH);
    if (!configuredRuntime.trimmed().isEmpty()) {
        candidates << configuredRuntime;
    }

    const QStringList libraryNames{
#if defined(Q_OS_WIN)
        QStringLiteral("NvFRUC.dll"),
        QStringLiteral("NvOFFRUC.dll")
#else
        QStringLiteral("libNvFRUC.so"),
        QStringLiteral("libNvOFFRUC.so")
#endif
    };

    for (const QString& rootPath : frucRootCandidates()) {
        if (rootPath.trimmed().isEmpty()) {
            continue;
        }
        const QDir root(rootPath);
        for (const QString& name : libraryNames) {
            candidates << root.absoluteFilePath(name)
                       << root.absoluteFilePath(QStringLiteral("bin/%1").arg(name))
                       << root.absoluteFilePath(QStringLiteral("bin/win64/%1").arg(name))
                       << root.absoluteFilePath(QStringLiteral("lib/%1").arg(name))
                       << root.absoluteFilePath(QStringLiteral("lib64/%1").arg(name))
                       << root.absoluteFilePath(QStringLiteral("NvFRUCSample/bin/win64/%1").arg(name))
                       << root.absoluteFilePath(QStringLiteral("NvOFFRUC/NvOFFRUCSample/bin/win64/%1").arg(name))
                       << root.absoluteFilePath(QStringLiteral("NvOFFRUCSample/bin/win64/%1").arg(name))
                       << root.absoluteFilePath(QStringLiteral("NvFRUCSample/bin/ubuntu/%1").arg(name))
                       << root.absoluteFilePath(QStringLiteral("NvOFFRUC/NvOFFRUCSample/bin/ubuntu/%1").arg(name))
                       << root.absoluteFilePath(QStringLiteral("NvOFFRUCSample/bin/ubuntu/%1").arg(name));
        }
    }

#if defined(Q_OS_WIN)
    candidates << QStringLiteral("NvFRUC") << QStringLiteral("NvOFFRUC");
#else
    candidates << QStringLiteral("NvFRUC") << QStringLiteral("NvOFFRUC")
               << QStringLiteral("libNvFRUC.so") << QStringLiteral("libNvOFFRUC.so");
#endif
    candidates.removeDuplicates();
    return candidates;
}

class FrucLibrary {
public:
    ~FrucLibrary() {
        unloadLoadedLibrary();
    }

    bool load(QString* errorMessage) {
        QStringList errors;
        for (const QString& candidate : runtimeCandidates()) {
            if (candidate.trimmed().isEmpty()) {
                continue;
            }
#if defined(Q_OS_WIN)
            nativeLibrary = loadWindowsLibrary(candidate, &errors);
            if (nativeLibrary == nullptr) {
                continue;
            }
#else
            library.setFileName(candidate);
            if (!library.load()) {
                errors << QStringLiteral("%1: %2").arg(candidate, library.errorString());
                continue;
            }
#endif

            create = resolveAny<CreateFn>({QStringLiteral("NvOFFRUCCreate"), QStringLiteral("NvFRUCCreate")});
            registerResource = resolveAny<RegisterFn>({QStringLiteral("NvOFFRUCRegisterResource"), QStringLiteral("NvFRUCRegisterResource")});
            unregisterResource = resolveAny<UnregisterFn>({QStringLiteral("NvOFFRUCUnregisterResource"), QStringLiteral("NvFRUCUnregisterResource")});
            process = resolveAny<ProcessFn>({QStringLiteral("NvOFFRUCProcess"), QStringLiteral("NvFRUCProcess")});
            destroy = resolveAny<DestroyFn>({QStringLiteral("NvOFFRUCDestroy"), QStringLiteral("NvFRUCDestroy")});

            if (create == nullptr || registerResource == nullptr || unregisterResource == nullptr ||
                process == nullptr || destroy == nullptr) {
                errors << QStringLiteral("%1: required FRUC API symbols were not found").arg(candidate);
                unloadLoadedLibrary();
                clear();
                continue;
            }

            loadedPath = candidate;
            return true;
        }

        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("NVIDIA FRUC runtime could not be loaded. %1").arg(errors.join(QStringLiteral("; ")));
        }
        return false;
    }

    QString path() const {
        return loadedPath;
    }

#if !defined(Q_OS_WIN)
    QLibrary library;
#endif
    QString loadedPath;
    CreateFn create = nullptr;
    DestroyFn destroy = nullptr;
    ProcessFn process = nullptr;
    RegisterFn registerResource = nullptr;
    UnregisterFn unregisterResource = nullptr;

private:
#if defined(Q_OS_WIN)
    HMODULE nativeLibrary = nullptr;

    static HMODULE loadWindowsLibrary(const QString& candidate, QStringList* errors) {
        const std::wstring nativePath = QDir::toNativeSeparators(candidate).toStdWString();
        HMODULE handle = nullptr;
        if (QFileInfo(candidate).isAbsolute()) {
            handle = LoadLibraryExW(
                nativePath.c_str(),
                nullptr,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if (handle == nullptr && GetLastError() == ERROR_INVALID_PARAMETER) {
                handle = LoadLibraryExW(nativePath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            }
        } else {
            handle = LoadLibraryW(nativePath.c_str());
        }
        if (handle == nullptr && errors != nullptr) {
            errors->append(QStringLiteral("%1: %2").arg(candidate, windowsErrorText(GetLastError())));
        }
        return handle;
    }
#endif

    template <typename Fn>
    Fn resolveAny(const QStringList& names) {
        for (const QString& name : names) {
#if defined(Q_OS_WIN)
            if (auto symbol = GetProcAddress(nativeLibrary, name.toLatin1().constData())) {
                return reinterpret_cast<Fn>(symbol);
            }
#else
            if (auto symbol = library.resolve(name.toLatin1().constData())) {
                return reinterpret_cast<Fn>(symbol);
            }
#endif
        }
        return nullptr;
    }

    void unloadLoadedLibrary() {
#if defined(Q_OS_WIN)
        if (nativeLibrary != nullptr) {
            FreeLibrary(nativeLibrary);
            nativeLibrary = nullptr;
        }
#else
        if (library.isLoaded()) {
            library.unload();
        }
#endif
    }

    void clear() {
        create = nullptr;
        destroy = nullptr;
        process = nullptr;
        registerResource = nullptr;
        unregisterResource = nullptr;
    }
};

class CudaContext {
public:
    bool initialize(QString* errorMessage) {
        if (!checkCuda(cuInit(0), QObject::tr("CUDA initialization"), errorMessage)) {
            return false;
        }

        int deviceCount = 0;
        if (!checkCuda(cuDeviceGetCount(&deviceCount), QObject::tr("CUDA device query"), errorMessage)) {
            return false;
        }
        if (deviceCount <= 0) {
            if (errorMessage != nullptr) {
                *errorMessage = QObject::tr("No CUDA-capable NVIDIA GPU was found.");
            }
            return false;
        }

        CUdevice device = 0;
        if (!checkCuda(cuDeviceGet(&device, 0), QObject::tr("CUDA device selection"), errorMessage)) {
            return false;
        }

#if CUDA_VERSION >= 13000
        if (!checkCuda(cuCtxCreate(&context_, nullptr, 0, device), QObject::tr("CUDA context creation"), errorMessage)) {
            return false;
        }
#else
        if (!checkCuda(cuCtxCreate(&context_, 0, device), QObject::tr("CUDA context creation"), errorMessage)) {
            return false;
        }
#endif
        CUcontext previous = nullptr;
        cuCtxPopCurrent(&previous);
        return true;
    }

    ~CudaContext() {
        if (context_ != nullptr) {
            cuCtxDestroy(context_);
        }
    }

    CUcontext handle() const {
        return context_;
    }

private:
    CUcontext context_ = nullptr;
};

class ScopedCurrentContext {
public:
    ScopedCurrentContext(CUcontext context, QString* errorMessage)
        : active_(checkCuda(cuCtxPushCurrent(context), QObject::tr("CUDA context activation"), errorMessage)) {}

    ~ScopedCurrentContext() {
        if (active_) {
            CUcontext previous = nullptr;
            cuCtxPopCurrent(&previous);
        }
    }

    bool active() const {
        return active_;
    }

private:
    bool active_ = false;
};

class CudaArray {
public:
    bool create(int width, int height, QString* errorMessage) {
        CUDA_ARRAY_DESCRIPTOR descriptor{};
        descriptor.Width = static_cast<size_t>(width);
        descriptor.Height = static_cast<size_t>(height);
        descriptor.Format = CU_AD_FORMAT_UNSIGNED_INT8;
        descriptor.NumChannels = 4;
        return checkCuda(cuArrayCreate(&array_, &descriptor), QObject::tr("CUDA array allocation"), errorMessage);
    }

    ~CudaArray() {
        destroy();
    }

    void destroy() {
        if (array_ != nullptr) {
            cuArrayDestroy(array_);
            array_ = nullptr;
        }
    }

    CUarray handle() const {
        return array_;
    }

private:
    CUarray array_ = nullptr;
};

QImage normalizedFrame(const QImage& image) {
    return image.convertToFormat(QImage::Format_RGBA8888);
}

bool copyImageToArray(const QImage& image, CUarray destination, QString* errorMessage) {
    const QImage source = normalizedFrame(image);
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = source.constBits();
    copy.srcPitch = static_cast<size_t>(source.bytesPerLine());
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = destination;
    copy.WidthInBytes = static_cast<size_t>(source.width() * 4);
    copy.Height = static_cast<size_t>(source.height());
    return checkCuda(cuMemcpy2D(&copy), QObject::tr("CUDA upload"), errorMessage);
}

bool copyArrayToImage(CUarray source, int width, int height, QImage* image, QString* errorMessage) {
    QImage output(width, height, QImage::Format_RGBA8888);
    if (output.isNull()) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Could not allocate interpolated frame image.");
        }
        return false;
    }

    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = source;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = output.bits();
    copy.dstPitch = static_cast<size_t>(output.bytesPerLine());
    copy.WidthInBytes = static_cast<size_t>(width * 4);
    copy.Height = static_cast<size_t>(height);
    if (!checkCuda(cuMemcpy2D(&copy), QObject::tr("CUDA download"), errorMessage)) {
        return false;
    }

    *image = output;
    return true;
}

class FrucSession {
public:
    bool initialize(int width, int height, QString* errorMessage) {
        width_ = width;
        height_ = height;
        if (!library_.load(errorMessage) || !context_.initialize(errorMessage)) {
            return false;
        }

        ScopedCurrentContext current(context_.handle(), errorMessage);
        if (!current.active()) {
            return false;
        }

        if (!inputA_.create(width, height, errorMessage) ||
            !inputB_.create(width, height, errorMessage) ||
            !output_.create(width, height, errorMessage)) {
            return false;
        }

        FrucCreateParam createParam{};
        createParam.uiWidth = static_cast<uint32_t>(width);
        createParam.uiHeight = static_cast<uint32_t>(height);
        createParam.pDevice = nullptr;
        createParam.eResourceType = kCudaResource;
        createParam.eSurfaceFormat = kArgbSurface;
        createParam.eCUDAResourceType = kCudaResourceCuArray;

        FrucStatus status = library_.create(&createParam, &handle_);
        if (status != kFrucSuccess || handle_ == nullptr) {
            if (errorMessage != nullptr) {
                *errorMessage = QObject::tr("NVIDIA FRUC initialization failed with status %1.").arg(static_cast<int>(status));
            }
            return false;
        }

        FrucRegisterParam registerParam{};
        registerParam.pArrResource[0] = inputA_.handle();
        registerParam.pArrResource[1] = inputB_.handle();
        registerParam.pArrResource[2] = output_.handle();
        registerParam.uiCount = 3;
        status = library_.registerResource(handle_, &registerParam);
        if (status != kFrucSuccess) {
            if (errorMessage != nullptr) {
                *errorMessage = QObject::tr("NVIDIA FRUC resource registration failed with status %1.").arg(static_cast<int>(status));
            }
            return false;
        }
        registered_ = true;
        return true;
    }

    ~FrucSession() {
        if (context_.handle() == nullptr) {
            return;
        }
        QString ignored;
        ScopedCurrentContext current(context_.handle(), &ignored);
        if (current.active()) {
            if (registered_ && handle_ != nullptr) {
                FrucUnregisterParam unregisterParam{};
                unregisterParam.pArrResource[0] = inputA_.handle();
                unregisterParam.pArrResource[1] = inputB_.handle();
                unregisterParam.pArrResource[2] = output_.handle();
                unregisterParam.uiCount = 3;
                library_.unregisterResource(handle_, &unregisterParam);
            }
            if (handle_ != nullptr) {
                library_.destroy(handle_);
            }
            output_.destroy();
            inputB_.destroy();
            inputA_.destroy();
        }
    }

    bool prime(const QImage& firstFrame, QString* errorMessage) {
        return processFrame(firstFrame, 0.0, 0.0, nullptr, errorMessage);
    }

    bool interpolateNext(const QImage& nextFrame,
                         double inputTimestamp,
                         double outputTimestamp,
                         QImage* interpolated,
                         QString* errorMessage) {
        return processFrame(nextFrame, inputTimestamp, outputTimestamp, interpolated, errorMessage);
    }

private:
    bool processFrame(const QImage& frame,
                      double inputTimestamp,
                      double outputTimestamp,
                      QImage* interpolated,
                      QString* errorMessage) {
        ScopedCurrentContext current(context_.handle(), errorMessage);
        if (!current.active()) {
            return false;
        }

        CUarray input = useInputA_ ? inputA_.handle() : inputB_.handle();
        useInputA_ = !useInputA_;
        if (!copyImageToArray(frame, input, errorMessage)) {
            return false;
        }

        bool repetitionFlag = false;
        FrucProcessInParam inputParam{};
        inputParam.stFrameDataInput.pFrame = input;
        inputParam.stFrameDataInput.nTimeStamp = inputTimestamp;
        FrucProcessOutParam outputParam{};
        outputParam.stFrameDataOutput.pFrame = output_.handle();
        outputParam.stFrameDataOutput.nTimeStamp = outputTimestamp;
        outputParam.stFrameDataOutput.bHasFrameRepetitionOccurred = &repetitionFlag;

        const FrucStatus status = library_.process(handle_, &inputParam, &outputParam);
        if (status != kFrucSuccess) {
            if (errorMessage != nullptr) {
                *errorMessage = QObject::tr("NVIDIA FRUC processing failed with status %1.").arg(static_cast<int>(status));
            }
            return false;
        }

        if (interpolated != nullptr) {
            return copyArrayToImage(output_.handle(), width_, height_, interpolated, errorMessage);
        }
        return true;
    }

    FrucLibrary library_;
    CudaContext context_;
    CudaArray inputA_;
    CudaArray inputB_;
    CudaArray output_;
    FrucHandle handle_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    bool registered_ = false;
    bool useInputA_ = true;
};

bool interpolate2x(const QVector<QImage>& inputFrames,
                   QVector<QImage>* outputFrames,
                   QString* errorMessage,
                   const InterpolationProgressCallback& progressCallback,
                   double progressStart,
                   double progressSpan) {
    if (inputFrames.size() < 2) {
        *outputFrames = inputFrames;
        return true;
    }

    const int width = inputFrames.first().width();
    const int height = inputFrames.first().height();
    FrucSession session;
    if (!session.initialize(width, height, errorMessage)) {
        return false;
    }
    if (!session.prime(inputFrames.first(), errorMessage)) {
        return false;
    }

    outputFrames->clear();
    outputFrames->reserve((inputFrames.size() - 1) * 2 + 1);
    outputFrames->push_back(normalizedFrame(inputFrames.first()));

    for (int i = 1; i < inputFrames.size(); ++i) {
        if (inputFrames[i].width() != width || inputFrames[i].height() != height) {
            if (errorMessage != nullptr) {
                *errorMessage = QObject::tr("Frame interpolation requires all frames to have the same resolution.");
            }
            return false;
        }

        QImage interpolated;
        if (!session.interpolateNext(inputFrames[i],
                                     static_cast<double>(i),
                                     static_cast<double>(i) - 0.5,
                                     &interpolated,
                                     errorMessage)) {
            return false;
        }
        outputFrames->push_back(interpolated);
        outputFrames->push_back(normalizedFrame(inputFrames[i]));

        if (progressCallback) {
            const double local = static_cast<double>(i) / static_cast<double>(inputFrames.size() - 1);
            const double percent = progressStart + progressSpan * local;
            if (!progressCallback(percent,
                                  QObject::tr("Interpolating frame pair %1 of %2...")
                                      .arg(i)
                                      .arg(inputFrames.size() - 1))) {
                if (errorMessage != nullptr) {
                    *errorMessage = QObject::tr("Animation export interrupted.");
                }
                return false;
            }
        }
    }
    return true;
}

RuntimeStatus probeRuntime() {
    FrucLibrary library;
    QString error;
    if (!library.load(&error)) {
        return {false, error};
    }

    if (!checkCuda(cuInit(0), QObject::tr("CUDA initialization"), &error)) {
        return {false, error};
    }
    int deviceCount = 0;
    if (!checkCuda(cuDeviceGetCount(&deviceCount), QObject::tr("CUDA device query"), &error)) {
        return {false, error};
    }
    if (deviceCount <= 0) {
        return {false, QObject::tr("No CUDA-capable NVIDIA GPU was found.")};
    }

    return {true, QObject::tr("NVIDIA FRUC runtime is available: %1").arg(library.path())};
}

#else

RuntimeStatus probeRuntime() {
    return {
        false,
        QObject::tr("NVIDIA FRUC support is not enabled in this build. Reconfigure with STREAMCENTERPLUS_ENABLE_NVOF_FRUC=ON and provide STREAMCENTERPLUS_NVOF_SDK_ROOT.")
    };
}

#endif

}  // namespace

RuntimeStatus NvidiaFrucRuntime::status() {
    static const RuntimeStatus cached = probeRuntime();
    return cached;
}

bool NvidiaFrucRuntime::interpolateFrames(const QVector<QImage>& inputFrames,
                                          int multiplier,
                                          QVector<QImage>* outputFrames,
                                          QString* errorMessage,
                                          const InterpolationProgressCallback& progressCallback) {
    if (outputFrames == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QObject::tr("Internal error: missing interpolation output buffer.");
        }
        return false;
    }

    const int normalizedMultiplier = (multiplier == 2 || multiplier == 4) ? multiplier : 1;
    if (normalizedMultiplier == 1 || inputFrames.size() < 2) {
        *outputFrames = inputFrames;
        return true;
    }

#if STREAMCENTERPLUS_ENABLE_NVOF_FRUC
    const RuntimeStatus runtime = status();
    if (!runtime.available) {
        if (errorMessage != nullptr) {
            *errorMessage = runtime.message;
        }
        return false;
    }

    QVector<QImage> current = inputFrames;
    const int passCount = normalizedMultiplier == 4 ? 2 : 1;
    for (int pass = 0; pass < passCount; ++pass) {
        QVector<QImage> next;
        const double progressStart = 100.0 * static_cast<double>(pass) / static_cast<double>(passCount);
        const double progressSpan = 100.0 / static_cast<double>(passCount);
        if (!interpolate2x(current, &next, errorMessage, progressCallback, progressStart, progressSpan)) {
            return false;
        }
        current = std::move(next);
    }

    for (int i = 1; i < normalizedMultiplier; ++i) {
        current.push_back(current.last());
    }
    *outputFrames = std::move(current);
    return true;
#else
    Q_UNUSED(inputFrames);
    Q_UNUSED(progressCallback);
    if (errorMessage != nullptr) {
        *errorMessage = status().message;
    }
    return false;
#endif
}

}  // namespace Streamcenter::Fruc
