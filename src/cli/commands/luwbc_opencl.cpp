#include "luw/luwbc_opencl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <utility>

#ifndef LUW_HAS_OPENCL
#define LUW_HAS_OPENCL 0
#endif

#if LUW_HAS_OPENCL
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>
#ifndef CL_PLATFORM_NOT_FOUND_KHR
#define CL_PLATFORM_NOT_FOUND_KHR -1001
#endif
#endif

namespace luw::cli {

#if LUW_HAS_OPENCL
namespace {

#ifndef LUW_LUWBC_MAX_WORK_ITEMS
#define LUW_LUWBC_MAX_WORK_ITEMS (1U << 22U)
#endif
constexpr std::size_t kMaximumWorkItemsPerSubmission =
    static_cast<std::size_t>(LUW_LUWBC_MAX_WORK_ITEMS);
static_assert(kMaximumWorkItemsPerSubmission > 0U);
// Spatial plan kernels carry substantially more private state than the fused
// resamplers.  A smaller deterministic submission ceiling bounds driver-side
// latency and lets production callers stream large city grids in modest
// chunks without changing numerical order within a target.
constexpr std::size_t kMaximumSpatialTargetsPerSubmission = 1U << 15U;

void requireSuccess(cl_int status, std::string_view operation) {
    if (status != CL_SUCCESS) {
        throw OpenClLuwbcError(std::string(operation) +
                               " failed (OpenCL error " +
                               std::to_string(status) + ")");
    }
}

std::string infoString(cl_device_id device, cl_device_info property) {
    std::size_t length = 0U;
    requireSuccess(clGetDeviceInfo(device, property, 0, nullptr, &length),
                   "querying OpenCL device string size");
    if (length == 0U) return "unknown";
    std::string value(length, '\0');
    requireSuccess(clGetDeviceInfo(device, property, length, value.data(),
                                   nullptr),
                   "querying OpenCL device string");
    while (!value.empty() && value.back() == '\0') value.pop_back();
    return value.empty() ? "unknown" : value;
}

std::size_t checkedMultiply(std::size_t first, std::size_t second,
                            std::string_view description) {
    if (first != 0U && second > std::numeric_limits<std::size_t>::max() / first) {
        throw OpenClLuwbcError(std::string(description) + " size overflow");
    }
    return first * second;
}

std::size_t checkedAdd(std::size_t first, std::size_t second,
                       std::string_view description) {
    if (second > std::numeric_limits<std::size_t>::max() - first) {
        throw OpenClLuwbcError(std::string(description) + " size overflow");
    }
    return first + second;
}

std::size_t bytesFor(std::size_t count, std::size_t elementSize,
                     std::string_view description) {
    return checkedMultiply(count, elementSize, description);
}

struct EnumeratedGpuDevice {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
};

std::vector<EnumeratedGpuDevice> enumerateGpuDevices() {
    cl_uint platformCount = 0U;
    const cl_int countStatus = clGetPlatformIDs(0, nullptr, &platformCount);
    if (countStatus == CL_PLATFORM_NOT_FOUND_KHR ||
        (countStatus == CL_SUCCESS && platformCount == 0U)) {
        return {};
    }
    requireSuccess(countStatus, "counting OpenCL platforms");

    std::vector<cl_platform_id> platforms(platformCount);
    requireSuccess(clGetPlatformIDs(platformCount, platforms.data(), nullptr),
                   "enumerating OpenCL platforms");
    std::vector<EnumeratedGpuDevice> result;
    for (const cl_platform_id platform : platforms) {
        cl_uint deviceCount = 0U;
        const cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0,
                                             nullptr, &deviceCount);
        if (status == CL_DEVICE_NOT_FOUND ||
            (status == CL_SUCCESS && deviceCount == 0U)) {
            continue;
        }
        requireSuccess(status, "counting OpenCL GPU devices");
        std::vector<cl_device_id> devices(deviceCount);
        requireSuccess(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount,
                                      devices.data(), nullptr),
                       "enumerating OpenCL GPU devices");
        for (const cl_device_id device : devices) result.push_back({platform, device});
    }
    return result;
}

EnumeratedGpuDevice selectGpuDevice(int requestedIndex) {
    const std::vector<EnumeratedGpuDevice> devices = enumerateGpuDevices();
    if (requestedIndex < 0 ||
        static_cast<std::size_t>(requestedIndex) >= devices.size()) {
        throw OpenClLuwbcError(
            "requested OpenCL GPU device index " +
            std::to_string(requestedIndex) + ", but only " +
            std::to_string(devices.size()) +
            " OpenCL GPU device(s) are available");
    }
    return devices[static_cast<std::size_t>(requestedIndex)];
}

class Backend final {
public:
    explicit Backend(int deviceIndex) : deviceIndex_(deviceIndex) {
        try {
            const EnumeratedGpuDevice selected = selectGpuDevice(deviceIndex);
            platform_ = selected.platform;
            device_ = selected.device;
            cl_device_fp_config doubleConfiguration = 0U;
            const cl_int doubleStatus = clGetDeviceInfo(
                device_, CL_DEVICE_DOUBLE_FP_CONFIG,
                sizeof(doubleConfiguration), &doubleConfiguration, nullptr);
            if (doubleStatus != CL_SUCCESS || doubleConfiguration == 0U) {
                throw OpenClLuwbcError(
                    "selected OpenCL GPU does not support the double-precision "
                    "arithmetic required to preserve luwbc CPU interpolation semantics");
            }
            const cl_context_properties contextProperties[] = {
                CL_CONTEXT_PLATFORM,
                reinterpret_cast<cl_context_properties>(platform_), 0};
            cl_int status = CL_SUCCESS;
            context_ = clCreateContext(contextProperties, 1, &device_, nullptr,
                                       nullptr, &status);
            requireSuccess(status, "creating OpenCL luwbc context");
            // Each pipeline slot owns an ordinary in-order queue.  Cross-slot
            // overlap comes from the two independent queues rather than an
            // optional out-of-order execution capability.
            for (std::size_t slot = 0U; slot < queues_.size(); ++slot) {
                queues_[slot] =
                    clCreateCommandQueue(context_, device_, 0, &status);
                requireSuccess(
                    status,
                    "creating OpenCL luwbc pipeline command queue " +
                        std::to_string(slot));
            }

            const char* source = kernelSource();
            const std::size_t sourceLength = std::strlen(source);
            program_ = clCreateProgramWithSource(context_, 1, &source,
                                                 &sourceLength, &status);
            requireSuccess(status, "creating OpenCL luwbc program");
            // Deliberately omit every fast-math option: interpolation behavior
            // must retain normal IEEE single-precision semantics.
            status = clBuildProgram(program_, 1, &device_, "-cl-std=CL1.2",
                                    nullptr, nullptr);
            if (status != CL_SUCCESS) {
                std::size_t logLength = 0U;
                (void)clGetProgramBuildInfo(program_, device_,
                                            CL_PROGRAM_BUILD_LOG, 0, nullptr,
                                            &logLength);
                std::string log(logLength, '\0');
                if (logLength != 0U) {
                    (void)clGetProgramBuildInfo(program_, device_,
                                                CL_PROGRAM_BUILD_LOG,
                                                logLength, log.data(), nullptr);
                }
                throw OpenClLuwbcError(
                    "OpenCL luwbc kernel compilation failed: " + log);
            }
            for (std::size_t slot = 0U; slot < queues_.size(); ++slot) {
                horizontalKernels_[slot] =
                    clCreateKernel(program_, "horizontal_resample", &status);
                requireSuccess(
                    status,
                    "creating OpenCL luwbc horizontal pipeline kernel " +
                        std::to_string(slot));
                verticalKernels_[slot] =
                    clCreateKernel(program_, "vertical_resample", &status);
                requireSuccess(
                    status,
                    "creating OpenCL luwbc vertical pipeline kernel " +
                        std::to_string(slot));
            }
            demIdwKernel_ =
                clCreateKernel(program_, "dem_idw", &status);
            requireSuccess(status, "creating OpenCL luwbc DEM IDW kernel");
            mlsKernel_ =
                clCreateKernel(program_, "mls_weights", &status);
            requireSuccess(status, "creating OpenCL luwbc MLS kernel");

            requireSuccess(clGetDeviceInfo(device_, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                                           sizeof(maximumAllocation_),
                                           &maximumAllocation_, nullptr),
                           "querying OpenCL maximum allocation");
            requireSuccess(clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE,
                                           sizeof(globalMemory_),
                                           &globalMemory_, nullptr),
                           "querying OpenCL global memory");
            if (maximumAllocation_ == 0U || globalMemory_ == 0U) {
                throw OpenClLuwbcError(
                    "OpenCL GPU reports zero maximum allocation or global memory");
            }
        } catch (...) {
            releaseResources();
            throw;
        }
    }

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    ~Backend() { releaseResources(); }

    [[nodiscard]] std::string deviceName() const {
        return infoString(device_, CL_DEVICE_NAME);
    }

    [[nodiscard]] int deviceIndex() const noexcept { return deviceIndex_; }

    std::vector<float> runHorizontal(
        const OpenClLuwbcHorizontalRequest& request) {
        std::scoped_lock lock(runMutex_);
        validateHorizontal(request);
        const std::size_t totalPlanes = checkedMultiply(
            request.componentCount, request.layerCount,
            "horizontal component/layer");
        const std::size_t outputCount = checkedMultiply(
            totalPlanes, request.targetPointCount, "horizontal output");
        std::vector<float> result(outputCount);
        if (request.targetPointCount == 0U) return result;

        const std::size_t sourcePlaneBytes =
            bytesFor(request.sourceLayerSize, sizeof(float),
                     "horizontal source plane");
        const std::size_t allocationLimit = maximumAllocation();
        const std::size_t memoryBudget = globalMemoryBudget();
        if (sourcePlaneBytes > allocationLimit ||
            sourcePlaneBytes > memoryBudget) {
            throw OpenClLuwbcError(
                "one luwbc horizontal source plane exceeds the selected GPU's "
                "maximum allocation/global-memory budget");
        }

        const std::size_t fixedPerTarget =
            3U * sizeof(std::uint32_t) + 3U * sizeof(double) +
            sizeof(std::uint32_t);
        const auto maximumPlanesForBudget = [&](std::size_t slotBudget) {
            std::size_t maximumPlanes = std::min(
                {totalPlanes, allocationLimit / sourcePlaneBytes,
                 kMaximumWorkItemsPerSubmission});
            while (maximumPlanes > 0U) {
                const std::size_t sourceBytes = checkedMultiply(
                    maximumPlanes, sourcePlaneBytes,
                    "horizontal source chunk");
                const std::size_t oneTargetBytes = checkedAdd(
                    fixedPerTarget,
                    bytesFor(maximumPlanes, sizeof(float),
                             "horizontal output target"),
                    "horizontal per-target memory");
                if (sourceBytes <= slotBudget &&
                    oneTargetBytes <= slotBudget - sourceBytes &&
                    bytesFor(maximumPlanes, sizeof(float),
                             "horizontal output target") <= allocationLimit) {
                    break;
                }
                --maximumPlanes;
            }
            return maximumPlanes;
        };
        const auto targetChunkForBudget =
            [&](std::size_t planeCount, std::size_t slotBudget) {
            const std::size_t sourceBytes = checkedMultiply(
                planeCount, sourcePlaneBytes, "horizontal source chunk");
            if (sourceBytes >= slotBudget) return std::size_t{0U};
            const std::size_t outputPerTarget = bytesFor(
                planeCount, sizeof(float), "horizontal output per target");
            const std::size_t perTarget = checkedAdd(
                fixedPerTarget, outputPerTarget,
                "horizontal total per-target memory");
            return std::min(
                {request.targetPointCount,
                 allocationLimit / (3U * sizeof(std::uint32_t)),
                 allocationLimit / (3U * sizeof(double)),
                 allocationLimit / sizeof(std::uint32_t),
                 allocationLimit / outputPerTarget,
                 (slotBudget - sourceBytes) / perTarget,
                 kMaximumWorkItemsPerSubmission /
                     std::max<std::size_t>(planeCount, 1U)});
        };

        // Keep a one-submission request synchronous and let it use the whole
        // conservative budget.  Multi-submission requests prefer two equally
        // bounded slots; exceptionally large single-slot work remains
        // supported when half the budget cannot hold one item.
        std::size_t slotBudget = memoryBudget;
        std::size_t activeSlotCount = 1U;
        std::size_t maximumPlanes = maximumPlanesForBudget(slotBudget);
        if (maximumPlanes == 0U ||
            targetChunkForBudget(maximumPlanes, slotBudget) == 0U) {
            throw OpenClLuwbcError(
                "selected OpenCL GPU cannot fit one fused luwbc horizontal work item");
        }
        const bool oneSubmission =
            maximumPlanes == totalPlanes &&
            targetChunkForBudget(maximumPlanes, slotBudget) ==
                request.targetPointCount;
        if (!oneSubmission) {
            const std::size_t pipelineBudget = memoryBudget / 2U;
            const std::size_t pipelinePlanes =
                maximumPlanesForBudget(pipelineBudget);
            if (pipelinePlanes != 0U &&
                targetChunkForBudget(pipelinePlanes, pipelineBudget) != 0U) {
                activeSlotCount = 2U;
                slotBudget = pipelineBudget;
                maximumPlanes = pipelinePlanes;
            }
        }

        std::array<HorizontalPipelineSlot, 2U> slots{
            HorizontalPipelineSlot(queues_[0], horizontalKernels_[0]),
            HorizontalPipelineSlot(queues_[1], horizontalKernels_[1])};

        const auto ensureSlotBuffers =
            [&](HorizontalPipelineSlot& slot, std::size_t planeStart,
                std::size_t planeCount, std::size_t targetCapacity) {
                if (slot.pending) {
                    throw OpenClLuwbcError(
                        "internal luwbc horizontal pipeline slot reuse before completion");
                }
                if (slot.sourcePlaneStart == planeStart &&
                    slot.sourcePlaneCount == planeCount &&
                    slot.targetCapacity >= targetCapacity) {
                    return;
                }

                // The previous read event has completed before a slot becomes
                // reusable, so release its old capacity before reserving the
                // next plane group.  This keeps the transient peak within the
                // same two-slot budget used by the chunk calculation.
                slot.sourceReady.reset();
                slot.source.reset();
                slot.indices.reset();
                slot.weights.reset();
                slot.counts.reset();
                slot.output.reset();
                slot.sourcePlaneCount = 0U;
                slot.targetCapacity = 0U;

                const std::size_t sourceElements = checkedMultiply(
                    planeCount, request.sourceLayerSize,
                    "horizontal source chunk");
                const std::size_t sourceBytes = bytesFor(
                    sourceElements, sizeof(float), "horizontal source chunk");
                const std::size_t tripleCapacity = checkedMultiply(
                    targetCapacity, 3U, "horizontal pipeline plan capacity");
                const std::size_t outputCapacity = checkedMultiply(
                    planeCount, targetCapacity,
                    "horizontal pipeline output capacity");

                Buffer source(createBuffer(
                    CL_MEM_READ_ONLY, sourceBytes, nullptr,
                    "allocating GPU luwbc horizontal pipeline source buffer"));
                Buffer indices(createBuffer(
                    CL_MEM_READ_ONLY,
                    bytesFor(tripleCapacity, sizeof(std::uint32_t),
                             "horizontal pipeline index capacity"),
                    nullptr,
                    "allocating GPU luwbc horizontal pipeline index buffer"));
                Buffer weights(createBuffer(
                    CL_MEM_READ_ONLY,
                    bytesFor(tripleCapacity, sizeof(double),
                             "horizontal pipeline weight capacity"),
                    nullptr,
                    "allocating GPU luwbc horizontal pipeline weight buffer"));
                Buffer counts(createBuffer(
                    CL_MEM_READ_ONLY,
                    bytesFor(targetCapacity, sizeof(std::uint32_t),
                             "horizontal pipeline count capacity"),
                    nullptr,
                    "allocating GPU luwbc horizontal pipeline count buffer"));
                Buffer output(createBuffer(
                    CL_MEM_WRITE_ONLY,
                    bytesFor(outputCapacity, sizeof(float),
                             "horizontal pipeline output capacity"),
                    nullptr,
                    "allocating GPU luwbc horizontal pipeline output buffer"));

                slot.source = std::move(source);
                slot.indices = std::move(indices);
                slot.weights = std::move(weights);
                slot.counts = std::move(counts);
                slot.output = std::move(output);
                slot.sourcePlaneStart = planeStart;
                slot.sourcePlaneCount = planeCount;
                slot.targetCapacity = targetCapacity;
                slot.pending = true;
                requireSuccess(
                    clEnqueueWriteBuffer(
                        slot.queue, slot.source.value(), CL_FALSE, 0,
                        sourceBytes,
                        request.values.data() +
                            planeStart * request.sourceLayerSize,
                        0, nullptr, slot.sourceReady.put()),
                    "uploading GPU luwbc horizontal pipeline source");
            };

        const auto submit =
            [&](HorizontalPipelineSlot& slot, std::size_t planeStart,
                std::size_t planeCount, std::size_t targetStart,
                std::size_t count, std::size_t targetCapacity) {
                ensureSlotBuffers(slot, planeStart, planeCount,
                                  targetCapacity);
                const std::size_t tripleCount = checkedMultiply(
                    count, 3U, "horizontal weight chunk");
                const std::size_t outputElements = checkedMultiply(
                    planeCount, count, "horizontal output chunk");
                slot.hostOutput.resize(outputElements);
                slot.targetStart = targetStart;
                slot.targetCount = count;
                for (Event& event : slot.inputReady) event.reset();
                slot.kernelDone.reset();
                slot.readDone.reset();
                slot.pending = true;

                requireSuccess(
                    clEnqueueWriteBuffer(
                        slot.queue, slot.indices.value(), CL_FALSE, 0,
                        bytesFor(tripleCount, sizeof(std::uint32_t),
                                 "horizontal index upload"),
                        request.indices.data() + targetStart * 3U, 0, nullptr,
                        slot.inputReady[0].put()),
                    "uploading GPU luwbc horizontal indices");
                requireSuccess(
                    clEnqueueWriteBuffer(
                        slot.queue, slot.weights.value(), CL_FALSE, 0,
                        bytesFor(tripleCount, sizeof(double),
                                 "horizontal weight upload"),
                        request.weights.data() + targetStart * 3U, 0, nullptr,
                        slot.inputReady[1].put()),
                    "uploading GPU luwbc horizontal weights");
                requireSuccess(
                    clEnqueueWriteBuffer(
                        slot.queue, slot.counts.value(), CL_FALSE, 0,
                        bytesFor(count, sizeof(std::uint32_t),
                                 "horizontal count upload"),
                        request.counts.data() + targetStart, 0, nullptr,
                        slot.inputReady[2].put()),
                    "uploading GPU luwbc horizontal counts");

                const cl_ulong sourceLayerSize =
                    static_cast<cl_ulong>(request.sourceLayerSize);
                const cl_ulong targetCount = static_cast<cl_ulong>(count);
                cl_uint argument = 0U;
                setKernelArgument(slot.kernel, argument++, slot.source.value(),
                                  "setting luwbc horizontal source");
                setKernelArgument(slot.kernel, argument++, slot.indices.value(),
                                  "setting luwbc horizontal indices");
                setKernelArgument(slot.kernel, argument++, slot.weights.value(),
                                  "setting luwbc horizontal weights");
                setKernelArgument(slot.kernel, argument++, slot.counts.value(),
                                  "setting luwbc horizontal counts");
                setKernelArgument(slot.kernel, argument++, sourceLayerSize,
                                  "setting luwbc horizontal source size");
                setKernelArgument(slot.kernel, argument++, targetCount,
                                  "setting luwbc horizontal target count");
                setKernelArgument(slot.kernel, argument++, slot.output.value(),
                                  "setting luwbc horizontal output");

                const std::array<cl_event, 4U> ready{
                    slot.sourceReady.value(), slot.inputReady[0].value(),
                    slot.inputReady[1].value(), slot.inputReady[2].value()};
                const std::size_t global = outputElements;
                requireSuccess(
                    clEnqueueNDRangeKernel(
                        slot.queue, slot.kernel, 1, nullptr, &global, nullptr,
                        static_cast<cl_uint>(ready.size()), ready.data(),
                        slot.kernelDone.put()),
                    "running GPU luwbc horizontal pipeline kernel");
                const cl_event kernelDone = slot.kernelDone.value();
                requireSuccess(
                    clEnqueueReadBuffer(
                        slot.queue, slot.output.value(), CL_FALSE, 0,
                        bytesFor(outputElements, sizeof(float),
                                 "horizontal output read"),
                        slot.hostOutput.data(), 1U, &kernelDone,
                        slot.readDone.put()),
                    "reading GPU luwbc horizontal pipeline output");
                requireSuccess(clFlush(slot.queue),
                               "submitting GPU luwbc horizontal pipeline slot");
            };

        const auto collect = [&](HorizontalPipelineSlot& slot) {
            const cl_event readDone = slot.readDone.value();
            if (readDone == nullptr) {
                throw OpenClLuwbcError(
                    "internal luwbc horizontal pipeline completion event is missing");
            }
            requireSuccess(clWaitForEvents(1U, &readDone),
                           "waiting for GPU luwbc horizontal pipeline slot");
            for (std::size_t plane = 0U; plane < slot.sourcePlaneCount;
                 ++plane) {
                std::copy_n(
                    slot.hostOutput.data() + plane * slot.targetCount,
                    slot.targetCount,
                    result.data() +
                        (slot.sourcePlaneStart + plane) *
                            request.targetPointCount +
                        slot.targetStart);
            }
            slot.pending = false;
            for (Event& event : slot.inputReady) event.reset();
            slot.kernelDone.reset();
            slot.readDone.reset();
        };

        std::size_t submission = 0U;
        for (std::size_t planeStart = 0U; planeStart < totalPlanes;
             planeStart += maximumPlanes) {
            const std::size_t planeCount =
                std::min(maximumPlanes, totalPlanes - planeStart);
            const std::size_t targetChunk =
                targetChunkForBudget(planeCount, slotBudget);
            if (targetChunk == 0U) {
                throw OpenClLuwbcError(
                    "selected OpenCL GPU cannot fit one luwbc horizontal target");
            }

            for (std::size_t targetStart = 0U;
                 targetStart < request.targetPointCount;
                 targetStart += targetChunk) {
                const std::size_t count = std::min(
                    targetChunk, request.targetPointCount - targetStart);
                HorizontalPipelineSlot& current =
                    slots[submission % activeSlotCount];
                // Both slots may stay in flight.  Drain only the slot about
                // to be reused; the other queue can continue while this wait
                // completes, and no slot metadata/buffer is overwritten early.
                if (current.pending) collect(current);
                submit(current, planeStart, planeCount, targetStart, count,
                       targetChunk);
                ++submission;
            }
        }
        // submission % slot-count identifies the oldest still-pending slot,
        // so the final drain also materializes chunks in submission order.
        for (std::size_t offset = 0U; offset < activeSlotCount; ++offset) {
            HorizontalPipelineSlot& slot =
                slots[(submission + offset) % activeSlotCount];
            if (slot.pending) collect(slot);
        }
        return result;
    }

    std::vector<float> runVertical(
        const OpenClLuwbcVerticalRequest& request) {
        std::scoped_lock lock(runMutex_);
        validateVertical(request);
        const std::size_t sourceLevelCount = request.sourceLevels.size();
        const std::size_t targetLevelCount = request.targetLevels.size();
        const std::size_t outputPlanes = checkedMultiply(
            request.componentCount, targetLevelCount,
            "vertical output planes");
        const std::size_t outputCount = checkedMultiply(
            outputPlanes, request.pointCount, "vertical output");
        std::vector<float> result(outputCount);
        if (request.pointCount == 0U || targetLevelCount == 0U) return result;

        const std::shared_ptr<const VerticalPlan> verticalPlan =
            verticalPlanFor(request.sourceLevels, request.targetLevels);
        const std::vector<std::uint32_t>& lower = verticalPlan->lower;
        const std::vector<std::uint32_t>& upper = verticalPlan->upper;
        const std::vector<double>& fraction = verticalPlan->fraction;

        const std::size_t allocationLimit = maximumAllocation();
        const std::size_t memoryBudget = globalMemoryBudget();
        const std::size_t planBytes = checkedMultiply(
            targetLevelCount,
            2U * sizeof(std::uint32_t) + sizeof(double),
            "vertical interpolation plan");
        const std::size_t singlePlanAllocation = bytesFor(
            targetLevelCount, sizeof(std::uint32_t),
            "vertical plan allocation");
        if (singlePlanAllocation > allocationLimit ||
            bytesFor(targetLevelCount, sizeof(double),
                     "vertical fraction allocation") > allocationLimit ||
            planBytes >= memoryBudget) {
            throw OpenClLuwbcError(
                "luwbc vertical interpolation plan exceeds the selected GPU's "
                "maximum allocation/global-memory budget");
        }

        const std::size_t sourcePerPoint = checkedMultiply(
            checkedMultiply(request.componentCount, sourceLevelCount,
                            "vertical source planes"),
            sizeof(float), "vertical source bytes per point");
        const std::size_t outputPerPoint = checkedMultiply(
            outputPlanes, sizeof(float), "vertical output bytes per point");
        const std::size_t bytesPerPoint = checkedAdd(
            sourcePerPoint, outputPerPoint, "vertical bytes per point");
        const auto pointChunkForBudget = [&](std::size_t slotBudget) {
            if (planBytes >= slotBudget) return std::size_t{0U};
            return std::min(
                {request.pointCount,
                 allocationLimit / sourcePerPoint,
                 allocationLimit / outputPerPoint,
                 (slotBudget - planBytes) / bytesPerPoint,
                 kMaximumWorkItemsPerSubmission /
                     std::max<std::size_t>(outputPlanes, 1U)});
        };

        std::size_t slotBudget = memoryBudget;
        std::size_t activeSlotCount = 1U;
        std::size_t pointChunk = pointChunkForBudget(slotBudget);
        if (pointChunk == 0U) {
            throw OpenClLuwbcError(
                "selected OpenCL GPU cannot fit one fused luwbc vertical point");
        }
        if (pointChunk != request.pointCount) {
            const std::size_t pipelineBudget = memoryBudget / 2U;
            const std::size_t pipelineChunk =
                pointChunkForBudget(pipelineBudget);
            if (pipelineChunk != 0U) {
                activeSlotCount = 2U;
                pointChunk = pipelineChunk;
            }
        }

        std::array<VerticalPipelineSlot, 2U> slots{
            VerticalPipelineSlot(queues_[0], verticalKernels_[0]),
            VerticalPipelineSlot(queues_[1], verticalKernels_[1])};

        const auto ensureSlotBuffers =
            [&](VerticalPipelineSlot& slot, std::size_t pointCapacity) {
                if (slot.pending) {
                    throw OpenClLuwbcError(
                        "internal luwbc vertical pipeline slot reuse before completion");
                }
                if (slot.planUploaded &&
                    slot.pointCapacity >= pointCapacity) {
                    return;
                }

                const std::size_t sourceCapacity = checkedMultiply(
                    request.componentCount,
                    checkedMultiply(sourceLevelCount, pointCapacity,
                                    "vertical packed source capacity"),
                    "vertical packed source component capacity");
                const std::size_t outputCapacity = checkedMultiply(
                    outputPlanes, pointCapacity,
                    "vertical pipeline output capacity");
                Buffer lowerBuffer(createBuffer(
                    CL_MEM_READ_ONLY,
                    bytesFor(targetLevelCount, sizeof(std::uint32_t),
                             "vertical lower plan"),
                    nullptr,
                    "allocating GPU luwbc vertical lower-index buffer"));
                Buffer upperBuffer(createBuffer(
                    CL_MEM_READ_ONLY,
                    bytesFor(targetLevelCount, sizeof(std::uint32_t),
                             "vertical upper plan"),
                    nullptr,
                    "allocating GPU luwbc vertical upper-index buffer"));
                Buffer fractionBuffer(createBuffer(
                    CL_MEM_READ_ONLY,
                    bytesFor(targetLevelCount, sizeof(double),
                             "vertical fraction plan"),
                    nullptr,
                    "allocating GPU luwbc vertical fraction buffer"));
                Buffer source(createBuffer(
                    CL_MEM_READ_ONLY,
                    bytesFor(sourceCapacity, sizeof(float),
                             "vertical packed source capacity"),
                    nullptr,
                    "allocating GPU luwbc vertical pipeline source buffer"));
                Buffer output(createBuffer(
                    CL_MEM_WRITE_ONLY,
                    bytesFor(outputCapacity, sizeof(float),
                             "vertical output capacity"),
                    nullptr,
                    "allocating GPU luwbc vertical pipeline output buffer"));

                slot.lower = std::move(lowerBuffer);
                slot.upper = std::move(upperBuffer);
                slot.fraction = std::move(fractionBuffer);
                slot.source = std::move(source);
                slot.output = std::move(output);
                slot.pointCapacity = pointCapacity;
                for (Event& event : slot.planReady) event.reset();
                slot.planUploaded = false;
                slot.pending = true;
                requireSuccess(
                    clEnqueueWriteBuffer(
                        slot.queue, slot.lower.value(), CL_FALSE, 0,
                        bytesFor(targetLevelCount, sizeof(std::uint32_t),
                                 "vertical lower plan upload"),
                        lower.data(), 0, nullptr, slot.planReady[0].put()),
                    "uploading GPU luwbc vertical lower plan");
                requireSuccess(
                    clEnqueueWriteBuffer(
                        slot.queue, slot.upper.value(), CL_FALSE, 0,
                        bytesFor(targetLevelCount, sizeof(std::uint32_t),
                                 "vertical upper plan upload"),
                        upper.data(), 0, nullptr, slot.planReady[1].put()),
                    "uploading GPU luwbc vertical upper plan");
                requireSuccess(
                    clEnqueueWriteBuffer(
                        slot.queue, slot.fraction.value(), CL_FALSE, 0,
                        bytesFor(targetLevelCount, sizeof(double),
                                 "vertical fraction plan upload"),
                        fraction.data(), 0, nullptr, slot.planReady[2].put()),
                    "uploading GPU luwbc vertical fraction plan");
                slot.planUploaded = true;
            };

        const auto submit =
            [&](VerticalPipelineSlot& slot, std::size_t pointStart,
                std::size_t count) {
                ensureSlotBuffers(slot, pointChunk);
                const std::size_t sourceElements = checkedMultiply(
                    request.componentCount,
                    checkedMultiply(sourceLevelCount, count,
                                    "vertical packed source levels"),
                    "vertical packed source components");
                slot.hostSource.resize(sourceElements);
                for (std::size_t component = 0U;
                     component < request.componentCount; ++component) {
                    for (std::size_t level = 0U; level < sourceLevelCount;
                         ++level) {
                        const std::size_t sourceOffset =
                            (component * sourceLevelCount + level) *
                                request.pointCount +
                            pointStart;
                        const std::size_t packedOffset =
                            (component * sourceLevelCount + level) * count;
                        std::copy_n(request.values.data() + sourceOffset,
                                    count,
                                    slot.hostSource.data() + packedOffset);
                    }
                }

                const std::size_t outputElements = checkedMultiply(
                    outputPlanes, count, "vertical output chunk");
                slot.hostOutput.resize(outputElements);
                slot.pointStart = pointStart;
                slot.pointCount = count;
                slot.sourceReady.reset();
                slot.kernelDone.reset();
                slot.readDone.reset();
                slot.pending = true;
                requireSuccess(
                    clEnqueueWriteBuffer(
                        slot.queue, slot.source.value(), CL_FALSE, 0,
                        bytesFor(sourceElements, sizeof(float),
                                 "vertical packed source upload"),
                        slot.hostSource.data(), 0, nullptr,
                        slot.sourceReady.put()),
                    "uploading GPU luwbc vertical packed source");

                const cl_ulong sourceLevels =
                    static_cast<cl_ulong>(sourceLevelCount);
                const cl_ulong targetLevels =
                    static_cast<cl_ulong>(targetLevelCount);
                const cl_ulong pointCount = static_cast<cl_ulong>(count);
                cl_uint argument = 0U;
                setKernelArgument(slot.kernel, argument++, slot.source.value(),
                                  "setting luwbc vertical source");
                setKernelArgument(slot.kernel, argument++, slot.lower.value(),
                                  "setting luwbc vertical lower indices");
                setKernelArgument(slot.kernel, argument++, slot.upper.value(),
                                  "setting luwbc vertical upper indices");
                setKernelArgument(slot.kernel, argument++, slot.fraction.value(),
                                  "setting luwbc vertical fractions");
                setKernelArgument(slot.kernel, argument++, sourceLevels,
                                  "setting luwbc vertical source level count");
                setKernelArgument(slot.kernel, argument++, targetLevels,
                                  "setting luwbc vertical target level count");
                setKernelArgument(slot.kernel, argument++, pointCount,
                                  "setting luwbc vertical point count");
                setKernelArgument(slot.kernel, argument++, slot.output.value(),
                                  "setting luwbc vertical output");

                const std::array<cl_event, 4U> ready{
                    slot.planReady[0].value(), slot.planReady[1].value(),
                    slot.planReady[2].value(), slot.sourceReady.value()};
                const std::size_t global = outputElements;
                requireSuccess(
                    clEnqueueNDRangeKernel(
                        slot.queue, slot.kernel, 1, nullptr, &global, nullptr,
                        static_cast<cl_uint>(ready.size()), ready.data(),
                        slot.kernelDone.put()),
                    "running GPU luwbc vertical pipeline kernel");
                const cl_event kernelDone = slot.kernelDone.value();
                requireSuccess(
                    clEnqueueReadBuffer(
                        slot.queue, slot.output.value(), CL_FALSE, 0,
                        bytesFor(outputElements, sizeof(float),
                                 "vertical output read"),
                        slot.hostOutput.data(), 1U, &kernelDone,
                        slot.readDone.put()),
                    "reading GPU luwbc vertical pipeline output");
                requireSuccess(clFlush(slot.queue),
                               "submitting GPU luwbc vertical pipeline slot");
            };

        const auto collect = [&](VerticalPipelineSlot& slot) {
            const cl_event readDone = slot.readDone.value();
            if (readDone == nullptr) {
                throw OpenClLuwbcError(
                    "internal luwbc vertical pipeline completion event is missing");
            }
            requireSuccess(clWaitForEvents(1U, &readDone),
                           "waiting for GPU luwbc vertical pipeline slot");
            for (std::size_t plane = 0U; plane < outputPlanes; ++plane) {
                std::copy_n(
                    slot.hostOutput.data() + plane * slot.pointCount,
                    slot.pointCount,
                    result.data() + plane * request.pointCount +
                        slot.pointStart);
            }
            slot.pending = false;
            slot.sourceReady.reset();
            slot.kernelDone.reset();
            slot.readDone.reset();
        };

        std::size_t submission = 0U;
        for (std::size_t pointStart = 0U; pointStart < request.pointCount;
             pointStart += pointChunk) {
            const std::size_t count =
                std::min(pointChunk, request.pointCount - pointStart);
            VerticalPipelineSlot& current =
                slots[submission % activeSlotCount];
            if (current.pending) collect(current);
            submit(current, pointStart, count);
            ++submission;
        }
        for (std::size_t offset = 0U; offset < activeSlotCount; ++offset) {
            VerticalPipelineSlot& slot =
                slots[(submission + offset) % activeSlotCount];
            if (slot.pending) collect(slot);
        }
        return result;
    }

    std::vector<double> runDemIdw(
        const OpenClLuwbcDemIdwRequest& request) {
        std::scoped_lock lock(runMutex_);
        validateDemIdw(request);
        const std::size_t targetCount = request.targetX.size();
        std::vector<double> result(targetCount);
        if (targetCount == 0U) return result;

        const std::size_t sourceCount = request.sourceX.size();
        const std::size_t allocationLimit = maximumAllocation();
        const std::size_t memoryBudget = globalMemoryBudget();
        const std::size_t sourceArrayBytes =
            bytesFor(sourceCount, sizeof(double), "DEM IDW source array");
        const std::size_t sourceBytes = checkedMultiply(
            sourceArrayBytes, 3U, "DEM IDW source arrays");
        if (sourceArrayBytes > allocationLimit || sourceBytes >= memoryBudget) {
            throw OpenClLuwbcError(
                "luwbc DEM IDW source data exceeds the selected GPU's "
                "maximum allocation/global-memory budget");
        }

        Buffer sourceX(createBuffer(
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sourceArrayBytes,
            const_cast<double*>(request.sourceX.data()),
            "allocating GPU luwbc DEM IDW source-X buffer"));
        Buffer sourceY(createBuffer(
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sourceArrayBytes,
            const_cast<double*>(request.sourceY.data()),
            "allocating GPU luwbc DEM IDW source-Y buffer"));
        Buffer sourceValues(createBuffer(
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sourceArrayBytes,
            const_cast<double*>(request.sourceValues.data()),
            "allocating GPU luwbc DEM IDW source-value buffer"));

        const std::size_t indexBytesPerTarget = bytesFor(
            request.neighborCount, sizeof(std::uint32_t),
            "DEM IDW indices per target");
        const std::size_t bytesPerTarget = checkedAdd(
            checkedAdd(2U * sizeof(double), indexBytesPerTarget,
                       "DEM IDW target input"),
            sizeof(double), "DEM IDW target input/output");
        const std::size_t available = memoryBudget - sourceBytes;
        const std::size_t targetChunk = std::min(
            {targetCount,
             allocationLimit / sizeof(double),
             allocationLimit / indexBytesPerTarget,
             available / bytesPerTarget,
             kMaximumSpatialTargetsPerSubmission});
        if (targetChunk == 0U) {
            throw OpenClLuwbcError(
                "selected OpenCL GPU cannot fit one luwbc DEM IDW target");
        }

        for (std::size_t targetStart = 0U; targetStart < targetCount;
             targetStart += targetChunk) {
            const std::size_t count =
                std::min(targetChunk, targetCount - targetStart);
            const std::size_t coordinateBytes =
                bytesFor(count, sizeof(double), "DEM IDW target coordinates");
            const std::size_t indexCount = checkedMultiply(
                count, request.neighborCount, "DEM IDW target indices");
            Buffer targetX(createBuffer(
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, coordinateBytes,
                const_cast<double*>(request.targetX.data() + targetStart),
                "allocating GPU luwbc DEM IDW target-X buffer"));
            Buffer targetY(createBuffer(
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, coordinateBytes,
                const_cast<double*>(request.targetY.data() + targetStart),
                "allocating GPU luwbc DEM IDW target-Y buffer"));
            Buffer indices(createBuffer(
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                bytesFor(indexCount, sizeof(std::uint32_t),
                         "DEM IDW target index chunk"),
                const_cast<std::uint32_t*>(
                    request.neighborIndices.data() +
                    targetStart * request.neighborCount),
                "allocating GPU luwbc DEM IDW index buffer"));
            Buffer output(createBuffer(
                CL_MEM_WRITE_ONLY, coordinateBytes, nullptr,
                "allocating GPU luwbc DEM IDW output buffer"));

            const cl_uint neighborCount =
                static_cast<cl_uint>(request.neighborCount);
            cl_uint argument = 0U;
            setKernelArgument(demIdwKernel_, argument++, sourceX.value(),
                              "setting luwbc DEM IDW source X");
            setKernelArgument(demIdwKernel_, argument++, sourceY.value(),
                              "setting luwbc DEM IDW source Y");
            setKernelArgument(demIdwKernel_, argument++, sourceValues.value(),
                              "setting luwbc DEM IDW source values");
            setKernelArgument(demIdwKernel_, argument++, targetX.value(),
                              "setting luwbc DEM IDW target X");
            setKernelArgument(demIdwKernel_, argument++, targetY.value(),
                              "setting luwbc DEM IDW target Y");
            setKernelArgument(demIdwKernel_, argument++, indices.value(),
                              "setting luwbc DEM IDW neighbor indices");
            setKernelArgument(demIdwKernel_, argument++, neighborCount,
                              "setting luwbc DEM IDW neighbor count");
            setKernelArgument(demIdwKernel_, argument++, output.value(),
                              "setting luwbc DEM IDW output");
            const std::size_t global = count;
            requireSuccess(clEnqueueNDRangeKernel(
                               queues_[0], demIdwKernel_, 1, nullptr, &global,
                               nullptr, 0, nullptr, nullptr),
                           "running GPU luwbc DEM IDW kernel");
            requireSuccess(clEnqueueReadBuffer(
                               queues_[0], output.value(), CL_TRUE, 0,
                               coordinateBytes, result.data() + targetStart,
                               0, nullptr, nullptr),
                           "reading GPU luwbc DEM IDW output");
        }
        return result;
    }

    std::vector<float> runMlsWeights(
        const OpenClLuwbcMlsRequest& request) {
        std::scoped_lock lock(runMutex_);
        validateMls(request);
        const std::size_t targetCount = request.targetX.size();
        const std::size_t outputCount = checkedMultiply(
            targetCount, request.neighborCount, "MLS output weights");
        std::vector<float> result(outputCount);
        if (targetCount == 0U) return result;

        const std::size_t sourceCount = request.sourceX.size();
        const std::size_t allocationLimit = maximumAllocation();
        const std::size_t memoryBudget = globalMemoryBudget();
        const std::size_t sourceArrayBytes =
            bytesFor(sourceCount, sizeof(double), "MLS source array");
        const std::size_t sourceBytes = checkedMultiply(
            sourceArrayBytes, 2U, "MLS source arrays");
        if (sourceArrayBytes > allocationLimit || sourceBytes >= memoryBudget) {
            throw OpenClLuwbcError(
                "luwbc MLS source data exceeds the selected GPU's maximum "
                "allocation/global-memory budget");
        }

        Buffer sourceX(createBuffer(
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sourceArrayBytes,
            const_cast<double*>(request.sourceX.data()),
            "allocating GPU luwbc MLS source-X buffer"));
        Buffer sourceY(createBuffer(
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sourceArrayBytes,
            const_cast<double*>(request.sourceY.data()),
            "allocating GPU luwbc MLS source-Y buffer"));

        const std::size_t indexBytesPerTarget = bytesFor(
            request.neighborCount, sizeof(std::uint32_t),
            "MLS indices per target");
        const std::size_t outputBytesPerTarget = bytesFor(
            request.neighborCount, sizeof(float), "MLS output per target");
        const std::size_t bytesPerTarget = checkedAdd(
            checkedAdd(2U * sizeof(double), indexBytesPerTarget,
                       "MLS target input"),
            outputBytesPerTarget, "MLS target input/output");
        const std::size_t available = memoryBudget - sourceBytes;
        const std::size_t targetChunk = std::min(
            {targetCount,
             allocationLimit / sizeof(double),
             allocationLimit / indexBytesPerTarget,
             allocationLimit / outputBytesPerTarget,
             available / bytesPerTarget,
             kMaximumSpatialTargetsPerSubmission});
        if (targetChunk == 0U) {
            throw OpenClLuwbcError(
                "selected OpenCL GPU cannot fit one luwbc MLS target");
        }

        for (std::size_t targetStart = 0U; targetStart < targetCount;
             targetStart += targetChunk) {
            const std::size_t count =
                std::min(targetChunk, targetCount - targetStart);
            const std::size_t coordinateBytes =
                bytesFor(count, sizeof(double), "MLS target coordinates");
            const std::size_t planCount = checkedMultiply(
                count, request.neighborCount, "MLS target plan");
            Buffer targetX(createBuffer(
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, coordinateBytes,
                const_cast<double*>(request.targetX.data() + targetStart),
                "allocating GPU luwbc MLS target-X buffer"));
            Buffer targetY(createBuffer(
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, coordinateBytes,
                const_cast<double*>(request.targetY.data() + targetStart),
                "allocating GPU luwbc MLS target-Y buffer"));
            Buffer indices(createBuffer(
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                bytesFor(planCount, sizeof(std::uint32_t),
                         "MLS target index chunk"),
                const_cast<std::uint32_t*>(
                    request.neighborIndices.data() +
                    targetStart * request.neighborCount),
                "allocating GPU luwbc MLS index buffer"));
            Buffer output(createBuffer(
                CL_MEM_WRITE_ONLY,
                bytesFor(planCount, sizeof(float), "MLS output chunk"),
                nullptr, "allocating GPU luwbc MLS output buffer"));

            const cl_uint neighborCount =
                static_cast<cl_uint>(request.neighborCount);
            cl_uint argument = 0U;
            setKernelArgument(mlsKernel_, argument++, sourceX.value(),
                              "setting luwbc MLS source X");
            setKernelArgument(mlsKernel_, argument++, sourceY.value(),
                              "setting luwbc MLS source Y");
            setKernelArgument(mlsKernel_, argument++, targetX.value(),
                              "setting luwbc MLS target X");
            setKernelArgument(mlsKernel_, argument++, targetY.value(),
                              "setting luwbc MLS target Y");
            setKernelArgument(mlsKernel_, argument++, indices.value(),
                              "setting luwbc MLS neighbor indices");
            setKernelArgument(mlsKernel_, argument++, neighborCount,
                              "setting luwbc MLS neighbor count");
            setKernelArgument(mlsKernel_, argument++, output.value(),
                              "setting luwbc MLS output");
            const std::size_t global = count;
            requireSuccess(clEnqueueNDRangeKernel(
                               queues_[0], mlsKernel_, 1, nullptr, &global,
                               nullptr, 0, nullptr, nullptr),
                           "running GPU luwbc MLS kernel");
            requireSuccess(clEnqueueReadBuffer(
                               queues_[0], output.value(), CL_TRUE, 0,
                               bytesFor(planCount, sizeof(float),
                                        "MLS output read"),
                               result.data() +
                                   targetStart * request.neighborCount,
                               0, nullptr, nullptr),
                           "reading GPU luwbc MLS output");
        }
        return result;
    }

private:
    class Buffer final {
    public:
        Buffer() = default;
        explicit Buffer(cl_mem value) : value_(value) {}
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        Buffer(Buffer&& other) noexcept
            : value_(std::exchange(other.value_, nullptr)) {}
        Buffer& operator=(Buffer&& other) noexcept {
            if (this != &other) {
                reset();
                value_ = std::exchange(other.value_, nullptr);
            }
            return *this;
        }
        ~Buffer() { reset(); }

        void reset() noexcept {
            if (value_ != nullptr) clReleaseMemObject(value_);
            value_ = nullptr;
        }
        [[nodiscard]] cl_mem value() const noexcept { return value_; }

    private:
        cl_mem value_ = nullptr;
    };

    class Event final {
    public:
        Event() = default;
        Event(const Event&) = delete;
        Event& operator=(const Event&) = delete;
        Event(Event&& other) noexcept
            : value_(std::exchange(other.value_, nullptr)) {}
        Event& operator=(Event&& other) noexcept {
            if (this != &other) {
                reset();
                value_ = std::exchange(other.value_, nullptr);
            }
            return *this;
        }
        ~Event() { reset(); }

        void reset() noexcept {
            if (value_ != nullptr) clReleaseEvent(value_);
            value_ = nullptr;
        }

        [[nodiscard]] cl_event value() const noexcept { return value_; }

        // Reset before handing storage to OpenCL so an enqueue failure cannot
        // leak either a previous event or a partially returned new event.
        [[nodiscard]] cl_event* put() noexcept {
            reset();
            return &value_;
        }

    private:
        cl_event value_ = nullptr;
    };

    struct VerticalPlan final {
        std::vector<double> sourceLevels;
        std::vector<double> targetLevels;
        std::vector<std::uint32_t> lower;
        std::vector<std::uint32_t> upper;
        std::vector<double> fraction;
        std::size_t payloadBytes = 0U;
    };

    [[nodiscard]] static bool exactAxisMatch(
        const std::vector<double>& cached,
        const std::vector<double>& requested) noexcept {
        return cached.size() == requested.size() &&
               (cached.empty() ||
                std::memcmp(cached.data(), requested.data(),
                            cached.size() * sizeof(double)) == 0);
    }

    [[nodiscard]] std::shared_ptr<const VerticalPlan> verticalPlanFor(
        const std::vector<double>& sourceLevels,
        const std::vector<double>& targetLevels) {
        // runMutex_ serializes this per-device LRU together with kernel use.
        // Bytewise comparison deliberately distinguishes every bit pattern
        // (including +0/-0), so a changed axis can never reuse a stale plan.
        for (auto iterator = verticalPlanCache_.begin();
             iterator != verticalPlanCache_.end(); ++iterator) {
            if (exactAxisMatch((*iterator)->sourceLevels, sourceLevels) &&
                exactAxisMatch((*iterator)->targetLevels, targetLevels)) {
                const std::shared_ptr<const VerticalPlan> hit = *iterator;
                verticalPlanCache_.splice(verticalPlanCache_.begin(),
                                          verticalPlanCache_, iterator);
                return hit;
            }
        }

        auto plan = std::make_shared<VerticalPlan>();
        plan->sourceLevels = sourceLevels;
        plan->targetLevels = targetLevels;
        plan->lower.resize(targetLevels.size());
        plan->upper.resize(targetLevels.size());
        plan->fraction.resize(targetLevels.size());
        buildVerticalPlan(sourceLevels, targetLevels, plan->lower, plan->upper,
                          plan->fraction);
        plan->payloadBytes = checkedAdd(
            bytesFor(sourceLevels.size(), sizeof(double),
                     "vertical plan cache source axis"),
            checkedAdd(
                bytesFor(targetLevels.size(), sizeof(double),
                         "vertical plan cache target axis"),
                checkedMultiply(
                    targetLevels.size(),
                    2U * sizeof(std::uint32_t) + sizeof(double),
                    "vertical plan cache interpolation plan"),
                "vertical plan cache target payload"),
            "vertical plan cache payload");

        constexpr std::size_t kMaximumVerticalPlanCacheEntries = 8U;
        constexpr std::size_t kMaximumVerticalPlanCacheBytes = 8U << 20U;
        if (plan->payloadBytes <= kMaximumVerticalPlanCacheBytes) {
            while (!verticalPlanCache_.empty() &&
                   (verticalPlanCache_.size() >=
                        kMaximumVerticalPlanCacheEntries ||
                    verticalPlanCacheBytes_ >
                        kMaximumVerticalPlanCacheBytes -
                            plan->payloadBytes)) {
                verticalPlanCacheBytes_ -=
                    verticalPlanCache_.back()->payloadBytes;
                verticalPlanCache_.pop_back();
            }
            verticalPlanCacheBytes_ += plan->payloadBytes;
            verticalPlanCache_.push_front(plan);
        }
        return plan;
    }

    struct HorizontalPipelineSlot final {
        HorizontalPipelineSlot(cl_command_queue slotQueue, cl_kernel slotKernel)
            : queue(slotQueue), kernel(slotKernel) {}
        HorizontalPipelineSlot(const HorizontalPipelineSlot&) = delete;
        HorizontalPipelineSlot& operator=(const HorizontalPipelineSlot&) = delete;
        ~HorizontalPipelineSlot() {
            // Non-blocking transfers borrow host vector storage.  If any
            // enqueue below throws, drain this slot before those vectors are
            // destroyed; OpenCL command objects retain their device buffers.
            if (pending && queue != nullptr) (void)clFinish(queue);
        }

        cl_command_queue queue = nullptr;
        cl_kernel kernel = nullptr;
        Buffer source;
        Buffer indices;
        Buffer weights;
        Buffer counts;
        Buffer output;
        Event sourceReady;
        std::array<Event, 3U> inputReady;
        Event kernelDone;
        Event readDone;
        std::vector<float> hostOutput;
        std::size_t sourcePlaneStart = 0U;
        std::size_t sourcePlaneCount = 0U;
        std::size_t targetCapacity = 0U;
        std::size_t targetStart = 0U;
        std::size_t targetCount = 0U;
        bool pending = false;
    };

    struct VerticalPipelineSlot final {
        VerticalPipelineSlot(cl_command_queue slotQueue, cl_kernel slotKernel)
            : queue(slotQueue), kernel(slotKernel) {}
        VerticalPipelineSlot(const VerticalPipelineSlot&) = delete;
        VerticalPipelineSlot& operator=(const VerticalPipelineSlot&) = delete;
        ~VerticalPipelineSlot() {
            if (pending && queue != nullptr) (void)clFinish(queue);
        }

        cl_command_queue queue = nullptr;
        cl_kernel kernel = nullptr;
        Buffer lower;
        Buffer upper;
        Buffer fraction;
        Buffer source;
        Buffer output;
        std::array<Event, 3U> planReady;
        Event sourceReady;
        Event kernelDone;
        Event readDone;
        std::vector<float> hostSource;
        std::vector<float> hostOutput;
        std::size_t pointCapacity = 0U;
        std::size_t pointStart = 0U;
        std::size_t pointCount = 0U;
        bool planUploaded = false;
        bool pending = false;
    };

    void releaseResources() noexcept {
        if (mlsKernel_ != nullptr) clReleaseKernel(mlsKernel_);
        if (demIdwKernel_ != nullptr) clReleaseKernel(demIdwKernel_);
        for (cl_kernel& kernel : verticalKernels_) {
            if (kernel != nullptr) clReleaseKernel(kernel);
            kernel = nullptr;
        }
        for (cl_kernel& kernel : horizontalKernels_) {
            if (kernel != nullptr) clReleaseKernel(kernel);
            kernel = nullptr;
        }
        if (program_ != nullptr) clReleaseProgram(program_);
        for (cl_command_queue& queue : queues_) {
            if (queue != nullptr) clReleaseCommandQueue(queue);
            queue = nullptr;
        }
        if (context_ != nullptr) clReleaseContext(context_);
        mlsKernel_ = nullptr;
        demIdwKernel_ = nullptr;
        program_ = nullptr;
        context_ = nullptr;
    }

    [[nodiscard]] std::size_t maximumAllocation() const {
        return static_cast<std::size_t>(std::min<cl_ulong>(
            maximumAllocation_,
            static_cast<cl_ulong>(std::numeric_limits<std::size_t>::max())));
    }

    [[nodiscard]] std::size_t globalMemoryBudget() const {
        const cl_ulong conservative = globalMemory_ - globalMemory_ / 4U;
        return static_cast<std::size_t>(std::min<cl_ulong>(
            conservative,
            static_cast<cl_ulong>(std::numeric_limits<std::size_t>::max())));
    }

    cl_mem createBuffer(cl_mem_flags flags, std::size_t bytes, void* host,
                        std::string_view operation) {
        if (bytes == 0U || bytes > maximumAllocation()) {
            throw OpenClLuwbcError(
                std::string(operation) +
                " rejected by the selected GPU's maximum allocation limit");
        }
        cl_int status = CL_SUCCESS;
        cl_mem buffer = clCreateBuffer(context_, flags, bytes, host, &status);
        requireSuccess(status, operation);
        if (buffer == nullptr) {
            throw OpenClLuwbcError(std::string(operation) +
                                   " returned a null OpenCL buffer");
        }
        return buffer;
    }

    template <typename T>
    static void setKernelArgument(cl_kernel kernel, cl_uint index,
                                  const T& value,
                                  std::string_view operation) {
        requireSuccess(clSetKernelArg(kernel, index, sizeof(T), &value),
                       operation);
    }

    static void validateComponentCount(std::size_t componentCount) {
        if (componentCount != 3U && componentCount != 4U) {
            throw OpenClLuwbcError(
                "luwbc OpenCL componentCount must be 3 (U/V/W) or 4 "
                "(U/V/W/T)");
        }
    }

    static void validateHorizontal(
        const OpenClLuwbcHorizontalRequest& request) {
        validateComponentCount(request.componentCount);
        if (request.sourceLayerSize == 0U || request.layerCount == 0U) {
            throw OpenClLuwbcError(
                "luwbc OpenCL horizontal sourceLayerSize and layerCount "
                "must be positive");
        }
        if (request.sourceLayerSize >
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max())) {
            throw OpenClLuwbcError(
                "luwbc OpenCL sourceLayerSize exceeds uint32 index capacity");
        }
        const std::size_t planes = checkedMultiply(
            request.componentCount, request.layerCount,
            "horizontal component/layer");
        const std::size_t expectedValues = checkedMultiply(
            planes, request.sourceLayerSize, "horizontal input");
        if (request.values.size() != expectedValues) {
            throw OpenClLuwbcError(
                "inconsistent luwbc OpenCL horizontal value dimensions");
        }
        const std::size_t expectedTriples = checkedMultiply(
            request.targetPointCount, 3U, "horizontal target weights");
        if (request.indices.size() != expectedTriples ||
            request.weights.size() != expectedTriples ||
            request.counts.size() != request.targetPointCount) {
            throw OpenClLuwbcError(
                "inconsistent luwbc OpenCL horizontal plan dimensions");
        }
        for (std::size_t target = 0U;
             target < request.targetPointCount; ++target) {
            const std::uint32_t count = request.counts[target];
            if (count != 1U && count != 3U) {
                throw OpenClLuwbcError(
                    "luwbc OpenCL horizontal target " +
                    std::to_string(target) +
                    " has count " + std::to_string(count) +
                    "; expected nearest count 1 or barycentric count 3");
            }
            for (std::uint32_t corner = 0U; corner < count; ++corner) {
                const std::size_t offset = target * 3U + corner;
                if (request.indices[offset] >= request.sourceLayerSize) {
                    throw OpenClLuwbcError(
                        "luwbc OpenCL horizontal target " +
                        std::to_string(target) + " references source index " +
                        std::to_string(request.indices[offset]) +
                        " outside sourceLayerSize " +
                        std::to_string(request.sourceLayerSize));
                }
                if (!std::isfinite(request.weights[offset])) {
                    throw OpenClLuwbcError(
                        "luwbc OpenCL horizontal target " +
                        std::to_string(target) +
                        " has a non-finite interpolation weight");
                }
            }
        }
    }

    static void validateVertical(const OpenClLuwbcVerticalRequest& request) {
        validateComponentCount(request.componentCount);
        if (request.sourceLevels.empty()) {
            throw OpenClLuwbcError(
                "luwbc OpenCL vertical sourceLevels must not be empty");
        }
        if (request.sourceLevels.size() >
                static_cast<std::size_t>(
                    std::numeric_limits<std::uint32_t>::max()) ||
            request.targetLevels.size() >
                static_cast<std::size_t>(
                    std::numeric_limits<std::uint32_t>::max())) {
            throw OpenClLuwbcError(
                "luwbc OpenCL vertical level count exceeds uint32 capacity");
        }
        for (std::size_t level = 0U; level < request.sourceLevels.size();
             ++level) {
            if (!std::isfinite(request.sourceLevels[level])) {
                throw OpenClLuwbcError(
                    "luwbc OpenCL sourceLevels contain a non-finite value");
            }
            if (level != 0U &&
                request.sourceLevels[level] <= request.sourceLevels[level - 1U]) {
                throw OpenClLuwbcError(
                    "luwbc OpenCL sourceLevels must be strictly increasing");
            }
        }
        if (!std::all_of(request.targetLevels.begin(),
                         request.targetLevels.end(),
                         [](double value) { return std::isfinite(value); })) {
            throw OpenClLuwbcError(
                "luwbc OpenCL targetLevels contain a non-finite value");
        }
        const std::size_t sourcePlanes = checkedMultiply(
            request.componentCount, request.sourceLevels.size(),
            "vertical source planes");
        const std::size_t expectedValues = checkedMultiply(
            sourcePlanes, request.pointCount, "vertical input");
        if (request.values.size() != expectedValues) {
            throw OpenClLuwbcError(
                "inconsistent luwbc OpenCL vertical value dimensions");
        }
    }

    static void validateNeighborPlan(
        const std::vector<double>& sourceX,
        const std::vector<double>& sourceY,
        const std::vector<double>& targetX,
        const std::vector<double>& targetY,
        const std::vector<std::uint32_t>& neighborIndices,
        std::size_t neighborCount,
        std::string_view description) {
        if (sourceX.empty() || sourceX.size() != sourceY.size()) {
            throw OpenClLuwbcError(
                "luwbc OpenCL " + std::string(description) +
                " source coordinates must be nonempty and equally sized");
        }
        if (sourceX.size() > static_cast<std::size_t>(
                                 std::numeric_limits<std::uint32_t>::max())) {
            throw OpenClLuwbcError(
                "luwbc OpenCL " + std::string(description) +
                " source point count exceeds uint32 index capacity");
        }
        if (targetX.size() != targetY.size()) {
            throw OpenClLuwbcError(
                "luwbc OpenCL " + std::string(description) +
                " target coordinates must be equally sized");
        }
        if (neighborCount == 0U || neighborCount > 24U ||
            neighborCount > sourceX.size()) {
            throw OpenClLuwbcError(
                "luwbc OpenCL " + std::string(description) +
                " neighborCount must be in 1..min(24, sourcePointCount)");
        }
        const std::size_t expectedIndices = checkedMultiply(
            targetX.size(), neighborCount,
            std::string(description) + " neighbor plan");
        if (neighborIndices.size() != expectedIndices) {
            throw OpenClLuwbcError(
                "inconsistent luwbc OpenCL " + std::string(description) +
                " neighbor-plan dimensions");
        }
        const auto finite = [](double value) { return std::isfinite(value); };
        if (!std::all_of(sourceX.begin(), sourceX.end(), finite) ||
            !std::all_of(sourceY.begin(), sourceY.end(), finite) ||
            !std::all_of(targetX.begin(), targetX.end(), finite) ||
            !std::all_of(targetY.begin(), targetY.end(), finite)) {
            throw OpenClLuwbcError(
                "luwbc OpenCL " + std::string(description) +
                " coordinates contain a non-finite value");
        }
        for (std::size_t offset = 0U; offset < neighborIndices.size();
             ++offset) {
            if (neighborIndices[offset] >= sourceX.size()) {
                throw OpenClLuwbcError(
                    "luwbc OpenCL " + std::string(description) +
                    " neighbor plan references source index " +
                    std::to_string(neighborIndices[offset]) +
                    " outside source point count " +
                    std::to_string(sourceX.size()));
            }
        }
    }

    static void validateDemIdw(const OpenClLuwbcDemIdwRequest& request) {
        validateNeighborPlan(
            request.sourceX, request.sourceY, request.targetX,
            request.targetY, request.neighborIndices, request.neighborCount,
            "DEM IDW");
        if (request.sourceValues.size() != request.sourceX.size()) {
            throw OpenClLuwbcError(
                "inconsistent luwbc OpenCL DEM IDW source-value dimensions");
        }
        if (!std::all_of(request.sourceValues.begin(),
                         request.sourceValues.end(),
                         [](double value) { return std::isfinite(value); })) {
            throw OpenClLuwbcError(
                "luwbc OpenCL DEM IDW source values contain a non-finite value");
        }
    }

    static void validateMls(const OpenClLuwbcMlsRequest& request) {
        validateNeighborPlan(
            request.sourceX, request.sourceY, request.targetX,
            request.targetY, request.neighborIndices, request.neighborCount,
            "MLS");
    }

    static void buildVerticalPlan(
        const std::vector<double>& sourceLevels,
        const std::vector<double>& targetLevels,
        std::vector<std::uint32_t>& lower,
        std::vector<std::uint32_t>& upper,
        std::vector<double>& fraction) {
        const std::size_t last = sourceLevels.size() - 1U;
        for (std::size_t target = 0U; target < targetLevels.size(); ++target) {
            const double value = targetLevels[target];
            if (value <= sourceLevels.front()) {
                lower[target] = 0U;
                upper[target] = 0U;
                fraction[target] = 0.0;
                continue;
            }
            if (value >= sourceLevels.back()) {
                lower[target] = static_cast<std::uint32_t>(last);
                upper[target] = static_cast<std::uint32_t>(last);
                fraction[target] = 0.0;
                continue;
            }
            const std::size_t high = static_cast<std::size_t>(
                std::lower_bound(sourceLevels.begin(), sourceLevels.end(),
                                 value) -
                sourceLevels.begin());
            const std::size_t low = high - 1U;
            const double lowerDistance = std::abs(value - sourceLevels[low]);
            const double upperDistance = std::abs(value - sourceLevels[high]);
            if (lowerDistance < 1.0e-6) {
                lower[target] = static_cast<std::uint32_t>(low);
                upper[target] = static_cast<std::uint32_t>(low);
                fraction[target] = 0.0;
            } else if (upperDistance < 1.0e-6) {
                lower[target] = static_cast<std::uint32_t>(high);
                upper[target] = static_cast<std::uint32_t>(high);
                fraction[target] = 0.0;
            } else {
                lower[target] = static_cast<std::uint32_t>(low);
                upper[target] = static_cast<std::uint32_t>(high);
                fraction[target] = std::clamp(
                    lowerDistance / (lowerDistance + upperDistance), 0.0, 1.0);
            }
        }
    }

    static const char* kernelSource() {
        return R"CLC(
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error OpenCL double precision is required by the luwbc interpolation kernels
#endif
#pragma OPENCL FP_CONTRACT OFF
__kernel void horizontal_resample(
    __global const float* values,
    __global const uint* indices,
    __global const double* weights,
    __global const uint* counts,
    const ulong source_layer_size,
    const ulong target_count,
    __global float* output) {
    const size_t global_index = get_global_id(0);
    const ulong target = (ulong)global_index % target_count;
    const ulong plane = (ulong)global_index / target_count;
    const ulong plan_offset = target * 3UL;
    const ulong source_offset = plane * source_layer_size;
    const uint count = counts[target];
    double value = 0.0;
    for (uint corner = 0U; corner < count; ++corner) {
        value += weights[plan_offset + corner] *
                 (double)values[source_offset +
                                (ulong)indices[plan_offset + corner]];
    }
    output[global_index] = (float)value;
}

__kernel void vertical_resample(
    __global const float* values,
    __global const uint* lower,
    __global const uint* upper,
    __global const double* fractions,
    const ulong source_level_count,
    const ulong target_level_count,
    const ulong point_count,
    __global float* output) {
    const size_t global_index = get_global_id(0);
    const ulong point = (ulong)global_index % point_count;
    const ulong plane = (ulong)global_index / point_count;
    const ulong target_level = plane % target_level_count;
    const ulong component = plane / target_level_count;
    const ulong low = (ulong)lower[target_level];
    const ulong high = (ulong)upper[target_level];
    const ulong component_offset =
        component * source_level_count * point_count;
    const float first =
        values[component_offset + low * point_count + point];
    const double fraction = fractions[target_level];
    if (low == high || fraction <= 0.0) {
        output[global_index] = first;
        return;
    }
    const float second =
        values[component_offset + high * point_count + point];
    if (fraction >= 1.0) {
        output[global_index] = second;
        return;
    }
    output[global_index] = (float)((double)first * (1.0 - fraction) +
                                   (double)second * fraction);
}

__kernel void dem_idw(
    __global const double* source_x,
    __global const double* source_y,
    __global const double* source_values,
    __global const double* target_x,
    __global const double* target_y,
    __global const uint* neighbor_indices,
    const uint neighbor_count,
    __global double* output) {
    const size_t target = get_global_id(0);
    const size_t plan_offset = target * (size_t)neighbor_count;
    const double query_x = target_x[target];
    const double query_y = target_y[target];
    double weighted = 0.0;
    double weight_sum = 0.0;
    for (uint entry = 0U; entry < neighbor_count; ++entry) {
        const uint source = neighbor_indices[plan_offset + (size_t)entry];
        const double dx = source_x[source] - query_x;
        const double dy = source_y[source] - query_y;
        const double distance = fmax(sqrt(dx * dx + dy * dy), 1.0e-10);
        const double weight = 1.0 / (distance * distance);
        weighted += weight * source_values[source];
        weight_sum += weight;
    }
    output[target] = weighted / weight_sum;
}

__kernel void mls_weights(
    __global const double* source_x,
    __global const double* source_y,
    __global const double* target_x,
    __global const double* target_y,
    __global const uint* neighbor_indices,
    const uint neighbor_count,
    __global float* output) {
    const size_t target = get_global_id(0);
    const size_t plan_offset = target * (size_t)neighbor_count;
    const double query_x = target_x[target];
    const double query_y = target_y[target];
    const uint dimension = neighbor_count >= 6U ? 6U :
                           (neighbor_count >= 3U ? 3U : 1U);
    double h = 0.0;
    for (uint entry = 0U; entry < neighbor_count; ++entry) {
        const uint source = neighbor_indices[plan_offset + (size_t)entry];
        const double dx = source_x[source] - query_x;
        const double dy = source_y[source] - query_y;
        h = fmax(h, sqrt(dx * dx + dy * dy));
    }
    if (!isfinite(h) || h <= 1.0e-12) h = 1.0;
    h *= 1.000001;

    double kernel_values[24];
    double kernel_sum = 0.0;
    for (uint entry = 0U; entry < neighbor_count; ++entry) {
        const uint source = neighbor_indices[plan_offset + (size_t)entry];
        const double xn = (source_x[source] - query_x) / h;
        const double yn = (source_y[source] - query_y) / h;
        const double radius = sqrt(xn * xn + yn * yn);
        const double compact = fmax(0.0, 1.0 - radius);
        const double value = pow(compact, 4.0) * (4.0 * radius + 1.0);
        kernel_values[entry] = value;
        kernel_sum += value;
    }
    if (!isfinite(kernel_sum) || kernel_sum <= 1.0e-14) {
        kernel_sum = (double)neighbor_count;
        for (uint entry = 0U; entry < neighbor_count; ++entry) {
            kernel_values[entry] = 1.0;
        }
    }

    double coefficients[24];
    if (dimension == 1U) {
        for (uint entry = 0U; entry < neighbor_count; ++entry) {
            coefficients[entry] = kernel_values[entry] / kernel_sum;
        }
    } else {
        double matrix[36];
        double right[6];
        for (uint entry = 0U; entry < dimension * dimension; ++entry) {
            matrix[entry] = 0.0;
        }
        for (uint entry = 0U; entry < dimension; ++entry) right[entry] = 0.0;
        right[0] = 1.0;

        for (uint entry = 0U; entry < neighbor_count; ++entry) {
            const uint source = neighbor_indices[plan_offset + (size_t)entry];
            const double xn = (source_x[source] - query_x) / h;
            const double yn = (source_y[source] - query_y) / h;
            double basis[6];
            basis[0] = 1.0;
            basis[1] = xn;
            basis[2] = yn;
            if (dimension == 6U) {
                basis[3] = xn * xn;
                basis[4] = xn * yn;
                basis[5] = yn * yn;
            }
            for (uint row = 0U; row < dimension; ++row) {
                for (uint column = 0U; column < dimension; ++column) {
                    matrix[row * dimension + column] +=
                        kernel_values[entry] * basis[row] * basis[column];
                }
            }
        }
        double trace = 0.0;
        for (uint diagonal = 0U; diagonal < dimension; ++diagonal) {
            trace += matrix[diagonal * dimension + diagonal];
        }
        const double regularization =
            1.0e-10 * trace / (double)dimension + 1.0e-12;
        for (uint diagonal = 0U; diagonal < dimension; ++diagonal) {
            matrix[diagonal * dimension + diagonal] += regularization;
        }

        int solved = 1;
        for (uint pivot = 0U; pivot < dimension; ++pivot) {
            uint best = pivot;
            for (uint row = pivot + 1U; row < dimension; ++row) {
                if (fabs(matrix[row * dimension + pivot]) >
                    fabs(matrix[best * dimension + pivot])) {
                    best = row;
                }
            }
            if (fabs(matrix[best * dimension + pivot]) <= 1.0e-18) {
                solved = 0;
                break;
            }
            if (best != pivot) {
                for (uint column = pivot; column < dimension; ++column) {
                    const double temporary =
                        matrix[pivot * dimension + column];
                    matrix[pivot * dimension + column] =
                        matrix[best * dimension + column];
                    matrix[best * dimension + column] = temporary;
                }
                const double temporary = right[pivot];
                right[pivot] = right[best];
                right[best] = temporary;
            }
            const double diagonal = matrix[pivot * dimension + pivot];
            for (uint column = pivot; column < dimension; ++column) {
                matrix[pivot * dimension + column] /= diagonal;
            }
            right[pivot] /= diagonal;
            for (uint row = 0U; row < dimension; ++row) {
                if (row == pivot) continue;
                const double factor = matrix[row * dimension + pivot];
                if (factor == 0.0) continue;
                for (uint column = pivot; column < dimension; ++column) {
                    matrix[row * dimension + column] -=
                        factor * matrix[pivot * dimension + column];
                }
                right[row] -= factor * right[pivot];
            }
        }

        if (solved) {
            double coefficient_sum = 0.0;
            for (uint entry = 0U; entry < neighbor_count; ++entry) {
                const uint source =
                    neighbor_indices[plan_offset + (size_t)entry];
                const double xn = (source_x[source] - query_x) / h;
                const double yn = (source_y[source] - query_y) / h;
                double dot = right[0];
                if (dimension >= 3U) {
                    dot += xn * right[1];
                    dot += yn * right[2];
                }
                if (dimension == 6U) {
                    dot += xn * xn * right[3];
                    dot += xn * yn * right[4];
                    dot += yn * yn * right[5];
                }
                coefficients[entry] = kernel_values[entry] * dot;
                coefficient_sum += coefficients[entry];
            }
            if (!isfinite(coefficient_sum) ||
                fabs(coefficient_sum) <= 1.0e-14) {
                solved = 0;
            } else {
                for (uint entry = 0U; entry < neighbor_count; ++entry) {
                    coefficients[entry] /= coefficient_sum;
                }
                double negative_mass = 0.0;
                for (uint entry = 0U; entry < neighbor_count; ++entry) {
                    if (coefficients[entry] < 0.0) {
                        negative_mass += fabs(coefficients[entry]);
                    }
                }
                if (negative_mass > 0.08) {
                    const double alpha = fmin(
                        1.0, (negative_mass - 0.08) / 0.30);
                    double limited_sum = 0.0;
                    for (uint entry = 0U; entry < neighbor_count; ++entry) {
                        coefficients[entry] =
                            (1.0 - alpha) * coefficients[entry] +
                            alpha * kernel_values[entry] / kernel_sum;
                        limited_sum += coefficients[entry];
                    }
                    for (uint entry = 0U; entry < neighbor_count; ++entry) {
                        coefficients[entry] /= limited_sum;
                    }
                }
            }
        }
        if (!solved) {
            for (uint entry = 0U; entry < neighbor_count; ++entry) {
                coefficients[entry] = kernel_values[entry] / kernel_sum;
            }
        }
    }
    for (uint entry = 0U; entry < neighbor_count; ++entry) {
        output[plan_offset + (size_t)entry] = (float)coefficients[entry];
    }
}
)CLC";
    }

    int deviceIndex_ = 0;
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    std::array<cl_command_queue, 2U> queues_{nullptr, nullptr};
    cl_program program_ = nullptr;
    std::array<cl_kernel, 2U> horizontalKernels_{nullptr, nullptr};
    std::array<cl_kernel, 2U> verticalKernels_{nullptr, nullptr};
    cl_kernel demIdwKernel_ = nullptr;
    cl_kernel mlsKernel_ = nullptr;
    cl_ulong maximumAllocation_ = 0U;
    cl_ulong globalMemory_ = 0U;
    std::list<std::shared_ptr<const VerticalPlan>> verticalPlanCache_;
    std::size_t verticalPlanCacheBytes_ = 0U;
    std::mutex runMutex_;
};

std::unordered_map<int, std::unique_ptr<Backend>>& backendInstances() {
    static std::unordered_map<int, std::unique_ptr<Backend>> backends;
    return backends;
}

std::mutex& backendInstancesMutex() {
    static std::mutex mutex;
    return mutex;
}

} // namespace
#endif

bool nativeOpenClLuwbcRequiresWarmup(int deviceIndex) noexcept {
#if LUW_HAS_OPENCL
    try {
        std::scoped_lock lock(backendInstancesMutex());
        return !backendInstances().contains(deviceIndex);
    } catch (...) {
        return true;
    }
#else
    (void)deviceIndex;
    return true;
#endif
}

OpenClLuwbcBackendInfo requireNativeOpenClLuwbcBackend(int deviceIndex) {
#if LUW_HAS_OPENCL
    std::scoped_lock lock(backendInstancesMutex());
    auto& backends = backendInstances();
    auto match = backends.find(deviceIndex);
    const bool compileNow = match == backends.end();
    if (compileNow) {
        match = backends.emplace(deviceIndex,
                                 std::make_unique<Backend>(deviceIndex)).first;
    }
    return {match->second->deviceName(), match->second->deviceIndex(),
            compileNow};
#else
    (void)deviceIndex;
    throw OpenClLuwbcError("OpenCL support is not enabled in this build");
#endif
}

OpenClLuwbcResult runNativeOpenClLuwbcHorizontal(
    const OpenClLuwbcHorizontalRequest& request) {
#if LUW_HAS_OPENCL
    const OpenClLuwbcBackendInfo info =
        requireNativeOpenClLuwbcBackend(request.deviceIndex);
    Backend* backend = nullptr;
    {
        std::scoped_lock lock(backendInstancesMutex());
        backend = backendInstances().at(request.deviceIndex).get();
    }
    return {backend->runHorizontal(request), info.device, info.deviceIndex,
            info.compiledThisCall};
#else
    (void)request;
    throw OpenClLuwbcError("OpenCL support is not enabled in this build");
#endif
}

OpenClLuwbcResult runNativeOpenClLuwbcVertical(
    const OpenClLuwbcVerticalRequest& request) {
#if LUW_HAS_OPENCL
    const OpenClLuwbcBackendInfo info =
        requireNativeOpenClLuwbcBackend(request.deviceIndex);
    Backend* backend = nullptr;
    {
        std::scoped_lock lock(backendInstancesMutex());
        backend = backendInstances().at(request.deviceIndex).get();
    }
    return {backend->runVertical(request), info.device, info.deviceIndex,
            info.compiledThisCall};
#else
    (void)request;
    throw OpenClLuwbcError("OpenCL support is not enabled in this build");
#endif
}

OpenClLuwbcDoubleResult runNativeOpenClLuwbcDemIdw(
    const OpenClLuwbcDemIdwRequest& request) {
#if LUW_HAS_OPENCL
    const OpenClLuwbcBackendInfo info =
        requireNativeOpenClLuwbcBackend(request.deviceIndex);
    Backend* backend = nullptr;
    {
        std::scoped_lock lock(backendInstancesMutex());
        backend = backendInstances().at(request.deviceIndex).get();
    }
    return {backend->runDemIdw(request), info.device, info.deviceIndex,
            info.compiledThisCall};
#else
    (void)request;
    throw OpenClLuwbcError("OpenCL support is not enabled in this build");
#endif
}

OpenClLuwbcResult runNativeOpenClLuwbcMlsWeights(
    const OpenClLuwbcMlsRequest& request) {
#if LUW_HAS_OPENCL
    const OpenClLuwbcBackendInfo info =
        requireNativeOpenClLuwbcBackend(request.deviceIndex);
    Backend* backend = nullptr;
    {
        std::scoped_lock lock(backendInstancesMutex());
        backend = backendInstances().at(request.deviceIndex).get();
    }
    return {backend->runMlsWeights(request), info.device, info.deviceIndex,
            info.compiledThisCall};
#else
    (void)request;
    throw OpenClLuwbcError("OpenCL support is not enabled in this build");
#endif
}

} // namespace luw::cli
