#include "driver.hpp"

#include "../core/core.hpp"

#include <mutex>

namespace culbm::gpu {

std::string describe(const CUresult result) {
	const char* name = nullptr;
	const char* text = nullptr;
	cuGetErrorName(result, &name);
	cuGetErrorString(result, &text);
	std::string message = name!=nullptr ? name : "CUDA_ERROR_UNKNOWN";
	message += " (";
	message += text!=nullptr ? text : "no description";
	message += ")";
	return message;
}

void check(const CUresult result, const char* const what) {
	if(result==CUDA_SUCCESS) return;
	fatal_error(std::string(what)+" failed: "+describe(result)+".");
}

void check(const nvrtcResult result, const char* const what) {
	if(result==NVRTC_SUCCESS) return;
	fatal_error(std::string(what)+" failed: "+nvrtcGetErrorString(result)+".");
}

void make_current(const CUcontext context) {
	check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
}

void ensure_driver() {
	static std::once_flag initialised;
	std::call_once(initialised, [] { check(cuInit(0u), "cuInit"); });
}

} // namespace culbm::gpu
