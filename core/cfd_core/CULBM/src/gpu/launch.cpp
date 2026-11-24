#include "launch.hpp"

#include <cstring>

namespace culbm::gpu {

Launch::Launch(const Device& device, const string& kernel, const std::uint64_t threads)
	: owner_(device.state()), name_(kernel), threads_(threads) {
	if(!owner_) fatal_error("Kernel \""+kernel+"\" needs an opened CUDA device.");
	make_current(owner_->context);
	if(cuModuleGetFunction(&function_, owner_->module, name_.c_str())!=CUDA_SUCCESS)
		fatal_error("The CULBM CUDA module has no kernel named \""+name_+"\".");
}

Launch& Launch::block(const uint threads_per_block) & {
	if(threads_per_block==0u || threads_per_block>1024u)
		fatal_error("Kernel \""+name_+"\": invalid block size "+to_string(threads_per_block)+".");
	block_ = threads_per_block;
	return *this;
}

void Launch::write(const uint slot, const void* const data, const std::size_t size) {
	if(size>sizeof(Slot::bytes))
		fatal_error("Kernel \""+name_+"\": argument "+to_string(slot)+" is wider than 16 bytes.");
	if(slot>=slots_.size()) {
		slots_.resize((std::size_t)slot+1u, Slot{{}, false});
		pointers_stale_ = true;
	}
	Slot& target = slots_[slot];
	std::memset(target.bytes, 0, sizeof(target.bytes));
	std::memcpy(target.bytes, data, size);
	target.set = true;
}

void Launch::submit(const uint repeat) {
	if(!owner_) return;
	submit_on(owner_->stream, repeat);
}

void Launch::submit_on(const CUstream stream, const uint repeat) {
	if(threads_==0u || repeat==0u || !owner_) return;
	make_current(owner_->context);
	const std::uint64_t blocks = (threads_+block_-1u)/block_;
	if(blocks>0x7FFFFFFFu) fatal_error("Kernel \""+name_+"\": launch grid is too large.");
	if(pointers_stale_) {
		pointers_.clear();
		for(uint slot=0u; slot<slot_count(); ++slot) {
			if(!slots_[slot].set)
				fatal_error("Kernel \""+name_+"\": argument "+to_string(slot)+" was never set.");
			pointers_.push_back(slots_[slot].bytes);
		}
		pointers_stale_ = false;
	}
	for(uint i=0u; i<repeat; ++i) {
		const CUresult status = cuLaunchKernel(function_, (unsigned)blocks, 1u, 1u, block_, 1u, 1u,
			0u, stream, pointers_.data(), nullptr);
		if(status!=CUDA_SUCCESS) check(status, ("cuLaunchKernel("+name_+")").c_str());
	}
}

void Launch::submit_and_wait() {
	submit();
	if(!owner_) return;
	make_current(owner_->context);
	check(cuStreamSynchronize(owner_->stream), ("cuStreamSynchronize("+name_+")").c_str());
}

} // namespace culbm::gpu
