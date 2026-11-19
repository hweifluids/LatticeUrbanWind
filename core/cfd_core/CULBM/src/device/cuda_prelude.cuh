
#include <cuda_runtime.h>
#include <cuda_fp16.h>

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long long ulong;

#define cuda_threads_per_block 64u
#define CUDA_LOCAL_MEM_FENCE 0
#define CUDA_GLOBAL_MEM_FENCE 0

#define cuda_global_thread_id(dim) ((ulong)blockIdx.x * (ulong)blockDim.x + (ulong)threadIdx.x)
#define cuda_global_thread_count(dim) ((ulong)gridDim.x * (ulong)blockDim.x)
#define cuda_local_thread_id(dim) ((uint)threadIdx.x)
#define cuda_local_thread_count(dim) ((uint)blockDim.x)
#define cuda_block_count(dim) ((uint)gridDim.x)
#define cuda_block_id(dim) ((uint)blockIdx.x)
#define cuda_global_offset(dim) 0u
#define cuda_block_barrier(x) __syncthreads()
#define cuda_memory_fence(x) __threadfence_block()
#define cuda_read_memory_fence(x) __threadfence_block()
#define cuda_write_memory_fence(x) __threadfence_block()

__device__ __forceinline__ float2 cuda_float2(float v) { return make_float2(v, v); }
__device__ __forceinline__ float2 cuda_float2(float x, float y) { return make_float2(x, y); }
__device__ __forceinline__ float3 cuda_float3(float v) { return make_float3(v, v, v); }
__device__ __forceinline__ float3 cuda_float3(float x, float y, float z) { return make_float3(x, y, z); }
__device__ __forceinline__ uint3 cuda_uint3(uint v) { return make_uint3(v, v, v); }
__device__ __forceinline__ uint3 cuda_uint3(uint x, uint y, uint z) { return make_uint3(x, y, z); }
__device__ __forceinline__ int3 cuda_int3(int v) { return make_int3(v, v, v); }
__device__ __forceinline__ int3 cuda_int3(int x, int y, int z) { return make_int3(x, y, z); }
__device__ __forceinline__ uchar4 cuda_uchar4(uchar v) { return make_uchar4(v, v, v, v); }
__device__ __forceinline__ uchar4 cuda_uchar4(uchar x, uchar y, uchar z, uchar w) { return make_uchar4(x, y, z, w); }

__device__ __forceinline__ float3 operator+(float3 a, float3 b) { return cuda_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 operator-(float3 a, float3 b) { return cuda_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 operator-(float3 a) { return cuda_float3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ float3 operator*(float3 a, float3 b) { return cuda_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ __forceinline__ float3 operator*(float3 a, float b) { return cuda_float3(a.x * b, a.y * b, a.z * b); }
__device__ __forceinline__ float3 operator*(float a, float3 b) { return b * a; }
__device__ __forceinline__ float3 operator/(float3 a, float b) { return cuda_float3(a.x / b, a.y / b, a.z / b); }
__device__ __forceinline__ float3 &operator+=(float3 &a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}
__device__ __forceinline__ float3 &operator-=(float3 &a, float3 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}
__device__ __forceinline__ float3 &operator*=(float3 &a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  return a;
}
__device__ __forceinline__ float3 &operator/=(float3 &a, float b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  return a;
}

__device__ __forceinline__ uint3 operator+(uint3 a, uint3 b) { return cuda_uint3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ uint3 operator-(uint3 a, uint3 b) { return cuda_uint3(a.x - b.x, a.y - b.y, a.z - b.z); }

template <typename T> __device__ __forceinline__ T cuda_min(T a, T b) { return a < b ? a : b; }
template <typename T> __device__ __forceinline__ T cuda_max(T a, T b) { return a > b ? a : b; }
template <typename T> __device__ __forceinline__ T cuda_clamp(T x, T lo, T hi) { return cuda_min(cuda_max(x, lo), hi); }
__device__ __forceinline__ float3 cuda_clamp(float3 x, float lo, float hi) {
  return cuda_float3(cuda_clamp(x.x, lo, hi), cuda_clamp(x.y, lo, hi), cuda_clamp(x.z, lo, hi));
}

__device__ __forceinline__ float cuda_fma(float a, float b, float c) { return fmaf(a, b, c); }
__device__ __forceinline__ float3 cuda_fma(float a, float3 b, float3 c) {
  return cuda_float3(fmaf(a, b.x, c.x), fmaf(a, b.y, c.y), fmaf(a, b.z, c.z));
}
__device__ __forceinline__ float3 cuda_fma(float3 a, float b, float3 c) {
  return cuda_float3(fmaf(a.x, b, c.x), fmaf(a.y, b, c.y), fmaf(a.z, b, c.z));
}
__device__ __forceinline__ float3 cuda_fma(float3 a, float3 b, float3 c) {
  return cuda_float3(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y), fmaf(a.z, b.z, c.z));
}

__device__ __forceinline__ float dot(float2 a, float2 b) { return fmaf(a.x, b.x, a.y * b.y); }
__device__ __forceinline__ float dot(float3 a, float3 b) { return fmaf(a.x, b.x, fmaf(a.y, b.y, a.z * b.z)); }
__device__ __forceinline__ float3 cross(float3 a, float3 b) {
  return cuda_float3(fmaf(a.y, b.z, -a.z * b.y), fmaf(a.z, b.x, -a.x * b.z), fmaf(a.x, b.y, -a.y * b.x));
}
__device__ __forceinline__ float length(float2 v) { return sqrtf(dot(v, v)); }
__device__ __forceinline__ float length(float3 v) { return sqrtf(dot(v, v)); }
__device__ __forceinline__ float distance(float3 a, float3 b) { return length(b - a); }
__device__ __forceinline__ float3 normalize(float3 v) {
  const float inv = rsqrtf(dot(v, v));
  return v * inv;
}
__device__ __forceinline__ float sign(float v) { return (float)((v > 0.0f) - (v < 0.0f)); }
__device__ __forceinline__ float3 floor(float3 v) { return cuda_float3(floorf(v.x), floorf(v.y), floorf(v.z)); }
__device__ __forceinline__ float acospi(float v) { return acosf(v) * 0.31830988618379067154f; }

__device__ __forceinline__ int cuda_as_int(float value) { return __float_as_int(value); }
__device__ __forceinline__ float cuda_as_float(int value) { return __int_as_float(value); }
__device__ __forceinline__ float cuda_as_float(uint value) { return __uint_as_float(value); }
__device__ __forceinline__ uint cuda_as_uint(float value) { return __float_as_uint(value); }
__device__ __forceinline__ uchar4 cuda_as_uchar4(int value) {
  union Bits {
    int i;
    uchar4 v;
  } bits;
  bits.i = value;
  return bits.v;
}
__device__ __forceinline__ int cuda_as_int(uchar4 value) {
  union Bits {
    int i;
    uchar4 v;
  } bits;
  bits.v = value;
  return bits.i;
}

__device__ __forceinline__ float cuda_vload_half(size_t offset, const half *pointer) {
  return __half2float(pointer[offset]);
}
__device__ __forceinline__ void cuda_vstore_half_rn(float value, size_t offset, half *pointer) {
  pointer[offset] = __float2half_rn(value);
}

__device__ __forceinline__ int cuda_atomic_max(volatile int *address, int value) {
  return atomicMax((int *)address, value);
}
__device__ __forceinline__ uint cuda_atomic_add(volatile uint *address, uint value) {
  return atomicAdd((uint *)address, value);
}
__device__ __forceinline__ float cuda_atomic_xchg(volatile float *address, float value) {
  return atomicExch((float *)address, value);
}

__device__ __forceinline__ float cuda_fast_rsqrt(float value) { return rsqrtf(value); }
__device__ __forceinline__ float cuda_fast_sqrt(float value) { return sqrtf(value); }
__device__ __forceinline__ float cuda_rsqrt(float value) { return rsqrtf(value); }
__device__ __forceinline__ float cuda_pown(float value, int exponent) { return powf(value, (float)exponent); }
