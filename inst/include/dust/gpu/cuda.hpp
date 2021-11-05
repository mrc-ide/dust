#ifndef DUST_GPU_CUDA_HPP
#define DUST_GPU_CUDA_HPP

// Important: some key defines occur in
// dust/random/cuda_compatibility.hpp rather than here; they need to
// be there so that the standalone random library will work.

#ifdef __NVCC__
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

// CUDA 11 cooperative groups
#if __CUDACC_VER_MAJOR__ >= 11
#include <cooperative_groups/memcpy_async.h>
#endif

#endif

#include "dust/gpu/call.hpp"
#include "dust/random/cuda_compatibility.hpp"

// Prevent accidentally enabling profiling on non-nvcc platforms
#ifndef __NVCC__
#undef DUST_CUDA_ENABLE_PROFILER
#endif

namespace dust {
namespace gpu {

const int warp_size = 32;

#ifdef __NVCC__
template <typename T>
__device__
void shared_mem_cpy(cooperative_groups::thread_block& block, T* shared_ptr,
                    const T* global_ptr, size_t n_elem) {
#if __CUDACC_VER_MAJOR__ >= 11
  cooperative_groups::memcpy_async(block,
                                   shared_ptr,
                                   global_ptr,
                                   sizeof(T) * n_elem);
#else
  if (threadIdx.x < warp_size) {
    for (int lidx = threadIdx.x; lidx < n_elem; lidx += warp_size) {
      shared_ptr[lidx] = global_ptr[lidx];
    }
  }
#endif
}

__device__ void shared_mem_wait(cooperative_groups::thread_block& block) {
#if __CUDACC_VER_MAJOR__ >= 11
  cooperative_groups::wait(block);
#else
  __syncthreads();
#endif
}
#endif

// Having more ifdefs here makes code elsewhere clearer, as this can be included
// as type in function arguments
class cuda_stream {
public:
  cuda_stream() {
#ifdef __NVCC__
    // Handle error manually, as this may be called when nvcc has been used
    // to compile, but no device is present on the executing system
    cudaError_t status = cudaStreamCreate(&stream_);
    if (status == cudaErrorNoDevice) {
      stream_ = nullptr;
    } else if (status != cudaSuccess) {
      dust::gpu::throw_cuda_error(__FILE__, __LINE__, status);
    }
#endif
  }

#ifdef __NVCC__
  ~cuda_stream() {
    if (stream_ != nullptr) {
      CUDA_CALL_NOTHROW(cudaStreamDestroy(stream_));
    }
  }

  cudaStream_t stream() {
    return stream_;
  }
#endif

  void sync() {
#ifdef __NVCC__
    CUDA_CALL(cudaStreamSynchronize(stream_));
#endif
  }

  bool query() const {
    bool ready = true;
#ifdef __NVCC__
    if (cudaStreamQuery(stream_) != cudaSuccess) {
      ready = false;
    }
#endif
    return ready;
  }

private:
  // Delete copy and move
  cuda_stream ( const cuda_stream & ) = delete;
  cuda_stream ( cuda_stream && ) = delete;

#ifdef __NVCC__
  cudaStream_t stream_;
#endif
};

}
}

#endif
