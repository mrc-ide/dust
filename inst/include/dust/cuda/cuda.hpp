#ifndef DUST_CUDA_CUDA_HPP
#define DUST_CUDA_CUDA_HPP

#ifdef __NVCC__
#define DEVICE __device__
#define HOST __host__
#define HOSTDEVICE __host__ __device__
#define KERNEL __global__
#define ALIGN(n) __align__(n)

#ifdef DUST_ENABLE_CUDA_PROFILER
#define DUST_USING_CUDA_PROFILER
#endif

// This is necessary due to templates which are __host__ __device__;
// whenever a HOSTDEVICE function is called from another HOSTDEVICE
// function the compiler gets confused as it can't tell which one it's
// going to use. This suppresses the warning as it is ok here.
#define __nv_exec_check_disable__ _Pragma("nv_exec_check_disable")

#include <dust/cuda/call.hpp>

#include <device_launch_parameters.h>

#include <cooperative_groups.h>
// CUDA 11 cooperative groups
#if __CUDACC_VER_MAJOR__ >= 11
#include <cooperative_groups/memcpy_async.h>
#endif

// cub functions (included with CUDA>=11)
#include <cub/cub.cuh>

#else
#define DEVICE
#define HOST
#define HOSTDEVICE
#define KERNEL
#undef DUST_CUDA_ENABLE_PROFILER
#define __nv_exec_check_disable__
#define ALIGN(n)
#endif

// const definition depends on __host__/__device__
#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#define SYNCWARP __syncwarp();
#else
#define CONSTANT const
#define SYNCWARP
#endif

namespace dust {
namespace cuda {

const int warp_size = 32;

#ifdef __NVCC__
template <typename T>
DEVICE void shared_mem_cpy(cooperative_groups::thread_block& block,
                           T* shared_ptr,
                           const T* global_ptr,
                           size_t n_elem) {
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

DEVICE void shared_mem_wait(cooperative_groups::thread_block& block) {
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
      dust::cuda::throw_cuda_error(__FILE__, __LINE__, status);
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
