// -*- c++ -*-
#ifndef DUST_CUDA_CUH
#define DUST_CUDA_CUH

// Align structs
// https://stackoverflow.com/a/12779757
#if defined(__CUDACC__) // NVCC
#define ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

#ifdef __NVCC__
#define DEVICE __device__
#define HOST __host__
#define HOSTDEVICE __host__ __device__
#define KERNEL __global__

// This is necessary due to templates which are __host__ __device__;
// whenever a HOSTDEVICE function is called from another HOSTDEVICE
// function the compiler gets confused as it can't tell which one it's
// going to use. This suppresses the warning as it is ok here.
#define __nv_exec_check_disable__ _Pragma("nv_exec_check_disable")

#include <dust/cuda_call.cuh>

#include <device_launch_parameters.h>

#include <cooperative_groups.h>
// CUDA 11 cooperative groups
#if __CUDACC_VER_MAJOR__ >= 11
#include <cooperative_groups/memcpy_async.h>
#endif

// cub functions (included with CUDA>=11)
#include <cub/cub.cuh>

namespace dust {
namespace cuda {

const int warp_size = 32;

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

}
}

#else
#define DEVICE
#define HOST
#define HOSTDEVICE
#define KERNEL
#undef DUST_CUDA_ENABLE_PROFILER
#define __nv_exec_check_disable__
#endif

#endif
