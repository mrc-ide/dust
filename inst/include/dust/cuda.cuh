// -*- c++ -*-
#ifndef DUST_CUDA_CUH
#define DUST_CUDA_CUH

#ifdef __NVCC__
#define DEVICE __device__
#define HOST __host__
#define HOSTDEVICE __host__ __device__
#define KERNEL __global__

// This is necessary due to templates which are __host__ __device__
#define __nv_exec_check_disable__ _Pragma("nv_exec_check_disable")

#include <stdio.h>
#include <sstream>

// Standard cuda library functions
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <device_launch_parameters.h>

#include <cooperative_groups.h>
// CUDA 11 cooperative groups
#if __CUDACC_VER_MAJOR__ >= 11
#include <cooperative_groups/memcpy_async.h>
#endif

// cub functions (included with CUDA>=11)
#include <cub/device/device_select.cuh>

const int warp_size = 32;

static void HandleCUDAError(const char *file, int line,
                            cudaError_t status = cudaGetLastError()) {
#ifdef _DEBUG
  cudaDeviceSynchronize();
#endif

  if (status != CUDA_SUCCESS || (status = cudaGetLastError()) != CUDA_SUCCESS) {
    std::stringstream msg;
    if (status == cudaErrorUnknown) {
      msg << file << "(" << line << ") An Unknown CUDA Error Occurred :(";
    } else {
      msg << file << "(" << line << ") CUDA Error Occurred:\n" <<
        cudaGetErrorString(status);
    }
    cudaProfilerStop();
    throw std::runtime_error(msg.str());
  }
}

#define CUDA_CALL( err ) (HandleCUDAError(__FILE__, __LINE__ , err))

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

#else
#define DEVICE
#define HOST
#define HOSTDEVICE
#define KERNEL
#define __nv_exec_check_disable__
#endif

#endif
