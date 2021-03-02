// -*- c++ -*-
#ifndef DUST_CUDA_CALL_CUH
#define DUST_CUDA_CALL_CUH

#ifdef __NVCC__

#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

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

#endif

#endif
