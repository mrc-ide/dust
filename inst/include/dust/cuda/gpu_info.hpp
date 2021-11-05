#ifndef DUST_CUDA_DEVICE_INFO_HPP
#define DUST_CUDA_DEVICE_INFO_HPP

#include <vector>
#include <climits>
#include "dust/cuda/call.hpp"

namespace dust {
namespace gpu {

inline int devices_count() {
  int device_count = 0;
#ifdef __NVCC__
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess && status != cudaErrorNoDevice) {
    throw_cuda_error(__FILE__, __LINE__, status);
  }
#endif
  return device_count;
}

}
}

#endif
