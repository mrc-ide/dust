#ifndef DUST_DEVICE_INFO_HPP
#define DUST_DEVICE_INFO_HPP

#include <cpp11/integers.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/list.hpp>
#include <dust/cuda.cuh>

template <typename T>
cpp11::sexp dust_device_info() {
#ifdef __NVCC__
  int device_count;
  CUDA_CALL(cudaGetDeviceCount(&device_count));

  cpp11::writable::integers ids(device_count);
  cpp11::writable::doubles memory(device_count);
  cpp11::writable::strings names(device_count);

  if (device_count > 0) {
    for (int i = 0; i < device_count; ++i) {
      cudaDeviceProp properties;
      CUDA_CALL(cudaGetDeviceProperties(&properties, i));
      ids[i] = i;
      names[i] = properties.name;
      memory[i] = static_cast<double>(properties.totalGlobalMem) / (1024 * 1024);
    }
  }
  using namespace cpp11::literals;
  return cpp11::writable::list({
    "id"_nm = ids,
    "name"_nm = names,
    "memory_mb"_nm = memory
  });
#else
  return R_NilValue;
#endif
}

#endif
