#ifndef DUST_DEVICE_INFO_HPP
#define DUST_DEVICE_INFO_HPP

#include <vector>

#include <cpp11/integers.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/logicals.hpp>
#include <cpp11/list.hpp>
#include <cpp11/data_frame.hpp>
#include <dust/cuda_call.cuh>

template <typename T>
cpp11::sexp dust_device_info() {
  using namespace cpp11::literals;
  cpp11::writable::logicals has_cuda(1);
#ifdef __NVCC__
  int device_count;
  cudaError_t status = cudaGetDeviceCount(&device_count);

  if (status == cudaErrorNoDevice) {
    device_count = 0;
  } else if (status != cudaSuccess) {
    throw_cuda_error(__FILE__, __LINE__, status);
  }

  cpp11::writable::integers ids(device_count);
  cpp11::writable::doubles memory(device_count);
  cpp11::writable::strings names(device_count);
  cpp11::writable::integers version(device_count);

  if (device_count > 0) {
    for (int i = 0; i < device_count; ++i) {
      cudaDeviceProp properties;
      CUDA_CALL(cudaGetDeviceProperties(&properties, i));
      ids[i] = i;
      names[i] = properties.name;
      memory[i] = static_cast<double>(properties.totalGlobalMem) /
        (1024 * 1024);
      version[i] = properties.major * 10 + properties.minor;
    }
  }

  cpp11::writable::integers cuda_version_int(3);
  cuda_version_int[0] = __CUDACC_VER_MAJOR__;
  cuda_version_int[1] = __CUDACC_VER_MINOR__;
  cuda_version_int[2] = __CUDACC_VER_BUILD__;
  cpp11::writable::list cuda_version({cuda_version_int});
  cuda_version.attr("class") = "numeric_version";

  cpp11::writable::data_frame devices({
    "id"_nm = ids,
    "name"_nm = names,
    "memory"_nm = memory,
    "version"_nm = version
    });

  has_cuda[0] = true;
#else
  has_cuda[0] = false;
  cpp11::sexp cuda_version = R_NilValue;
  cpp11::sexp devices = R_NilValue;
#endif
  return cpp11::writable::list({"has_cuda"_nm = has_cuda,
                                "cuda_version"_nm = cuda_version,
                                "devices"_nm = devices});
}

#endif
