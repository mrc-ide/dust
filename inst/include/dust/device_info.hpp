#ifndef DUST_DEVICE_INFO_HPP
#define DUST_DEVICE_INFO_HPP

#include <vector>
#include <climits>

#include <cpp11/integers.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/logicals.hpp>
#include <cpp11/list.hpp>
#include <cpp11/data_frame.hpp>
#include <dust/cuda_call.cuh>

namespace dust {
namespace cuda {

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

template <typename real_t>
cpp11::sexp device_info() {
  using namespace cpp11::literals;
  cpp11::writable::logicals has_cuda(1);
  int device_count = devices_count();

  cpp11::writable::integers ids(device_count);
  cpp11::writable::doubles memory(device_count);
  cpp11::writable::strings names(device_count);
  cpp11::writable::integers version(device_count);

#ifdef __NVCC__
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

  has_cuda[0] = true;
  cpp11::writable::integers cuda_version_int(3);
  cuda_version_int[0] = __CUDACC_VER_MAJOR__;
  cuda_version_int[1] = __CUDACC_VER_MINOR__;
  cuda_version_int[2] = __CUDACC_VER_BUILD__;
  cpp11::writable::list cuda_version({cuda_version_int});
  cuda_version.attr("class") = "numeric_version";
#else
  has_cuda[0] = false;
  cpp11::sexp cuda_version = R_NilValue;
#endif

  cpp11::writable::data_frame devices({
    "id"_nm = ids,
    "name"_nm = names,
    "memory"_nm = memory,
    "version"_nm = version
    });

  cpp11::writable::integers real_bits =
    cpp11::as_sexp(sizeof(real_t) * CHAR_BIT);

  return cpp11::writable::list({"has_cuda"_nm = has_cuda,
                                "cuda_version"_nm = cuda_version,
                                "devices"_nm = devices,
                                "real_bits"_nm = real_bits});
}

}
}

#endif
