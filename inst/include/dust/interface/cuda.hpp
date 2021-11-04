#ifndef DUST_INTERFACE_CUDA_HPP
#define DUST_INTERFACE_CUDA_HPP

#include <cpp11/data_frame.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/integers.hpp>
#include <cpp11/list.hpp>
#include <cpp11/logicals.hpp>
#include <cpp11/strings.hpp>

#include "dust/cuda/device_info.hpp"
#include "dust/cuda/launch_control.hpp"

namespace dust {
namespace cuda {
namespace interface {

inline int check_device_id(cpp11::sexp r_device_id) {
#ifdef __NVCC__
  const int device_id_max = dust::cuda::devices_count() - 1;
#else
  // We allow a device_id set to 0 to allow us to test the device
  // storage. However, if compiling with nvcc the device id must be
  // valid.
  const int device_id_max = 0;
#endif
  int device_id = cpp11::as_cpp<int>(r_device_id);
  // TODO: would be nice to use validate_size here, but helpers.hpp
  // can't be include because the headers are still in a tangle. See #306
  if (device_id < 0) {
    cpp11::stop("Invalid 'device_id' %d, must be positive",
                device_id);
  }
  if (device_id > device_id_max) {
    cpp11::stop("Invalid 'device_id' %d, must be at most %d",
                device_id, device_id_max);
  }
  return device_id;
}

inline dust::cuda::device_config device_config(cpp11::sexp r_device_config) {
  size_t run_block_size = 128;

  cpp11::sexp r_device_id = r_device_config;
  if (TYPEOF(r_device_config) == VECSXP) {
    cpp11::list r_device_config_l = cpp11::as_cpp<cpp11::list>(r_device_config);
    r_device_id = r_device_config_l["device_id"]; // could error if missing?
    cpp11::sexp r_run_block_size = r_device_config_l["run_block_size"];
    if (r_run_block_size != R_NilValue) {
      int run_block_size_int = cpp11::as_cpp<int>(r_run_block_size);
      if (run_block_size_int < 0) {
        cpp11::stop("'run_block_size' must be positive (but was %d)",
                    run_block_size_int);
      }
      if (run_block_size_int % 32 != 0) {
        cpp11::stop("'run_block_size' must be a multiple of 32 (but was %d)",
                    run_block_size_int);
      }
      run_block_size = run_block_size_int;
    }
  }
  return dust::cuda::device_config(check_device_id(r_device_id),
                                   run_block_size);
}

inline
cpp11::sexp device_config_as_sexp(const dust::cuda::device_config& config) {
  using namespace cpp11::literals;
  return cpp11::writable::list({"real_gpu"_nm = config.real_gpu_,
                                "device_id"_nm = config.device_id_,
                                "shared_size"_nm = config.shared_size_,
                                "run_block_size"_nm = config.run_block_size_});
}

}


// TODO(306): For historical reasons, this is not in the interface
// namespace, will look into that later.
template <typename real_type>
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
    cpp11::as_sexp(sizeof(real_type) * CHAR_BIT);

  return cpp11::writable::list({"has_cuda"_nm = has_cuda,
                                "cuda_version"_nm = cuda_version,
                                "devices"_nm = devices,
                                "real_bits"_nm = real_bits});
}

}
}


#endif
