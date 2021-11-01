#ifndef DUST_INTERFACE_CUDA_HPP
#define DUST_INTERFACE_CUDA_HPP

#include <dust/cuda/device_info.hpp>
#include <dust/cuda/launch_control.hpp>

namespace dust {
namespace cuda {
namespace interface {

inline int check_device_id(cpp11::sexp r_device_id) {
#ifdef __NVCC__
  const int n_devices = dust::cuda::devices_count();
#else
  // We allow a device_id set to 0 to allow us to test the device
  // storage. However, if compiling with nvcc the device id must be
  // valid.
  const int n_devices = 1;
#endif
  const int device_id_max = n_devices - 1;
  int device_id = -1;
  if (r_device_id == R_NilValue) {
    device_id = device_id_max > 0 ? 0 : -1;
  } else {
    device_id = cpp11::as_cpp<int>(r_device_id);
    if (device_id > device_id_max) {
      cpp11::stop("Invalid 'device_id' %d, must be at most %d",
                  device_id, device_id_max);
    }
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
}
}


#endif
