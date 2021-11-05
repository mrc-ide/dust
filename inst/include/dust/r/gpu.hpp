#ifndef DUST_R_GPU_HPP
#define DUST_R_GPU_HPP

#include <cpp11/data_frame.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/integers.hpp>
#include <cpp11/list.hpp>
#include <cpp11/logicals.hpp>
#include <cpp11/strings.hpp>

#include "dust/gpu/launch_control.hpp"
#include "dust/r/gpu_info.hpp"
#include "dust/r/helpers.hpp"

namespace dust {
namespace gpu {
namespace r {

inline int check_device_id(cpp11::sexp r_device_id) {
#ifdef __NVCC__
  const int device_id_max = dust::gpu::devices_count() - 1;
#else
  // We allow a device_id set to 0 to allow us to test the device
  // storage. However, if compiling with nvcc the device id must be
  // valid.
  const int device_id_max = 0;
#endif
  int device_id = cpp11::as_cpp<int>(r_device_id);
  dust::r::validate_size(device_id, "device_id");
  if (device_id > device_id_max) {
    cpp11::stop("Invalid 'device_id' %d, must be at most %d",
                device_id, device_id_max);
  }
  return device_id;
}

inline dust::gpu::gpu_config gpu_config(cpp11::sexp r_gpu_config) {
  size_t run_block_size = 128;

  cpp11::sexp r_device_id = r_gpu_config;
  if (TYPEOF(r_gpu_config) == VECSXP) {
    cpp11::list r_gpu_config_l = cpp11::as_cpp<cpp11::list>(r_gpu_config);
    r_device_id = r_gpu_config_l["device_id"]; // could error if missing?
    cpp11::sexp r_run_block_size = r_gpu_config_l["run_block_size"];
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
  return dust::gpu::gpu_config(check_device_id(r_device_id), run_block_size);
}

inline
cpp11::sexp gpu_config_as_sexp(const dust::gpu::gpu_config& config) {
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
