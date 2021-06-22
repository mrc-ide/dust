#include <cpp11.hpp>
#include <dust/dust.hpp>
#include <dust/cuda_pars.hpp>

[[cpp11::register]]
SEXP test_cuda_pars(int device_id, int n_particles, int n_particles_each,
                    int n_state, int n_state_full,
                    int n_shared_int, int n_shared_real, int data_size,
                    int shared_size) {

  // n_pars_effective can be worked out from n_particles and each

  auto pars = dust::set_cuda_pars<float>(device_id, n_particles, n_particles_each,
                                         n_state, n_state_full,
                                         n_shared_int, n_shared_real, data_size,
                                         shared_size);

  using namespace cpp11::literals;
  return cpp11::writable::list({"run_blockSize"_nm = pars.run_blockSize,
                                "run_blockCount"_nm = pars.run_blockCount,
                                "run_shared_size_bytes"_nm = pars.run_shared_size_bytes,
                                "run_L1_int"_nm = pars.run_L1_int,
                                "run_L1_real"_nm = pars.run_L1_real,

                                "compare_blockSize"_nm = pars.compare_blockSize,
                                "compare_blockCount"_nm = pars.compare_blockCount,
                                "compare_shared_size_bytes"_nm = pars.compare_shared_size_bytes,
                                "compare_L1_int"_nm = pars.compare_L1_int,
                                "compare_L1_real"_nm = pars.compare_L1_real,
    });
}
