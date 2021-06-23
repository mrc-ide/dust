#include <cpp11.hpp>
#include <dust/dust.hpp>
#include <dust/cuda_pars.hpp>

cpp11::list launch_r_list(const dust::cuda_launch_control& p) {
  using namespace cpp11::literals;
  return cpp11::writable::list({"block_size"_nm = p.block_size,
                                "block_count"_nm = p.block_count,
                                "shared_size_bytes"_nm = p.shared_size_bytes,
                                "shared_int"_nm = p.shared_int,
                                "shared_real"_nm = p.shared_real});
}

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
  return cpp11::writable::list({"run"_nm = launch_r_list(pars.run),
                                "compare"_nm = launch_r_list(pars.compare)});
}
