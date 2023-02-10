/// IMPORTANT; changes here must be reflected into inst/template/dust.hpp
#include <dust/r/dust.hpp>

// This is temporary, while we handle the merge
#include <mode/r/mode.hpp>
namespace dust {

namespace r {
template <typename T>
cpp11::list dust_ode_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp time,
                           cpp11::sexp r_n_particles, int n_threads,
                           cpp11::sexp r_seed, bool deterministic,
                           cpp11::sexp r_gpu_config,
                           cpp11::sexp r_ode_control) {
  return mode::r::mode_alloc<T>(r_pars, pars_multi, time, r_n_particles,
                                n_threads, r_seed, deterministic, r_gpu_config,
                                r_ode_control);
}

}

template <typename T>
using dust_ode = mode::dust_ode<T>;

}

/// Can we move this later? in that case we can simplify a little.
{{model}}

cpp11::sexp dust_{{name}}_capabilities() {
  return dust::r::dust_capabilities<{{class}}>();
}

cpp11::sexp dust_{{name}}_gpu_info() {
  return dust::gpu::r::gpu_info();
}
