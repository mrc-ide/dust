#ifndef MODE_R_MODE_HPP
#define MODE_R_MODE_HPP

#include <cpp11/external_pointer.hpp>
#include <cpp11/strings.hpp> // required to avoid link error only
#include <cpp11/list.hpp>
#include <cpp11/integers.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/protect.hpp>

#include <dust/random/random.hpp>
#include <dust/r/helpers.hpp>
#include <dust/r/random.hpp>
#include <dust/r/utils.hpp>
#include <dust/types.hpp>
#include <dust/utils.hpp>

#include <dust/dust_ode.hpp>
#include <mode/r/helpers.hpp>

namespace mode {
namespace r {

template <typename T>
cpp11::list mode_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time,
                       cpp11::sexp r_n_particles, size_t n_threads,
                       cpp11::sexp r_seed, bool deterministic,
                       cpp11::sexp r_gpu_config, cpp11::sexp r_ode_control) {
  if (deterministic) {
    cpp11::stop("Deterministic mode not supported for mode models");
  }
  auto pars = dust::dust_pars<T>(r_pars);
  auto seed = dust::random::r::as_rng_seed<typename T::rng_state_type>(r_seed);
  auto ctl = mode::r::validate_ode_control(r_ode_control);
  const double t0 = 0;
  const auto time = dust::r::validate_time<double>(r_time, t0, "time");
  cpp11::sexp info = dust::dust_info(pars);
  dust::r::validate_positive(n_threads, "n_threads");
  auto n_particles = cpp11::as_cpp<int>(r_n_particles);
  dust::r::validate_positive(n_particles, "n_particles");
  dust_ode<T> *d = new mode::dust_ode<T>(pars, time, n_particles,
                                           n_threads, ctl, seed);
  cpp11::external_pointer<dust_ode<T>> ptr(d, true, false);
  cpp11::writable::integers r_shape =
    dust::r::vector_size_to_int(ptr->shape());
  auto r_ctl = mode::r::control(ctl);
  return cpp11::writable::list({ptr, info, r_shape, r_gpu_config, r_ctl});
}

}
}

#endif
