#ifndef DUST_R_HELPERS_ODE_HPP
#define DUST_R_HELPERS_ODE_HPP

#include <vector>

#include <cpp11/doubles.hpp>
#include <cpp11/integers.hpp>
#include <cpp11/list.hpp>

#include "dust/ode/control.hpp"
#include "dust/r/helpers.hpp"
#include "dust/r/utils.hpp"

namespace dust {
namespace r {

inline cpp11::sexp ode_statistics_array(const std::vector<size_t>& dat,
                                        size_t n_particles) {
  cpp11::writable::integers ret(dat.size());
  std::copy(dat.begin(), dat.end(), ret.begin());
  ret.attr("dim") = cpp11::writable::integers{3, static_cast<int>(n_particles)};
  ret.attr("class") = "ode_statistics";
  auto row_names = cpp11::writable::strings{"n_steps",
                                            "n_steps_accepted",
                                            "n_steps_rejected"};
  ret.attr("dimnames") = cpp11::writable::list{row_names, R_NilValue};
  return ret;
}

inline
bool validate_reset_step_size(SEXP r_time,
                              SEXP r_pars,
                              SEXP r_reset_step_size) {
  bool reset_step_size = false;
  if (r_reset_step_size == R_NilValue) {
    reset_step_size = r_time != R_NilValue || r_pars != R_NilValue;
  } else {
    reset_step_size = cpp11::as_cpp<bool>(r_reset_step_size);
  }
  return reset_step_size;
}

template <typename real_type>
dust::ode::control<real_type> validate_ode_control(cpp11::sexp r_control) {
  const auto defaults = dust::ode::control<real_type>();
  if (r_control == R_NilValue) {
    return defaults;
  }
  else {
    auto control = cpp11::as_cpp<cpp11::list>(r_control);
    auto max_steps = dust::r::validate_integer(control["max_steps"],
                                               defaults.max_steps,
                                               "max_steps");
    auto atol = dust::r::validate_double(control["atol"],
                                         defaults.atol,
                                         "atol");
    auto rtol = dust::r::validate_double(control["rtol"],
                                         defaults.rtol,
                                         "rtol");
    auto step_size_min = dust::r::validate_double(control["step_size_min"],
                                                  defaults.step_size_min,
                                                  "step_size_min");
    auto step_size_max = dust::r::validate_double(control["step_size_max"],
                                                  defaults.step_size_max,
                                                  "step_size_max");
    auto debug_record_step_times =
        dust::r::validate_logical(control["debug_record_step_times"],
                                  defaults.debug_record_step_times,
                                  "debug_record_step_times");
    return dust::ode::control<real_type>(max_steps, atol, rtol, step_size_min,
                                         step_size_max,
                                         debug_record_step_times);
  }
}

template <typename real_type>
cpp11::sexp ode_control(const dust::ode::control<real_type> ctl) {
  using namespace cpp11::literals;
  auto ret = cpp11::writable::list({"max_steps"_nm = ctl.max_steps,
                                    "atol"_nm = ctl.atol,
                                    "rtol"_nm = ctl.rtol,
                                    "step_size_min"_nm = ctl.step_size_min,
                                    "step_size_max"_nm = ctl.step_size_max,
                                    "debug_record_step_times"_nm = ctl.debug_record_step_times});

  ret.attr("class") = "dust_ode_control";
  return ret;
}

}
}

#endif
