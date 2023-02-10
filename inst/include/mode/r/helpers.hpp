#ifndef MODE_R_HELPERS_HPP
#define MODE_R_HELPERS_HPP

#include <cpp11/external_pointer.hpp>
#include <cpp11/strings.hpp> // required to avoid link error only
#include <cpp11/list.hpp>
#include <cpp11/integers.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/logicals.hpp>

#include <vector>
#include "dust/ode/control.hpp"
#include "dust/r/helpers.hpp"

namespace mode {
namespace r {

inline
mode::control validate_ode_control(cpp11::sexp r_control) {
  const auto defaults = mode::control();
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
    return mode::control(max_steps, atol, rtol, step_size_min,
                         step_size_max, debug_record_step_times);
  }
}

inline
cpp11::sexp control(const mode::control ctl) {
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
