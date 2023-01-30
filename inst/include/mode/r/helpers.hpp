#include <cpp11/external_pointer.hpp>
#include <cpp11/strings.hpp> // required to avoid link error only
#include <cpp11/list.hpp>
#include <cpp11/integers.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/logicals.hpp>

namespace mode {
namespace r {

inline
int r_index_check(int x, int max) {
  if (x < 1 || x > max) {
    cpp11::stop("All elements of 'index' must lie in [1, %d]", max);
  }
  return x - 1;
}

inline
void validate_positive(int x, const char *name) {
  if (x <= 0) {
    cpp11::stop("'%s' must be positive (was given %d)", name, x);
  }
}

inline
cpp11::integers as_integer(cpp11::sexp x, const char * name) {
  if (TYPEOF(x) == INTSXP) {
    return cpp11::as_cpp<cpp11::integers>(x);
  } else if (TYPEOF(x) == REALSXP) {
    cpp11::doubles xn = cpp11::as_cpp<cpp11::doubles>(x);
    size_t len = xn.size();
    cpp11::writable::integers ret = cpp11::writable::integers(len);
    for (size_t i = 0; i < len; ++i) {
      double el = xn[i];
      if (!cpp11::is_convertable_without_loss_to_integer(el)) {
        cpp11::stop("All elements of '%s' must be integer-like",
                    name, i + 1);
      }
      ret[i] = static_cast<int>(el);
    }
    return ret;
  } else {
    cpp11::stop("Expected a numeric vector for '%s'", name);
    return cpp11::integers(); // never reached
  }
}

inline
std::vector<size_t> r_index_to_index(cpp11::sexp r_index, size_t nmax) {
  cpp11::integers r_index_int = as_integer(r_index, "index");
  const int n = r_index_int.size();
  std::vector<size_t> index;
  index.reserve(n);
  for (int i = 0; i < n; ++i) {
    int x = r_index_check(r_index_int[i], nmax);
    index.push_back(x);
  }
  return index;
}

inline
cpp11::integers object_dimensions(cpp11::sexp obj, size_t obj_size) {
  cpp11::integers dim;
  auto r_dim = obj.attr("dim");
  if (r_dim == R_NilValue) {
    dim = cpp11::writable::integers{static_cast<int>(obj_size)};
  } else {
    dim = cpp11::as_cpp<cpp11::integers>(r_dim);
  }
  return dim;
}

inline
cpp11::sexp state_array(const std::vector<double>& dat,
                        size_t n_state,
                        size_t n_particles) {
  cpp11::writable::doubles ret(dat.size());
  std::copy(dat.begin(), dat.end(), REAL(ret));

  ret.attr("dim") = cpp11::writable::integers{static_cast<int>(n_state),
                                              static_cast<int>(n_particles)};

  return ret;
}

inline
cpp11::sexp state_array(const std::vector<double>& dat,
                        size_t n_state, size_t n_particles, size_t n_time) {
  cpp11::writable::doubles ret(dat.size());
  std::copy(dat.begin(), dat.end(), REAL(ret));

  ret.attr("dim") = cpp11::writable::integers{static_cast<int>(n_state),
                                              static_cast<int>(n_particles),
                                              static_cast<int>(n_time)};

  return ret;
}

inline
cpp11::sexp stats_array(const std::vector<size_t>& dat,
                        size_t n_particles) {
  cpp11::writable::integers ret(dat.size());
  std::copy(dat.begin(), dat.end(), ret.begin());
  ret.attr("dim") = cpp11::writable::integers{3, static_cast<int>(n_particles)};
  ret.attr("class") = "mode_statistics";
  auto row_names = cpp11::writable::strings{"n_steps",
                                            "n_steps_accepted",
                                            "n_steps_rejected"};
  ret.attr("dimnames") = cpp11::writable::list{row_names, R_NilValue};
  return ret;
}

inline
std::vector<double> validate_state(cpp11::sexp r_state,
                                   int n_var,
                                   int n_state_full,
                                   int n_particles) {
  std::vector<double> ret;
  if (r_state == R_NilValue) {
    return ret;
  }
  cpp11::doubles r_state_data = cpp11::as_cpp<cpp11::doubles>(r_state);
  size_t state_len = r_state_data.size();
  auto dim = object_dimensions(r_state, n_var);
  if (dim.size() > 2) {
    cpp11::stop("Expected 'state' to have at most 2 dimensions");
  }
  if (dim.size() == 2) {
    auto n_state = dim[0];
    if (n_state == n_state_full) {
      n_state = n_var;
      state_len = n_state * n_particles;
    }
    if ((n_state != n_var && n_state != n_state_full) || dim[1] != n_particles) {
      cpp11::stop("Expected 'state' to be a %d by %d matrix but was %d by %d",
                  n_var, n_particles, dim[0], dim[1]);
    }
    std::vector<double> ret(state_len);
    auto data = REAL(r_state_data.data());
    auto it = ret.begin();
    for (int i = 0; i < n_particles; ++i) {
        it = std::copy_n(data, n_state, it);
        data += dim[0];
    }
    return ret;
  }
  if (dim.size() == 1) {
    auto len = static_cast<int>(state_len);
    if (len == n_state_full) {
      state_len = n_var;
    } else if (len != n_var) {
      cpp11::stop(
          "Expected 'state' to be a vector of length %d but was length %d",
          n_var, state_len);
    }
    std::vector<double> ret(state_len);
    std::copy_n(REAL(r_state_data.data()), state_len, ret.begin());
    return ret;
  }
  // Never get here but keeps compiler happy
  return ret; // #nocov
}

inline
std::vector<double> validate_time(cpp11::sexp r_time) {
  if (r_time == R_NilValue) {
    return std::vector<double>(0);
  }
  // See similar issues in as_integer, where we need to drop down to
  // the C macros.
  if (LENGTH(r_time) != 1) {
    cpp11::stop("Expected 'time' to be a scalar value");
  }
  double time;
  if (TYPEOF(r_time) == INTSXP) {
    time = INTEGER(r_time)[0];
  } else {
    time = cpp11::as_cpp<cpp11::doubles>(r_time)[0];
  }
  return std::vector<double> {time};
}

inline
bool validate_set_initial_state(SEXP r_state, SEXP r_pars, SEXP r_time,
                                SEXP r_set_initial_state) {
  bool set_initial_state = false;
  if (r_set_initial_state == R_NilValue) {
    set_initial_state = r_state == R_NilValue &&
        (r_time != R_NilValue || r_pars != R_NilValue);
  } else {
    set_initial_state = cpp11::as_cpp<bool>(r_set_initial_state);
    if (set_initial_state && r_state != R_NilValue) {
      cpp11::stop("'set_initial_state' cannot be TRUE unless 'state' is NULL");
    }
  }
  return set_initial_state;
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

inline
size_t validate_int(SEXP r_value, size_t default_value, const char * name) {
  size_t final_value = default_value;
  if (r_value == R_NilValue) {
    return final_value;
  }
  cpp11::integers values_int = as_integer(r_value, name);
  if (values_int.size() != 1) {
    cpp11::stop("Expected '%s' to be a scalar value", name);
  }
  return static_cast<size_t>(values_int[0]);
}

inline
double validate_double(SEXP r_value, double default_value, const char * name) {
  double final_value = default_value;
  if (r_value == R_NilValue) {
    return final_value;
  }
  cpp11::doubles values = cpp11::as_cpp<cpp11::doubles>(r_value);
  if (values.size() != 1) {
    cpp11::stop("Expected '%s' to be a scalar value", name);
  }
  return values[0];
}

inline
bool validate_logical(SEXP r_value, bool default_value, const char * name) {
  bool final_value = default_value;
  if (r_value == R_NilValue) {
    return final_value;
  }
  cpp11::logicals values = cpp11::as_cpp<cpp11::logicals>(r_value);
  if (values.size() != 1) {
    cpp11::stop("Expected '%s' to be a scalar value", name);
  }
  return values[0];
}

inline
mode::control validate_ode_control(cpp11::sexp r_control) {
  const auto defaults = mode::control();
  if (r_control == R_NilValue) {
    return defaults;
  }
  else {
    auto control = cpp11::as_cpp<cpp11::list>(r_control);
    auto max_steps = mode::r::validate_int(control["max_steps"],
                                           defaults.max_steps,
                                           "max_steps");
    auto atol = mode::r::validate_double(control["atol"],
                                         defaults.atol,
                                         "atol");
    auto rtol = mode::r::validate_double(control["rtol"],
                                         defaults.rtol,
                                         "rtol");
    auto step_size_min = mode::r::validate_double(control["step_size_min"],
                                                  defaults.step_size_min,
                                                  "step_size_min");
    auto step_size_max = mode::r::validate_double(control["step_size_max"],
                                                  defaults.step_size_max,
                                                  "step_size_max");
    auto debug_record_step_times =
        mode::r::validate_logical(control["debug_record_step_times"],
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
