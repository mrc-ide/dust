#ifndef DUST_R_HELPERS_HPP
#define DUST_R_HELPERS_HPP

#include <map>
#include <sstream>
#include <vector>

#include <cpp11/doubles.hpp>
#include <cpp11/integers.hpp>
#include <cpp11/list.hpp>
#include <cpp11/strings.hpp>

#include "dust/r/random.hpp"
#include "dust/r/utils.hpp"

namespace dust {
namespace r {

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
      if (!cpp11::is_convertible_without_loss_to_integer(el)) {
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

// Originally from mode, would be nice if this was not needed.
template <typename T, typename U>
std::vector<T> copy_vector(U x) {
  std::vector<T> ret;
  const auto len = x.size();
  ret.reserve(len);
  for (int i = 0; i < len; ++i) {
    ret.push_back(x[i]);
  }
  return ret;
}

template <typename real_type>
inline std::vector<real_type> as_vector_real(cpp11::sexp x, const char * name) {
  if (TYPEOF(x) != INTSXP && TYPEOF(x) != REALSXP) {
    cpp11::stop("Expected a numeric vector for '%s'", name);
  }
  if (TYPEOF(x) == INTSXP) {
    return copy_vector<real_type>(cpp11::as_cpp<cpp11::integers>(x));
  } else {
    return copy_vector<real_type>(cpp11::as_cpp<cpp11::doubles>(x));
  }
}

inline
void validate_size(int x, const char * name) {
  if (x < 0) {
    cpp11::stop("'%s' must be non-negative (was given %d)", name, x);
  }
}

inline
bool validate_logical(SEXP x, bool default_value, const char * name) {
  if (x == R_NilValue) {
    return default_value;
  }
  if (TYPEOF(x) != LGLSXP || LENGTH(x) != 1) { // TODO: ideally check not missing
    cpp11::stop("'%s' must be a non-missing scalar logical", name);
  }
  return INTEGER(x)[0];
}

inline
double validate_double(SEXP x, double default_value, const char * name) {
  if (x == R_NilValue) {
    return default_value;
  }
  cpp11::doubles values = cpp11::as_cpp<cpp11::doubles>(x);
  if (values.size() != 1) {
    cpp11::stop("Expected '%s' to be a scalar value", name);
  }
  return values[0];
}

// TODO: template on return?
inline
size_t validate_integer(SEXP x, size_t default_value, const char * name) {
  if (x == R_NilValue) {
    return default_value;
  }
  cpp11::integers values = dust::r::as_integer(x, name);
  if (values.size() != 1) {
    cpp11::stop("Expected '%s' to be a scalar value", name);
  }
  validate_size(values[0], name);
  return static_cast<size_t>(values[0]);
}

inline
void validate_positive(int x, const char *name) {
  if (x <= 0) {
    cpp11::stop("'%s' must be positive (was given %d)", name, x);
  }
}

inline
std::vector<size_t> validate_size(cpp11::sexp r_x, const char * name) {
  cpp11::integers r_xi = as_integer(r_x, name);
  const size_t n = static_cast<size_t>(r_xi.size());
  std::vector<size_t> x;
  x.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    int el = r_xi[i];
    if (el < 0) {
      cpp11::stop("All elements of '%s' must be non-negative", name);
    }
    x.push_back(el);
  }
  return x;
}


inline
int r_index_check(int x, int max) {
  if (x < 1 || x > max) {
    cpp11::stop("All elements of 'index' must lie in [1, %d]", max);
  }
  return x - 1;
}


// Converts an R vector of integers (in base-1) to a C++ std::vector
// of size_t values in base-0 having checked that the values of the
// vectors are approproate; that they will not fall outside of the
// range [1, nmax] in base-1.
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
void check_dimensions_rank(size_t dim_len, size_t shape_len, bool is_list,
                           const char * name) {
  if (dim_len != shape_len) {
    std::stringstream msg;
    if (shape_len == 1) {
      if (is_list) {
        msg << "Expected a list with no dimension attribute";
      } else {
        msg << "Expected a vector";
      }
    } else if (shape_len == 2 && !is_list) {
      msg << "Expected a matrix";
    } else {
      const char * what = is_list ? "a list array" : "an array";
      msg << "Expected " << what << " of rank " << shape_len;
    }
    msg << " for '" << name << "'";
    throw std::invalid_argument(msg.str());
  }
}

inline
void check_dimensions_size(cpp11::integers dim,
                           const std::vector<size_t>& shape,
                           bool is_list,
                           const char * name) {
  for (size_t i = 0; i < shape.size(); ++i) {
    const size_t found = dim[i], expected = shape[i];
    if (found != expected) {
      std::stringstream msg;
      if (shape.size() == 1) {
        const char * what = is_list ? "list" : "vector";
        msg << "Expected a " << what << " of length " << expected <<
          " for '" << name << "' but given " << found;
      } else if (shape.size() == 2 && !is_list) {
        const char * what = i == 0 ? "rows" : "cols";
        msg << "Expected a matrix with " << expected << " " << what <<
          " for '" << name << "' but given " << found;
      } else {
        msg << "Expected dimension " << i + 1 << " of '" << name <<
          "' to be " << expected << " but given " << found;
      }
      throw std::invalid_argument(msg.str());
    }
  }
}

inline
void check_dimensions(cpp11::sexp obj, size_t obj_size,
                      const std::vector<size_t>& shape,
                      bool is_list,
                      const char * name) {
  cpp11::integers dim = object_dimensions(obj, obj_size);
  check_dimensions_rank(dim.size(), shape.size(), is_list, name);
  check_dimensions_size(dim, shape, is_list, name);
}

inline
cpp11::writable::integers state_array_dim(size_t n_state,
                                          const std::vector<size_t>& shape) {
  cpp11::writable::integers dim(shape.size() + 1);
  dim[0] = n_state;
  std::copy(shape.begin(), shape.end(), dim.begin() + 1);
  return dim;
}

inline
cpp11::writable::integers state_array_dim(size_t n_state,
                                          const std::vector<size_t>& shape,
                                          size_t n_time) {
  cpp11::writable::integers dim(shape.size() + 2);
  dim[0] = n_state;
  std::copy(shape.begin(), shape.end(), dim.begin() + 1);
  dim[dim.size() - 1] = n_time;
  return dim;
}

template <typename real_type>
cpp11::sexp state_array(const std::vector<real_type>& dat, size_t n_state,
                        const std::vector<size_t>& shape) {
  cpp11::writable::doubles ret(dat.size());
  std::copy(dat.begin(), dat.end(), REAL(ret));

  ret.attr("dim") = state_array_dim(n_state, shape);

  return ret;
}

template <typename real_type>
cpp11::sexp state_array(const std::vector<real_type>& dat, size_t n_state,
                        const std::vector<size_t>& shape, size_t n_time) {
  cpp11::writable::doubles ret(dat.size());
  std::copy(dat.begin(), dat.end(), REAL(ret));

  ret.attr("dim") = state_array_dim(n_state, shape, n_time);

  return ret;
}

inline
cpp11::writable::integers vector_size_to_int(const std::vector<size_t> & x) {
  cpp11::writable::integers ret(x.size());
  std::copy(x.begin(), x.end(), ret.begin());
  return ret;
}


template <typename T, typename U>
T validate_time(cpp11::sexp time, U time_min, const char* name);

// This bit makes sure that time vectors have at least one point and
// that they have non-decreasing times; this all holds true regardless
// of the actual types.
template <typename T>
void validate_time_vector(std::vector<T> time, T time_min, const char *name) {
  const size_t n_time = time.size();
  if (n_time == 0) {
    cpp11::stop("'%s' must have at least one element", name);
  }
  if (time[0] < time_min) {
    cpp11::stop("'%s[1]' must be at least %s",
                name, std::to_string(time_min).c_str());
  }
  for (size_t i = 1; i < n_time; ++i) {
    if (time[i] < time[i - 1]) {
      cpp11::stop("'%s' must be non-decreasing (error on element %d)",
                  name, i + 1);
    }
  }
}

template <>
inline std::vector<size_t> validate_time(cpp11::sexp r_time, size_t time_min,
                                         const char* name) {
  const std::vector<size_t> time = validate_size(r_time, name);
  validate_time_vector<size_t>(time, time_min, name);
  return time;
}

// Used only in ode models
template <>
inline std::vector<double> validate_time(cpp11::sexp r_time, double time_min,
                                         const char *name) {
  const std::vector<double> time = as_vector_real<double>(r_time, name);
  validate_time_vector<double>(time, time_min, name);
  return time;
}

template <>
inline std::vector<float> validate_time(cpp11::sexp r_time, float time_min,
                                        const char *name) {
  const std::vector<float> time = as_vector_real<float>(r_time, name);
  validate_time_vector<float>(time, time_min, name);
  return time;
}

template <>
inline size_t validate_time(cpp11::sexp r_time, size_t time_min,
                            const char* name) {
  const int time_int = cpp11::as_cpp<int>(r_time);
  dust::r::validate_size(time_int, name);
  const size_t time = static_cast<size_t>(time_int);
  if (time < time_min) {
    cpp11::stop("'%s' must be at least %s",
                name, std::to_string(time_min).c_str());
  }
  return time;
}

template <>
inline float validate_time(cpp11::sexp r_time, float time_min,
                           const char* name) {
  const float time = static_cast<float>(cpp11::as_cpp<double>(r_time));
  if (time < time_min) {
    cpp11::stop("'%s' must be at least %s",
                name, std::to_string(time_min).c_str());
  }
  return time;
}

template <>
inline double validate_time(cpp11::sexp r_time, double time_min,
                            const char* name) {
  const double time = cpp11::as_cpp<double>(r_time);
  if (time < time_min) {
    cpp11::stop("'%s' must be at least %s",
                name, std::to_string(time_min).c_str());
  }
  return time;
}

inline void check_pars_multi(cpp11::list r_pars,
                             std::vector<size_t> shape,
                             const bool pars_are_shared) {
  if (r_pars.attr("names") != R_NilValue) {
    throw std::invalid_argument("Expected an unnamed list for 'pars' (given 'pars_multi')");
  }
  if (pars_are_shared) {
    shape = std::vector<size_t>(shape.begin() + 1, shape.end());
  }
  check_dimensions(r_pars, r_pars.size(), shape, true, "pars");
}

// This version used on initialisation where we are trying to find
// dim, not check it. There are far fewer constraints in this case.
inline void check_pars_multi(cpp11::list r_pars) {
  if (r_pars.attr("names") != R_NilValue) {
    throw std::invalid_argument("Expected an unnamed list for 'pars' (given 'pars_multi')");
  }
  if (r_pars.size() == 0) {
    throw std::invalid_argument("Expected 'pars' to have at least one element");
  }
}

// TODO: consider behaviour of the recycling here (if given a vector
// of n_particles). Can this be extended to multiparameter sets
// reasonably? I guess the logic would be that *exactly* the
// n_particles dimension is missing? The other option is to set *all*
// to the same value.
//
// no pars: shape = {n_particles}
//   vector of length n_state => shared
//   matrix of n_state x n_particles => individual
//
// with pars: shape = {n_particles_each, n_pars}
//   matrix of n_state x n_pars => shared per parameter
//   array of n_state x n_particles_each x n_pars => individual
//
// with pars: shape = {n_particles_each, n_a, n_b}
//   matrix of n_state x n_a x n_b => shared per parameter
//   array of n_state x n_particles_each x n_a x n_b => individual
template <typename real_type>
std::vector<real_type> check_state(cpp11::sexp r_state, size_t n_state,
                                const std::vector<size_t>& shape,
                                const bool is_shared) {
  cpp11::doubles r_state_data = cpp11::as_cpp<cpp11::doubles>(r_state);
  const size_t state_len = r_state_data.size();

  std::vector<size_t> shape_check{n_state};
  if (is_shared) {
    const size_t dim_len = object_dimensions(r_state, state_len).size();
    const size_t len_shared = shape.size(), len_individual = shape.size() + 1;
    if (dim_len == len_individual) {
      shape_check.insert(shape_check.end(), shape.begin(), shape.end());
    } else if (dim_len == len_shared) {
      shape_check.insert(shape_check.end(), shape.begin() + 1, shape.end());
    } else {
      cpp11::stop("Expected array of rank %d or %d for 'state'",
                  len_shared, len_individual);
    }
  } else {
    shape_check.insert(shape_check.end(), shape.begin(), shape.end());
  }
  check_dimensions(r_state, state_len, shape_check, false, "state");

  std::vector<real_type> ret(state_len);
  std::copy_n(REAL(r_state_data.data()), state_len, ret.begin());
  return ret;
}

// There are few options here, and this transform applies to other
// things (reorder weights)
//
// single: expect a vector of length shape[0] (n_particles)
//
// groups: expect a matrix of size n_particles x n_groups (this is shape)
inline
std::vector<size_t> check_reorder_index(cpp11::sexp r_index,
                                        const std::vector<size_t>& shape) {
  cpp11::integers r_index_data = as_integer(r_index, "index");
  const size_t len = r_index_data.size();
  check_dimensions(r_index, len, shape, false, "index");

  const size_t n_particles = shape[0];
  const size_t n_groups = len  / n_particles;
  std::vector<size_t> index;
  index.reserve(len);
  for (size_t i = 0, j = 0; i < n_groups; ++i) {
    for (size_t k = 0; k < n_particles; ++j, ++k) {
      int x = r_index_check(r_index_data[j], n_particles);
      index.push_back(i * n_particles + x);
    }
  }

  return index;
}

template <typename real_type>
std::vector<real_type> check_resample_weights(cpp11::doubles r_weights,
                                           const std::vector<size_t>& shape) {
  const size_t len = r_weights.size();
  check_dimensions(r_weights, len, shape, false, "weights");
  if (*std::min_element(r_weights.begin(), r_weights.end()) < 0) {
    cpp11::stop("All weights must be positive");
  }
  const std::vector<real_type>
    weights(r_weights.begin(), r_weights.end());
  return weights;
}

template <typename T>
std::vector<size_t> check_time_snapshot(cpp11::sexp r_time_snapshot,
                                        const std::map<size_t, T>& data) {
  std::vector<size_t> time_snapshot;
  if (r_time_snapshot == R_NilValue) {
    return time_snapshot;
  }

  cpp11::integers r_time_snapshot_int =
    as_integer(r_time_snapshot, "time_snapshot");

  time_snapshot.reserve(r_time_snapshot_int.size());
  for (int i = 0; i < r_time_snapshot_int.size(); ++i) {
    const int time = r_time_snapshot_int[i];
    if (time < 0) {
      cpp11::stop("'time_snapshot' must be positive");
    }
    if (i > 0 && time <= r_time_snapshot_int[i - 1]) {
      cpp11::stop("'time_snapshot' must be strictly increasing");
    }
    if (data.find(time) == data.end()) {
      cpp11::stop("'time_snapshot[%d]' (time %d) was not found in data",
                  i + 1, time);
    }
    time_snapshot.push_back(time);
  }

  return time_snapshot;
}

template <typename real_type>
std::vector<real_type>
check_min_log_likelihood(cpp11::sexp r_min_log_likelihood, size_t n_pars) {
  std::vector<real_type> min_log_likelihood;
  if (r_min_log_likelihood != R_NilValue) {
    cpp11::doubles r_min_log_likelihood_vec =
      cpp11::as_cpp<cpp11::doubles>(r_min_log_likelihood);
    if (n_pars == 0) {
      n_pars = 1;
    }
    const size_t n_given = r_min_log_likelihood_vec.size();
    if (n_given != 1 && n_given != n_pars) {
      if (n_pars <= 1) { // avoid unfriendly error message
        cpp11::stop("'min_log_likelihood' must have length 1 (but given %d)",
                    n_given);
      } else {
        cpp11::stop("'min_log_likelihood' must have length 1 or %d (but given %d)",
                    n_pars, n_given);
      }
    }
    min_log_likelihood.reserve(n_given);
    for (auto x : r_min_log_likelihood_vec) {
      min_log_likelihood.push_back(x);
    }
  }

  return min_log_likelihood;
}

// It's possible that we could create a helper that looks to see if we
// have a deriv or an update method and then remove the need for the
// second template parameter here, but it's not that bad.
template <typename T, typename time_type>
struct dust_inputs {
  std::vector<dust::pars_type<T>> pars;
  time_type time;
  size_t n_particles;
  size_t n_threads;
  std::vector<typename T::rng_state_type::int_type> seed;
  std::vector<size_t> shape;
  cpp11::sexp info;
};

template <typename T, typename time_type>
dust_inputs<T, time_type> process_inputs_single(cpp11::list r_pars, cpp11::sexp r_time,
                                                cpp11::sexp r_n_particles,
                                                int n_threads, cpp11::sexp r_seed) {
  const time_type t0 = 0;
  const time_type time = dust::r::validate_time<time_type>(r_time, t0, "time");
  dust::r::validate_positive(n_threads, "n_threads");
  std::vector<typename T::rng_state_type::int_type> seed =
    dust::random::r::as_rng_seed<typename T::rng_state_type>(r_seed);

  std::vector<dust::pars_type<T>> pars;
  pars.push_back(dust::dust_pars<T>(r_pars));
  auto info = dust::dust_info<T>(pars[0]);
  auto n_particles = cpp11::as_cpp<int>(r_n_particles);
  dust::r::validate_positive(n_particles, "n_particles");
  std::vector<size_t> shape; // empty

  return dust_inputs<T, time_type>{
    pars,
    time,
    static_cast<size_t>(n_particles),
    static_cast<size_t>(n_threads),
    seed,
    shape,
    info
  };
}

template <typename T, typename time_type>
dust_inputs<T, time_type> process_inputs_multi(cpp11::list r_pars, cpp11::sexp r_time,
                                    cpp11::sexp r_n_particles,
                                    int n_threads, cpp11::sexp r_seed) {
  const time_type t0 = 0;
  const time_type time = dust::r::validate_time<time_type>(r_time, t0, "time");

  dust::r::validate_positive(n_threads, "n_threads");
  std::vector<typename T::rng_state_type::int_type> seed =
    dust::random::r::as_rng_seed<typename T::rng_state_type>(r_seed);

  dust::r::check_pars_multi(r_pars);
  std::vector<dust::pars_type<T>> pars;
  cpp11::writable::list info = cpp11::writable::list(r_pars.size());
  for (int i = 0; i < r_pars.size(); ++i) {
    pars.push_back(dust_pars<T>(r_pars[i]));
    info[i] = dust_info<T>(pars[i]);
  }
  cpp11::sexp dim_pars = r_pars.attr("dim");
  std::vector<size_t> shape;
  if (dim_pars == R_NilValue) {
    shape.push_back(pars.size());
  } else {
    cpp11::integers dim_pars_int = cpp11::as_cpp<cpp11::integers>(dim_pars);
    for (int i = 0; i < dim_pars_int.size(); ++i) {
      shape.push_back(dim_pars_int[i]);
    }
  }
  size_t n_particles = 0;
  if (r_n_particles != R_NilValue) {
    n_particles = cpp11::as_cpp<int>(r_n_particles);
    dust::r::validate_size(n_particles, "n_particles");
  }
  return dust_inputs<T, time_type>{
    pars,
    time,
    static_cast<size_t>(n_particles),
    static_cast<size_t>(n_threads),
    seed,
    shape,
    info};
}

// Can replace with std::make_integer_sequence(n) with C++14
inline std::vector<size_t> sequence(size_t n) {
  std::vector<size_t> ret;
  ret.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    ret.push_back(i);
  }
  return ret;
}

// Really unpleasant bit of C++ hackery to test whether or not the
// 'adjoint_size' method exists. Other approaches exist, but before
// C++20 they're all pretty ugly. This one works with C++11 and up.
// https://stackoverflow.com/a/257382
template <typename T>
class has_adjoint {
  typedef char one;
  struct two { char x[2]; };

  template <typename C> static one test( decltype(&C::adjoint_size) ) ;
  template <typename C> static two test(...);

public:
  enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

}
}

#endif
