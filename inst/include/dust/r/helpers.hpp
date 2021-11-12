#ifndef DUST_R_HELPERS_HPP
#define DUST_R_HELPERS_HPP

#include <map>
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
void validate_size(int x, const char * name) {
  if (x < 0) {
    cpp11::stop("'%s' must be non-negative (was given %d)", name, x);
  }
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
    int x = r_index_int[i];
    if (x < 1 || x > (int)nmax) {
      cpp11::stop("All elements of 'index' must lie in [1, %d]", nmax);
    }
    index.push_back(x - 1);
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
    if (shape_len == 1) {
      if (is_list) {
        cpp11::stop("Expected a list with no dimension attribute for '%s'",
                    name);
      } else {
        cpp11::stop("Expected a vector for '%s'", name);
      }
    } else if (shape_len == 2 && !is_list) {
      cpp11::stop("Expected a matrix for '%s'", name);
    } else {
      const char * type = is_list ? "a list array" : "an array";
      cpp11::stop("Expected %s of rank %d for '%s'",
                  type, shape_len, name);
    }
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
      if (shape.size() == 1) {
        const char * type = is_list ? "list" : "vector";
        cpp11::stop("Expected a %s of length %d for '%s' but given %d",
                    type, expected, name, found);
      } else if (shape.size() == 2 && !is_list) {
        const char * what = i == 0 ? "rows" : "cols";
        cpp11::stop("Expected a matrix with %d %s for '%s' but given %d",
                    expected, what, name, found);
      } else {
        cpp11::stop("Expected dimension %d of '%s' to be %d but given %d",
                    i + 1, name, expected, found);
      }
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

inline void check_pars_multi(cpp11::list r_pars,
                             std::vector<size_t> shape,
                             const bool pars_are_shared) {
  if (r_pars.attr("names") != R_NilValue) {
    cpp11::stop("Expected an unnamed list for 'pars' (given 'pars_multi')");
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
    cpp11::stop("Expected an unnamed list for 'pars' (given 'pars_multi')");
  }
  if (r_pars.size() == 0) {
    cpp11::stop("Expected 'pars' to have at least one element");
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
      int x = r_index_data[j];
      if (x < 1 || x > (int)n_particles) {
        cpp11::stop("All elements of 'index' must lie in [1, %d]", n_particles);
      }
      index.push_back(i * n_particles + x - 1);
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
std::vector<size_t> check_step_snapshot(cpp11::sexp r_step_snapshot,
                                        const std::map<size_t, T>& data) {
  std::vector<size_t> step_snapshot;
  if (r_step_snapshot == R_NilValue) {
    return step_snapshot;
  }

  cpp11::integers r_step_snapshot_int =
    as_integer(r_step_snapshot, "step_snapshot");

  step_snapshot.reserve(r_step_snapshot_int.size());
  for (int i = 0; i < r_step_snapshot_int.size(); ++i) {
    const int step = r_step_snapshot_int[i];
    if (step < 0) {
      cpp11::stop("'step_snapshot' must be positive");
    }
    if (i > 0 && step <= r_step_snapshot_int[i - 1]) {
      cpp11::stop("'step_snapshot' must be strictly increasing");
    }
    if (data.find(step) == data.end()) {
      cpp11::stop("'step_snapshot[%d]' (step %d) was not found in data",
                  i + 1, step);
    }
    step_snapshot.push_back(step);
  }

  return step_snapshot;
}

template <typename T>
struct dust_inputs {
  std::vector<dust::pars_type<T>> pars;
  size_t step;
  size_t n_particles;
  size_t n_threads;
  std::vector<typename T::rng_state_type::int_type> seed;
  std::vector<size_t> shape;
  cpp11::sexp info;
};

template <typename T>
dust_inputs<T> process_inputs_single(cpp11::list r_pars, int step,
                                     cpp11::sexp r_n_particles,
                                     int n_threads, cpp11::sexp r_seed) {
  dust::r::validate_size(step, "step");
  dust::r::validate_positive(n_threads, "n_threads");
  std::vector<typename T::rng_state_type::int_type> seed =
    dust::random::r::as_rng_seed<typename T::rng_state_type>(r_seed);

  std::vector<dust::pars_type<T>> pars;
  pars.push_back(dust::dust_pars<T>(r_pars));
  auto info = dust::dust_info<T>(pars[0]);
  auto n_particles = cpp11::as_cpp<int>(r_n_particles);
  dust::r::validate_positive(n_particles, "n_particles");
  std::vector<size_t> shape; // empty
  return dust_inputs<T>{
    pars,
    static_cast<size_t>(step),
    static_cast<size_t>(n_particles),
    static_cast<size_t>(n_threads),
    seed,
    shape,
    info};
}

template <typename T>
dust_inputs<T> process_inputs_multi(cpp11::list r_pars, int step,
                                    cpp11::sexp r_n_particles,
                                    int n_threads, cpp11::sexp r_seed) {
  dust::r::validate_size(step, "step");
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
  return dust_inputs<T>{
    pars,
    static_cast<size_t>(step),
    static_cast<size_t>(n_particles),
    static_cast<size_t>(n_threads),
    seed,
    shape,
    info};
}

}
}

#endif
