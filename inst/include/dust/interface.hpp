#ifndef DUST_INTERFACE_HPP
#define DUST_INTERFACE_HPP

#include <cstring>
#include <unordered_map>
#include <cpp11/doubles.hpp>
#include <cpp11/external_pointer.hpp>
#include <cpp11/integers.hpp>
#include <cpp11/list.hpp>
#include <cpp11/matrix.hpp>
#include <cpp11/raws.hpp>
#include <cpp11/strings.hpp>

#include <dust/rng_interface.hpp>

template <typename T>
typename dust::pars_t<T> dust_pars(cpp11::list pars);

template <typename T>
typename T::data_t dust_data(cpp11::list data);

template <typename T>
typename cpp11::sexp dust_info(const dust::pars_t<T>& pars);

inline void validate_size(int x, const char * name);
inline void validate_positive(int x, const char *name);
inline std::vector<size_t> validate_size(cpp11::sexp x, const char *name);
inline std::vector<size_t> r_index_to_index(cpp11::sexp r_index, size_t nmax);
inline std::vector<size_t> r_index_to_index_default(size_t n);
inline std::vector<size_t> r_index_reorder_matrix(cpp11::sexp r_index,
                                                  const size_t n_particles,
                                                  const size_t n_pars);
inline cpp11::integers as_integer(cpp11::sexp x, const char * name);

template <typename T>
std::vector<T> matrix_to_vector(cpp11::doubles_matrix x);

template <typename T>
cpp11::sexp create_matrix(size_t nrow, size_t ncol, T& pars);

template <typename T>
cpp11::sexp create_array(const std::vector<size_t>& dim, T& pars);

template <typename T>
cpp11::list dust_alloc(cpp11::list r_pars, bool pars_multi, int step,
                       int n_particles, int n_threads,
                       cpp11::sexp r_seed) {
  validate_size(step, "step");
  validate_positive(n_particles, "n_particles");
  validate_positive(n_threads, "n_threads");
  std::vector<uint64_t> seed = as_rng_seed<typename T::real_t>(r_seed);

  Dust<T> *d = nullptr;
  cpp11::sexp info;
  if (pars_multi) {
    std::vector<dust::pars_t<T>> pars;
    cpp11::writable::list info_list = cpp11::writable::list(r_pars.size());
    for (int i = 0; i < r_pars.size(); ++i) {
      pars.push_back(dust_pars<T>(r_pars[i]));
      info_list[i] = dust_info<T>(pars[i]);
    }
    info = info_list;
    d = new Dust<T>(pars, step, n_particles, n_threads, seed);
  } else {
    dust::pars_t<T> pars = dust_pars<T>(r_pars);
    d = new Dust<T>(pars, step, n_particles, n_threads, seed);
    info = dust_info<T>(pars);
  }
  cpp11::external_pointer<Dust<T>> ptr(d, true, false);

  return cpp11::writable::list({ptr, info});
}

// think about rng seeding here carefully; should accept either a raw
// vector or an integer I think.
template <typename T>
cpp11::writable::doubles dust_simulate(cpp11::sexp r_steps,
                                       cpp11::list r_pars,
                                       cpp11::doubles_matrix r_state,
                                       cpp11::sexp r_index,
                                       const size_t n_threads,
                                       cpp11::sexp r_seed,
                                       bool return_state) {
  typedef typename T::real_t real_t;
  std::vector<size_t> steps = validate_size(r_steps, "steps");
  std::vector<real_t> state = matrix_to_vector<real_t>(r_state);
  std::vector<size_t> index = r_index_to_index(r_index, r_state.nrow());
  std::vector<uint64_t> seed = as_rng_seed<typename T::real_t>(r_seed);

  if (r_pars.size() != r_state.ncol()) {
    cpp11::stop("Expected 'state' to be a matrix with %d columns",
                r_pars.size());
  }

  std::vector<dust::pars_t<T>> pars;
  pars.reserve(r_pars.size());
  for (int i = 0; i < r_pars.size(); ++i) {
    pars.push_back(dust_pars<T>(r_pars[i]));
  }

  cpp11::writable::doubles ret(index.size() * pars.size() * steps.size());

  std::vector<real_t> dat =
    dust_simulate<T>(steps, pars, state, index, n_threads, seed, return_state);
  std::copy(dat.begin(), dat.end(), REAL(ret));

  ret.attr("dim") = cpp11::writable::integers({(int)index.size(),
                                               (int)pars.size(),
                                               (int)steps.size()});

  if (return_state) {
    cpp11::writable::doubles r_state_end(state.size());
    std::copy(state.begin(), state.end(), REAL(r_state_end));
    r_state_end.attr("dim") = cpp11::writable::integers({
        r_state.nrow(), r_state.ncol()});
    ret.attr("state") = r_state_end;

    size_t n_rng_state = sizeof(uint64_t) * seed.size();
    cpp11::writable::raws r_rng_state(n_rng_state);
    std::memcpy(RAW(r_rng_state), seed.data(), n_rng_state);
    ret.attr("rng_state") = r_rng_state;
  }

  return ret;
}

template <typename T>
void dust_set_index(SEXP ptr, cpp11::sexp r_index) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  const size_t index_max = obj->n_state_full();
  const std::vector<size_t> index = r_index_to_index(r_index, index_max);
  obj->set_index(index);
}

template <typename T>
void dust_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();

  // Do the validation here so that we leave this function having
  // dealt with both or neither (i.e., do not fail on step after
  // succeeding on state).
  std::vector<size_t> step;
  if (r_step != R_NilValue) {
    step = validate_size(r_step, "step");
    if (!(step.size() == 1 || step.size() == obj->n_particles())) {
      cpp11::stop("Expected 'step' to be scalar or length %d",
                  obj->n_particles());
    }
  }

  if (r_state != R_NilValue) {
    if (obj->n_pars() > 0) {
      dust_set_state_multi(obj, cpp11::as_cpp<cpp11::doubles>(r_state));
    } else if (Rf_isMatrix(r_state)) {
      dust_set_state(obj, cpp11::as_cpp<cpp11::doubles_matrix>(r_state));
    } else {
      dust_set_state(obj, cpp11::as_cpp<cpp11::doubles>(r_state));
    }
  }

  if (step.size() == 1) {
    obj->set_step(step[0]);
  } else if (step.size() > 1) {
    obj->set_step(step);
  }
}

template <typename T>
void dust_set_state(Dust<T> *obj, cpp11::doubles r_state) {
  const size_t n_state = obj->n_state_full();

  if (static_cast<size_t>(r_state.size()) != n_state) {
    cpp11::stop("Expected a vector with %d elements for 'state'", n_state);
  }

  const std::vector<typename T::real_t> state(r_state.begin(), r_state.end());
  obj->set_state(state, false);
}

template <typename T>
void dust_set_state(Dust<T> *obj, cpp11::doubles_matrix r_state) {
  typedef typename T::real_t real_t;
  const size_t n_state = obj->n_state_full();
  const size_t n_particles = obj->n_particles();

  if (static_cast<size_t>(r_state.nrow()) != n_state) {
    cpp11::stop("Expected a matrix with %d rows for 'state'", n_state);
  }
  if (static_cast<size_t>(r_state.ncol()) != n_particles) {
    cpp11::stop("Expected a matrix with %d columns for 'state'", n_particles);
  }

  std::vector<real_t> state = matrix_to_vector<real_t>(r_state);

  obj->set_state(state, true);
}

// name could be improved!
//
// NOTE: because recycling the state is ambiguous here, we require a
// full 3d matrix of state, at least for now. We might relax this
// later once tests are in place.
template <typename T>
void dust_set_state_multi(Dust<T> *obj, cpp11::doubles r_state) {
  const size_t n_state = obj->n_state_full();
  const size_t n_pars = obj->n_pars();
  const size_t n_particles_each = obj->n_particles() / n_pars;

  cpp11::sexp r_dim_sexp = r_state.attr("dim");
  if (r_dim_sexp == R_NilValue) {
    cpp11::stop("Expected a 3d array for 'state' (but recieved a vector)");
  }

  cpp11::integers r_dim = cpp11::as_cpp<cpp11::integers>(r_dim_sexp);
  if (r_dim.size() != 3) {
    cpp11::stop("Expected a 3d array for 'state'");
  }
  if (static_cast<size_t>(r_dim[0]) != n_state) {
    cpp11::stop("Expected a 3d array with %d rows for 'state'", n_state);
  }
  if (static_cast<size_t>(r_dim[1]) != n_particles_each) {
    cpp11::stop("Expected a 3d array with %d columns for 'state'",
                n_particles_each);
  }
  if (static_cast<size_t>(r_dim[2]) != n_pars) {
    cpp11::stop("Expected a 3d array with dim[3] == %d for 'state'", n_pars);
  }

  const std::vector<typename T::real_t> state(r_state.begin(), r_state.end());

  obj->set_state(state, true);
}

template <typename T>
cpp11::sexp dust_run(SEXP ptr, int step_end) {
  validate_size(step_end, "step_end");
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  obj->run(step_end);

  const size_t n_state = obj->n_state();
  const size_t n_particles = obj->n_particles();
  const size_t n_pars = obj->n_pars();
  const size_t len = n_state * n_particles;

  std::vector<typename T::real_t> dat(len);
  obj->state(dat);

  if (n_pars == 0) {
    return create_matrix(n_state, n_particles, dat);
  } else {
    std::vector<size_t> dim{n_state, n_particles / n_pars, n_pars};
    return create_array(dim, dat);
  }
}

template <typename T>
cpp11::sexp dust_reset(SEXP ptr, cpp11::list r_pars, int step) {
  validate_size(step, "step");
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  cpp11::sexp info;
  if (obj->n_pars() == 0) {
    dust::pars_t<T> pars = dust_pars<T>(r_pars);
    obj->reset(pars, step);
    info = dust_info<T>(pars);
  } else {
    std::vector<dust::pars_t<T>> pars;
    cpp11::writable::list info_list = cpp11::writable::list(r_pars.size());
    for (int i = 0; i < r_pars.size(); ++i) {
      pars.push_back(dust_pars<T>(r_pars[i]));
      info_list[i] = dust_info<T>(pars[i]);
    }
    obj->reset(pars, step);
    info = info_list;
  }
  return info;
}

template <typename T>
cpp11::sexp dust_set_pars(SEXP ptr, cpp11::list r_pars) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  cpp11::sexp info;
  if (obj->n_pars() == 0) {
    dust::pars_t<T> pars = dust_pars<T>(r_pars);
    obj->set_pars(pars);
    info = dust_info<T>(pars);
  } else {
    // The underlying implementation should be tidied up, as the
    // single case leaves us with inconsistent pars already, and the
    // error management is tricky (#125)
    cpp11::stop("set_pars() with pars_multi not yet supported");
  }
  return info;
}

template <typename T>
SEXP dust_state(SEXP ptr, SEXP r_index) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  if (r_index == R_NilValue) {
    return dust_state_full(obj);
  } else {
    return dust_state_select(obj, r_index);
  }
}

template <typename T>
SEXP dust_state_full(Dust<T> *obj) {
  const size_t n_state_full = obj->n_state_full();
  const size_t n_particles = obj->n_particles();
  const size_t n_pars = obj->n_pars();
  const size_t len = n_state_full * n_particles;

  std::vector<typename T::real_t> dat(len);
  obj->state_full(dat);

  cpp11::sexp ret;
  if (n_pars == 0) {
    ret = create_matrix(n_state_full, n_particles, dat);
  } else {
    std::vector<size_t> dim{n_state_full, n_particles / n_pars, n_pars};
    ret = create_array(dim, dat);
  }

  return ret;
}

template <typename T>
SEXP dust_state_select(Dust<T> *obj, cpp11::sexp r_index) {
  const size_t index_max = obj->n_state_full();
  const std::vector<size_t> index = r_index_to_index(r_index, index_max);
  const size_t n_state = static_cast<size_t>(index.size());
  const size_t n_particles = obj->n_particles();
  const size_t n_pars = obj->n_pars();
  const size_t len = n_state * n_particles;

  std::vector<typename T::real_t> dat(len);
  obj->state(index, dat);

  cpp11::sexp ret;
  if (n_pars == 0) {
    ret = create_matrix(n_state, n_particles, dat);
  } else {
    std::vector<size_t> dim{n_state, n_particles / n_pars, n_pars};
    ret = create_array(dim, dat);
  }

  return ret;
}

template <typename T>
size_t dust_step(SEXP ptr) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  return obj->step();
}

template <typename T>
void dust_reorder(SEXP ptr, cpp11::sexp r_index) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  size_t n_particles = obj->n_particles();
  size_t n_pars = obj->n_pars();

  std::vector<size_t> index;
  if (n_pars == 0) {
    index = r_index_to_index(r_index, n_particles);
    if ((size_t)index.size() != n_particles) {
      cpp11::stop("Expected a vector of length %d for 'index'", n_particles);
    }
  } else {
    index = r_index_reorder_matrix(r_index, n_particles / n_pars, n_pars);
  }

  obj->reorder(index);
}

template <typename T>
SEXP dust_resample(SEXP ptr, cpp11::doubles r_weights) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  size_t n_particles = obj->n_particles();
  size_t n_pars = obj->n_pars();
  size_t n_particles_each = n_particles;
  if (n_pars == 0) {
    if (static_cast<size_t>(r_weights.size()) != n_particles) {
      cpp11::stop("Expected a vector with %d elements for 'weights'",
                  n_particles);
    }
  } else {
    cpp11::sexp r_dim = r_weights.attr("dim");
    if (r_dim == R_NilValue) {
      cpp11::stop("Expected a matrix for 'weights', but given vector");
    }
    cpp11::integers dim = cpp11::as_cpp<cpp11::integers>(r_dim);
    if (dim.size() != 2) {
      cpp11::stop("Expected a matrix for 'weights'");
    }
    n_particles_each /= n_pars;
    if (static_cast<size_t>(dim[0]) != n_particles_each) {
      cpp11::stop("Expected a matrix with %d rows for 'weights'",
                  n_particles_each);
    }
    if (static_cast<size_t>(dim[1]) != n_pars) {
      cpp11::stop("Expected a matrix with %d columns for 'weights'", n_pars);
    }
  }
  if (*std::min_element(r_weights.begin(), r_weights.end()) < 0) {
    cpp11::stop("All weights must be positive");
  }
  const std::vector<typename T::real_t>
    weights(r_weights.begin(), r_weights.end());

  std::vector<size_t> idx = obj->resample(weights);

  cpp11::writable::integers ret(n_particles);
  for (size_t i = 0; i < n_particles; ++i) {
    ret[i] = idx[i] % n_particles_each + 1;
  }

  // Same shape as on exit; we rescale the index so that it is
  // equivalent to the order that reorder would accept
  if (n_pars > 0) {
    ret.attr("dim") = r_weights.attr("dim");
  }
  return ret;
}

template <typename T>
SEXP dust_rng_state(SEXP ptr, bool first_only) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  auto state = obj->rng_state();
  size_t n = first_only ? dust::rng_state_t<double>::size() : state.size();
  size_t len = sizeof(uint64_t) * n;
  cpp11::writable::raws ret(len);
  std::memcpy(RAW(ret), state.data(), len);
  return ret;
}

template <typename T>
void dust_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  auto prev_state = obj->rng_state();
  size_t len = prev_state.size() * sizeof(uint64_t);
  if ((size_t)rng_state.size() != len) {
    cpp11::stop("'rng_state' must be a raw vector of length %d (but was %d)",
                len, rng_state.size());
  }
  std::vector<uint64_t> pars(prev_state.size());
  std::memcpy(pars.data(), RAW(rng_state), len);
  obj->set_rng_state(pars);
}

template <typename T>
void dust_set_n_threads(SEXP ptr, int n_threads) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  validate_positive(n_threads, "n_threads");
  obj->set_n_threads(n_threads);
}

// Trivial default implementation of a method for getting back
// arbitrary information from the object.
template <typename T>
cpp11::sexp dust_info(const dust::pars_t<T>& pars) {
  return R_NilValue;
}

template <typename T, typename std::enable_if<!std::is_same<dust::no_data, typename T::data_t>::value, int>::type = 0>
void dust_set_data(SEXP ptr, cpp11::list r_data) {
  typedef typename T::data_t data_t;
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  const size_t n_pars = obj->n_pars() == 0 ? 1 : obj->n_pars();

  const size_t len = r_data.size();
  std::unordered_map<size_t, std::vector<data_t>> data;

  for (size_t i = 0; i < len; ++i) {
    cpp11::list el = r_data[i];
    if (el.size() != n_pars + 1) {
      cpp11::stop("Expected a list of length %d for element %d of 'data'",
                  n_pars + 1, i + 1);
    }
    const size_t step_i = cpp11::as_cpp<int>(el[0]);
    std::vector<data_t> data_i;
    data_i.reserve(n_pars);
    for (size_t j = 0; j < n_pars; ++j) {
      data_i.push_back(dust_data<T>(cpp11::as_cpp<cpp11::list>(el[j + 1])));
    }
    data[step_i] = data_i;
  }
  obj->set_data(data);
}

template <typename T, typename std::enable_if<!std::is_same<dust::no_data, typename T::data_t>::value, int>::type = 0>
cpp11::sexp dust_compare_data(SEXP ptr) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  std::vector<typename T::real_t> ret = obj->compare_data();
  if (ret.size() == 0) {
    return R_NilValue;
  }
  cpp11::writable::doubles ret_r(ret.size());
  std::copy(ret.begin(), ret.end(), REAL(ret_r));

  const size_t n_pars = obj->n_pars();
  if (n_pars > 0) {
    const size_t n_particles_each = obj->n_particles() / n_pars;
    ret_r.attr("dim") = cpp11::writable::integers({(int)n_particles_each,
                                                   (int)n_pars});
  }

  return ret_r;
}

template <typename T, typename std::enable_if<!std::is_same<dust::no_data, typename T::data_t>::value, int>::type = 0>
cpp11::sexp dust_filter(SEXP ptr) {
  return R_NilValue;
}

// Based on the value of the data_t in the underlying model class we
// might use these functions for set_data and compare_data which give
// reasonable errors back to R, as we can't use the full versions
// above.
inline void disable_method(const char * name) {
  cpp11::stop("The '%s' method is not supported for this class", name);
}

template <typename T, typename std::enable_if<std::is_same<dust::no_data, typename T::data_t>::value, int>::type = 0>
void dust_set_data(SEXP ptr, cpp11::list r_data) {
  disable_method("set_data");
}

template <typename T, typename std::enable_if<std::is_same<dust::no_data, typename T::data_t>::value, int>::type = 0>
cpp11::sexp dust_compare_data(SEXP ptr) {
  disable_method("compare_data");
  return R_NilValue; // never gets here
}

template <typename T, typename std::enable_if<std::is_same<dust::no_data, typename T::data_t>::value, int>::type = 0>
cpp11::sexp dust_filter(SEXP ptr) {
  disable_method("filter");
  return R_NilValue; // never gets here
}

template <typename T>
cpp11::sexp dust_capabilities() {
  using namespace cpp11::literals;
#ifdef _OPENMP
  bool openmp = true;
#else
  bool openmp = false;
#endif
  bool compare = !std::is_same<dust::no_data, typename T::data_t>::value;
  return cpp11::writable::list({"openmp"_nm = openmp, "compare"_nm = compare});
}

template <typename T>
int dust_n_state(SEXP ptr) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  return obj->n_state_full();
}

inline void validate_size(int x, const char * name) {
  if (x < 0) {
    cpp11::stop("'%s' must be non-negative", name);
  }
}

inline void validate_positive(int x, const char *name) {
  if (x <= 0) {
    cpp11::stop("'%s' must be positive", name);
  }
}

inline std::vector<size_t> validate_size(cpp11::sexp r_x, const char * name) {
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
inline std::vector<size_t> r_index_to_index(cpp11::sexp r_index, size_t nmax) {
  if (r_index == R_NilValue) {
    return r_index_to_index_default(nmax);
  }

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

inline std::vector<size_t> r_index_reorder_matrix(cpp11::sexp r_index,
                                                  const size_t n_particles,
                                                  const size_t n_pars) {
  if (!Rf_isMatrix(r_index)) {
    cpp11::stop("Expected a matrix for 'index'");
  }
  cpp11::integers_matrix r_index_mat =
    cpp11::as_cpp<cpp11::integers_matrix>(r_index);
  if (static_cast<size_t>(r_index_mat.nrow()) != n_particles) {
    cpp11::stop("Expected a matrix with %d rows for 'index'", n_particles);
  }
  if (static_cast<size_t>(r_index_mat.ncol()) != n_pars) {
    cpp11::stop("Expected a matrix with %d columns for 'index'", n_pars);
  }

  const int * index_pars = INTEGER(r_index_mat);

  std::vector<size_t> index;
  index.reserve(n_particles * n_pars);
  for (size_t i = 0, j = 0; i < n_pars; ++i) {
    for (size_t k = 0; k < n_particles; ++j, ++k) {
      int x = index_pars[j];
      if (x < 1 || x > (int)n_particles) {
        cpp11::stop("All elements of 'index' must lie in [1, %d]", n_particles);
      }
      index.push_back(i * n_particles + x - 1);
    }
  }
  return index;
}

// Helper for the above; in the case where index is not given we
// assume it would have been given as 1..n so generate out 0..(n-1)
inline std::vector<size_t> r_index_to_index_default(size_t n) {
  std::vector<size_t> index;
  index.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    index.push_back(i);
  }
  return index;
}

template <typename T>
cpp11::sexp create_matrix(size_t nrow, size_t ncol, T& pars) {
  const size_t len = pars.size();
  cpp11::writable::doubles ret(static_cast<R_xlen_t>(len));
  double * dest = REAL(ret);
  for (size_t i = 0; i < len; ++i) {
    dest[i] = pars[i];
  }

  ret.attr("dim") = cpp11::writable::integers({(int)nrow, (int)ncol});
  return ret;
}

template <typename T>
cpp11::sexp create_array(const std::vector<size_t>& dim, T& pars) {
  const size_t len = pars.size();
  cpp11::writable::doubles ret(static_cast<R_xlen_t>(len));
  double * dest = REAL(ret);
  for (size_t i = 0; i < len; ++i) {
    dest[i] = pars[i];
  }

  cpp11::writable::integers r_dim(dim.size());
  for (size_t i = 0; i < dim.size(); ++i) {
    r_dim[i] = dim[i];
  }
  ret.attr("dim") = r_dim;

  return ret;
}

inline cpp11::integers as_integer(cpp11::sexp x, const char * name) {
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

template <typename T>
std::vector<T> matrix_to_vector(cpp11::doubles_matrix x) {
  const size_t len = x.nrow() * x.ncol();
  const double * x_data = REAL(x.data());
  std::vector<T> ret(len);
  std::copy(x_data, x_data + len, ret.begin());
  return ret;
}

#endif
