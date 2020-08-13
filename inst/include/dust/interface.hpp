#include <cstring>
#include <cpp11/doubles.hpp>
#include <cpp11/external_pointer.hpp>
#include <cpp11/integers.hpp>
#include <cpp11/list.hpp>
#include <cpp11/matrix.hpp>
#include <cpp11/raws.hpp>
#include <cpp11/strings.hpp>
#include <dust/dust.hpp>
#include <R_ext/Random.h>

template <typename T>
typename T::init_t dust_data(cpp11::list data);

template <typename T>
typename cpp11::sexp dust_info(const typename T::init_t& data);

inline void validate_size(int x, const char * name);
inline std::vector<size_t> validate_size(cpp11::sexp x, const char *name);
inline std::vector<size_t> r_index_to_index(cpp11::sexp r_index, size_t nmax);
inline std::vector<size_t> r_index_to_index_default(size_t n);
inline cpp11::integers as_integer(cpp11::sexp x, const char * name);
template <typename T>
std::vector<uint64_t> as_rng_seed(cpp11::sexp r_seed);

template <typename T>
std::vector<T> matrix_to_vector(cpp11::doubles_matrix x);

template <typename T>
cpp11::writable::doubles_matrix create_matrix(size_t nrow, size_t ncol,
                                              const T& data);

template <typename T>
cpp11::list dust_alloc(cpp11::list r_data, int step,
                       int n_particles, int n_threads,
                       int seed) {
  validate_size(step, "step");
  validate_size(n_particles, "n_particles");
  validate_size(n_threads, "n_threads");
  validate_size(seed, "seed");

  typename T::init_t data = dust_data<T>(r_data);

  Dust<T> *d = new Dust<T>(data, step, n_particles, n_threads, seed);
  cpp11::external_pointer<Dust<T>> ptr(d, true, false);
  cpp11::sexp info = dust_info<T>(data);

  return cpp11::writable::list({ptr, info});
}

// think about rng seeding here carefully; should accept either a raw
// vector or an integer I think.
template <typename T>
cpp11::writable::doubles dust_simulate(cpp11::sexp r_steps,
                                       cpp11::list r_data,
                                       cpp11::doubles_matrix r_state,
                                       cpp11::sexp r_index,
                                       const size_t n_threads,
                                       const size_t seed) {
  typedef typename T::real_t real_t;
  std::vector<size_t> steps = validate_size(r_steps, "steps");
  std::vector<real_t> state = matrix_to_vector<real_t>(r_state);
  std::vector<size_t> index = r_index_to_index(r_index, r_state.nrow());

  if (r_data.size() != r_state.ncol()) {
    cpp11::stop("Expected 'state' to be a matrix with %d columns",
                r_data.size());
  }

  std::vector<typename T::init_t> data;
  data.reserve(r_data.size());
  for (int i = 0; i < r_data.size(); ++i) {
    data.push_back(dust_data<T>(r_data[i]));
  }

  cpp11::writable::doubles ret(index.size() * data.size() * steps.size());

  std::vector<real_t> dat =
    dust_simulate<T>(steps, data, state, index, n_threads, seed);
  std::copy(dat.begin(), dat.end(), REAL(ret));

  ret.attr("dim") = cpp11::writable::integers({(int)index.size(),
                                               (int)data.size(),
                                               (int)steps.size()});
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
      cpp11::stop("Expected 'size' to be scalar or length %d",
                  obj->n_particles());
    }
  }

  if (r_state != R_NilValue) {
    if (Rf_isMatrix(r_state)) {
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

template <typename T>
cpp11::writable::doubles_matrix dust_run(SEXP ptr, int step_end) {
  validate_size(step_end, "step_end");
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  obj->run(step_end);

  const size_t n_state = obj->n_state();
  const size_t n_particles = obj->n_particles();
  const size_t len = n_state * n_particles;

  std::vector<typename T::real_t> dat(len);
  obj->state(dat);

  return create_matrix(n_state, n_particles, dat);
}

template <typename T>
cpp11::sexp dust_reset(SEXP ptr, cpp11::list r_data, int step) {
  validate_size(step, "step");
  typename T::init_t data = dust_data<T>(r_data);
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  obj->reset(data, step);
  return dust_info<T>(data);
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
  const size_t len = n_state_full * n_particles;

  std::vector<typename T::real_t> dat(len);
  obj->state_full(dat);

  return create_matrix(n_state_full, n_particles, dat);
}

template <typename T>
SEXP dust_state_select(Dust<T> *obj, cpp11::sexp r_index) {
  const size_t index_max = obj->n_state_full();
  const std::vector<size_t> index = r_index_to_index(r_index, index_max);
  const size_t n_state = static_cast<size_t>(index.size());
  const size_t n_particles = obj->n_particles();
  const size_t len = n_state * n_particles;

  std::vector<typename T::real_t> dat(len);
  obj->state(index, dat);

  return create_matrix(n_state, n_particles, dat);
}

template <typename T>
size_t dust_step(SEXP ptr) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  return obj->step();
}

template <typename T>
void dust_reorder(SEXP ptr, cpp11::sexp r_index) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  size_t n = obj->n_particles();
  std::vector<size_t> index = r_index_to_index(r_index, n);
  if ((size_t)index.size() != obj->n_particles()) {
    cpp11::stop("Expected a vector of length %d for 'index'", n);
  }

  obj->reorder(index);
}

template <typename T>
SEXP dust_rng_state(SEXP ptr) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  auto state = obj->rng_state();
  size_t len = sizeof(uint64_t) * state.size();
  cpp11::writable::raws ret(len);
  std::memcpy(RAW(ret), state.data(), len);
  return ret;
}

// Trivial default implementation of a method for getting back
// arbitrary information from the object.
template <typename T>
cpp11::sexp dust_info(const typename T::init_t& data) {
  return R_NilValue;
}

inline void validate_size(int x, const char * name) {
  if (x < 0) {
    cpp11::stop("%s must be non-negative", name);
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
cpp11::writable::doubles_matrix create_matrix(size_t nrow, size_t ncol,
                                              const T& data) {
  cpp11::writable::doubles_matrix ret(nrow, ncol);
  double * dest = REAL(ret);
  const size_t len = data.size();
  for (size_t i = 0; i < len; ++i) {
    dest[i] = data[i];
  }
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

template <typename T>
std::vector<uint64_t> as_rng_seed(cpp11::sexp r_seed) {
  auto seed_type = TYPEOF(r_seed);
  std::vector<uint64_t> seed;
  if (seed_type == INTSXP || seed_type == REALSXP) {
    size_t seed_int = cpp11::as_cpp<int>(r_seed);
    validate_size(seed_int, "seed");
    seed = dust::xoshiro_initial_seed<T>(seed_int);
  } else if (seed_type == RAWSXP) {
    cpp11::raws seed_data = cpp11::as_cpp<cpp11::raws>(r_seed);
    const size_t len = sizeof(uint64_t) * dust::rng_state_t<T>::size();
    if (seed_data.size() == 0 || seed_data.size() % len != 0) {
      cpp11::stop("Expected raw vector of length as multiple of %d for 'seed'",
                  len);
    }
    seed.resize(seed_data.size() / sizeof(uint64_t));
    std::memcpy(seed.data(), RAW(seed_data), seed_data.size());
  } else if (seed_type == NILSXP) {
    GetRNGstate();
    int seed_int =
      std::ceil(std::abs(unif_rand()) * std::numeric_limits<int>::max());
    PutRNGstate();
    seed = dust::xoshiro_initial_seed<T>(seed_int);
  } else {
    cpp11::stop("Invalid type for 'seed'");
  }
  return seed;
}
