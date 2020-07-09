#include <Rcpp.h>

template <typename T>
typename T::init_t dust_data(Rcpp::List data);

template <typename T>
typename Rcpp::RObject dust_info(const typename T::init_t& data);

inline void validate_n(size_t n_generators, size_t n_threads);
inline void validate_size(int x, const char * name);
inline std::vector<size_t> validate_size(Rcpp::IntegerVector x,
                                         const char *name);
inline std::vector<size_t> r_index_to_index(Rcpp::IntegerVector r_index,
                                            size_t nmax);

template <typename T>
Rcpp::List dust_alloc(Rcpp::List r_data, int step,
                      int n_particles, int n_threads,
                      int n_generators, int seed) {
  validate_size(step, "step");
  validate_size(n_particles, "n_particles");
  validate_size(n_threads, "n_threads");
  validate_size(n_generators, "n_generators");
  validate_size(seed, "seed");
  validate_n(n_generators, n_threads);

  typename T::init_t data = dust_data<T>(r_data);

  Dust<T> *d =
    new Dust<T>(data, step, n_particles, n_threads, n_generators, seed);
  Rcpp::XPtr<Dust<T>> ptr(d, false);
  Rcpp::RObject info = dust_info<T>(data);

  return Rcpp::List::create(ptr, info);
}

template <typename T>
void dust_set_index(SEXP ptr, Rcpp::IntegerVector r_index) {
  Dust<T> *obj = Rcpp::as<Rcpp::XPtr<Dust<T>>>(ptr);
  const size_t index_max = obj->n_state_full();
  const std::vector<size_t> index = r_index_to_index(r_index, index_max);
  obj->set_index(index);
}

template <typename T>
void dust_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  Dust<T> *obj = Rcpp::as<Rcpp::XPtr<Dust<T>>>(ptr);

  // Do the validation here so that we leave this function having
  // dealt with both or neither (i.e., do not fail on step after
  // succeeding on state).
  std::vector<size_t> step;
  if (r_step != R_NilValue) {
    step = validate_size(Rcpp::as<Rcpp::IntegerVector>(r_step), "step");
    if (!(step.size() == 1 || step.size() == obj->n_particles())) {
      Rcpp::stop("Expected 'size' to be scalar or length %d",
                 obj->n_particles());
    }
  }

  if (r_state != R_NilValue) {
    if (Rf_isMatrix(r_state)) {
      dust_set_state(obj, Rcpp::as<Rcpp::NumericMatrix>(r_state));
    } else {
      dust_set_state(obj, Rcpp::as<Rcpp::NumericVector>(r_state));
    }
  }

  if (step.size() == 1) {
    obj->set_step(step[0]);
  } else if (step.size() > 1) {
    obj->set_step(step);
  }
}

template <typename T>
void dust_set_state(Dust<T> *obj, Rcpp::NumericVector r_state) {
  const size_t n_state = obj->n_state_full();
  if (static_cast<size_t>(r_state.size()) != n_state) {
    Rcpp::stop("Expected a vector with %d elements for 'state'", n_state);
  }
  const std::vector<typename T::real_t> state =
    Rcpp::as<std::vector<typename T::real_t>>(r_state);
  obj->set_state(state, false);
}

template <typename T>
void dust_set_state(Dust<T> *obj, Rcpp::NumericMatrix r_state) {
  const size_t n_state = obj->n_state_full();
  const size_t n_particles = obj->n_particles();

  if (static_cast<size_t>(r_state.nrow()) != n_state) {
    Rcpp::stop("Expected a matrix with %d rows for 'state'", n_state);
  }
  if (static_cast<size_t>(r_state.ncol()) != n_particles) {
    Rcpp::stop("Expected a matrix with %d columns for 'state'", n_particles);
  }

  const std::vector<typename T::real_t> state =
    Rcpp::as<std::vector<typename T::real_t>>(r_state);
  obj->set_state(state, true);
}

template <typename T>
Rcpp::NumericMatrix dust_run(SEXP ptr, int step_end) {
  validate_size(step_end, "step_end");
  Dust<T> *obj = Rcpp::as<Rcpp::XPtr<Dust<T>>>(ptr);
  obj->run(step_end);

  const size_t n_state = obj->n_state();
  const size_t n_particles = obj->n_particles();
  const size_t len = n_state * n_particles;

  std::vector<typename T::real_t> dat(len);
  obj->state(dat);

  return Rcpp::NumericMatrix(n_state, n_particles, dat.begin());
}

template <typename T>
Rcpp::RObject dust_reset(SEXP ptr, Rcpp::List r_data, int step) {
  validate_size(step, "step");
  typename T::init_t data = dust_data<T>(r_data);
  Dust<T> *obj = Rcpp::as<Rcpp::XPtr<Dust<T>>>(ptr);
  obj->reset(data, step);
  return dust_info<T>(data);
}

template <typename T>
SEXP dust_state(SEXP ptr, SEXP r_index) {
  Dust<T> *obj = Rcpp::as<Rcpp::XPtr<Dust<T>>>(ptr);
  if (r_index == R_NilValue) {
    return dust_state_full(obj);
  } else {
    return dust_state_select(obj, Rcpp::as<Rcpp::IntegerVector>(r_index));
  }
}

template <typename T>
SEXP dust_state_full(Dust<T> *obj) {
  const size_t n_state_full = obj->n_state_full();
  const size_t n_particles = obj->n_particles();
  const size_t len = n_state_full * n_particles;

  std::vector<typename T::real_t> dat(len);
  obj->state_full(dat);

  return Rcpp::NumericMatrix(n_state_full, n_particles, dat.begin());
}

template <typename T>
SEXP dust_state_select(Dust<T> *obj, Rcpp::IntegerVector r_index) {
  const size_t n_state = static_cast<size_t>(r_index.size());
  const size_t n_particles = obj->n_particles();
  const size_t len = n_state * n_particles;
  const size_t index_max = obj->n_state_full();
  const std::vector<size_t> index = r_index_to_index(r_index, index_max);

  std::vector<typename T::real_t> dat(len);
  obj->state(index, dat);

  return Rcpp::NumericMatrix(n_state, n_particles, dat.begin());
}

template <typename T>
size_t dust_step(SEXP ptr) {
  Dust<T> *obj = Rcpp::as<Rcpp::XPtr<Dust<T>>>(ptr);
  return obj->step();
}

template <typename T>
void dust_reorder(SEXP ptr, Rcpp::IntegerVector r_index) {
  Dust<T> *obj = Rcpp::as<Rcpp::XPtr<Dust<T>>>(ptr);
  size_t n = obj->n_particles();
  if ((size_t)r_index.size() != obj->n_particles()) {
    Rcpp::stop("Expected a vector of length %d for 'index'", n);
  }

  obj->reorder(r_index_to_index(r_index, n));
}

// Trivial default implementation of a method for getting back
// arbitrary information from the object.
template <typename T>
Rcpp::RObject dust_info(const typename T::init_t& data) {
  return R_NilValue;
}

inline void validate_n(size_t n_generators, size_t n_threads) {
  if (n_generators < n_threads) {
    Rcpp::stop("n_generators must be at least n_threads");
  }
  if (n_generators % n_threads > 0) {
    Rcpp::stop("n_generators must be a multiple of n_threads");
  }
}

inline void validate_size(int x, const char * name) {
  if (x < 0) {
    Rcpp::stop("%s must be non-negative", name);
  }
}

inline std::vector<size_t> validate_size(Rcpp::IntegerVector r_x,
                                         const char * name) {
  const size_t n = static_cast<size_t>(r_x.size());
  std::vector<size_t> x;
  x.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    int el = r_x[i];
    if (el < 0) {
      Rcpp::stop("All elements of '%s' must be non-negative", name);
    }
    x.push_back(el);
  }
  return x;
}

inline std::vector<size_t> r_index_to_index(Rcpp::IntegerVector r_index,
                                            size_t nmax) {
  const size_t n = static_cast<size_t>(r_index.size());
  std::vector<size_t> index;
  index.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    int x = r_index[i];
    if (x < 1 || (size_t)x > nmax) {
      Rcpp::stop("All elements of 'index' must lie in [1, %d]", nmax);
    }
    index.push_back(x - 1);
  }
  return index;
}
