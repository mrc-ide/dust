#include <Rcpp.h>

template <typename T>
typename T::init_t dust_data(Rcpp::List data);

template <typename T>
typename Rcpp::RObject dust_info(const typename T::init_t& data);

inline void validate_n(size_t n_generators, size_t n_threads);
inline void validate_size(int x, const char * name);
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
void dust_set_index_y(SEXP ptr, Rcpp::IntegerVector r_index_y) {
  Dust<T> *obj = Rcpp::as<Rcpp::XPtr<Dust<T>>>(ptr);
  const size_t index_max = obj->n_state_full();
  const std::vector<size_t> index = r_index_to_index(r_index_y, index_max);
  obj->set_index_y(index);
}

template <typename T>
void dust_set_state(SEXP ptr, Rcpp::NumericVector r_state) {
  Dust<T> *obj = Rcpp::as<Rcpp::XPtr<Dust<T>>>(ptr);
  const size_t n_state = obj->n_state_full();
  if (static_cast<size_t>(r_state.size()) != n_state) {
    Rcpp::stop("Expected a vector with %d elements for 'state'", n_state);
  }
  // TODO: make flexible with types
  const std::vector<double> state = Rcpp::as<std::vector<double>>(r_state);
  obj->set_state(state);
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
