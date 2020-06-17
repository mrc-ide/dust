#include <Rcpp.h>

template <typename T>
typename T::init_t dust_data(Rcpp::List data);
inline void validate_n(size_t n_generators, size_t n_threads);
inline void validate_size(int x, const char * name);

template <typename T>
Rcpp::XPtr<Dust<T>> dust_alloc(Rcpp::List r_data, int step,
                               int n_particles, int n_threads,
                               int n_generators, int seed) {
  validate_size(step, "step");
  validate_size(n_particles, "n_particles");
  validate_size(n_threads, "n_threads");
  validate_size(n_generators, "n_generators");
  validate_size(seed, "seed");
  validate_n(n_generators, n_threads);

  typename T::init_t data = dust_data<T>(r_data);
  std::vector<size_t> index_y = {0};

  // TODO: can't customise initial state
  // TODO: can't customise index y

  Dust<T> *d =
    new Dust<T>(data, step, index_y, n_particles, n_threads, n_generators,
                seed);
  Rcpp::XPtr<Dust<T>> ptr(d, false);
  return ptr;
}

template <typename T>
Rcpp::NumericMatrix dust_run(SEXP ptr, int step_end) {
  validate_size(step_end, "step_end");
  Dust<T> *obj = Rcpp::as<Rcpp::XPtr<Dust<T>>>(ptr);
  obj->run(step_end);

  const size_t n_state = obj->n_state();
  const size_t n_particles = obj->n_particles();
  const size_t len = n_state * n_particles;

  std::vector<double> dat(len);
  obj->state(dat);

  return Rcpp::NumericMatrix(n_state, n_particles, dat.begin());
}

template <typename T>
void dust_reset(SEXP ptr, Rcpp::List r_data, int step) {
  validate_size(step, "step");
  typename T::init_t data = dust_data<T>(r_data);
  Dust<T> *obj = Rcpp::as<Rcpp::XPtr<Dust<T>>>(ptr);
  obj->reset(data, step);
}

template <typename T>
SEXP dust_state(SEXP ptr) {
  Dust<T> *obj = Rcpp::as<Rcpp::XPtr<Dust<T>>>(ptr);

  const size_t n_state_full = obj->n_state_full();
  const size_t n_particles = obj->n_particles();
  const size_t len = n_state_full * n_particles;

  std::vector<double> dat(len);
  obj->state_full(dat);

  return Rcpp::NumericMatrix(n_state_full, n_particles, dat.begin());
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

  std::vector<size_t> index;
  index.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    int x = r_index[i];
    if (x < 1 || (size_t)x > n) {
      Rcpp::stop("All elements of 'index' must lie in [1, %d]", n);
    }
    index.push_back(x - 1);
  }

  obj->reorder(index);
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
