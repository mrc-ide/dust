#include <cstring>
#include <cpp11/external_pointer.hpp>
#include <cpp11/raws.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/integers.hpp>
#include <dust/rng.hpp>
#include <dust/rng_interface.hpp>

typedef dust::pRNG<double> dust_rng_t;
typedef cpp11::external_pointer<dust_rng_t> dust_rng_ptr_t;

[[cpp11::register]]
SEXP dust_rng_alloc(cpp11::sexp r_seed, int n_generators) {
  std::vector<uint64_t> seed = as_rng_seed<double>(r_seed);
  dust_rng_t *rng = new dust_rng_t(n_generators, seed);
  return cpp11::external_pointer<dust_rng_t>(rng);
}

[[cpp11::register]]
int dust_rng_size(SEXP ptr) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  return static_cast<int>(rng->size());
}

[[cpp11::register]]
void dust_rng_jump(SEXP ptr) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  rng->jump();
}

[[cpp11::register]]
void dust_rng_long_jump(SEXP ptr) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  rng->long_jump();
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_unif_rand(SEXP ptr, int n) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  const size_t n_generators = rng->size();

  cpp11::writable::doubles ret = cpp11::writable::doubles(n);
  double * y = REAL(ret);

  for (size_t i = 0; i < (size_t)n; ++i) {
    dust::rng_state_t<double> s = rng->state(i % n_generators);
    y[i] = dust::unif_rand(s);
  }

  return ret;
}

// NOTE: no special treatment (yet) for this
[[cpp11::register]]
cpp11::writable::doubles dust_rng_norm_rand(SEXP ptr, int n) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  const size_t n_generators = rng->size();

  cpp11::writable::doubles ret = cpp11::writable::doubles(n);
  double * y = REAL(ret);

  for (size_t i = 0; i < (size_t)n; ++i) {
    dust::rng_state_t<double> s = rng->state(i % n_generators);
    y[i] = dust::distr::rnorm(s, 0, 1);
  }

  return ret;
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_runif(SEXP ptr, int n,
                                         cpp11::doubles r_min,
                                         cpp11::doubles r_max) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  const double * min = REAL(r_min);
  const double * max = REAL(r_max);
  const size_t n_generators = rng->size();

  cpp11::writable::doubles ret = cpp11::writable::doubles(n);
  double * y = REAL(ret);

  for (size_t i = 0; i < (size_t)n; ++i) {
    dust::rng_state_t<double> s = rng->state(i % n_generators);
    y[i] = dust::distr::runif(s, min[i], max[i]);
  }

  return ret;
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_rnorm(SEXP ptr, int n,
                                        cpp11::doubles r_mean,
                                        cpp11::doubles r_sd) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  const double * mean = REAL(r_mean);
  const double * sd = REAL(r_sd);
  const size_t n_generators = rng->size();

  cpp11::writable::doubles ret = cpp11::writable::doubles(n);
  double * y = REAL(ret);

  for (size_t i = 0; i < (size_t)n; ++i) {
    dust::rng_state_t<double> s = rng->state(i % n_generators);
    y[i] = dust::distr::rnorm(s, mean[i], sd[i]);
  }

  return ret;
}

[[cpp11::register]]
cpp11::writable::integers dust_rng_rbinom(SEXP ptr, int n,
                                          cpp11::integers r_size,
                                          cpp11::doubles r_prob) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  const int * size = INTEGER(r_size);
  const double * prob = REAL(r_prob);

  cpp11::writable::integers ret = cpp11::writable::integers(n);
  int * y = INTEGER(ret);

  const size_t n_generators = rng->size();
  for (size_t i = 0; i < (size_t)n; ++i) {
    dust::rng_state_t<double> s = rng->state(i % n_generators);
    y[i] = dust::distr::rbinom(s, size[i], prob[i]);
  }

  return ret;
}

[[cpp11::register]]
cpp11::writable::integers dust_rng_rpois(SEXP ptr, int n,
                                         cpp11::doubles r_lambda) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  const double * lambda = REAL(r_lambda);
  const size_t n_generators = rng->size();

  cpp11::writable::integers ret = cpp11::writable::integers(n);
  int * y = INTEGER(ret);

  for (size_t i = 0; i < (size_t)n; ++i) {
    dust::rng_state_t<double> s = rng->state(i % n_generators);
    y[i] = dust::distr::rpois(s, lambda[i]);
  }

  return ret;
}

[[cpp11::register]]
cpp11::writable::raws dust_rng_state(SEXP ptr) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  auto state = rng->export_state();
  size_t len = sizeof(uint64_t) * state.size();
  cpp11::writable::raws ret(len);
  std::memcpy(RAW(ret), state.data(), len);
  return ret;
}

[[cpp11::register]]
void dust_rng_set_state(SEXP ptr, cpp11::raws state) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();

  auto len_state = rng->size() * dust::rng_state_t<double>::size();
  size_t len = len_state * sizeof(uint64_t);

  if ((size_t)state.size() != len) {
    cpp11::stop("'state' must be a raw vector of length %d (but was %d)",
                len, state.size());
  }
  std::vector<uint64_t> data(len_state);
  std::memcpy(data.data(), RAW(state), len);
  rng->import_state(data);
}
