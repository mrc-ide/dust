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
    y[i] = dust::unif_rand(rng->state(i % n_generators));
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
    y[i] = dust::distr::rnorm(rng->state(i % n_generators), 0, 1);
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
    y[i] = dust::distr::runif(rng->state(i % n_generators), min[i], max[i]);
  }

  return ret;
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_rexp(SEXP ptr, int n,
                                       cpp11::doubles r_rate) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  const double * rate = REAL(r_rate);
  const size_t n_generators = rng->size();

  cpp11::writable::doubles ret = cpp11::writable::doubles(n);
  double * y = REAL(ret);

  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::distr::rexp(rng->state(i % n_generators), rate[i]);
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
    y[i] = dust::distr::rnorm(rng->state(i % n_generators), mean[i], sd[i]);
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
    y[i] = dust::distr::rbinom(rng->state(i % n_generators), size[i], prob[i]);
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
    y[i] = dust::distr::rpois(rng->state(i % n_generators), lambda[i]);
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

// This is is not for general use, trying to trigger a bug
[[cpp11::register]]
cpp11::sexp test_rbinom_float(cpp11::sexp r_seed, int n_samples,
                              int size, double p) {
  std::vector<uint64_t> seed = as_rng_seed<float>(r_seed);
  dust::pRNG<float> rng(1, seed);
  cpp11::writable::integers ret = cpp11::writable::integers(n_samples);
  int * y = INTEGER(ret);
  const float p_f = p;
  for (size_t i = 0; i < (size_t)n_samples; ++i) {
    y[i] = dust::distr::rbinom(rng.state(0), size, p_f);
  }
  return ret;
}
