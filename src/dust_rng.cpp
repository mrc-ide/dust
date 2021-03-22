#include <cstring>
#include <cpp11/external_pointer.hpp>
#include <cpp11/raws.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/integers.hpp>
#include <dust/rng.hpp>
#include <dust/rng_interface.hpp>
#include <dust/interface_helpers.hpp>

template <typename real_t>
using dust_rng_ptr_t = cpp11::external_pointer<dust::pRNG<real_t>>;

namespace dust {
namespace rng {

template <typename real_t>
SEXP dust_rng_alloc(cpp11::sexp r_seed, int n_generators) {
  std::vector<uint64_t> seed = dust::interface::as_rng_seed<real_t>(r_seed);
  dust::pRNG<real_t> *rng = new dust::pRNG<real_t>(n_generators, seed);
  return dust_rng_ptr_t<real_t>(rng);
}

template <typename real_t>
void dust_rng_jump(SEXP ptr) {
  dust::pRNG<real_t> *rng = cpp11::as_cpp<dust_rng_ptr_t<real_t>>(ptr).get();
  rng->jump();
}

template <typename real_t>
void dust_rng_long_jump(SEXP ptr) {
  dust::pRNG<real_t> *rng = cpp11::as_cpp<dust_rng_ptr_t<real_t>>(ptr).get();
  rng->long_jump();
}

template <typename real_t>
cpp11::writable::doubles dust_rng_unif_rand(SEXP ptr, int n) {
  dust::pRNG<real_t> *rng = cpp11::as_cpp<dust_rng_ptr_t<real_t>>(ptr).get();
  const size_t n_generators = rng->size();

  cpp11::writable::doubles ret = cpp11::writable::doubles(n);
  double * y = REAL(ret);

  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::unif_rand(rng->state(i % n_generators));
  }

  return ret;
}

// NOTE: no special treatment (yet) for this
template <typename real_t>
cpp11::writable::doubles dust_rng_norm_rand(SEXP ptr, int n) {
  dust::pRNG<real_t> *rng = cpp11::as_cpp<dust_rng_ptr_t<real_t>>(ptr).get();
  const size_t n_generators = rng->size();

  cpp11::writable::doubles ret = cpp11::writable::doubles(n);
  double * y = REAL(ret);

  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::distr::rnorm(rng->state(i % n_generators), 0, 1);
  }

  return ret;
}

template <typename real_t>
cpp11::writable::doubles dust_rng_runif(SEXP ptr, int n,
                                        cpp11::doubles r_min,
                                        cpp11::doubles r_max) {
  dust::pRNG<real_t> *rng = cpp11::as_cpp<dust_rng_ptr_t<real_t>>(ptr).get();
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

template <typename real_t>
cpp11::writable::doubles dust_rng_rexp(SEXP ptr, int n,
                                       cpp11::doubles r_rate) {
  dust::pRNG<real_t> *rng = cpp11::as_cpp<dust_rng_ptr_t<real_t>>(ptr).get();
  const double * rate = REAL(r_rate);
  const size_t n_generators = rng->size();

  cpp11::writable::doubles ret = cpp11::writable::doubles(n);
  double * y = REAL(ret);

  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::distr::rexp(rng->state(i % n_generators), rate[i]);
  }

  return ret;
}

template <typename real_t>
cpp11::writable::doubles dust_rng_rnorm(SEXP ptr, int n,
                                        cpp11::doubles r_mean,
                                        cpp11::doubles r_sd) {
  dust::pRNG<real_t> *rng = cpp11::as_cpp<dust_rng_ptr_t<real_t>>(ptr).get();
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

template <typename real_t>
cpp11::writable::integers dust_rng_rbinom(SEXP ptr, int n,
                                          cpp11::sexp r_size,
                                          cpp11::doubles r_prob) {
  dust::pRNG<real_t> *rng = cpp11::as_cpp<dust_rng_ptr_t<real_t>>(ptr).get();
  cpp11::integers r_size_int = dust::interface::as_integer(r_size, "size");
  const int * size = INTEGER(r_size_int);
  const double * prob = REAL(r_prob);

  cpp11::writable::integers ret = cpp11::writable::integers(n);
  int * y = INTEGER(ret);

  const size_t n_generators = rng->size();
  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::distr::rbinom(rng->state(i % n_generators), size[i], prob[i]);
  }

  return ret;
}

template <typename real_t>
cpp11::writable::integers dust_rng_rpois(SEXP ptr, int n,
                                         cpp11::doubles r_lambda) {
  dust::pRNG<real_t> *rng = cpp11::as_cpp<dust_rng_ptr_t<real_t>>(ptr).get();
  const double * lambda = REAL(r_lambda);
  const size_t n_generators = rng->size();

  cpp11::writable::integers ret = cpp11::writable::integers(n);
  int * y = INTEGER(ret);

  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::distr::rpois(rng->state(i % n_generators), lambda[i]);
  }

  return ret;
}

template <typename real_t>
cpp11::writable::raws dust_rng_state(SEXP ptr) {
  dust::pRNG<real_t> *rng = cpp11::as_cpp<dust_rng_ptr_t<real_t>>(ptr).get();
  auto state = rng->export_state();
  size_t len = sizeof(uint64_t) * state.size();
  cpp11::writable::raws ret(len);
  std::memcpy(RAW(ret), state.data(), len);
  return ret;
}

}
}

[[cpp11::register]]
SEXP dust_rng_alloc(cpp11::sexp r_seed, int n_generators, bool is_float) {
  return is_float ?
    dust::rng::dust_rng_alloc<float>(r_seed, n_generators) :
    dust::rng::dust_rng_alloc<double>(r_seed, n_generators);
}

[[cpp11::register]]
void dust_rng_jump(SEXP ptr, bool is_float) {
  if (is_float) {
    dust::rng::dust_rng_jump<float>(ptr);
  } else {
    dust::rng::dust_rng_jump<double>(ptr);
  }
}

[[cpp11::register]]
void dust_rng_long_jump(SEXP ptr, bool is_float) {
  if (is_float) {
    dust::rng::dust_rng_long_jump<float>(ptr);
  } else {
    dust::rng::dust_rng_long_jump<double>(ptr);
  }
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_unif_rand(SEXP ptr, int n, bool is_float) {
  return is_float ?
    dust::rng::dust_rng_unif_rand<float>(ptr, n) :
    dust::rng::dust_rng_unif_rand<double>(ptr, n);
}

// NOTE: no special treatment (yet) for this
[[cpp11::register]]
cpp11::writable::doubles dust_rng_norm_rand(SEXP ptr, int n, bool is_float) {
  return is_float ?
    dust::rng::dust_rng_norm_rand<float>(ptr, n) :
    dust::rng::dust_rng_norm_rand<double>(ptr, n);
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_runif(SEXP ptr, int n,
                                        cpp11::doubles r_min,
                                        cpp11::doubles r_max,
                                        bool is_float) {
  return is_float ?
    dust::rng::dust_rng_runif<float>(ptr, n, r_min, r_max) :
    dust::rng::dust_rng_runif<double>(ptr, n, r_min, r_max);
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_rexp(SEXP ptr, int n, cpp11::doubles r_rate,
                                       bool is_float) {
  return is_float ?
    dust::rng::dust_rng_rexp<float>(ptr, n, r_rate) :
    dust::rng::dust_rng_rexp<double>(ptr, n, r_rate);
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_rnorm(SEXP ptr, int n, cpp11::doubles r_mean,
                                        cpp11::doubles r_sd, bool is_float) {
  return is_float ?
    dust::rng::dust_rng_rnorm<float>(ptr, n, r_mean, r_sd) :
    dust::rng::dust_rng_rnorm<double>(ptr, n, r_mean, r_sd);
}

[[cpp11::register]]
cpp11::writable::integers dust_rng_rbinom(SEXP ptr, int n, cpp11::sexp r_size,
                                          cpp11::doubles r_prob,
                                          bool is_float) {
  return is_float ?
    dust::rng::dust_rng_rbinom<float>(ptr, n, r_size, r_prob) :
    dust::rng::dust_rng_rbinom<double>(ptr, n, r_size, r_prob);
}

[[cpp11::register]]
cpp11::writable::integers dust_rng_rpois(SEXP ptr, int n,
                                         cpp11::doubles r_lambda,
                                         bool is_float) {
  return is_float ?
    dust::rng::dust_rng_rpois<float>(ptr, n, r_lambda) :
    dust::rng::dust_rng_rpois<double>(ptr, n, r_lambda);
}

[[cpp11::register]]
cpp11::writable::raws dust_rng_state(SEXP ptr, bool is_float) {
  return is_float ?
    dust::rng::dust_rng_state<float>(ptr) :
    dust::rng::dust_rng_state<double>(ptr);
}
