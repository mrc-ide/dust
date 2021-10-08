#include <cstring>
#include <cpp11/external_pointer.hpp>
#include <cpp11/raws.hpp>
#include <cpp11/doubles.hpp>

#include <dust/random/random.hpp>
#include <dust/interface/random.hpp>

using dust_rng64 = dust::random::prng<dust::random::xoshiro256starstar_state>;
using dust_rng32 = dust::random::prng<dust::random::xoshiro128starstar_state>;

template <typename T>
SEXP dust_rng_alloc(cpp11::sexp r_seed, int n_generators, bool deterministic) {
  auto seed = dust::interface::as_rng_seed<typename T::rng_state>(r_seed);
  T *rng = new T(n_generators, seed, deterministic);
  return cpp11::external_pointer<T>(rng);
}

template <typename T>
void dust_rng_jump(SEXP ptr) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  rng->jump();
}

template <typename T>
void dust_rng_long_jump(SEXP ptr) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  rng->long_jump();
}

template <typename real_type, typename T>
cpp11::writable::doubles dust_rng_random_real(SEXP ptr, int n) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  const size_t n_generators = rng->size();

  cpp11::writable::doubles ret = cpp11::writable::doubles(n);
  double * y = REAL(ret);

  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::random::random_real<real_type>(rng->state(i % n_generators));
  }

  return ret;
}

template <typename real_type, typename T>
cpp11::writable::doubles dust_rng_uniform(SEXP ptr, int n,
                                          cpp11::doubles r_min,
                                          cpp11::doubles r_max) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  const double * min = REAL(r_min);
  const double * max = REAL(r_max);
  const size_t n_generators = rng->size();

  cpp11::writable::doubles ret = cpp11::writable::doubles(n);
  double * y = REAL(ret);

  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::random::uniform<real_type>(rng->state(i % n_generators),
                                            min[i], max[i]);
  }

  return ret;
}

template <typename real_type, typename T>
cpp11::writable::doubles dust_rng_exponential(SEXP ptr, int n,
                                              cpp11::doubles r_rate) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  const double * rate = REAL(r_rate);
  const size_t n_generators = rng->size();

  cpp11::writable::doubles ret = cpp11::writable::doubles(n);
  double * y = REAL(ret);

  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::random::exponential<real_type>(rng->state(i % n_generators),
                                                rate[i]);
  }

  return ret;
}

template <typename real_type, typename T>
cpp11::writable::doubles dust_rng_normal(SEXP ptr, int n,
                                         cpp11::doubles r_mean,
                                         cpp11::doubles r_sd) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  const double * mean = REAL(r_mean);
  const double * sd = REAL(r_sd);
  const size_t n_generators = rng->size();

  cpp11::writable::doubles ret = cpp11::writable::doubles(n);
  double * y = REAL(ret);

  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::random::normal<real_type>(rng->state(i % n_generators),
                                           mean[i], sd[i]);
  }

  return ret;
}

template <typename real_type, typename T>
cpp11::writable::doubles dust_rng_binomial(SEXP ptr, int n,
                                           cpp11::doubles r_size,
                                           cpp11::doubles r_prob) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  const double * size = REAL(r_size);
  const double * prob = REAL(r_prob);

  cpp11::writable::doubles ret = cpp11::writable::doubles(n);
  double * y = REAL(ret);

  const size_t n_generators = rng->size();
  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::random::binomial<real_type>(rng->state(i % n_generators),
                                             size[i], prob[i]);
  }

  return ret;
}

template <typename real_type, typename T>
cpp11::writable::doubles dust_rng_poisson(SEXP ptr, int n,
                                          cpp11::doubles r_lambda) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  const double * lambda = REAL(r_lambda);
  const size_t n_generators = rng->size();

  cpp11::writable::doubles ret = cpp11::writable::doubles(n);
  double * y = REAL(ret);

  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::random::poisson<real_type>(rng->state(i % n_generators),
                                            lambda[i]);
  }

  return ret;
}

template <typename T>
cpp11::writable::raws dust_rng_state(SEXP ptr) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  auto state = rng->export_state();
  size_t len = sizeof(typename T::int_type) * state.size();
  cpp11::writable::raws ret(len);
  std::memcpy(RAW(ret), state.data(), len);
  return ret;
}

[[cpp11::register]]
SEXP dust_rng_alloc(cpp11::sexp r_seed, int n_generators, bool deterministic,
                    bool is_float) {
  return is_float ?
    dust_rng_alloc<dust_rng32>(r_seed, n_generators, deterministic) :
    dust_rng_alloc<dust_rng64>(r_seed, n_generators, deterministic);
}

[[cpp11::register]]
void dust_rng_jump(SEXP ptr, bool is_float) {
  if (is_float) {
    dust_rng_jump<dust_rng32>(ptr);
  } else {
    dust_rng_jump<dust_rng64>(ptr);
  }
}

[[cpp11::register]]
void dust_rng_long_jump(SEXP ptr, bool is_float) {
  if (is_float) {
    dust_rng_long_jump<dust_rng32>(ptr);
  } else {
    dust_rng_long_jump<dust_rng64>(ptr);
  }
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_random_real(SEXP ptr, int n, bool is_float) {
  return is_float ?
    dust_rng_random_real<float, dust_rng32>(ptr, n) :
    dust_rng_random_real<double, dust_rng64>(ptr, n);
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_uniform(SEXP ptr, int n,
                                          cpp11::doubles r_min,
                                          cpp11::doubles r_max,
                                          bool is_float) {
  return is_float ?
    dust_rng_uniform<float, dust_rng32>(ptr, n, r_min, r_max) :
    dust_rng_uniform<double, dust_rng64>(ptr, n, r_min, r_max);
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_exponential(SEXP ptr, int n,
                                              cpp11::doubles r_rate,
                                              bool is_float) {
  return is_float ?
    dust_rng_exponential<float, dust_rng32>(ptr, n, r_rate) :
    dust_rng_exponential<double, dust_rng64>(ptr, n, r_rate);
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_normal(SEXP ptr, int n, cpp11::doubles r_mean,
                                         cpp11::doubles r_sd, bool is_float) {
  return is_float ?
    dust_rng_normal<float, dust_rng32>(ptr, n, r_mean, r_sd) :
    dust_rng_normal<double, dust_rng64>(ptr, n, r_mean, r_sd);
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_binomial(SEXP ptr, int n,
                                           cpp11::doubles r_size,
                                           cpp11::doubles r_prob,
                                           bool is_float) {
  return is_float ?
    dust_rng_binomial<float, dust_rng32>(ptr, n, r_size, r_prob) :
    dust_rng_binomial<double, dust_rng64>(ptr, n, r_size, r_prob);
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_poisson(SEXP ptr, int n,
                                          cpp11::doubles r_lambda,
                                          bool is_float) {
  return is_float ?
    dust_rng_poisson<float, dust_rng32>(ptr, n, r_lambda) :
    dust_rng_poisson<double, dust_rng64>(ptr, n, r_lambda);
}

[[cpp11::register]]
cpp11::writable::raws dust_rng_state(SEXP ptr, bool is_float) {
  return is_float ?
    dust_rng_state<dust_rng32>(ptr) :
    dust_rng_state<dust_rng64>(ptr);
}
