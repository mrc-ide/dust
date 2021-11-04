#include <cstring>
#include <cpp11/external_pointer.hpp>
#include <cpp11/raws.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/integers.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <dust/random/random.hpp>
#include <dust/interface/random.hpp>
#include <dust/utils.hpp>

using dust_rng64 = dust::random::prng<dust::random::generator<double>>;
using dust_rng32 = dust::random::prng<dust::random::generator<float>>;

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

// Little helper for returning x as a vector (m == 1) or matrix (n * m)
cpp11::writable::doubles double_matrix(cpp11::writable::doubles x,
                                       int n, int m) {
  if (m > 1) {
    x.attr("dim") = cpp11::writable::integers{n, m};
  }
  return x;
}

template <typename real_type, typename T>
cpp11::writable::doubles dust_rng_random_real(SEXP ptr, int n, int n_threads) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  const int n_generators = rng->size();

  cpp11::writable::doubles ret = cpp11::writable::doubles(n * n_generators);
  double * y = REAL(ret);

#ifdef _OPENMP
   #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (int i = 0; i < n_generators; ++i) {
    auto &state = rng->state(i);
    auto y_i = y + n * i;
    for (size_t j = 0; j < (size_t)n; ++j) {
      y_i[j] = dust::random::random_real<real_type>(state);
    }
  }

  return double_matrix(ret, n, n_generators);
}

struct input_vary {
  size_t len;
  size_t offset;
  bool draw;
  bool generator;
};
                  
input_vary check_input_type(cpp11::doubles x, int n, int m, const char *name) {
  input_vary ret {1, 1, false, false};
  if (Rf_isMatrix(x)) {
    if (Rf_ncols(x) != m) {
      cpp11::stop("If '%s' is a matrix, it must have %d columns", name, m);
    }
    ret.generator = true;
    if (Rf_nrows(x) == n) {
      ret.draw = true;
    } else if (Rf_nrows(x) != 1) {
      cpp11::stop("If '%s' is a matrix, it must have 1 or %d rows", name, n);
    }
  } else {
    if (x.size() == n) {
      ret.draw = true;
    } else if (x.size() != 1) {
      cpp11::stop("If '%s' is a vector, it must have 1 or %d elements",
                  name, n);
    }
  }

  if (ret.draw) {
    ret.offset = n;
  }
  
  return ret;
}

// n = n draws
// m = n generators

// normally: scalar - don't vary
// vector: length n, vary over draws, shared across draws
// matrix:
//   - n x m: vary over both
//   - 1 x m: vary over generators, shared over draws

// If we add a dimension to this, and letting 'len' be the number of classes,
// we get:
// normally: vector - don't vary
// matrix len x n: vary over draws
// array:
//   len x n x m: vary over both
//   len x 1 x m: vary over generators

input_vary check_input_type2(cpp11::doubles x, int n, int m, const char *name) {
  input_vary ret {1, 1, false, false};
  cpp11::sexp r_dim = x.attr("dim");
  if (r_dim == R_NilValue) {
    ret.len = x.size();
  } else if (LENGTH(r_dim) == 2) { // matrix
    auto dim = cpp11::as_cpp<cpp11::integers>(r_dim);
    ret.len = dim[0];
    if (dim[1] == n) {
      ret.draw = true;
    } else if (dim[1] != 1) {
      cpp11::stop("If '%s' is a matrix, it must have 1 or %d columns",
                  name, n);
    }
  } else if (LENGTH(r_dim) == 3) {
    auto dim = cpp11::as_cpp<cpp11::integers>(r_dim);
    ret.len = dim[0];
    if (dim[1] == n) {
      ret.draw = true;
    } else if (dim[1] != 1) {
      cpp11::stop("If '%s' is a 3d array, it must have 1 or %d columns",
                  name, n);
    }
    if (dim[2] != m) {
      cpp11::stop("If '%s' is a 3d array, it must have %d layers", name, m);
    }
    ret.generator = true;
  } else {
    cpp11::stop("'%s' must be a vector, matrix or 3d array", name);
  }

  if (ret.len < 2) {
    cpp11::stop("Input parameters imply length of '%s' of only %d (< 2)",
                name, ret.len);
  }

  if (ret.draw) {
    ret.offset = n * ret.len;
  } else {
    ret.offset = ret.len;
  }

  return ret;
}

// Below here is very repetitive, and could probably be deduplicated
// with some clever template magic. Most of the faff is because we
// want to support 4 modes of taking 1 or 2 parameters (each varying
// or not over draws and generators)
template <typename real_type, typename T>
cpp11::writable::doubles dust_rng_uniform(SEXP ptr, int n,
                                          cpp11::doubles r_min,
                                          cpp11::doubles r_max,
                                          int n_threads) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  const int n_generators = rng->size();
  cpp11::writable::doubles ret = cpp11::writable::doubles(n * n_generators);
  double * y = REAL(ret);

  const double * min = REAL(r_min);
  const double * max = REAL(r_max);
  auto min_vary = check_input_type(r_min, n, n_generators, "min");
  auto max_vary = check_input_type(r_max, n, n_generators, "max");

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (int i = 0; i < n_generators; ++i) {
    auto &state = rng->state(i);
    auto y_i = y + n * i;
    auto min_i = min_vary.generator ? min + min_vary.offset * i : min;
    auto max_i = max_vary.generator ? max + max_vary.offset * i : max;
    for (size_t j = 0; j < (size_t)n; ++j) {
      auto min_ij = min_vary.draw ? min_i[j] : min_i[0];
      auto max_ij = max_vary.draw ? max_i[j] : max_i[0];
      y_i[j] = dust::random::uniform<real_type>(state, min_ij, max_ij);
    }
  }

  return double_matrix(ret, n, n_generators);
}

template <typename real_type, typename T>
cpp11::writable::doubles dust_rng_exponential(SEXP ptr, int n,
                                              cpp11::doubles r_rate,
                                              int n_threads) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  const int n_generators = rng->size();
  cpp11::writable::doubles ret = cpp11::writable::doubles(n * n_generators);
  double * y = REAL(ret);

  const double * rate = REAL(r_rate);
  auto rate_vary = check_input_type(r_rate, n, n_generators, "rate");

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (int i = 0; i < n_generators; ++i) {
    auto &state = rng->state(i);
    auto y_i = y + n * i;
    auto rate_i = rate_vary.generator ? rate + rate_vary.offset * i : rate;
    for (size_t j = 0; j < (size_t)n; ++j) {
      auto rate_ij = rate_vary.draw ? rate_i[j] : rate_i[0];
      y_i[j] = dust::random::exponential<real_type>(state, rate_ij);
    }
  }

  return double_matrix(ret, n, n_generators);
}


template <typename real_type, typename T>
cpp11::writable::doubles dust_rng_normal(SEXP ptr, int n,
                                         cpp11::doubles r_mean,
                                         cpp11::doubles r_sd,
                                         int n_threads) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  const int n_generators = rng->size();
  cpp11::writable::doubles ret = cpp11::writable::doubles(n * n_generators);
  double * y = REAL(ret);

  const double * mean = REAL(r_mean);
  const double * sd = REAL(r_sd);
  auto mean_vary = check_input_type(r_mean, n, n_generators, "mean");
  auto sd_vary = check_input_type(r_sd, n, n_generators, "sd");

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (int i = 0; i < n_generators; ++i) {
    auto &state = rng->state(i);
    auto y_i = y + n * i;
    auto mean_i = mean_vary.generator ? mean + mean_vary.offset * i : mean;
    auto sd_i = sd_vary.generator ? sd + sd_vary.offset * i : sd;
    for (size_t j = 0; j < (size_t)n; ++j) {
      auto mean_ij = mean_vary.draw ? mean_i[j] : mean_i[0];
      auto sd_ij = sd_vary.draw ? sd_i[j] : sd_i[0];
      y_i[j] = dust::random::normal<real_type>(state, mean_ij, sd_ij);
    }
  }

  return double_matrix(ret, n, n_generators);
}

template <typename real_type, typename T>
cpp11::writable::doubles dust_rng_binomial(SEXP ptr, int n,
                                           cpp11::doubles r_size,
                                           cpp11::doubles r_prob,
                                           int n_threads) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  const int n_generators = rng->size();
  cpp11::writable::doubles ret = cpp11::writable::doubles(n * n_generators);
  double * y = REAL(ret);

  const double * size = REAL(r_size);
  const double * prob = REAL(r_prob);
  auto size_vary = check_input_type(r_size, n, n_generators, "size");
  auto prob_vary = check_input_type(r_prob, n, n_generators, "prob");

  dust::utils::openmp_errors errors(n_generators);

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (int i = 0; i < n_generators; ++i) {
    try {
      auto &state = rng->state(i);
      auto y_i = y + n * i;
      auto size_i = size_vary.generator ? size + size_vary.offset * i : size;
      auto prob_i = prob_vary.generator ? prob + prob_vary.offset * i : prob;
      for (size_t j = 0; j < (size_t)n; ++j) {
        auto size_ij = size_vary.draw ? size_i[j] : size_i[0];
        auto prob_ij = prob_vary.draw ? prob_i[j] : prob_i[0];
        y_i[j] = dust::random::binomial<real_type>(state, size_ij, prob_ij);
      }
    } catch (std::exception const& e) {
      errors.capture(e, i);
    }
  }

  errors.report("generators");

  return double_matrix(ret, n, n_generators);
}

template <typename real_type, typename T>
cpp11::writable::doubles dust_rng_poisson(SEXP ptr, int n,
                                          cpp11::doubles r_lambda,
                                          int n_threads) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  const int n_generators = rng->size();
  cpp11::writable::doubles ret = cpp11::writable::doubles(n * n_generators);
  double * y = REAL(ret);

  const double * lambda = REAL(r_lambda);
  auto lambda_vary = check_input_type(r_lambda, n, n_generators, "lambda");

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (int i = 0; i < n_generators; ++i) {
    auto &state = rng->state(i);
    auto y_i = y + n * i;
    auto lambda_i = lambda_vary.generator ? lambda + lambda_vary.offset * i :
      lambda;
    for (size_t j = 0; j < (size_t)n; ++j) {
      auto lambda_ij = lambda_vary.draw ? lambda_i[j] : lambda_i[0];
      y_i[j] = dust::random::poisson<real_type>(state, lambda_ij);
    }
  }

  return double_matrix(ret, n, n_generators);
}

template <typename real_type, typename T>
cpp11::writable::doubles dust_rng_multinomial(SEXP ptr, int n,
                                              cpp11::doubles r_size,
                                              cpp11::doubles r_prob,
                                              int n_threads) {
  T *rng = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  const int n_generators = rng->size();
  // What do we assume here about prob and size? We need to require
  // that 'prob' has identical length for all cases I think.  For now
  // we will assume a single vector!

  const double * size = REAL(r_size);
  const double * prob = REAL(r_prob);
  auto size_vary = check_input_type(r_size, n, n_generators, "size");
  auto prob_vary = check_input_type2(r_prob, n, n_generators, "prob");
  const int len = prob_vary.len;

  // Normally we return a block of doubles with the first 'n' entries
  // being the results for the first generator, the second 'n' for the
  // second, and so on. Here the first n * len are the first generator
  // (with the first 'len' being the first sample.
  cpp11::writable::doubles ret =
    cpp11::writable::doubles(len * n * n_generators);
  double * y = REAL(ret);

  dust::utils::openmp_errors errors(n_generators);

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (int i = 0; i < n_generators; ++i) {
    try {
      auto &state = rng->state(i);
      auto y_i = y + len * n * i;
      auto size_i = size_vary.generator ? size + size_vary.offset * i : size;
      auto prob_i = prob_vary.generator ? prob + prob_vary.offset * i : prob;
      for (size_t j = 0; j < (size_t)n; ++j) {
        auto size_ij = size_vary.draw ? size_i[j]        : size_i[0];
        auto prob_ij = prob_vary.draw ? prob_i + j * len : prob_i;
        auto y_ij = y_i + j * len;
        dust::random::multinomial<real_type>(state, size_ij, prob_ij, len,
                                             y_ij);
      }
    } catch (std::exception const& e) {
      errors.capture(e, i);
    }
  }
  errors.report("generators");

  if (n_generators == 1) {
    ret.attr("dim") = cpp11::writable::integers{len, n};
  } else {
    ret.attr("dim") = cpp11::writable::integers{len, n, n_generators};
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
cpp11::writable::doubles dust_rng_random_real(SEXP ptr, int n, int n_threads,
                                              bool is_float) {
  return is_float ?
    dust_rng_random_real<float, dust_rng32>(ptr, n, n_threads) :
    dust_rng_random_real<double, dust_rng64>(ptr, n, n_threads);
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_uniform(SEXP ptr, int n,
                                          cpp11::doubles r_min,
                                          cpp11::doubles r_max,
                                          int n_threads,
                                          bool is_float) {
  return is_float ?
    dust_rng_uniform<float, dust_rng32>(ptr, n, r_min, r_max, n_threads) :
    dust_rng_uniform<double, dust_rng64>(ptr, n, r_min, r_max, n_threads);
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_exponential(SEXP ptr, int n,
                                              cpp11::doubles r_rate,
                                              int n_threads,
                                              bool is_float) {
  return is_float ?
    dust_rng_exponential<float, dust_rng32>(ptr, n, r_rate, n_threads) :
    dust_rng_exponential<double, dust_rng64>(ptr, n, r_rate, n_threads);
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_normal(SEXP ptr, int n, cpp11::doubles r_mean,
                                         cpp11::doubles r_sd, int n_threads,
                                         bool is_float) {
  return is_float ?
    dust_rng_normal<float, dust_rng32>(ptr, n, r_mean, r_sd, n_threads) :
    dust_rng_normal<double, dust_rng64>(ptr, n, r_mean, r_sd, n_threads);
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_binomial(SEXP ptr, int n,
                                           cpp11::doubles r_size,
                                           cpp11::doubles r_prob,
                                           int n_threads,
                                           bool is_float) {
  return is_float ?
    dust_rng_binomial<float, dust_rng32>(ptr, n, r_size, r_prob, n_threads) :
    dust_rng_binomial<double, dust_rng64>(ptr, n, r_size, r_prob, n_threads);
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_poisson(SEXP ptr, int n,
                                          cpp11::doubles r_lambda,
                                          int n_threads,
                                          bool is_float) {
  return is_float ?
    dust_rng_poisson<float, dust_rng32>(ptr, n, r_lambda, n_threads) :
    dust_rng_poisson<double, dust_rng64>(ptr, n, r_lambda, n_threads);
}

[[cpp11::register]]
cpp11::writable::doubles dust_rng_multinomial(SEXP ptr, int n,
                                              cpp11::doubles r_size,
                                              cpp11::doubles r_prob,
                                              int n_threads,
                                              bool is_float) {
  return is_float ?
    dust_rng_multinomial<float, dust_rng32>(ptr, n, r_size, r_prob, n_threads) :
    dust_rng_multinomial<double, dust_rng64>(ptr, n, r_size, r_prob, n_threads);
}

[[cpp11::register]]
cpp11::writable::raws dust_rng_state(SEXP ptr, bool is_float) {
  return is_float ?
    dust_rng_state<dust_rng32>(ptr) :
    dust_rng_state<dust_rng64>(ptr);
}
