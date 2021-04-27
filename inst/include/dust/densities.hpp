#ifndef DUST_DENSITIES_HPP
#define DUST_DENSITIES_HPP

#include <cmath>
#include <limits>
#include <type_traits>
#include <dust/cuda.cuh>
#include <dust/utils.hpp>

CONSTANT double m_ln_sqrt_2pi_dbl = 0.918938533204672741780329736406;
CONSTANT float m_ln_sqrt_2pi_flt = 0.918938533204672741780329736406f;

// Returns m_ln_sqrt_2pi
template <typename real_t>
HOSTDEVICE real_t norm_integral();

template<>
HOSTDEVICE inline double norm_integral() {
  return m_ln_sqrt_2pi_dbl;
}

template<>
HOSTDEVICE inline float norm_integral() {
  return m_ln_sqrt_2pi_flt;
}

namespace dust {

__nv_exec_check_disable__
template <typename T>
HOSTDEVICE T maybe_log(T x, bool log) {
  return log ? x : std::exp(x);
}

template <typename T>
HOSTDEVICE T lchoose(T n, T k) {
  return dust::utils::lgamma(static_cast<T>(n + 1)) -
    dust::utils::lgamma(static_cast<T>(k + 1)) -
    dust::utils::lgamma(static_cast<T>(n - k + 1));
}

template <typename T>
HOSTDEVICE T lbeta(T x, T y) {
  return dust::utils::lgamma(x) + dust::utils::lgamma(y) - dust::utils::lgamma(x + y);
}

template <typename T>
HOSTDEVICE T dbinom(int x, int size, T prob, bool log) {
#ifndef __CUDA_ARCH__
  static_assert(std::is_floating_point<T>::value,
                "dbinom should only be used with real types");
#endif
  if (x == 0 && size == 0) {
    return maybe_log(0, log);
  }
  const T ret = lchoose<T>(size, x) +
    x * std::log(prob) +
    (size - x) * std::log(1 - prob);
  return maybe_log(ret, log);
}

template <typename T>
HOSTDEVICE T ddelta(T x, bool log) {
  const T inf = dust::utils::infinity<T>();
  return maybe_log(x == 0 ? inf : -inf, log);
}

template <typename T>
HOSTDEVICE T dnorm(T x, T mu, T sd, bool log) {
  if (sd == 0) {
    return ddelta(x - mu, log);
  }
  const T dx = x - mu;
  const T ret = - dx * dx / (2 * sd * sd) - norm_integral<T>() - std::log(sd);
  return maybe_log(ret, log);
}

template <typename T>
HOSTDEVICE T dnbinom(int x, T size, T mu, bool log) {
#ifndef __CUDA_ARCH__
  static_assert(std::is_floating_point<T>::value,
                "dnbinom should only be used with real types");
#endif
  const T prob = size / (size + mu);
  if (x == 0 && size == 0) {
    return maybe_log(0, log);
  }
  if (x < 0 || size == 0) {
    return maybe_log(-dust::utils::infinity<T>(), log);
  }
  if (mu == 0) {
    return maybe_log(x == 0 ? 0 : -dust::utils::infinity<T>(), log);
  }
  const T ret = dust::utils::lgamma(static_cast<T>(x + size)) -
    dust::utils::lgamma(static_cast<T>(size)) -
    dust::utils::lgamma(static_cast<T>(x + 1)) +
    size * std::log(prob) + x * std::log(1 - prob);
  return maybe_log(ret, log);
}

// A note on this parametrisation:
//
//   prob = alpha / (alpha + beta)
//   rho = 1 / (alpha + beta + 1)
//
// Where alpha and beta have (0, Inf) support
template <typename T>
HOSTDEVICE T dbetabinom(int x, int size, T prob, T rho, bool log) {
#ifndef __CUDA_ARCH__
  static_assert(std::is_floating_point<T>::value,
                "dbetabinom should only be used with real types");
#endif
  if (x == 0 && size == 0) {
    return maybe_log(0, log);
  }
  const T a = prob * (1 / rho - 1);
  const T b = (1 - prob) * (1 / rho - 1);
  const T ret = lchoose<T>(size, x) + lbeta(x + a, size - x + b) - lbeta(a, b);
  return maybe_log(ret, log);
}

template <typename T>
HOSTDEVICE T dpois(int x, T lambda, bool log) {
#ifndef __CUDA_ARCH__
  static_assert(std::is_floating_point<T>::value,
                "dpois should only be used with real types");
#endif
  if (x == 0 && lambda == 0) {
    return maybe_log(0, log);
  }
  const T ret = x * std::log(lambda) - lambda -
    dust::utils::lgamma(static_cast<T>(x + 1));
  return maybe_log(ret, log);
}

}

#endif
