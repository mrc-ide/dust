#ifndef DUST_RANDOM_DENSITY_HPP
#define DUST_RANDOM_DENSITY_HPP

#include <cmath>
#include <limits>
#include <type_traits>
#include <dust/random/cuda.hpp>
#include <dust/random/numeric.hpp>

namespace dust {
namespace density {

namespace {

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

__nv_exec_check_disable__
template <typename T>
HOSTDEVICE T maybe_log(T x, bool log) {
  return log ? x : std::exp(x);
}

template <typename T>
HOSTDEVICE T lchoose(T n, T k) {
  return random::utils::lgamma(static_cast<T>(n + 1)) -
    random::utils::lgamma(static_cast<T>(k + 1)) -
    random::utils::lgamma(static_cast<T>(n - k + 1));
}

template <typename T>
HOSTDEVICE T lbeta(T x, T y) {
  return random::utils::lgamma(x) + random::utils::lgamma(y) -
    random::utils::lgamma(x + y);
}

}

template <typename T>
HOSTDEVICE T binomial(int x, int size, T prob, bool log) {
#ifndef __CUDA_ARCH__
  static_assert(std::is_floating_point<T>::value,
                "binomial should only be used with real types");
#endif
  T ret;
  if (x == 0 && size == 0) {
    ret = 0;
  } else {
    ret = lchoose<T>(size, x) +
      x * std::log(prob) +
      (size - x) * std::log(1 - prob);
  }

  SYNCWARP
  return maybe_log(ret, log);
}

template <typename T>
HOSTDEVICE T dirac_delta(T x, bool log) {
  const T inf = random::utils::infinity<T>();
  return maybe_log(x == 0 ? inf : -inf, log);
}

template <typename T>
HOSTDEVICE T normal(T x, T mu, T sd, bool log) {
  T ret;
  if (sd == 0) {
    ret = dirac_delta(x - mu, log); // This does maybe_log
  } else {
    const T dx = x - mu;
    ret = - dx * dx / (2 * sd * sd) - norm_integral<T>() - std::log(sd);
    ret = maybe_log(ret, log);
  }

  SYNCWARP
  return ret;
}

template <typename T>
HOSTDEVICE T negative_binomial_mu(int x, T size, T mu, bool log) {
#ifndef __CUDA_ARCH__
  static_assert(std::is_floating_point<T>::value,
                "negative_binomial should only be used with real types");
#endif
  T ret;
  if (x == 0 && size == 0) {
    ret = 0;
  }
  else if (x < 0 || size == 0) {
    ret = -random::utils::infinity<T>();
  }
  else if (mu == 0) {
    ret = x == 0 ? 0 : -random::utils::infinity<T>();
  } else {
    // Avoid size / (size + mu) when size is close to zero, and this
    // would cause prob to be equal to zero. Somewhat arbitrarily,
    // taking 100 * floating point eps as the change over.
    const T ratio = random::utils::epsilon<T>() * 100;
    if (mu < ratio * size) {
      const T log_prob = std::log(mu / (1 + mu / size));
      ret = x * log_prob - mu - random::utils::lgamma(static_cast<T>(x + 1)) +
        std::log1p(x * (x - 1) / (2 * size));
    } else {
      const T prob = size / (size + mu);
      ret = random::utils::lgamma(static_cast<T>(x + size)) -
        random::utils::lgamma(static_cast<T>(size)) -
        random::utils::lgamma(static_cast<T>(x + 1)) +
        size * std::log(prob) + x * std::log(1 - prob);
    }
  }

  SYNCWARP
  return maybe_log(ret, log);
}

// This may not be stable for all size and prob, but provides
// compatibility with R's C-level function. See ?dnbinom for details.
template <typename T>
HOSTDEVICE T negative_binomial_prob(int x, T size, T prob, bool log) {
  const T mu = size * (1 - prob) / prob;
  return negative_binomial_mu(x, size, mu, log);
}

// A note on this parametrisation:
//
//   prob = alpha / (alpha + beta)
//   rho = 1 / (alpha + beta + 1)
//
// Where alpha and beta have (0, Inf) support
template <typename T>
HOSTDEVICE T beta_binomial(int x, int size, T prob, T rho, bool log) {
#ifndef __CUDA_ARCH__
  static_assert(std::is_floating_point<T>::value,
                "beta_binomial should only be used with real types");
#endif
  T ret;
  if (x == 0 && size == 0) {
    ret = 0;
  } else {
    const T a = prob * (1 / rho - 1);
    const T b = (1 - prob) * (1 / rho - 1);
    ret = lchoose<T>(size, x) + lbeta(x + a, size - x + b) - lbeta(a, b);
  }

  SYNCWARP
  return maybe_log(ret, log);
}

template <typename T>
HOSTDEVICE T poisson(int x, T lambda, bool log) {
#ifndef __CUDA_ARCH__
  static_assert(std::is_floating_point<T>::value,
                "poisson should only be used with real types");
#endif
  T ret;
  if (x == 0 && lambda == 0) {
    ret = 0;
  } else {
    ret = x * std::log(lambda) - lambda -
      random::utils::lgamma(static_cast<T>(x + 1));
  }

  SYNCWARP
  return maybe_log(ret, log);
}

}
}

#endif
