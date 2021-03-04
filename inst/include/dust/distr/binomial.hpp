#ifndef DUST_DISTR_BINOMIAL_HPP
#define DUST_DISTR_BINOMIAL_HPP

#include <cmath>
#include <dust/utils.hpp>
#include <dust/distr/gamma_table.hpp>

namespace dust {
namespace distr {

// Faster version of pow(x, n) for integer 'n' by using
// "exponentiation by squaring"
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
template <typename real_t>
HOSTDEVICE real_t fast_pow(real_t x, int n) {
  real_t pow = 1.0;
  if (n != 0) {
    while (true) {
      if(n & 01) {
        pow *= x;
      }
      if(n >>= 1) {
        x *= x;
      } else {
        break;
      }
    }
  }
  return pow;
}

__nv_exec_check_disable__
template <typename real_t>
real_t HOSTDEVICE binomial_inversion_calc(real_t u, int n, real_t p) {
  const real_t q = 1 - p;
  const real_t r = p / q;
  const real_t g = r * (n + 1);
  real_t f = fast_pow(q, n);
  int k = 0;

  real_t f_prev = f;
  while (u >= f) {
    u -= f;
    k++;
    f *= (g / k - r);
    if (f == f_prev) {
      // This catches an issue seen running with floats where we end
      // up unable to decrease 'f' because we've run out of
      // precision. In this case we'll try again with a better u
      return -1;
    }
    f_prev = f;
  }

  return k;
}

// Binomial random numbers via inversion (for low np only!). Draw a
// random number from U(0, 1) and find the 'n' up the distribution
// (given p) that corresponds to this
__nv_exec_check_disable__
template <typename real_t>
real_t HOSTDEVICE binomial_inversion(rng_state_t<real_t>& rng_state, int n,
                                     real_t p) {
  real_t k = -1;
  do {
    real_t u = dust::unif_rand(rng_state);
    k = binomial_inversion_calc(u, n, p);
  } while (k < 0);
  return k;
}

template <typename real_t>
HOSTDEVICE real_t stirling_approx_tail(real_t k);

template <typename real_t>
HOSTDEVICE inline real_t stirling_approx_tail_calc(real_t k) {
  const real_t one = 1;
  real_t kp1sq = (k + 1) * (k + 1);
  return (one / 12 - (one / 360 - one / 1260 / kp1sq) / kp1sq) / (k + 1);
}

template <>
HOSTDEVICE inline float stirling_approx_tail(float k) {
  float tail;
  if (k <= k_tail_values_max_f) {
    tail = k_tail_values_f[static_cast<int>(k)];
  } else {
    tail = stirling_approx_tail_calc(k);
  }
  return tail;
}

template <>
HOSTDEVICE inline double stirling_approx_tail(double k) {
  double tail;
  if (k <= k_tail_values_max_d) {
    tail = k_tail_values_d[static_cast<int>(k)];
  } else {
    tail = stirling_approx_tail_calc(k);
  }
  return tail;
}

// https://www.tandfonline.com/doi/abs/10.1080/00949659308811496
__nv_exec_check_disable__
template <typename real_t>
inline HOSTDEVICE real_t btrs(rng_state_t<real_t>& rng_state, int n_int, real_t p) {
  const real_t n = static_cast<real_t>(n_int);
  const real_t one = 1.0;
  const real_t half = 0.5;

  // This is spq in the paper.
  const real_t stddev = std::sqrt(n * p * (1 - p));

  // Other coefficients for Transformed Rejection sampling.
  const real_t b = static_cast<real_t>(1.15) + static_cast<real_t>(2.53) * stddev;
  const real_t a = static_cast<real_t>(-0.0873) + static_cast<real_t>(0.0248) * b + static_cast<real_t>(0.01) * p;
  const real_t c = n * p + half;
  const real_t v_r = static_cast<real_t>(0.92) - static_cast<real_t>(4.2) / b;
  const real_t r = p / (1 - p);

  const real_t alpha = (static_cast<real_t>(2.83) + static_cast<real_t>(5.1) / b) * stddev;
  const real_t m = std::floor((n + 1) * p);

  real_t draw;
  while (true) {
    real_t u = dust::unif_rand<real_t, real_t>(rng_state);
    real_t v = dust::unif_rand<real_t, real_t>(rng_state);
    u -= half;
    real_t us = half - std::fabs(u);
    real_t k = std::floor((2 * a / us + b) * u + c);

    // Region for which the box is tight, and we
    // can return our calculated value This should happen
    // 0.86 * v_r times. In the limit as n * p is large,
    // the acceptance rate converges to ~79% (and in the lower
    // regime it is ~24%).
    if (us >= static_cast<real_t>(0.07) && v <= v_r) {
      draw = k;
      break;
    }
    // Reject non-sensical answers.
    if (k < 0 || k > n) {
      continue;
    }

    // This deviates from Hormann's BRTS algorithm, as there is a log missing.
    // For all (u, v) pairs outside of the bounding box, this calculates the
    // transformed-reject ratio.
    v = std::log(v * alpha / (a / (us * us) + b));
    real_t upperbound =
      ((m + half) * std::log((m + 1) / (r * (n - m + 1))) +
       (n + one) * std::log((n - m + 1) / (n - k + 1)) +
       (k + half) * std::log(r * (n - k + 1) / (k + 1)) +
       stirling_approx_tail(m) + stirling_approx_tail(n - m) -
       stirling_approx_tail(k) - stirling_approx_tail(n - k));
    if (v <= upperbound) {
      draw = k;
      break;
    }
  }
  return draw;
}

template <typename real_t>
HOSTDEVICE int rbinom(rng_state_t<real_t>& rng_state, int n,
                      typename rng_state_t<real_t>::real_t p) {
  int draw;

  // Early exit:
  if (n == 0 || p == 0) {
    draw = 0;
  } else if (p == 1) {
    draw = n;
  } else {
    // TODO: Should control for this too, but not really clear what we
    // need to do to safely deal.
    /*
      if (n < 0 || p < 0 || p > 1) {
      return NaN;
      }
    */

    real_t q = p;
    if (p > static_cast<real_t>(0.5)) {
      q = 1 - q;
    }

    if (n * q >= 10) {
      draw = static_cast<int>(btrs(rng_state, n, q));
    } else {
      draw = static_cast<int>(binomial_inversion(rng_state, n, q));
    }

    if (p > static_cast<real_t>(0.5)) {
      draw = n - draw;
    }
  }

#ifdef __CUDA_ARCH__
  __syncwarp();
#endif

  return draw;
}

}
}

#endif
