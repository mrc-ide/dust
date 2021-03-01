#ifndef DUST_DISTR_BINOMIAL_HPP
#define DUST_DISTR_BINOMIAL_HPP

#include <cmath>
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

// Binomial random numbers via inversion (for low np only!). Draw a
// random number from U(0, 1) and find the 'n' up the distribution
// (given p) that corresponds to this
__nv_exec_check_disable__
template <typename real_t>
real_t HOSTDEVICE binomial_inversion(rng_state_t<real_t>& rng_state, int n, real_t p) {
  real_t u = dust::unif_rand<real_t>(rng_state);

  // This is equivalent to qbinom(u, n, p)
  const real_t q = 1 - p;
  const real_t r = p / q;
  const real_t g = r * (n + 1);
  real_t f = fast_pow(q, n);
  int k = 0;
#ifdef __NVCC__
  // Assumes float for now - use DBL_EPSILON if real_t == double
  const real_t epsilon = FLT_EPSILON;
#else
  constexpr real_t epsilon = std::numeric_limits<real_t>::epsilon();
#endif
  // while (u >= f) { // TODO: fix this
  while (u - f >= 1E-6 && f > epsilon) {
    u -= f;
    k++;
    f *= (g / k - r);
  }

  return k;
}

template <typename real_t>
HOSTDEVICE real_t stirling_approx_tail(real_t k) {
  const real_t one = 1;
  real_t tail;
  if (k <= 127) {
#ifdef __CUDA_ARCH__
    tail = cudakTailValues[static_cast<int>(k)];
#else
    tail = kTailValues[static_cast<int>(k)];
#endif
  } else {
    real_t kp1sq = (k + 1) * (k + 1);
    tail = (one / 12 - (one / 360 - one / 1260 / kp1sq) / kp1sq) / (k + 1);
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
  const real_t b = static_cast<real_t>(1.15f) + static_cast<real_t>(2.53f) * stddev;
  const real_t a = static_cast<real_t>(-0.0873f) + static_cast<real_t>(0.0248f) * b + static_cast<real_t>(0.01f) * p;
  const real_t c = n * p + static_cast<real_t>(0.5f);
  const real_t v_r = static_cast<real_t>(0.92f) - static_cast<real_t>(4.2f) / b;
  const real_t r = p / (1 - p);

  const real_t alpha = (static_cast<real_t>(2.83f) + static_cast<real_t>(5.1f) / b) * stddev;
  const real_t m = std::floor((n + 1) * p);

  real_t draw;
  while (true) {
    real_t u = dust::unif_rand<real_t, real_t>(rng_state);
    real_t v = dust::unif_rand<real_t, real_t>(rng_state);
    u = u - static_cast<real_t>(0.5f);
    real_t us = static_cast<real_t>(0.5f) - std::fabs(u);
    real_t k = std::floor((2 * a / us + b) * u + c);

    // Region for which the box is tight, and we
    // can return our calculated value This should happen
    // 0.86 * v_r times. In the limit as n * p is large,
    // the acceptance rate converges to ~79% (and in the lower
    // regime it is ~24%).
    if (us >= static_cast<real_t>(0.07f) && v <= v_r) {
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
    return n;
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
