#ifndef DUST_RANDOM_BINOMIAL_HPP
#define DUST_RANDOM_BINOMIAL_HPP

#include <cmath>

#include "dust/random/gamma_table.hpp"
#include "dust/random/generator.hpp"

namespace dust {
namespace random {

// Faster version of pow(x, n) for integer 'n' by using
// "exponentiation by squaring"
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
template <typename real_type>
__host__ __device__
real_type fast_pow(real_type x, int n) {
  real_type pow = 1.0;
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
template <typename real_type>
__host__ __device__
real_type binomial_inversion_calc(real_type u, int n,
                                                      real_type p) {
  const real_type q = 1 - p;
  const real_type r = p / q;
  const real_type g = r * (n + 1);
  real_type f = fast_pow(q, n);
  int k = 0;

  real_type f_prev = f;
  while (u >= f) {
    u -= f;
    k++;
    f *= (g / k - r);
    if (f == f_prev || k > n) {
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
template <typename real_type, typename rng_state_type>
__host__ __device__
real_type binomial_inversion(rng_state_type& rng_state, int n, real_type p) {
  real_type k = -1;
  do {
    real_type u = random_real<real_type>(rng_state);
    k = binomial_inversion_calc(u, n, p);
  } while (k < 0);
  return k;
}

template <typename real_type>
__host__ __device__ real_type stirling_approx_tail(real_type k);

template <typename real_type>
__host__ __device__ inline real_type stirling_approx_tail_calc(real_type k) {
  const real_type one = 1;
  real_type kp1sq = (k + 1) * (k + 1);
  return (one / 12 - (one / 360 - one / 1260 / kp1sq) / kp1sq) / (k + 1);
}

template <>
__host__ __device__ inline float stirling_approx_tail(float k) {
  float tail;
  if (k <= k_tail_values_max_f) {
    tail = k_tail_values_f[static_cast<int>(k)];
  } else {
    // We've chosen our table length on the float case to never git
    // this branch; we'll come back and test this properly on dust
    // issue #191
    tail = stirling_approx_tail_calc(k); // #nocov
  }
  return tail;
}

template <>
__host__ __device__ inline double stirling_approx_tail(double k) {
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
template <typename real_type, typename rng_state_type>
inline __host__ __device__
real_type btrs(rng_state_type& rng_state, int n_int, real_type p) {
  const real_type n = static_cast<real_type>(n_int);
  const real_type one = 1.0;
  const real_type half = 0.5;

  // This is spq in the paper.
  const real_type stddev = std::sqrt(n * p * (1 - p));

  // Other coefficients for Transformed Rejection sampling.
  const real_type b = static_cast<real_type>(1.15) + static_cast<real_type>(2.53) * stddev;
  const real_type a = static_cast<real_type>(-0.0873) + static_cast<real_type>(0.0248) * b + static_cast<real_type>(0.01) * p;
  const real_type c = n * p + half;
  const real_type v_r = static_cast<real_type>(0.92) - static_cast<real_type>(4.2) / b;
  const real_type r = p / (1 - p);

  const real_type alpha = (static_cast<real_type>(2.83) +
                           static_cast<real_type>(5.1) / b) * stddev;
  const real_type m = std::floor((n + 1) * p);

  real_type draw;
  while (true) {
    real_type u = random_real<real_type>(rng_state);
    real_type v = random_real<real_type>(rng_state);
    u -= half;
    real_type us = half - std::fabs(u);
    real_type k = std::floor((2 * a / us + b) * u + c);

    // Region for which the box is tight, and we
    // can return our calculated value This should happen
    // 0.86 * v_r times. In the limit as n * p is large,
    // the acceptance rate converges to ~79% (and in the lower
    // regime it is ~24%).
    if (us >= static_cast<real_type>(0.07) && v <= v_r) {
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
    real_type upperbound =
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

template <typename real_type>
__host__ __device__
void binomial_validate(int n, real_type p) {
  if (n < 0 || p < 0 || p > 1) {
#ifdef __CUDA_ARCH__
    // This is unrecoverable
    printf("Invalid call to binomial with n = %d, p = %g, q = %g\n",
           n, p, 1 - p);
    __trap();
#else
    char buffer[256];
    snprintf(buffer, 256,
             "Invalid call to binomial with n = %d, p = %g, q = %g",
             n, p, 1 - p);
    throw std::runtime_error(buffer);
#endif
  }
}

template <typename real_type>
__host__ real_type binomial_deterministic(real_type n, real_type p) {
  if (n < 0) {
    if (n * n < std::numeric_limits<real_type>::epsilon()) {
      // Avoid small round-off errors here
      n = std::round(n);
    } else {
      char buffer[256];
      snprintf(buffer, 256, "Invalid call to binomial with n = %f", n);
      throw std::runtime_error(buffer);
    }
  }
  binomial_validate(static_cast<int>(n), p);
  return n * p;
}

// NOTE: we return a real, not an int, as with deterministic mode this
// will not necessarily be an integer
template <typename real_type, typename rng_state_type>
__host__ __device__
real_type binomial_stochastic(rng_state_type& rng_state, int n, real_type p) {
  binomial_validate(n, p);
  real_type draw;

  if (n == 0 || p == 0) {
    draw = 0;
  } else if (p == 1) {
    draw = n;
  } else {
    real_type q = p;
    if (p > static_cast<real_type>(0.5)) {
      q = 1 - q;
    }

    if (n * q >= 10) {
      draw = btrs(rng_state, n, q);
    } else {
      draw = binomial_inversion(rng_state, n, q);
    }

    if (p > static_cast<real_type>(0.5)) {
      draw = n - draw;
    }
  }

  SYNCWARP
  return draw;
}

template <typename real_type, typename rng_state_type>
__host__ __device__
real_type binomial(rng_state_type& rng_state, real_type n, real_type p) {
  static_assert(std::is_floating_point<real_type>::value,
                "Only valid for floating-point types; use binomial<real_type>()");
#ifndef __CUDA_ARCH__
  if (rng_state.deterministic) {
    return binomial_deterministic<real_type>(n, p);
  }
#endif
  // Avoid integer truncation (which a cast to int would cause) in
  // case of numerical error, instead taking the slightly lower but
  // more accurate round route. This means that `n - eps` becomes
  // `n` not `n - 1`.
  return binomial_stochastic<real_type>(rng_state, std::round(n), p);
}

}
}

#endif
