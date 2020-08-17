#ifndef DUST_DISTR_BINOMIAL_HPP
#define DUST_DISTR_BINOMIAL_HPP

#include <cmath>

namespace dust {
namespace distr {

// Faster version of pow(x, n) for integer 'n' by using
// "exponentiation by squaring"
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
template <typename T>
T fast_pow(T x, int n) {
  T pow = 1.0;
  if (n != 0) {
    for(;;) {
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
template <typename T>
T binomial_inversion(rng_state_t<T>& rng_state, int n, T p) {
  T u = dust::unif_rand<T>(rng_state);

  // This is equivalent to qbinom(u, n, p)
  const T q = 1 - p;
  const T r = p / q;
  const T g = r * (n + 1);
  const T f = fast_pow(q, n);
  int k = 0;
  for (;;) {
    if (u < f) {
      break;
    }
    u -= f;
    k++;
    f *= (g / k - r);
  }

  return k;
}

inline double stirling_approx_tail(double k) {
  static double kTailValues[] = {0.0810614667953272,  0.0413406959554092,
                                 0.0276779256849983,  0.02079067210376509,
                                 0.0166446911898211,  0.0138761288230707,
                                 0.0118967099458917,  0.0104112652619720,
                                 0.00925546218271273, 0.00833056343336287};
  if (k <= 9) {
    return kTailValues[static_cast<int>(k)];
  }
  double kp1sq = (k + 1) * (k + 1);
  return (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1);
}

// https://www.tandfonline.com/doi/abs/10.1080/00949659308811496
template <typename T>
inline double btrs(rng_state_t<T>& rng_state, double n, double p) {
  // This is spq in the paper.
  const double stddev = std::sqrt(n * p * (1 - p));

  // Other coefficients for Transformed Rejection sampling.
  const double b = 1.15 + 2.53 * stddev;
  const double a = -0.0873 + 0.0248 * b + 0.01 * p;
  const double c = n * p + 0.5;
  const double v_r = 0.92 - 4.2 / b;
  const double r = p / (1 - p);

  const double alpha = (2.83 + 5.1 / b) * stddev;
  const double m = std::floor((n + 1) * p);

  while (true) {
    double u = dust::unif_rand<T, double>(rng_state);
    double v = dust::unif_rand<T, double>(rng_state);
    u = u - 0.5;
    double us = 0.5 - std::fabs(u);
    double k = std::floor((2 * a / us + b) * u + c);

    // Region for which the box is tight, and we
    // can return our calculated value This should happen
    // 0.86 * v_r times. In the limit as n * p is large,
    // the acceptance rate converges to ~79% (and in the lower
    // regime it is ~24%).
    if (us >= 0.07 && v <= v_r) {
      return k;
    }
    // Reject non-sensical answers.
    if (k < 0 || k > n) {
      continue;
    }

    // This deviates from Hormann's BRTS algorithm, as there is a log missing.
    // For all (u, v) pairs outside of the bounding box, this calculates the
    // transformed-reject ratio.
    v = std::log(v * alpha / (a / (us * us) + b));
    double upperbound =
      ((m + 0.5) * std::log((m + 1) / (r * (n - m + 1))) +
       (n + 1) * std::log((n - m + 1) / (n - k + 1)) +
       (k + 0.5) * std::log(r * (n - k + 1) / (k + 1)) +
       stirling_approx_tail(m) + stirling_approx_tail(n - m) -
       stirling_approx_tail(k) - stirling_approx_tail(n - k));
    if (v <= upperbound) {
      return k;
    }
  }
}

template <typename real_t>
int rbinom(rng_state_t<real_t>& rng_state, int n,
           typename rng_state_t<real_t>::real_t p) {
  int draw;

  // Early exit:
  if (n == 0 || p == 0) {
    return 0;
  }
  if (p == 1) {
    return n;
  }

  // TODO: Should control for this too, but not really clear what we
  // need to do to safely deal.
  /*
    if (n < 0 || p < 0 || p > 1) {
    return NaN;
    }
  */

  real_t q = p;
  if (p > 0.5) {
    q = 1 - q;
  }

  if (n * q >= 10) {
    draw = static_cast<int>(btrs(rng_state, n, q));
  } else {
    draw = static_cast<int>(binomial_inversion(rng_state, n, q));
  }

  if (p > 0.5) {
    draw = n - draw;
  }

  return draw;
}

}
}

#endif
