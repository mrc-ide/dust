#ifndef DUST_DISTR_BINOMIAL_HPP
#define DUST_DISTR_BINOMIAL_HPP

#include <cmath>

namespace dust {
namespace distr {

template <typename T>
__device__
float binomial_inversion(rng_state_t<T>& rng_state,
                          float n, float prob) {
  float geom_sum = 0;
  float num_geom = 0;

  while (true) {
    float r = device_unif_rand(rng_state);
    float geom = ceil(log(r) / log1p(-prob));
    geom_sum += geom;
    if (geom_sum > n) {
      break;
    }
    ++num_geom;
  }
  //__syncwarp();
  return num_geom;
}

__device__
inline float stirling_approx_tail(float k) {
  static float kTailValues[] = {0.0810614667953272,  0.0413406959554092,
                                 0.0276779256849983,  0.02079067210376509,
                                 0.0166446911898211,  0.0138761288230707,
                                 0.0118967099458917,  0.0104112652619720,
                                 0.00925546218271273, 0.00833056343336287};
  float tail;
  if (k <= 9) {
    tail = kTailValues[static_cast<int>(k)];
  } else {
    float kp1sq = (k + 1) * (k + 1);
    tail = (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1);
  }
  //__syncwarp();
  return tail;
}

// https://www.tandfonline.com/doi/abs/10.1080/00949659308811496
template <typename T>
__device__
inline float btrs(rng_state_t<T>& rng_state, float n, float p) {
  // This is spq in the paper.
  const float stddev = sqrt(n * p * (1 - p));

  // Other coefficients for Transformed Rejection sampling.
  const float b = 1.15 + 2.53 * stddev;
  const float a = -0.0873 + 0.0248 * b + 0.01 * p;
  const float c = n * p + 0.5;
  const float v_r = 0.92 - 4.2 / b;
  const float r = p / (1 - p);

  const float alpha = (2.83 + 5.1 / b) * stddev;
  const float m = floor((n + 1) * p);

  float draw;
  while (true) {
    float u = device_unif_rand(rng_state);
    float v = device_unif_rand(rng_state);
    u = u - 0.5;
    float us = 0.5 - fabs(u);
    float k = floor((2 * a / us + b) * u + c);

    // Region for which the box is tight, and we
    // can return our calculated value This should happen
    // 0.86 * v_r times. In the limit as n * p is large,
    // the acceptance rate converges to ~79% (and in the lower
    // regime it is ~24%).
    if (us >= 0.07 && v <= v_r) {
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
    v = log(v * alpha / (a / (us * us) + b));
    float upperbound =
      ((m + 0.5) * log((m + 1) / (r * (n - m + 1))) +
       (n + 1) * log((n - m + 1) / (n - k + 1)) +
       (k + 0.5) * log(r * (n - k + 1) / (k + 1)) +
       stirling_approx_tail(m) + stirling_approx_tail(n - m) -
       stirling_approx_tail(k) - stirling_approx_tail(n - k));
    if (v <= upperbound) {
      draw = k;
      break;
    }
  }
  //__syncwarp();
  return draw;
}

template <typename real_t>
__device__
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

  //if (n * q >= 10) {
  draw = static_cast<int>(btrs(rng_state, n, q));
  //} else {
  //draw = static_cast<int>(binomial_inversion(rng_state, n, q));
  //}
  //__syncwarp();

  if (p > 0.5) {
    draw = n - draw;
  }

  return draw;
}

}
}

#endif
