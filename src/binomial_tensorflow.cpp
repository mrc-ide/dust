#include <math.h> 

#include "rng.hpp"

double binomial_inversion(double count, double prob,
                          RNG * gen, size_t thread_idx) {
  double geom_sum = 0;
  int num_geom = 0;

  while (true) {
    double geom = ceil(log(gen->runif(thread_idx)) / log1p(-prob));
    geom_sum += geom;
    if (geom_sum > count) {
      break;
    }
    ++num_geom;
  }
  return num_geom;
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
inline double btrs(double count, double prob, 
                   RNG * gen, size_t thread_idx) {
  // This is spq in the paper.
  const double stddev = sqrt(count * prob * (1 - prob));

  // Other coefficients for Transformed Rejection sampling.
  const double b = 1.15 + 2.53 * stddev;
  const double a = -0.0873 + 0.0248 * b + 0.01 * prob;
  const double c = count * prob + 0.5;
  const double v_r = 0.92 - 4.2 / b;
  const double r = prob / (1 - prob);

  const double alpha = (2.83 + 5.1 / b) * stddev;
  const double m = floor((count + 1) * prob);

  while (true) {
    double u = gen->runif(thread_idx);
    double v = gen->runif(thread_idx);
    u = u - 0.5;
    double us = 0.5 - abs(u);
    double k = floor((2 * a / us + b) * u + c);

    // Region for which the box is tight, and we
    // can return our calculated value This should happen
    // 0.86 * v_r times. In the limit as n * p is large,
    // the acceptance rate converges to ~79% (and in the lower
    // regime it is ~24%).
    if (us >= 0.07 && v <= v_r) {
      return k;
    }
    // Reject non-sensical answers.
    if (k < 0 || k > count) {
      continue;
    }

    // This deviates from Hormann's BRTS algorithm, as there is a log missing.
    // For all (u, v) pairs outside of the bounding box, this calculates the
    // transformed-reject ratio.
    v = log(v * alpha / (a / (us * us) + b));
    double upperbound =
        ((m + 0.5) * log((m + 1) / (r * (count - m + 1))) +
         (count + 1) * log((count - m + 1) / (count - k + 1)) +
         (k + 0.5) * log(r * (count - k + 1) / (k + 1)) +
         stirling_approx_tail(m) + stirling_approx_tail(count - m) -
         stirling_approx_tail(k) - stirling_approx_tail(count - k));
    if (v <= upperbound) {
      return k;
    }
  }
}

template <class T>
T RNG::rbinom(const size_t thread_idx, double p, int n) {
    T draw;
    double q = p;
    if (q > 0.5) {
      q = 1 - q;
    }

    if (n * p >= 10) {
        // Uses 256 random numbers
        draw = static_cast<T>(btrs(n, q, this, thread_idx));
    } else {
        // Uses 42 random numbers
        draw = static_cast<T>(binomial_inversion(n, q, this, thread_idx));
    }

    if (p > 0.5) {
      draw = n - draw;
    }

    return(draw);
}

template int RNG::rbinom<int>(const size_t thread_idx, double p, int n); 
