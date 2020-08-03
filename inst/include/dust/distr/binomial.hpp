#ifndef DUST_DISTR_BINOMIAL_HPP
#define DUST_DISTR_BINOMIAL_HPP

#include <cmath>

namespace dust {
namespace distr {

template <typename T>
double binomial_inversion(rng_state_t<T>& rng_state,
                          double n, double prob) {
  double geom_sum = 0;
  double num_geom = 0;

  while (true) {
    double r = dust::unif_rand<T, double>(rng_state);
    double geom = std::ceil(std::log(r) / std::log1p(-prob));
    geom_sum += geom;
    if (geom_sum > n) {
      break;
    }
    ++num_geom;
  }
  return num_geom;
}

inline double stirling_approx_tail(double k) {
  static double kTailValues[] = {0.08106146679532733,
                                  0.041340695955409457,
                                  0.027677925684998161,
                                  0.020790672103765173,
                                  0.016644691189821259,
                                  0.013876128823071099,
                                  0.011896709945892425,
                                  0.010411265261970115,
                                  0.009255462182705898,
                                  0.0083305634333594725,
                                  0.0075736754879489609,
                                  0.0069428401072073598,
                                  0.006408994188007,
                                  0.0059513701127578145,
                                  0.0055547335519605667,
                                  0.0052076559196052585,
                                  0.0049013959484298653,
                                  0.0046291537493345913,
                                  0.0043855602492257617,
                                  0.0041663196919898837,
                                  0.0039679542186377148,
                                  0.0037876180684577321,
                                  0.0036229602246820036,
                                  0.0034720213829828594,
                                  0.0033331556367386383,
                                  0.0032049702280616543,
                                  0.0030862786826162392,
                                  0.0029760639835672009,
                                  0.0028734493623545632,
                                  0.0027776749297458991,
                                  0.0026880788285268409,
                                  0.0026040819192587605,
                                  0.0025251752497723601,
                                  0.0024509097354297182,
                                  0.0023808876082398456,
                                  0.0023147552905129487,
                                  0.0022521974243261411,
                                  0.0021929318432967193,
                                  0.0021367053177385742,
                                  0.0020832899382980941,
                                  0.0020324800282622846,
                                  0.0019840894972418255,
                                  0.0019379495639952893,
                                  0.0018939067895757944,
                                  0.0018518213729947774,
                                  0.0018115656687029968,
                                  0.0017730228939285553,
                                  0.0017360859968675868,
                                  0.0017006566641839527,
                                  0.0016666444469990438,
                                  0.001633965989896069,
                                  0.0016025443491400893,
                                  0.0015723083877219324,
                                  0.0015431922375341856,
                                  0.0015151348208348736,
                                  0.0014880794221880933,
                                  0.0014619733060499129,
                                  0.001436767373576231,
                                  0.0014124158545030241,
                                  0.0013888760298357283,
                                  0.0013661079815676658,
                                  0.0013440743670685151,
                                  0.0013227402145616907,
                                  0.0013020727377295316,
                                  0.001282041167968373,
                                  0.0012626166012807971,
                                  0.0012437718593503178,
                                  0.0012254813623826522,
                                  0.0012077210134293637,
                                  0.0011904680924885724,
                                  0.0011737011595869262,
                                  0.0011573999656775413,
                                  0.0011415453712970702,
                                  0.0011261192715892321,
                                  0.001111104527126372,
                                  0.0010964849005574706,
                                  0.0010822449980310012,
                                  0.0010683702151936814,
                                  0.0010548466869408912,
                                  0.0010416612416292992,
                                  0.0010288013577337551,
                                  0.0010162551247958618,
                                  0.0010040112064189088,
                                  0.00099205880559338766,
                                  0.0009803876338878581,
                                  0.00096898788103771949,
                                  0.00095785018794458665,
                                  0.00094696562098306458,
                                  0.00093632564789913886,
                                  0.00092592211569808569,
                                  0.00091574722972609379,
                                  0.00090579353434350196,
                                  0.00089605389439384453,
                                  0.00088652147843504281,
                                  0.00087718974270956096,
                                  0.00086805241602405658,
                                  0.00085910348576589968,
                                  0.0008503371848291863,
                                  0.00084174797910918642,
                                  0.00083333055562206937,
                                  0.00082507981221624505,
                                  0.00081699084654474063,
                                  0.00080905894668603651,
                                  0.00080127958193543236,
                                  0.00079364839422169098,
                                  0.00078616118980789906,
                                  0.00077881393195866622,
                                  0.00077160273326626339,
                                  0.00076452384899994286,
                                  0.00075757367056894509,
                                  0.00075074871966762657,
                                  0.00074404564190899691,
                                  0.00073746120165196771,
                                  0.00073099227711281856,
                                  0.00072463585468085512,
                                  0.00071838902505305668,
                                  0.0007122489779476382,
                                  0.00070621299857975828,
                                  0.0007002784636256365,
                                  0.00069444283690245356,
                                  0.00068870366612827638,
                                  0.00068305857956829641,
                                  0.0006775052823400074,
                                  0.00067204155374156471,
                                  0.00066666524440961439,
                                  0.00066137427273815774,
                                  0.00065616662294587513,
                                  0.00065104034212026818,
                                  0.00064599353800076642};
  if (k <= 128) {
    return kTailValues[static_cast<int>(k)];
  }
  double kp1sq = (k + 1) * (k + 1);
  return (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1);
}

template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

template <typename real_t>
real_t recursive_f(real_t f, real_t nr, real_t n, real_t r, int k, int i) {
  if (i == k) {
    return f;
  } else {
    i++;
    f = f * (nr / i - r);
    return(recursive_f(f, nr, n, r, k, i));
  }
}

// https://www.tandfonline.com/doi/abs/10.1080/00949659308811496
template <typename T>
inline double btrd(rng_state_t<T>& rng_state, double n, double p) {
  // This is spq in the paper.
  const double npq = n * p * (1 - p);
  const double stddev = std::sqrt(npq);

  // Other coefficients for Transformed Rejection sampling.
  const double b = 1.15 + 2.53 * stddev;
  const double a = -0.0873 + 0.0248 * b + 0.01 * p;
  const double c = n * p + 0.5;
  const double v_r = 0.92 - 4.2 / b;
  const double r = p / (1 - p);
  const double u_r_u_v = 0.86 * v_r;

  const double alpha = (2.83 + 5.1 / b) * stddev;
  const double m = std::floor((n + 1) * p);
  const double nr = (n + 1) * r;

  while (true) {
    // Step 1
    double v = dust::unif_rand<T, double>(rng_state);
    if (v <= u_r_u_v) {
      double u = v / v_r - 0.43;
      return(std::floor(((2 * a) / (0.5 - std::fabs(u)) + b ) * u + c));
    }

    // Step 2
    double u;
    if (v >= v_r) {
      u = dust::distr::runif<double>(rng_state, -0.5, 0.5);
    } else {
      u = v / v_r - 0.93;
      u = sign(u) * 0.5 - u;
      v = dust::distr::runif<double>(rng_state, 0, v_r);
    }

    // Step 3.0
    double us = 0.5 - std::fabs(u);
    double k = std::floor((2 * a / us + b) * u + c);
    if (k < 0 || k > n) {
      continue;
    }
    v = v * alpha / (a / (us * us) + b); // TF code suggests this should be logged
    double km = std::fabs(k - m);

    if (km <= 15) {
      // Step 3.1
      double f = 1;
      if (m < k) {
        f = recursive_f<double>(f, nr, n, r, k, m);
      } else {
        v = recursive_f<double>(v, nr, n, r, m, k);
      }

      if (v <= f) {
        return k;
      }
    } else {
      // Step 3.2
      v = std::log(v);
      double rho = (km / npq) * (((km / 3 + 0.625) * km + 1/6)/ npq + 0.5);
      double t = -km * km / (2 * npq);
      if (v < t - rho) {
        return k;
      } else if (v > t + rho) {
        continue;
      }

      // Steps 3.3 and 3.4
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
    draw = static_cast<int>(btrd(rng_state, n, q));
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
