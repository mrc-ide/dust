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
  static double kTailValues[] = {0.081349206349206338,
                                  0.041344246031746031,
                                  0.02767816317199033,
                                  0.020790705605158728,
                                  0.01664469841269841,
                                  0.013876130870729635,
                                  0.01189671064871874,
                                  0.010411265539744544,
                                  0.0092554623050482409,
                                  0.0083305634920634915,
                                  0.0075736755181465241,
                                  0.0069428401236609179,
                                  0.006408994197411969,
                                  0.0059513701183652682,
                                  0.0055547335554249127,
                                  0.0052076559218149337,
                                  0.0049013959498782969,
                                  0.0046291537503020761,
                                  0.0043855602498956417,
                                  0.0041663196924603173,
                                  0.0039679542189702934,
                                  0.0037876180686823744,
                                  0.003622960224857451,
                                  0.003472021383108231,
                                  0.0033331556368253966,
                                  0.0032049702281289937,
                                  0.0030862786826655712,
                                  0.0029760639835944439,
                                  0.0028734493623869153,
                                  0.0027776749297798676,
                                  0.0026880788285527464,
                                  0.0026040819192689559,
                                  0.0025251752497707329,
                                  0.0024509097354494373,
                                  0.0023808876082433528,
                                  0.0023147552905222711,
                                  0.0022521974243437872,
                                  0.0021929318432930313,
                                  0.0021367053177568335,
                                  0.0020832899383060517,
                                  0.0020324800282596842,
                                  0.0019840894972483754,
                                  0.0019379495639979874,
                                  0.0018939067896024974,
                                  0.0018518213729947714,
                                  0.0018115656687209955,
                                  0.0017730228939140278,
                                  0.0017360859968776107,
                                  0.0017006566641969388,
                                  0.0016666444469841269,
                                  0.0016339659899085206,
                                  0.0016025443491768992,
                                  0.0015723083877166662,
                                  0.0015431922375552078,
                                  0.0015151348208439935,
                                  0.0014880794221974818,
                                  0.0014619733060455718,
                                  0.0014367673735673884,
                                  0.0014124158545106879,
                                  0.0013888760298272256,
                                  0.0013661079815880852,
                                  0.0013440743670992095,
                                  0.0013227402145284072,
                                  0.0013020727376911847,
                                  0.0012820411679322781,
                                  0.0012626166012898244,
                                  0.0012437718593455796,
                                  0.0012254813623524094,
                                  0.0012077210133936073,
                                  0.0011904680924709188,
                                  0.0011737011595424412,
                                  0.0011573999656403207,
                                  0.0011415453712935067,
                                  0.0011261192715645869,
                                  0.0011111045270834148,
                                  0.0010964849005252234,
                                  0.00108224499803829,
                                  0.0010683702151769901,
                                  0.0010548466869410387,
                                  0.0010416612415616474,
                                  0.0010288013577108034,
                                  0.0010162551248414635,
                                  0.001004011206394621,
                                  0.00099205880563434882,
                                  0.00098038763389440798,
                                  0.00096898788104013422,
                                  0.00095785018796737094,
                                  0.00094696562097641788,
                                  0.00093632564787352534,
                                  0.00092592211566557448,
                                  0.00091574722972539492,
                                  0.00090579353431582781,
                                  0.00089605389437026628,
                                  0.00088652147843610728,
                                  0.00087718974269543044,
                                  0.00086805241598436004,
                                  0.00085910348573903868,
                                  0.00085033718480203083,
                                  0.0008417479790283187,
                                  0.00083333055563492063,
                                  0.00082507981224259387,
                                  0.00081699084656213059,
                                  0.00080905894668143526,
                                  0.00080127958191295392,
                                  0.00079364839416409339,
                                  0.00078616118979609861,
                                  0.00077881393193943074,
                                  0.00077160273323606109,
                                  0.0007645238489812626,
                                  0.00075757367063947895,
                                  0.0007507487197106797,
                                  0.00074404564192529789,
                                  0.00073746120174739431,
                                  0.00073099227716712505,
                                  0.00072463585476489976,
                                  0.00071838902503083921,
                                  0.00071224897792425467,
                                  0.0007062129986589149,
                                  0.00070027846370081422,
                                  0.00069444283696605145,
                                  0.00068870366620723915,
                                  0.00068305857957762868,
                                  0.00067750528236283462,
                                  0.00067204155387070111,
                                  0.00066666524447045071,
                                  0.00066137427277282511,
                                  0.00065616662294344468,
                                  0.00065104034214210325,
                                  0.00064599353808116348};
  if (k <= 128) {
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

template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

template <typename real_t>
real_t recursive_f(real_t f, real_t nr, real_t n, real_t r, int k, int i) {
  if (i == k) {
    return f;
  } else {
    f = f * (nr / i - r);
    return(recursive_f(f, nr, n, r, i++));
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
      u = dust::distr::runif<T, double>(rng_state, -0.5, 0.5);
    } else {
      u = v / v_r - 0.93;
      u = sign(u) * 0.5 - u;
      v = dust::distr::runif<T, double>(rng_state, 0, v_r);
    }

    // Step 3.0
    double us = 0.5 - std::fabs(u);
    double k = std::floor((2 * a / us + b) * u + c);
    if (k < 0 || k > n) {
      continue;
    }
    v = (v * alpha / (a / (us * us) + b); // TF code suggests this should be logged
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
      double nk = n - k + 1;
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
