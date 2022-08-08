#ifndef DUST_RANDOM_HYPERGEOMETRIC_HPP
#define DUST_RANDOM_HYPERGEOMETRIC_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "dust/random/generator.hpp"
#include "dust/random/numeric.hpp"

// Implementation follows Kachitvichyanukul & Schmeiser (1985)
// https://www.tandfonline.com/doi/abs/10.1080/00949658508810839
// and follows the Rust implementation
// https://docs.rs/rand_distr/latest/src/rand_distr/hypergeometric.rs
// but adapted to fit our needs.
namespace dust {
namespace random {
namespace {

inline void hypergeometric_validate(int n1, int n2, int n, int k) {
  if (n1 < 0 || n2 < 0 || k < 0 || k > n) {
    char buffer[256];
    snprintf(buffer, 256,
             "Invalid call to hypergeometric with n1 = %d, n2 = %d, k = %d",
             n1, n2, k);
    dust::utils::fatal_error(buffer);
  }
}

template <typename real_type>
using h2pe_sample_result = std::pair<real_type, real_type>;
template <typename real_type>
using h2pe_test_result = std::pair<bool, real_type>;

template <typename real_type, typename rng_state_type>
int hypergeometric_hin(rng_state_type& rng_state, int n1, int n2, int n, int k);
template <typename real_type, typename rng_state_type>
int hypergeometric_h2pe(rng_state_type& rng_state, int n1, int n2, int n, int k, int m);
template <typename real_type, typename rng_state_type>
h2pe_sample_result<real_type>
h2pe_sample(rng_state_type& rng_state, int n1, int n2, int k,
            real_type p1, real_type p2, real_type p3,
            real_type x_l, real_type x_r, real_type lambda_l,
            real_type lambda_r);

template <typename real_type>
h2pe_test_result<real_type> h2pe_test_recursive(int n1, int n2, int k, int m,
                                                real_type y, real_type v);
template <typename real_type>
h2pe_test_result<real_type> h2pe_test_squeeze(int n1, int n2, int k, int m,
                                              real_type y, real_type v,
                                              real_type a);
template <typename real_type>
real_type fraction_of_products_of_factorials(int a, int b, int c, int d);
template <typename T>
T quad(T x);

// Generate hypergeometric random number via inversion (HIN), p 130 of
// reference.
template <typename real_type, typename rng_state_type>
int hypergeometric_hin(rng_state_type& rng_state, int n1, int n2, int n, int k) {
  real_type p;
  int x;
  if (k < n2) {
    p = fraction_of_products_of_factorials<real_type>(n2, n - k, n, n2 - k);
    x = 0;
  } else {
    p = fraction_of_products_of_factorials<real_type>(n1, k, n, k - n2);
    x = (k - n2);
  }

  real_type u = random_real<real_type>(rng_state);
  while (u > p && x < k) {
    // Comment in the Rust version:
    // > the paper erroneously uses `until n < p`, which doesn't make any sense
    u -= p;
    p *= ((n1 - x) * (k - x)) / ((x + 1) * (n2 - k + 1 + x));
    ++x;
  }
  return x;
}

template <typename real_type, typename rng_state_type>
int hypergeometric_h2pe(rng_state_type& rng_state, int n1, int n2, int n, int k, int m) {
  // Step 0 set up constants
  const real_type a = utils::lfactorial<real_type>(m) +
    utils::lfactorial<real_type>(n1 - m) +
    utils::lfactorial<real_type>(k - m) +
    utils::lfactorial<real_type>((n2 - k)  + m);

  const real_type d_numerator =
    (real_type)(n - k) * k * (real_type)n1  * (real_type)n2;
  const real_type d_denominator =
    (real_type)(n - 1) * (real_type)n * (real_type)n;
  const real_type d =
    std::floor(1.5 * std::sqrt(d_numerator / d_denominator)) + 0.5;

  // I think that here x_l and x_r are really integers and therefore
  // some of the +1.0s become +1s
  const real_type x_l = m - d + 0.5;
  const real_type x_r = m + d + 0.5;

  const real_type k_l = std::exp(a -
                                 utils::lfactorial<real_type>(x_l) -
                                 utils::lfactorial<real_type>(n1 - x_l) -
                                 utils::lfactorial<real_type>(k - x_l) -
                                 utils::lfactorial<real_type>((n2 - k)  + x_l));
  const real_type k_r = std::exp(a -
                                 utils::lfactorial<real_type>(x_r - 1.0) -
                                 utils::lfactorial<real_type>(n1 - x_r + 1.0) -
                                 utils::lfactorial<real_type>(k - x_r + 1.0) -
                                 utils::lfactorial<real_type>((n2 - k)  + x_r - 1.0));

  const real_type ll_numerator = x_l * ((n2 - k) + x_l);
  const real_type ll_denominator = (n1 - x_l + 1.0) * (k - x_l + 1.0);
  const real_type lambda_l = -std::log(ll_numerator / ll_denominator);

  const real_type lr_numerator = (n1 - x_r + 1.0) * (k - x_r + 1.0);
  const real_type lr_denominator = x_r * ((n2 - k) + x_r);
  const real_type lambda_r = -std::log(lr_numerator / lr_denominator);

  // Comment in the Rust version:
  // > the paper literally gives `p2 + kL/lambdaL` where it (probably)
  // > should have been `p2 = p1 + kL/lambdaL` another print error?!
  const real_type p1 = 2.0 * d;
  const real_type p2 = p1 + k_l / lambda_l;
  const real_type p3 = p2 + k_r / lambda_r;

  real_type x; // final result
  for (;;) {
    const auto vy = h2pe_sample(rng_state, n1, n2, k, p1, p2, p3,
                                x_l, x_r, lambda_l, lambda_r);
    const real_type v = vy.first;
    const real_type y = vy.second;

    const auto result = (m < 100.0 || y <= 50.0) ?
      h2pe_test_recursive(n1, n2, k, m, y, v) :
      h2pe_test_squeeze(n1, n2, k, m, y, v, a);
    if (result.first) {
      x = result.second;
      break;
    }
  }
  return x;
}

// Steps 1-3
template <typename real_type, typename rng_state_type>
h2pe_sample_result<real_type>
h2pe_sample(rng_state_type& rng_state, int n1, int n2, int k,
            real_type p1, real_type p2, real_type p3,
            real_type x_l, real_type x_r, real_type lambda_l,
            real_type lambda_r) {
  real_type y; // will become x on exit
  real_type v;
  // We get a false-negative coverage report on this line, which is
  // very definitely hit. I presume the compiler is converting it into
  // something wildly different (though not on the similar case above)?
  for (;;) { // #nocov
    // U(0, p3) for region selection
    const real_type u = random_real<real_type>(rng_state) * p3;
    // U(0, 1) for accept/reject
    v = random_real<real_type>(rng_state);
    if (u <= p1) {
      // Region 1, central bell (step 1)
      y = std::floor(x_l + u); // could make x and y int
      break;
    } else if (u <= p2) {
      // Region 2, left exponential tail (step 2)
      y = std::floor(x_l + std::log(v) / lambda_l);
      if (y >= std::max(0, k - n2)) {
        v *= (u - p1) * lambda_l;
        break;
      }
    } else {
      // Region 3, right exponential tail (step 3)
      y = std::floor(x_r - std::log(v) / lambda_r);
      if (y <= std::min(n1, k)) {
        v *= (u - p2) * lambda_r;
        break;
      }
    }
  }
  return h2pe_sample_result<real_type>{v, y};
}

// Step 4.1: Evaluate f(y) via recursive relationship
template <typename real_type>
h2pe_test_result<real_type> h2pe_test_recursive(int n1, int n2, int k, int m,
                                                real_type y, real_type v) {
  real_type f = 1;
  if (m < y) {
    for (int i = m + 1; i <= y; ++i) {
      f *= (n1 - i + 1) * (k - i + 1) / (real_type)((n2 - k + i) *  i);
    }
  } else if (m > y) {
    for (int i = y + 1; i <= m; ++i) {
      // The rust version does not have the + 1 on both parts of
      // the denominator, added in the R version and a fixable
      // bug in the Rust version.
      f *= i * (n2 - k + i) / (real_type)((n1 - i + 1) * (k - i + 1));
    }
  }
  const bool accept = v <= f;
  return h2pe_test_result<real_type>{accept, y};
}

// Step 4.2: Squeezing
template <typename real_type>
h2pe_test_result<real_type> h2pe_test_squeeze(int n1, int n2, int k, int m,
                                              real_type y, real_type v,
                                              real_type a) {
  const real_type y1 = y + 1.0;
  const real_type ym = y - m;
  const real_type yn = n1 - y + 1.0;
  const real_type yk = k - y + 1.0;
  const real_type nk = n2 - k + y1;
  const real_type r = -ym / y1;
  const real_type s = ym / yn;
  const real_type t = ym / yk;
  const real_type e = -ym / nk;
  const real_type g = yn * yk / (y1 * nk) - 1.0;
  const real_type dg = g < 0.0 ? 1 + g : 1;

  const real_type gu = g * (1.0 + g * (-0.5 + g / 3.0));
  const real_type gl = gu - quad(g) / (4.0 * dg);
  const real_type xm = m + 0.5;
  const real_type xn = n1 - m + 0.5;
  const real_type xk = k - m + 0.5;
  const real_type nm = n2 - k + xm;
  const real_type ub =
    xm * r * (1.0 + r * (-0.5 + r / 3.0)) +
    xn * s * (1.0 + s * (-0.5 + s / 3.0)) +
    xk * t * (1.0 + t * (-0.5 + t / 3.0)) +
    nm * e * (1.0 + e * (-0.5 + e / 3.0)) +
    y * gu - m * gl + 0.0034;
  const real_type av = std::log(v);
  if (av > ub) {
    return h2pe_test_result<real_type>{false, 0};
  }

  const real_type dr = r < 0 ? xm * quad(r) / (1.0 + r) : xm * quad(r);
  const real_type ds = s < 0 ? xn * quad(s) / (1.0 + s) : xn * quad(s);
  const real_type dt = t < 0 ? xk * quad(t) / (1.0 + t) : xk * quad(t);
  const real_type de = e < 0 ? nm * quad(e) / (1.0 + e) : nm * quad(e);

  if (av < ub - 0.25 * (dr + ds + dt + de) + (y + m) * (gl - gu) - 0.0078) {
    return h2pe_test_result<real_type>{true, y};
  }

  // Step 4.3: Final Acceptance/Rejection Test
  const real_type av_critical = a -
    utils::lfactorial<real_type>(y) -
    utils::lfactorial<real_type>(n1 - y) -
    utils::lfactorial<real_type>(k - y) - 
    utils::lfactorial<real_type>((n2 - k) + y);
  if (log(v) <= av_critical) {
    return h2pe_test_result<real_type>{true, y};
  }

  return h2pe_test_result<real_type>{false, y};
}

// This is subject to overflow for very large inputs with very small
// mode (on the order of n = 1e9, m < 10)
// https://bugs.r-project.org/show_bug.cgi?id=16489
template <typename real_type>
real_type fraction_of_products_of_factorials(int a, int b, int c, int d) {
  return std::exp(utils::lfactorial<real_type>(a) +
                  utils::lfactorial<real_type>(b) -
                  utils::lfactorial<real_type>(c) -
                  utils::lfactorial<real_type>(d));
}

template <typename T>
T quad(T x) {
  return x * x * x * x;
}

}

// NOTE: we return a real, not an int, as with deterministic mode this
// will not necessarily be an integer
template <typename real_type, typename rng_state_type>
real_type hypergeometric_stochastic(rng_state_type& rng_state, int n1, int n2, int k) {
  const int n = n1 + n2;
  hypergeometric_validate(n1, n2, n, k);

  int sign_x = 1;
  int offset_x = 0;

  if (n1 > n2) {
    sign_x = -1;
    offset_x = k;
    std::swap(n1, n2);
  }
  if (k > n / 2) {
    offset_x += n1 * sign_x;
    sign_x = -sign_x;
    k = n - k;
  }

  int x;
  // Same fast exits as for the binomial case, n == k case handled by
  // the transformation above.
  if (k == 0 || n1 == 0) {
    x = 0;
  } else {
    constexpr real_type hin_threshold = 10;
    const int m = std::floor((k + 1) * (n1 + 1) / (real_type)(n + 2));
    x = (m < hin_threshold) ?
      hypergeometric_hin<real_type>(rng_state, n1, n2, n, k) :
      hypergeometric_h2pe<real_type>(rng_state, n1, n2, n, k, m);
  }

  return offset_x + sign_x * x;
}

template <typename real_type>
real_type hypergeometric_deterministic(real_type n1, real_type n2, real_type k) {
  const real_type n = n1 + n2;
  hypergeometric_validate(static_cast<int>(n1), static_cast<int>(n2),
                          static_cast<int>(n), static_cast<int>(k));
  return n1 * k / n;
}

/// Draw random number from the hypergeometric distribution. This is
/// often descrribed as sampling `k` elements without replacement from
/// an urn with `n1` white balls and `n2` black balls, where the
/// outcome is the number of white balls retrieved.
///
/// @tparam real_type The underlying real number type, typically
/// `double` or `float`. A compile-time error will be thrown if you
/// attempt to use a non-floating point type (based on
/// `std::is_floating_point).
///
/// @tparam rng_state_type The random number state type
///
/// @param rng_state Reference to the random number state, will be
/// modified as a side-effect
///
/// @param n1 The number of white balls in the urn
///
/// @param n2 The number of black balls in the urn
///
/// @param k The number of draws
template <typename real_type, typename rng_state_type>
__host__ __device__
real_type hypergeometric(rng_state_type& rng_state, real_type n1, real_type n2, real_type k) {
  static_assert(std::is_floating_point<real_type>::value,
                "Only valid for floating-point types; use hypergeometric<real_type>()");
  // There are some issues around multiple returns and use of
  // std::pair that probably require some additional work to get this
  // behaving well on a GPU. Unlikely to be a lot of work, but better
  // not to assume that it does. Proper testing of the algorithm under
  // single precision would also be wise to prevent possible infinite
  // loops, that's easiest to do once we have some real use-cases.
#ifdef __CUDA_ARCH__
  static_assert("hypergeomeric() not implemented for GPU targets");
#endif

  if (rng_state.deterministic) {
    return hypergeometric_deterministic<real_type>(n1, n2, k);
  }
  // Avoid integer truncation (which a cast to int would cause) in
  // case of numerical error, instead taking the slightly lower but
  // more accurate round route. This means that `n - eps` becomes
  // `n` not `n - 1`.
  return hypergeometric_stochastic<real_type>(rng_state, std::round(n1), std::round(n2), std::round(k));
}

}
}

#endif
