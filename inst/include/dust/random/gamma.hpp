#ifndef DUST_RANDOM_GAMMA_HPP
#define DUST_RANDOM_GAMMA_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "dust/random/generator.hpp"
#include "dust/random/numeric.hpp"
#include "dust/random/exponential.hpp"
#include "dust/random/uniform.hpp"
#include "dust/random/normal.hpp"

// For 'shape >= 1' follows Ahrens, J. H. and Dieter, U. (1982). Generating gamma variates by
// a modified rejection technique. For '0 < shape < 1' uses Ahrens, J. H. and Dieter,
// U. (1974). Computer methods for sampling from gamma, beta, Poisson and binomial distributions.
// and follows the R implementation: https://github.com/wch/r-source/blob/trunk/src/nmath/rgamma.c
namespace dust {
namespace random {
namespace {

template <typename real_type>
void gamma_validate(real_type a, real_type b) {
  if (a < 0.0 || b < 0.0) {
    char buffer[256];
    snprintf(buffer, 256,
             "Invalid call to gamma with a = %g, b = %g",
             a, b);
    dust::utils::fatal_error(buffer);
  }
}

template <typename real_type, typename rng_state_type>
real_type gamma_large(rng_state_type& rng_state, real_type a) {
  real_type d = a - 1 / 3;
  real_type c = 1 / sqrt(9 * d);
  while(true) {
    real_type x = normal<real_type>(rng_state, 0, 1);
    real_type v_cbrt = 1.0 + c * x;
    if (v_cbrt <= 0.0) {
      continue;
    }
    real_type v = v_cbrt * v_cbrt * v_cbrt;
    real_type u = uniform<real_type>(rng_state, 0, 1);
    real_type x_sqr = x * x;
    if (u < 1.0 - 0.0331 * x_sqr * x_sqr ||
      log(u) < 0.5 * x_sqr + d * (1.0 - v + log(v))) {
      return d * v;
    }
  }
}

template <typename real_type, typename rng_state_type>
real_type gamma_small(rng_state_type& rng_state, real_type a) {
  real_type inv_shape = 1 / a;
  real_type u = uniform<real_type>(rng_state, 0, 1);
  return gamma_large(rng_state, a + 1.0) * pow(u, inv_shape);
}

template <typename real_type>
real_type gamma_deterministic(real_type a, real_type b) {
  return a * b;
}

/// Draw random number from the gamma distribution.
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
/// @param a Shape
///
/// @param b Scale
template <typename real_type, typename rng_state_type>
__host__ __device__
real_type gamma(rng_state_type& rng_state, real_type a, real_type b) {
  static_assert(std::is_floating_point<real_type>::value,
                "Only valid for floating-point types; use gamma<real_type>()");

  gamma_validate(a, b);

#ifdef __CUDA_ARCH__
  static_assert("gamma() not implemented for GPU targets");
#endif

  if (a == 0 || b == 0) {
    return 0;
  }

  if (rng_state.deterministic) {
    return gamma_deterministic<real_type>(a, b);
  }

  if (a < 1) {
    return gamma_small<real_type>(rng_state, a) * b;
  }

  if (a == 1) {
    return exponential(rng_state, 1 / b);
  }

  return gamma_large<real_type>(rng_state, a) * b;
}

}
}
}

#endif
