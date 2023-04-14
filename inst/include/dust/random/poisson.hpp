#ifndef DUST_RANDOM_POISSON_HPP
#define DUST_RANDOM_POISSON_HPP

#include <cmath>

#include "dust/random/generator.hpp"
#include "dust/random/numeric.hpp"
#include "dust/random/math.hpp"

namespace dust {
namespace random {

__nv_exec_check_disable__
template <typename real_type>
__host__ __device__
void poisson_validate(real_type lambda) {
  if (!std::isfinite(lambda) || lambda < 0) {
    char buffer[256];
    snprintf(buffer, 256,
             "Invalid call to Poisson with lambda = %g",
             lambda);
    dust::utils::fatal_error(buffer);
  }
}

__nv_exec_check_disable__
template <typename real_type, typename rng_state_type>
__host__ __device__
real_type poisson_knuth(rng_state_type& rng_state, real_type lambda) {
  int x = 0;
  // Knuth's algorithm for generating Poisson random variates.
  // Given a Poisson process, the time between events is exponentially
  // distributed. If we have a Poisson process with rate lambda, then,
  // the time between events is distributed Exp(lambda). If X ~
  // Uniform(0, 1), then Y ~ Exp(lambda), where Y = -log(X) / lambda.
  // Thus to simulate a Poisson draw, we can draw X_i ~ Exp(lambda),
  // and N ~ Poisson(lambda), where N is the least number such that
  // \sum_i^N X_i > 1.
  const real_type exp_neg_rate = dust::math::exp(-lambda);

  real_type prod = 1;

  // Keep trying until we surpass e^(-rate). This will take
  // expected time proportional to rate.
  while (true) {
    real_type u = random_real<real_type>(rng_state);
    prod = prod * u;
    if (prod <= exp_neg_rate && x <= utils::integer_max()) {
      break;
    }
    x++;
  }
  return x;
}

__nv_exec_check_disable__
template <typename real_type, typename rng_state_type>
__host__ __device__
real_type poisson_hormann(rng_state_type& rng_state, real_type lambda) {
  // Transformed rejection due to Hormann.
  //
  // Given a CDF F(x), and G(x), a dominating distribution chosen such
  // that it is close to the inverse CDF F^-1(x), compute the following
  // steps:
  //
  // 1) Generate U and V, two independent random variates. Set U = U - 0.5
  // (this step isn't strictly necessary, but is done to make some
  // calculations symmetric and convenient. Henceforth, G is defined on
  // [-0.5, 0.5]).
  //
  // 2) If V <= alpha * F'(G(U)) * G'(U), return floor(G(U)), else return
  // to step 1. alpha is the acceptance probability of the rejection
  // algorithm.
  //
  // For more details on transformed rejection, see:
  // https://doi.org/10.1016/0167-6687(93)90997-4
  //
  // The dominating distribution in this case:
  //
  // G(u) = (2 * a / (2 - |u|) + b) * u + c

  int x = 0;
  const real_type log_rate = dust::math::log(lambda);

  // Constants used to define the dominating distribution. Names taken
  // from Hormann's paper. Constants were chosen to define the tightest
  // G(u) for the inverse Poisson CDF.
  const real_type b = static_cast<real_type>(0.931) +
    static_cast<real_type>(2.53) * dust::math::sqrt(lambda);
  const real_type a = static_cast<real_type>(-0.059) +
    static_cast<real_type>(0.02483) * b;

  // This is the inverse acceptance rate. At a minimum (when rate = 10),
  // this corresponds to ~75% acceptance. As the rate becomes larger, this
  // approaches ~89%.
  const real_type inv_alpha = static_cast<real_type>(1.1239) +
    static_cast<real_type>(1.1328) / (b - static_cast<real_type>(3.4));

  while (true) {
    real_type u = random_real<real_type>(rng_state);
    u -= static_cast<real_type>(0.5);
    real_type v = random_real<real_type>(rng_state);

    real_type u_shifted = static_cast<real_type>(0.5) - dust::math::abs(u);
    real_type k = floor((2 * a / u_shifted + b) * u + lambda + static_cast<real_type>(0.43));

    if (k > utils::integer_max()) {
      // retry in case of overflow.
      continue; // # nocov
    }

    // When alpha * f(G(U)) * G'(U) is close to 1, it is possible to
    // find a rectangle (-u_r, u_r) x (0, v_r) under the curve, such
    // that if v <= v_r and |u| <= u_r, then we can accept.
    // Here v_r = 0.9227 - 3.6224 / (b - 2) and u_r = 0.43.
    if (u_shifted >= static_cast<real_type>(0.07) &&
        v <= static_cast<real_type>(0.9277) - static_cast<real_type>(3.6224) / (b - 2)) {
      x = k;
      break;
    }

    if (k < 0 || (u_shifted < static_cast<real_type>(0.013) && v > u_shifted)) {
      continue;
    }

    // The expression below is equivalent to the computation of step 2)
    // in transformed rejection (v <= alpha * F'(G(u)) * G'(u)).
    real_type s = dust::math::log(v * inv_alpha / (a / (u_shifted * u_shifted) + b));
    real_type t = -lambda + k * log_rate -
      utils::lgamma(static_cast<real_type>(k + 1));
    if (s <= t) {
      x = k;
      break;
    }
  }
  return x;
}

__nv_exec_check_disable__
template <typename real_type, typename rng_state_type>
__host__ __device__
real_type poisson_cauchy(rng_state_type& rng_state, real_type lambda) {
  // The algorithm as in the dust rand_distr crate
  // https://rust-random.github.io/rand/src/rand_distr/poisson.rs.html
  // using the fat tails of a Cauchy distribution to do generate
  // poisson values via rejection sampling.
  //
  // This algorithm is much less efficient than hormann (consistently
  // about 5x slower) but much simpler. The algorithm of Ahrens and
  // Dieter 1980 ("Sampling from Binomial and Poisson Distributions",
  // Computing 25 193-208) is meant to be the fastest with a
  // constantly changing lambda, but is more complex to implement.
  real_type result = 0;
  const real_type log_lambda = dust::math::log<real_type>(lambda);
  const real_type sqrt_2lambda = dust::math::sqrt<real_type>(2 * lambda);
  const real_type magic_val = lambda * log_lambda - dust::math::lgamma<real_type>(1 + lambda);
  for (;;) {
    real_type comp_dev;
    for (;;) {
      comp_dev = cauchy<real_type>(rng_state, 0, 1);
      result = sqrt_2lambda * comp_dev + lambda;
      if (result >= 0) {
        break;
      }
    }
    result = dust::math::trunc<real_type>(result);
    const real_type check = static_cast<real_type>(0.9) *
      (1 + comp_dev * comp_dev) *
      dust::math::exp<real_type>(result * log_lambda - dust::math::lgamma<real_type>(1 + result) - magic_val);
    const real_type u = random_real<real_type>(rng_state);
    if (u <= check) {
      break;
    }
  }
  return result;
}

/// Draw a Poisson distributed random number given a mean
/// parameter. Generation is performed using either Knuth's algorithm
/// (small lambda) or Hormann's rejection sampling algorithm (medium
/// lambda), or rejection based on the Cauchy (large lambda)
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
/// @param lambda The mean of the distribution
///
/// @return We return a `real_type`, not an `int`, as with
/// deterministic mode this will not necessarily be an integer
__nv_exec_check_disable__
template <typename real_type, typename rng_state_type>
__host__ __device__
real_type poisson(rng_state_type& rng_state, real_type lambda) {
  static_assert(std::is_floating_point<real_type>::value,
                "Only valid for floating-point types; use poisson<real_type>()");

  // This cut-off comes from p 42 of Hormann, and might only be
  // valid for double precision; for single precision we need to
  // check that we can go this high.
  constexpr real_type big_lambda =
    std::is_same<real_type, float>::value ? 1e4 : 1e8;

  poisson_validate(lambda);
  real_type x = 0;
  if (lambda == 0) {
    // do nothing, but leave this branch in to help the GPU
#ifndef __CUDA_ARCH__
  } else if (rng_state.deterministic) {
    x = lambda;
#endif
  } else if (lambda < 10) {
    x = poisson_knuth<real_type>(rng_state, lambda);
  } else if (lambda < big_lambda) {
    x = poisson_hormann<real_type>(rng_state, lambda);
  } else {
    x = poisson_cauchy<real_type>(rng_state, lambda);
  }

  SYNCWARP
  return x;
}

}
}

#endif
