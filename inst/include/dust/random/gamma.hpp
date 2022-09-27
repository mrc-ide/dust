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

// Generate gamma random number via GS, p 228 of reference.
template <typename real_type, typename rng_state_type>
real_type gamma_gs(rng_state_type& rng_state, real_type a) {
  real_type b = 1.0 + std::exp(-1) * a;
  while (TRUE) {
    real_type p = b * uniform<real_type>(rng_state, 0, 1);
    if (p >= 1.0) {
      real_type x = -log((b - p) / a);
      if (exponential_rand<real_type>(rng_state) >= (1.0 - a) * log(x))
        return x;
    } else {
      real_type x = exp(log(p) / a);
      if (exponential_rand<real_type>(rng_state) >= x)
        return x;
    }
  }
}

// Generate gamma random number via GD, p 53 of reference.
template <typename real_type, typename rng_state_type>
real_type gamma_gd(rng_state_type& rng_state, real_type a) {

  const real_type q1 = 0.04166669;
  const real_type q2 = 0.02083148;
  const real_type q3 = 0.00801191;
  const real_type q4 = 0.00144121;
  const real_type q5 = -7.388e-5;
  const real_type q6 = 2.4511e-4;
  const real_type q7 = 2.424e-4;

  const real_type a1 = 0.3333333;
  const real_type a2 = -0.250003;
  const real_type a3 = 0.2000062;
  const real_type a4 = -0.1662921;
  const real_type a5 = 0.1423657;
  const real_type a6 = -0.1367177;
  const real_type a7 = 0.1233795;

  real_type s, s2, d;    /* no. 1 (step 1) */
  real_type q0, b, si, c;/* no. 2 (step 4) */
    
  real_type e, p, q, r, t, u, v, w, x, ret_val;

  s2 = a - 0.5;
  s = sqrt(s2);
  d = sqrt(2) * 4 - s * 12.0;

  /* Step 2: t = standard normal deviate,
           x = (s,1/2) -normal deviate. */

  /* immediate acceptance (i) */
  t = random_normal<real_type>(rng_state);
  x = s + 0.5 * t;
  ret_val = x * x;
  if (t >= 0.0) {
    return ret_val;
  }

  /* Step 3: u = 0,1 - uniform sample. squeeze acceptance (s) */
  u = uniform<real_type>(rng_state, 0, 1);
  if (d * u <= t * t * t) {
    return ret_val;
  }

  /* Step 4: calculations of q0, b, si, c */
  r = 1.0 / a;
  q0 = ((((((q7 * r + q6) * r + q5) * r + q4) * r + q3) * r + q2) * r + q1) * r;

  /* Approximation depending on size of parameter a */
  /* The constants in the expressions for b, si and c */
  /* were established by numerical experiments */

  if (a <= 3.686) {
    b = 0.463 + s + 0.178 * s2;
    si = 1.235;
    c = 0.195 / s - 0.079 + 0.16 * s;
  } else if (a <= 13.022) {
    b = 1.654 + 0.0076 * s2;
    si = 1.68 / s + 0.275;
    c = 0.062 / s + 0.024;
  } else {
    b = 1.77;
    si = 0.75;
    c = 0.1515 / s;
  }

  /* Step 5: no quotient test if x not positive */

  if (x > 0.0) {
  	/* Step 6: calculation of v and quotient q */
  	v = t / (s + s);
  	if (fabs(v) <= 0.25) {
  	  q = q0 + 0.5 * t * t * ((((((a7 * v + a6) * v + a5) * v + a4) * v
  				      + a3) * v + a2) * v + a1) * v;
  	} else {
  	  q = q0 - s * t + 0.25 * t * t + (s2 + s2) * log(1.0 + v);
  	}

  	/* Step 7: quotient acceptance (q) */
  	if (log(1.0 - u) <= q)
  	    return ret_val;
    }

    while (TRUE) {
      /* Step 8: e = standard exponential deviate
       *	u =  0,1 -uniform deviate
       *	t = (b,si)-double exponential (laplace) sample */
      e = exponential_rand<real_type>(rng_state);
      u = uniform<real_type>(rng_state, 0, 1);
      u = u + u - 1.0;
      if (u < 0.0) {
        t = b - si * e;
      }  else {
        t = b + si * e;
      }

      /* Step	 9:  rejection if t < tau(1) = -0.71874483771719 */
      if (t >= -0.71874483771719) {
        /* Step 10:	 calculation of v and quotient q */
        v = t / (s + s);

        if (fabs(v) <= 0.25) {
          q = q0 + 0.5 * t * t *
            ((((((a7 * v + a6) * v + a5) * v + a4) * v + a3) * v
              + a2) * v + a1) * v;
        } else {
          q = q0 - s * t + 0.25 * t * t + (s2 + s2) * log(1.0 + v);
        }
        /* Step 11:	 hat acceptance (h) */
        /* (if q not positive go to step 8) */
        if (q > 0.0) {
          w = expm1(q);
          /* if t is rejected sample again at step 8 */
          if (c * fabs(u) <= w * exp(e - 0.5 * t * t)) break;
        }
      }
    } /* repeat .. until  `t' is accepted */
    x = s + 0.5 * t;
    return x * x;
}

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

  if (a >= 1.0) {
    return gamma_gd<real_type>(rng_state, a) * b;
  }

  return gamma_gs<real_type>(rng_state, a) * b;
}

}
}

#endif
