// Split a into the integer part m+-[a] and the fi'actional part f e-a-m.
//2. If m=O set y~O and go to 3. Otherwise take a sample y from the gamma (m)
//distribution using GM.
//3. If f=O set z~-O and go to 4. Otherwise take a sample z from the gamma (f)
//distribution using GS.
//4. Deliver x ~ y + z.

#ifndef DUST_RANDOM_GAMMA_HPP
#define DUST_RANDOM_GAMMA_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "dust/random/generator.hpp"
#include "dust/random/numeric.hpp"
#include "dust/random/exponential.hpp"
#include "dust/random/uniform.hpp"

namespace dust {
namespace random {
namespace {

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
  // There are some issues around multiple returns and use of
  // std::pair that probably require some additional work to get this
  // behaving well on a GPU. Unlikely to be a lot of work, but better
  // not to assume that it does. Proper testing of the algorithm under
  // single precision would also be wise to prevent possible infinite
  // loops, that's easiest to do once we have some real use-cases.
#ifdef __CUDA_ARCH__
  static_assert("gamma() not implemented for GPU targets");
#endif

  return gamma_gs<real_type>(rng_state, a) * b;
}

}
}

#endif
