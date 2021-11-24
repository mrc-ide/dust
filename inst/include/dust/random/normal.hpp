#ifndef DUST_RANDOM_NORMAL_HPP
#define DUST_RANDOM_NORMAL_HPP

#include <cmath>

#include "dust/random/generator.hpp"
#include "dust/random/numeric.hpp"
#include "dust/random/normal_box_muller.hpp"
#include "dust/random/normal_polar.hpp"
#include "dust/random/normal_ziggurat.hpp"

namespace dust {
namespace random {

namespace algorithm {
enum class normal {
                   box_muller, ///< Box-Muller method (transformation)
                   polar,      ///< Polar method (rejection)
                   ziggurat    ///< Ziggurat method (rejection)
};
}

/// Draw a standard normally distributed random number (with mean 0
/// and standard deviation 1)
///
/// @tparam T The real type to return, typically `double` or `float`;
/// because this affects the return value only it must be provided.
///
/// @tparam The algorithm to use; the default is Box-Muller which is
/// slowest on CPU but simple. Other alternatives are `ziggurat`
/// (fast) or `polar` (medium).
///
/// @tparam U The random number generator state type; this will be
/// inferred based on the argument
///
/// @param state The random number state, will be updated as a side effect
///
/// @return A real-valued random number on (-inf, inf)
__nv_exec_check_disable__
template <typename real_type,
          algorithm::normal algorithm = algorithm::normal::box_muller,
          typename rng_state_type>
__host__ __device__
real_type random_normal(rng_state_type& rng_state) {
  static_assert(std::is_floating_point<real_type>::value,
                "Only valid for floating-point types; use normal<real_type>()");
  switch(algorithm) {
  case algorithm::normal::box_muller:
    return random_normal_box_muller<real_type>(rng_state);
  case algorithm::normal::polar:
    return random_normal_polar<real_type>(rng_state);
  case algorithm::normal::ziggurat:
  default: // keeps compiler happy
    return random_normal_ziggurat<real_type>(rng_state);
  }
}


/// Draw a normally distributed random number with arbitrary bounds.
/// This function simply scales the output of
/// `dust::random::random_real`
///
/// @tparam real_type The underlying real number type, typically
/// `double` or `float`. A compile-time error will be thrown if you
/// attempt to use a non-floating point type (based on
/// `std::is_floating_point).
///
/// @tparam The algorithm to use; the default is Box-Muller which is
/// slowest on CPU but simple.
///
/// @tparam rng_state_type The random number state type
///
/// @param rng_state Reference to the random number state, will be
/// modified as a side-effect
///
/// @param mean The mean of the distribution
///
/// @param sd The standard deviation of the distribution
__nv_exec_check_disable__
template <typename real_type,
          algorithm::normal algorithm = algorithm::normal::box_muller,
          typename rng_state_type>
__host__ __device__
real_type normal(rng_state_type& rng_state, real_type mean, real_type sd) {
  static_assert(std::is_floating_point<real_type>::value,
                "Only valid for floating-point types; use normal<real_type>()");
#ifdef __CUDA_ARCH__
  real_type z = random_normal<real_type, algorithm>(rng_state);
#else
  real_type z = rng_state.deterministic ?
    0 : random_normal<real_type, algorithm>(rng_state);
#endif
  return z * sd + mean;
}

}
}

#endif
