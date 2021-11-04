#ifndef DUST_RANDOM_NORMAL_HPP
#define DUST_RANDOM_NORMAL_HPP

#include <cmath>

#include "dust/random/numeric.hpp"

namespace dust {
namespace random {

namespace {

__nv_exec_check_disable__
template <typename real_type, typename rng_state_type>
HOSTDEVICE
real_type box_muller(rng_state_type& rng_state) {
  // This function implements the Box-Muller transform:
  // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
  // Do not send a really small number to log().
  const real_type epsilon = utils::epsilon<real_type>();
  const real_type two_pi = 2 * M_PI;

  real_type u1, u2;
  do {
    u1 = random_real<real_type>(rng_state);
    u2 = random_real<real_type>(rng_state);
  } while (u1 <= epsilon);

  SYNCWARP
  return std::sqrt(-2 * std::log(u1)) * std::cos(two_pi * u2);
}

}

__nv_exec_check_disable__
template <typename real_type, typename rng_state_type>
HOSTDEVICE
real_type normal(rng_state_type& rng_state, real_type mean, real_type sd) {
  static_assert(std::is_floating_point<real_type>::value,
                "Only valid for floating-point types; use normal<real_type>()");
#ifdef __CUDA_ARCH__
  real_type z = box_muller<real_type>(rng_state);
#else
  real_type z = rng_state.deterministic ? 0 : box_muller<real_type>(rng_state);
#endif
  return z * sd + mean;
}

}
}

#endif
