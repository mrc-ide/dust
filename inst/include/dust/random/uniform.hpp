#ifndef DUST_RANDOM_UNIFORM_HPP
#define DUST_RANDOM_UNIFORM_HPP

#include "dust/random/generator.hpp"

namespace dust {
namespace random {

__nv_exec_check_disable__
template <typename real_type, typename rng_state_type>
__host__ __device__
real_type uniform(rng_state_type& rng_state, real_type min, real_type max) {
  static_assert(std::is_floating_point<real_type>::value,
                "Only valid for floating-point types; use uniform<real_type>()");
#ifndef __CUDA_ARCH__
  if (rng_state.deterministic) {
    return (max - min) / 2 + min;
  }
#endif
  return random_real<real_type>(rng_state) * (max - min) + min;
}

}
}

#endif
