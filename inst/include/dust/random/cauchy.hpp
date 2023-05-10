#ifndef DUST_RANDOM_CAUCHY_HPP
#define DUST_RANDOM_CAUCHY_HPP

#include <cmath>

#include "dust/random/generator.hpp"
#include "dust/random/numeric.hpp"
#include "dust/random/math.hpp"

namespace dust {
namespace random {

__nv_exec_check_disable__
template <typename real_type, typename rng_state_type>
__host__ __device__
real_type cauchy(rng_state_type& rng_state, real_type location,
                 real_type scale) {
  static_assert(std::is_floating_point<real_type>::value,
                "Only valid for floating-point types; use cauchy<real_type>()");
#ifndef __CUDA_ARCH__
  if (rng_state.deterministic) {
    throw std::runtime_error("Can't use Cauchy distribution deterministically; it has no mean");
  }
#endif
  const real_type u = random_real<real_type>(rng_state);
  return location + scale * dust::math::tan<real_type>(static_cast<real_type>(M_PI) * u);
}

}
}

#endif
