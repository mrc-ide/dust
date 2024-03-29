#ifndef DUST_RANDOM_NORMAL_POLAR_HPP
#define DUST_RANDOM_NORMAL_POLAR_HPP

#include "dust/random/generator.hpp"
#include "dust/random/math.hpp"

namespace dust {
namespace random {

__nv_exec_check_disable__
template <typename real_type, typename rng_state_type>
__host__ __device__
real_type random_normal_polar(rng_state_type& rng_state) {
  real_type s, x, y;
  do {
    x = 2 * random_real<real_type>(rng_state) - 1;
    y = 2 * random_real<real_type>(rng_state) - 1;
    s = x * x + y * y;
  } while (s > 1);
  SYNCWARP

  return x * dust::math::sqrt(-2 * dust::math::log(s) / s);
}

}
}

#endif
