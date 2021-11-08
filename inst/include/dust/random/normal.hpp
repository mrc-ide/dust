#ifndef DUST_RANDOM_NORMAL_HPP
#define DUST_RANDOM_NORMAL_HPP

#include <cmath>

#include "dust/random/generator.hpp"
#include "dust/random/numeric.hpp"
#include "dust/random/normal_box_muller.hpp"
#include "dust/random/normal_ziggurat.hpp"

namespace dust {
namespace random {

namespace algorithm {
enum class normal {box_muller, ziggurat};
}

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
  case algorithm::normal::ziggurat:
  default: // keeps compiler happy
    return random_normal_ziggurat<real_type>(rng_state);
  }
}

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
