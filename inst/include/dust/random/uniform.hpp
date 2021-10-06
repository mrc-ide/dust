#ifndef DUST_RANDOM_UNIFORM_HPP
#define DUST_RANDOM_UNIFORM_HPP

namespace dust {
namespace random {

__nv_exec_check_disable__
template <typename real_t, typename rng_state_t>
HOSTDEVICE real_t uniform(rng_state_t& rng_state, real_t min, real_t max) {
  static_assert(std::is_floating_point<real_t>::value,
                "Only valid for floating-point types; use runif<real_t>()");
#ifndef __CUDA_ARCH__
  if (rng_state.deterministic) {
    return (max - min) / 2 + min;
  }
#endif
  return random_real<real_t>(rng_state) * (max - min) + min;
}

}
}

#endif
