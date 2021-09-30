#ifndef DUST_DISTR_UNIFORM_HPP
#define DUST_DISTR_UNIFORM_HPP

namespace dust {
namespace distr {

__nv_exec_check_disable__
template <typename real_t>
HOSTDEVICE real_t runif(rng_state_t& rng_state, real_t min, real_t max) {
  static_assert(std::is_floating_point<real_t>::value,
                "Only valid for floating-point types; use runif<real_t>()");
  return dust::unif_rand<real_t>(rng_state) * (max - min) + min;
}

}
}

#endif
