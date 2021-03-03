#ifndef DUST_DISTR_UNIFORM_HPP
#define DUST_DISTR_UNIFORM_HPP

namespace dust {
namespace distr {

__nv_exec_check_disable__
template <typename real_t>
HOSTDEVICE real_t runif(rng_state_t<real_t>& rng_state,
                        typename rng_state_t<real_t>::real_t min,
                        typename rng_state_t<real_t>::real_t max) {
  return dust::unif_rand<real_t>(rng_state) * (max - min) + min;
}

}
}

#endif
