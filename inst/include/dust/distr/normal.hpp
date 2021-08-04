#ifndef DUST_DISTR_NORMAL_HPP
#define DUST_DISTR_NORMAL_HPP

#include <cmath>
#include <dust/utils.hpp>

namespace dust {
namespace distr {

__nv_exec_check_disable__
template <typename real_t>
HOSTDEVICE inline real_t box_muller(rng_state_t<real_t>& rng_state) {
  // This function implements the Box-Muller transform:
  // http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
  // Do not send a really small number to log().
  const real_t epsilon = dust::utils::epsilon<real_t>();
  const real_t two_pi = 2 * M_PI;

  real_t u1, u2;
  do {
    u1 = dust::unif_rand(rng_state);
    u2 = dust::unif_rand(rng_state);
  } while (u1 <= epsilon);

  SYNCWARP
  return std::sqrt(-2 * std::log(u1)) * std::cos(two_pi * u2);
}

// The type declarations for mean and sd are ugly but prevent the
// compiler complaining about conflicting inferred types for real_t
__nv_exec_check_disable__
template <typename real_t>
HOSTDEVICE real_t rnorm(rng_state_t<real_t>& rng_state,
                        typename rng_state_t<real_t>::real_t mean,
                        typename rng_state_t<real_t>::real_t sd) {
#ifdef __CUDA_ARCH__
  real_t z = box_muller<real_t>(rng_state);
#else
  real_t z = rng_state.deterministic ? 0 : box_muller<real_t>(rng_state);
#endif
  return z * sd + mean;
}

}
}

#endif
