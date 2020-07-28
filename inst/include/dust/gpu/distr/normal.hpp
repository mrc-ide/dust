#ifndef DUST_DISTR_NORMAL_HPP
#define DUST_DISTR_NORMAL_HPP

#include <cmath>

namespace dust {
namespace distr {

template <typename real_t>
__device__
inline real_t box_muller(rng_state_t<real_t>& rng_state) {
  // This function implements the Box-Muller transform:
  // http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
  // Do not send a really small number to log().
  constexpr real_t epsilon = 1.0e-7;
  constexpr real_t two_pi = 2 * M_PI;

  real_t u1 = device_unif_rand(rng_state);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  real_t u2 = device_unif_rand(rng_state);

  return std::sqrt(-static_cast<real_t>(2.0) * std::log(u1)) *
    std::cos(two_pi * u2);
}

template <typename real_t>
__device__
inline real_t rnorm(rng_state_t<real_t>& rng_state,
                    typename rng_state_t<real_t>::real_t mean,
                    typename rng_state_t<real_t>::real_t sd) {
  real_t z = box_muller<real_t>(rng_state);
  return z * sd + mean;
}

}
}

#endif
