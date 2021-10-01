#pragma once

#include <cmath>

namespace dust {
namespace random {

// Devroye 1986
// http://luc.devroye.org/rnbookindex.html
// Chapter 9, p 392
//
// > No method is shorter than the inversion method, which returns
// > -log(U) where U is a uniform [0,1] random variate
//
// Faster generators will exist but we can swap one in if it becomes
// important.
template <typename real_t, typename rng_state_t>
HOSTDEVICE real_t exponential_rand(rng_state_t& rng_state) {
#ifdef __CUDA_ARCH__
  return -std::log(random_real<real_t>(rng_state));
#else
  return rng_state.deterministic ? 1 :
    -std::log(random_real<real_t>(rng_state));
#endif
}

__nv_exec_check_disable__
template <typename real_t, typename rng_state_t>
HOSTDEVICE real_t exponential(rng_state_t& rng_state, real_t rate) {
  static_assert(std::is_floating_point<real_t>::value,
                "Only valid for floating-point types; use rexponential<real_t>()");
  return exponential_rand<real_t>(rng_state) / rate;
}

}
}
