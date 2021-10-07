#ifndef DUST_RANDOM_EXPONENTIAL_HPP
#define DUST_RANDOM_EXPONENTIAL_HPP

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
template <typename real_type, typename rng_state_type>
HOSTDEVICE real_type exponential_rand(rng_state_type& rng_state) {
#ifdef __CUDA_ARCH__
  return -std::log(random_real<real_type>(rng_state));
#else
  return rng_state.deterministic ? 1 :
    -std::log(random_real<real_type>(rng_state));
#endif
}

__nv_exec_check_disable__
template <typename real_type, typename rng_state_type>
HOSTDEVICE real_type exponential(rng_state_type& rng_state, real_type rate) {
  static_assert(std::is_floating_point<real_type>::value,
                "Only valid for floating-point types; use exponential<real_type>()");
  return exponential_rand<real_type>(rng_state) / rate;
}

}
}

#endif
