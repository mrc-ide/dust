#ifndef DUST_DISTR_EXPONENTIAL_HPP
#define DUST_DISTR_EXPONENTIAL_HPP

#include <cmath>

namespace dust {
namespace distr {

// Devroye 1986
// http://luc.devroye.org/rnbookindex.html
// Chapter 9, p 392
//
// > No method is shorter than the inversion method, which returns
// > -log(U) where U is a uniform [0,1] random variate
//
// Faster generators will exist but we can swap one in if it becomes
// important.
template <typename real_t>
HOSTDEVICE real_t exp_rand(rng_state_t<real_t>& rng_state) {
#ifdef __CUDA_ARCH__
  return -std::log(dust::unif_rand(rng_state));
#else
  return rng_state.deterministic ? 1 : -std::log(dust::unif_rand(rng_state));
#endif
}

__nv_exec_check_disable__
template <typename real_t>
HOSTDEVICE real_t rexp(rng_state_t<real_t>& rng_state,
                       typename rng_state_t<real_t>::real_t rate) {
  return exp_rand(rng_state) / rate;
}

}
}

#endif
