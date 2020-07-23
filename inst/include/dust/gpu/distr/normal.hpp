#ifndef DUST_DISTR_NORMAL_HPP
#define DUST_DISTR_NORMAL_HPP

#include <cmath>

namespace dust {
namespace distr {

template <typename real_t>
__device__
inline void box_muller(rng_state_t<real_t>& rng_state, real_t* d0, real_t* d1) {
  // This function implements the Box-Muller transform:
  // http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
  // Do not send a really small number to log().
  // We cannot mark "epsilon" as "static const" because NVCC would complain
  const double epsilon = 1.0e-7;
  double u1 = device_unif_rand(rng_state);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  const double v1 = 2 * M_PI * device_unif_rand(rng_state);
  const double u2 = sqrt(-2.0 * log(u1));
  sincos(v1, d0, d1);
  *d0 *= u2;
  *d1 *= u2;
}

template <typename real_t>
__device__
inline real_t rnorm(rng_state_t<real_t>& rng_state,
                    typename rng_state_t<real_t>::real_t mean,
                    typename rng_state_t<real_t>::real_t sd) {
  real_t r0, r1; // r1 currently thrown away
  box_muller(rng_state, &r0, &r1);
  return r0 * sd + mean;
}

// Device class which saves both values from the BoxMuller transform
// Not yet used
template <typename real_t>
class rnorm_buffer {
 public:
  __device__
  rnorm_buffer() : _buffered(false) {}

  __device__
  inline real_t operator()(rng_state_t<real_t>& rng_state, real_t mean, real_t sd) {
    real_t z0;
    if (_buffered) {
      _buffered = false;
      z0 = result[1];
    } else {
      BoxMuller<real_t>(rng_state, &result[0], &result[1]);
      _buffered = true;
      z0 = result[0];
    }
    __syncwarp();
    return(z0 * sd + mean);
  }

  private:
    bool _buffered;
    real_t result[2];
};

}
}

#endif
