#ifndef DUST_DISTR_UNIFORM_HPP
#define DUST_DISTR_UNIFORM_HPP

namespace dust {
namespace distr {

template <typename real_t, typename int_t>
__device__
real_t runif(RNGState& rng_state, real_t min, real_t max) {
    real_t u1 = device_unif_rand(rng_state) * (max - min) + min;
    return u1;
}

}
}

#endif
