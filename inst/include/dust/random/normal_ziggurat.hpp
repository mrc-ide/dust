#ifndef DUST_RANDOM_NORMAL_ZIGGURAT_HPP
#define DUST_RANDOM_NORMAL_ZIGGURAT_HPP

#include <cmath>

#include "dust/random/generator.hpp"
#include "dust/random/normal_ziggurat_tables.hpp"

namespace dust {
namespace random {

__nv_exec_check_disable__
namespace {
template <typename real_type, typename rng_state_type>
__host__ __device__
real_type normal_ziggurat_tail(rng_state_type& rng_state, real_type x1,
                               bool negative) {
  real_type ret;
  do {
    const auto u1 = random_real<real_type>(rng_state);
    const auto u2 = random_real<real_type>(rng_state);
    const auto x = std::log(u1) / x1;
    const auto y = std::log(u2);
    if (- 2 * y > x * x) {
      ret = negative ? x - x1 : x1 - x;
      break;
    }
  } while (true);
  return ret;
}

template <typename rng_state_type>
int ziggurat_layer_draw(rng_state_type& rng_state,
                        typename rng_state_type::int_type value, int n) {
  using int_type = typename rng_state_type::int_type;
  // We need to use log2(n) bits; this will be 8 while we use n = 256
  // and it's unlikely that we will use more than that. Doing `% 8` is
  // the same as `& 0xff` but less horrible
  //
  // We have either an uint64_t or a uint32_t to play with.
  //
  // If the former we can partition it into chunks of 3, 8 and 53; we
  // send the 53 bits off to generate our real number later (via
  // int_to_real, see below) use the middle 8 to compute our integer
  // on 0..255 and throw away the worst 3 bits.
  //
  // If we have an uint32_t we have no choice but to draw a second
  // number to compute the layer or we will overlap in bits between
  // the number used for the layer and that used for the position
  // within the layer (see Doornik 2005). So we draw another number,
  // shift that by 16 to get into the middle of the number and apply
  // the same mask.
  //
  // The reason for both shifts is to avoid the low-quality bits, see
  // https://vigna.di.unimi.it/ftp/papers/ScrambledLinear.pdf - the
  // whole process is discussed a bit further in the vignette.
  //
  // It might be worth rejecting this approach for xoroshiro128+ or
  // for all rngs with a + scrambler (this is gettable at compile
  // time).
  if (std::is_same<int_type, uint64_t>::value) {
    return (value >> 3) % n;
  } else {
    return (random_int<typename rng_state_type::int_type>(rng_state) >> 16) % n;
  }
}
}

// TODO: this will not work efficiently for float types because we
// don't have float tables for 'x' and 'y'; getting them is not easy
// without requiring c++14 either. The lower loop using 'x' could
// rewritten easily as a function though taking 'u1' and so allowing
// full template specialisation. However by most accounts this
// performs poorly onn a GPU to latency so it might be ok.
__nv_exec_check_disable__
template <typename real_type, typename rng_state_type>
__host__ __device__
real_type random_normal_ziggurat(rng_state_type& rng_state) {
  using ziggurat::x;
  using ziggurat::y;
  // This 'n' needs to match the length of 'y'. To change, update the
  // tables by editing and re-running ./scripts/update_ziggurat_tables
  //
  // Benchmarking on the CPU showed 256 to be the fastest of 32, 64,
  // 128, 258 (it was the largest tried though), but it is quite
  // possible that a different number will be better on the GPU. Once
  // #327 and #324 are dealt with we could template over the number of
  // bins.
  constexpr size_t n = 256;
  const real_type r = x[1];

  using int_type = typename rng_state_type::int_type;

  real_type ret;
  do {
    const auto value = random_int<int_type>(rng_state);
    const auto i = ziggurat_layer_draw(rng_state, value, n);
    const auto u0 = 2 * int_to_real<real_type>(value) - 1;

    if (std::abs(u0) < y[i]) {
      ret = u0 * x[i];
      break;
    }
    if (i == 0) {
      ret = normal_ziggurat_tail<real_type>(rng_state, r, u0 < 0);
      break;
    }
    const auto z = u0 * x[i];
    const auto f0 = std::exp(-0.5 * (x[i] * x[i] - z * z));
    const auto f1 = std::exp(-0.5 * (x[i + 1] * x[i + 1] - z * z));
    const auto u1 = random_real<real_type>(rng_state);
    if (f1 + u1 * (f0 - f1) < 1.0) {
      ret = z;
      break;
    }
  } while (true);
  SYNCWARP
  return ret;
}

}
}

#endif
