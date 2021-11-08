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
  const real_type r = x<real_type>[1];

  real_type ret;
  do {
    const auto i = (next(rng_state) >> 16) % n;
    const auto u0 = 2 * random_real<real_type>(rng_state) - 1;
    if (std::abs(u0) < y<real_type>[i]) {
      ret = u0 * x<real_type>[i];
      break;
    }
    if (i == 0) {
      ret = normal_ziggurat_tail<real_type>(rng_state, r, u0 < 0);
      break;
    }
    const auto z = u0 * x<real_type>[i];
    const auto f0 = std::exp(-0.5 * (x<real_type>[i] * x<real_type>[i] - z * z));
    const auto f1 = std::exp(-0.5 * (x<real_type>[i + 1] * x<real_type>[i + 1] - z * z));
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
