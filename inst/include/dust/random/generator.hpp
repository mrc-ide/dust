#ifndef DUST_RANDOM_GENERATOR_HPP
#define DUST_RANDOM_GENERATOR_HPP

// This is the random generator, holding rng state and providing
// support for generating reals on the interval 0..1
//
// Typically this will be too low level for most applications and you
// should use dust::random::prng which provides a parallel random
// number generator.
//
// The api is:
//
// * the dust::random::xoshiro_state type, plus all the
//   specific versions of it (e.g., xoshiro256starstar_state); these
//   objects can be created but should be considered opaque.
//
// * dust::random::random_real which yields a real (of the
//   requested type) given a xoshiro_state state
//
// * dust::random::seed which seeds a generator
//
// * dust::random::jump and dust::random::long_jump which "jump" the
//   generator state forward, a key part of the parallel generators.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "dust/random/cuda_compatibility.hpp"
#include "dust/random/utils.hpp"
#include "dust/random/xoshiro_state.hpp"

// 32 bit generators, 4 * uint32_t
#include "dust/random/xoshiro128.hpp"

// 64 bit generators, 2 * uint64_t
#include "dust/random/xoroshiro128.hpp"

// 64 bit generators, 4 * uint64_t
#include "dust/random/xoshiro256.hpp"

// 64 bit generators, 8 * uint64_t
#include "dust/random/xoshiro512.hpp"

namespace dust {
namespace random {

template <typename T>
inline __host__ void jump(T& state) {
  using int_type = typename T::int_type;
  constexpr auto N = T::size();
  constexpr std::array<int_type, N> jump = jump_constants<T>();
  rng_jump_state(state, jump);
}

template <typename T>
inline __host__ void long_jump(T& state) {
  using int_type = typename T::int_type;
  constexpr auto N = T::size();
  constexpr std::array<int_type, N> jump = long_jump_constants<T>();
  rng_jump_state(state, jump);
}

template <typename T>
inline __host__
void rng_jump_state(T& state,
                    std::array<typename T::int_type, T::size()> coef) {
  using int_type = typename T::int_type;
  constexpr auto N = T::size();
  int_type work[N] = { }; // enforced zero-initialisation
  constexpr int bits = bit_size<int_type>();
  for (size_t i = 0; i < N; ++i) {
    for (int b = 0; b < bits; b++) {
      if (coef[i] & static_cast<int_type>(1) << b) {
        for (size_t j = 0; j < N; ++j) {
          work[j] ^= state[j];
        }
      }
      next(state);
    }
  }
  for (size_t i = 0; i < N; ++i) {
    state[i] = work[i];
  }
}

template <typename T>
__host__ void seed(T& state, uint64_t seed) {
  constexpr size_t n = T::size();
  using int_type = typename T::int_type;
  for (size_t i = 0; i < n; ++i) {
    seed = splitmix64(seed);
    state[i] = static_cast<int_type>(seed);
  }
}

template <typename T>
__host__ T seed(uint64_t seed) {
  T state;
  dust::random::seed(state, seed);
  return state;
}

template <typename T>
__host__ std::vector<typename T::int_type> seed_data(uint64_t seed) {
  T state = dust::random::seed<T>(seed);
  const size_t n = state.size();
  std::vector<typename T::int_type> ret(n);
  std::copy_n(state.state, n, ret.begin());
  return ret;
}

// This is the workhorse function
template <typename T, typename U>
__host__ __device__
T random_real(U& state) {
  const auto value = next(state);
  return int_to_real<T>(value);
}

template <typename T, typename U>
__host__ __device__
T random_int(U& state) {
  static_assert(sizeof(T) <= sizeof(typename U::int_type),
                "requested integer too wide");
  static_assert(std::is_integral<T>::value,
                "integer type required for T");
  const auto value = next(state);
  return static_cast<T>(value);
}

}
}

#endif
