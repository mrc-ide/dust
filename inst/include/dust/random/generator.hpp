#pragma once

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

#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>

#include <dust/random/cuda.hpp>
#include <dust/random/utils.hpp>

namespace dust {
namespace random {

// Generic data storage, this is common
template <typename T, size_t N, xoshiro_mode M>
struct xoshiro_state {
  typedef T data_type;
  static HOSTDEVICE size_t size() {
    return N;
  }
  data_type state[N];
  // This flag indicates that the distributions should return the
  // deterministic expectation of the draw, and not use any random
  // numbers
  bool deterministic = false;
  HOSTDEVICE data_type& operator[](size_t i) {
    return state[i];
  }
};

template <typename T>
typename T::data_type next(T& state);

template <typename T, size_t N, xoshiro_mode M>
inline HOST
void jump(xoshiro_state<T, N, M>& state) {
  constexpr std::array<T, N> jump = jump_constants<T, N, M>();
  rng_jump_state(state, jump);
}

template <typename T, size_t N, xoshiro_mode M>
inline HOST
void long_jump(xoshiro_state<T, N, M>& state) {
  constexpr std::array<T, N> jump = long_jump_constants<T, N, M>();
  rng_jump_state(state, jump);
}

template <typename T, size_t N, xoshiro_mode M>
inline HOST
void rng_jump_state(xoshiro_state<T, N, M>& state, std::array<T, N> coef) {
  T work[N] = { }; // enforced zero-initialisation
  constexpr int bits = bit_size<T>();
  for (size_t i = 0; i < N; ++i) {
    for (int b = 0; b < bits; b++) {
      if (coef[i] & static_cast<T>(1) << b) {
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
HOST
void seed(T& state, uint64_t seed) {
  const size_t n = T::size();
  using data_type = typename T::data_type;
  for (size_t i = 0; i < n; ++i) {
    seed = splitmix64(seed);
    state[i] = static_cast<data_type>(seed);
  }
}

template <typename T>
HOST
T seed(uint64_t seed) {
  T state;
  dust::random::seed(state, seed);
  return state;
}

template <typename T>
HOST
std::vector<typename T::data_type> seed_data(uint64_t seed) {
  T state = dust::random::seed<T>(seed);
  const size_t n = state.size();
  std::vector<typename T::data_type> ret(n);
  std::copy_n(state.state, n, ret.begin());
  return ret;
}

// This is the workhorse function
template <typename T, typename U>
HOSTDEVICE
T random_real(U& state) {
  const auto value = next(state);
  return int_to_real<T>(value);
}

}
}

// Implementations

// 64 bit generators, 4 * uint64_t
#include "xoshiro256.hpp"

// 64 bit generators, 2 * uint64_t
#include "xoroshiro128.hpp"

// 32 bit generators, 4 * uint32_t
#include "xoshiro128.hpp"
