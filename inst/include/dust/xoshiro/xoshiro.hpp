#include <array>
#include <cstdint>
#include <cstddef>

// Dust stuff, we'll want this again later
#define HOST
#define HOSTDEVICE
#define DEVICE

enum xoshiro_mode {STARSTAR, PLUSPLUS, PLUS};

// Generic data storage, this is common
template <typename T, size_t N, xoshiro_mode M>
struct xoshiro_state {
  typedef T data_type;
  static size_t size() {
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
typename T::data_type rng_next(T& state);

template <typename T, size_t N, xoshiro_mode M>
std::array<T, N> jump_constants();

template <typename T, size_t N, xoshiro_mode M>
std::array<T, N> long_jump_constants();

// This is probably doable with sizeof(T) * 8 too?
template <typename T>
constexpr size_t bit_size();

template <>
constexpr size_t bit_size<uint32_t>() {
  return 32;
}

template <>
constexpr size_t bit_size<uint64_t>() {
  return 64;
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
      rng_next(state);
    }
  }
  for (size_t i = 0; i < N; ++i) {
    state[i] = work[i];
  }
}

template <typename T, size_t N, xoshiro_mode M>
inline HOST
void rng_jump(xoshiro_state<T, N, M>& state) {
  constexpr std::array<T, N> jump = jump_constants<T, N, M>();
  rng_jump_state(state, jump);
}

template <typename T, size_t N, xoshiro_mode M>
inline HOST
void rng_long_jump(xoshiro_state<T, N, M>& state) {
  constexpr std::array<T, N> jump = long_jump_constants<T, N, M>();
  rng_jump_state(state, jump);
}

inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

inline uint32_t rotl(const uint32_t x, int k) {
  return (x << k) | (x >> (32 - k));
}

// 64 bit generators, 4 * uint64_t
#include "xoshiro256.hpp"

// 64 bit generators, 2 * uint64_t
#include "xoroshiro128.hpp"

// 32 bit generators, 4 * uint32_t
#include "xoshiro128.hpp"

// Seeding
inline HOST uint64_t splitmix64(uint64_t seed) {
  uint64_t z = (seed += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

template <typename T>
void rng_seed(T& state, uint64_t seed) {
  const size_t n = T::size();
  using data_type = typename T::data_type;
  for (size_t i = 0; i < n; ++i) {
    seed = splitmix64(seed);
    state[i] = static_cast<data_type>(seed);
  }
}
