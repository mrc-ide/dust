#ifndef DUST_RANDOM_XOROSHIRO128_HPP
#define DUST_RANDOM_XOROSHIRO128_HPP

#include "dust/random/utils.hpp"
#include "dust/random/xoshiro_state.hpp"

// 64-bit generators, state is 2 * uint64_t
//
//  xoroshiro128**  | https://prng.di.unimi.it/xoroshiro128starstar.c
//  xoroshiro128++  | https://prng.di.unimi.it/xoroshiro128plusplus.c
//  xoroshiro128+   | https://prng.di.unimi.it/xoroshiro128plus.c

namespace dust {
namespace random {

using xoroshiro128starstar_state =
  xoshiro_state<uint64_t, 2, scrambler::starstar>;
using xoroshiro128plusplus_state =
  xoshiro_state<uint64_t, 2, scrambler::plusplus>;
using xoroshiro128plus_state =
  xoshiro_state<uint64_t, 2, scrambler::plus>;

// Jump coefficients.
// * 2^64 calls to next
// * 2^64 non-overlapping subsequences
template <>
constexpr
std::array<uint64_t, 2> jump_constants<xoroshiro128starstar_state>() {
  return std::array<uint64_t, 2>{{
      0xdf900294d8f554a5, 0x170865df4b3201fc }};
}

template <>
constexpr
std::array<uint64_t, 2> jump_constants<xoroshiro128plusplus_state>() {
  return std::array<uint64_t, 2>{{
      0x2bd7a6a6e99c2ddc, 0x0992ccaf6a6fca05 }};
}

template <>
constexpr
std::array<uint64_t, 2> jump_constants<xoroshiro128plus_state>() {
  return jump_constants<xoroshiro128starstar_state>();
}

// Long-jump coefficients.
// * 2^96 calls to next
// * 2^32 starting points, each with 2^32 subsequences
template <>
constexpr
std::array<uint64_t, 2> long_jump_constants<xoroshiro128starstar_state>() {
  return std::array<uint64_t, 2>{{
      0xd2a98b26625eee7b, 0xdddf9b1090aa7ac1 }};
}

template <>
constexpr
std::array<uint64_t, 2> long_jump_constants<xoroshiro128plusplus_state>() {
  return std::array<uint64_t, 2>{{
      0x360fd5f2cf8d5d99, 0x9c6e6877736c46e3 }};
}

template <>
constexpr
std::array<uint64_t, 2> long_jump_constants<xoroshiro128plus_state>() {
  return long_jump_constants<xoroshiro128starstar_state>();
}

template <>
inline __host__ __device__
uint64_t next(xoroshiro128starstar_state& state) {
  const uint64_t s0 = state[0];
  uint64_t s1 = state[1];
  const uint64_t result = rotl(s0 * 5, 7) * 9;
  s1 ^= s0;
  state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
  state[1] = rotl(s1, 37); // c
  return result;
}

template <>
inline __host__ __device__
uint64_t next(xoroshiro128plusplus_state& state) {
  const uint64_t s0 = state[0];
  uint64_t s1 = state[1];
  const uint64_t result = rotl(s0 + s1, 17) + s0;
  s1 ^= s0;
  state[0] = rotl(s0, 49) ^ s1 ^ (s1 << 21); // a, b
  state[1] = rotl(s1, 28); // c
  return result;
}

template <>
inline __host__ __device__
uint64_t next(xoroshiro128plus_state& state) {
  const uint64_t s0 = state[0];
  uint64_t s1 = state[1];
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
  state[1] = rotl(s1, 37); // c
  return result;
}

}
}

#endif
