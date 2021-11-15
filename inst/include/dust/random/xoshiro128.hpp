#ifndef DUST_RANDOM_XOSHIRO128_HPP
#define DUST_RANDOM_XOSHIRO128_HPP

#include "dust/random/utils.hpp"
#include "dust/random/xoshiro_state.hpp"

// 32-bit generators, state is 4 * uint32_t
//
//  xoshiro128**  | https://prng.di.unimi.it/xoshiro128starstar.c
//  xoshiro128++  | https://prng.di.unimi.it/xoshiro128plusplus.c
//  xoshiro128+   | https://prng.di.unimi.it/xoshiro128plus.c

namespace dust {
namespace random {

using xoshiro128starstar =
  xoshiro_state<uint32_t, 4, scrambler::starstar>;
using xoshiro128plusplus =
  xoshiro_state<uint32_t, 4, scrambler::plusplus>;
using xoshiro128plus =
  xoshiro_state<uint32_t, 4, scrambler::plus>;

// Jump coefficients.
// * 2^64 calls to next
// * 2^64 non-overlapping subsequences
template <>
constexpr std::array<uint32_t, 4> jump_constants<xoshiro128starstar>() {
  return std::array<uint32_t, 4>{{
      0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b }};
}

template <>
constexpr std::array<uint32_t, 4> jump_constants<xoshiro128plusplus>() {
  return jump_constants<xoshiro128starstar>();
}

template <>
constexpr std::array<uint32_t, 4> jump_constants<xoshiro128plus>() {
  return jump_constants<xoshiro128starstar>();
}

// Long-jump coefficients.
// * 2^96 calls to next
// * 2^32 starting points, each with 2^32 subsequences
template <>
constexpr
std::array<uint32_t, 4> long_jump_constants<xoshiro128starstar>() {
  return std::array<uint32_t, 4>{{
      0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662 }};
}

template <>
constexpr
std::array<uint32_t, 4> long_jump_constants<xoshiro128plusplus>() {
  return long_jump_constants<xoshiro128starstar>();
}

template <>
constexpr
std::array<uint32_t, 4> long_jump_constants<xoshiro128plus>() {
  return long_jump_constants<xoshiro128starstar>();
}

template <>
inline __host__ __device__ uint32_t next(xoshiro128starstar& state) {
  const uint32_t result = rotl(state[1] * 5, 7) * 9;
  const uint32_t t = state[1] << 9;
  state[2] ^= state[0];
  state[3] ^= state[1];
  state[1] ^= state[2];
  state[0] ^= state[3];
  state[2] ^= t;
  state[3] = rotl(state[3], 11);
  return result;
}

template <>
inline __host__ __device__ uint32_t next(xoshiro128plusplus& state) {
  const uint32_t result = rotl(state[0] + state[3], 7) + state[0];
  const uint32_t t = state[1] << 9;
  state[2] ^= state[0];
  state[3] ^= state[1];
  state[1] ^= state[2];
  state[0] ^= state[3];
  state[2] ^= t;
  state[3] = rotl(state[3], 11);
  return result;
}

template <>
inline __host__ __device__ uint32_t next(xoshiro128plus& state) {
  const uint32_t result = state[0] + state[3];
  const uint32_t t = state[1] << 9;
  state[2] ^= state[0];
  state[3] ^= state[1];
  state[1] ^= state[2];
  state[0] ^= state[3];
  state[2] ^= t;
  state[3] = rotl(state[3], 11);
  return result;
}

}
}

#endif
