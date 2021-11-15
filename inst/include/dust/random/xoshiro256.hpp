#ifndef DUST_RANDOM_XOSHIRO256_HPP
#define DUST_RANDOM_XOSHIRO256_HPP

#include "dust/random/utils.hpp"
#include "dust/random/xoshiro_state.hpp"

// 64-bit Generators, state is 4 * uint64_t
//
// xoshiro256**  | https://prng.di.unimi.it/xoshiro256starstar.c
// xoshiro256++  | https://prng.di.unimi.it/xoshiro256plusplus.c
// xoshiro256+   | https://prng.di.unimi.it/xoshiro256plus.c

namespace dust {
namespace random {

using xoshiro256starstar =
  xoshiro_state<uint64_t, 4, scrambler::starstar>;
using xoshiro256plusplus =
  xoshiro_state<uint64_t, 4, scrambler::plusplus>;
using xoshiro256plus =
  xoshiro_state<uint64_t, 4, scrambler::plus>;

// Jump coefficients.
// * 2^128 calls to next
// * 2^128 non-overlapping subsequences
template <>
constexpr std::array<uint64_t, 4> jump_constants<xoshiro256starstar>() {
  return std::array<uint64_t, 4>{{
      0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
      0xa9582618e03fc9aa, 0x39abdc4529b1661c }};
}

template <>
constexpr std::array<uint64_t, 4> jump_constants<xoshiro256plusplus>() {
  return jump_constants<xoshiro256starstar>();
}

template <>
constexpr std::array<uint64_t, 4> jump_constants<xoshiro256plus>() {
  return jump_constants<xoshiro256starstar>();
}

// Long-jump coefficients
// * 2^192 calls to next
// * 2^64 starting points, each with 2^64 subsequences
template <>
constexpr
std::array<uint64_t, 4> long_jump_constants<xoshiro256starstar>() {
  return std::array<uint64_t, 4>{{
      0x76e15d3efefdcbbf, 0xc5004e441c522fb3,
      0x77710069854ee241, 0x39109bb02acbe635 }};
}

template <>
constexpr
std::array<uint64_t, 4> long_jump_constants<xoshiro256plusplus>() {
  return long_jump_constants<xoshiro256starstar>();
}

template <>
constexpr
std::array<uint64_t, 4> long_jump_constants<xoshiro256plus>() {
  return long_jump_constants<xoshiro256starstar>();
}

template <>
inline __host__ __device__ uint64_t next(xoshiro256starstar& state) {
  const uint64_t result = rotl(state[1] * 5, 7) * 9;
  const uint64_t t = state[1] << 17;
  state[2] ^= state[0];
  state[3] ^= state[1];
  state[1] ^= state[2];
  state[0] ^= state[3];
  state[2] ^= t;
  state[3] = rotl(state[3], 45);
  return result;
}

template <>
inline __host__ __device__ uint64_t next(xoshiro256plusplus& state) {
  const uint64_t result = rotl(state[0] + state[3], 23) + state[0];
  const uint64_t t = state[1] << 17;
  state[2] ^= state[0];
  state[3] ^= state[1];
  state[1] ^= state[2];
  state[0] ^= state[3];
  state[2] ^= t;
  state[3] = rotl(state[3], 45);
  return result;
}

template <>
inline __host__ __device__ uint64_t next(xoshiro256plus& state) {
  const uint64_t result = state[0] + state[3];
  const uint64_t t = state[1] << 17;
  state[2] ^= state[0];
  state[3] ^= state[1];
  state[1] ^= state[2];
  state[0] ^= state[3];
  state[2] ^= t;
  state[3] = rotl(state[3], 45);
  return result;
}

}
}

#endif
