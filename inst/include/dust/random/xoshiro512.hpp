#ifndef DUST_RANDOM_XOSHIRO512_HPP
#define DUST_RANDOM_XOSHIRO512_HPP

#include "dust/random/utils.hpp"
#include "dust/random/xoshiro_state.hpp"

// 64-bit Generators, state is 8 * uint64_t
//
// xoshiro512**  | https://prng.di.unimi.it/xoshiro512starstar.c
// xoshiro512++  | https://prng.di.unimi.it/xoshiro512plusplus.c
// xoshiro512+   | https://prng.di.unimi.it/xoshiro512plus.c

namespace dust {
namespace random {

using xoshiro512starstar_state =
  xoshiro_state<uint64_t, 8, scrambler::starstar>;
using xoshiro512plusplus_state =
  xoshiro_state<uint64_t, 8, scrambler::plusplus>;
using xoshiro512plus_state =
  xoshiro_state<uint64_t, 8, scrambler::plus>;

// Jump coefficients.
// * 2^256 calls to next
// * 2^256 non-overlapping subsequences
template <>
constexpr
std::array<uint64_t, 8> jump_constants<xoshiro512starstar_state>() {
  return std::array<uint64_t, 8>{{
      0x33ed89b6e7a353f9, 0x760083d7955323be,
      0x2837f2fbb5f22fae, 0x4b8c5674d309511c,
      0xb11ac47a7ba28c25, 0xf1be7667092bcc1c,
      0x53851efdb6df0aaf, 0x1ebbc8b23eaf25db
    }};      
}

template <>
constexpr
std::array<uint64_t, 8> jump_constants<xoshiro512plusplus_state>() {
  return jump_constants<xoshiro512starstar_state>();
}

template <>
constexpr
std::array<uint64_t, 8> jump_constants<xoshiro512plus_state>() {
  return jump_constants<xoshiro512starstar_state>();
}

// Long-jump coefficients
// * 2^384 calls to next
// * 2^128 starting points, each with 2^64 subsequences
template <>
constexpr
std::array<uint64_t, 8> long_jump_constants<xoshiro512starstar_state>() {
  return std::array<uint64_t, 8>{{
      0x11467fef8f921d28, 0xa2a819f2e79c8ea8,
      0xa8299fc284b3959a, 0xb4d347340ca63ee1,
      0x1cb0940bedbff6ce, 0xd956c5c4fa1f8e17,
      0x915e38fd4eda93bc, 0x5b3ccdfa5d7daca5      
    }};
}

template <>
constexpr
std::array<uint64_t, 8> long_jump_constants<xoshiro512plusplus_state>() {
  return long_jump_constants<xoshiro512starstar_state>();
}

template <>
constexpr
std::array<uint64_t, 8> long_jump_constants<xoshiro512plus_state>() {
  return long_jump_constants<xoshiro512starstar_state>();
}

template <>
inline __host__ __device__ uint64_t next(xoshiro512starstar_state& state) {
  const uint64_t result = rotl(state[1] * 5, 7) * 9;
  const uint64_t t = state[1] << 11;
  state[2] ^= state[0];
  state[5] ^= state[1];
  state[1] ^= state[2];
  state[7] ^= state[3];
  state[3] ^= state[4];
  state[4] ^= state[5];
  state[0] ^= state[6];
  state[6] ^= state[7];
  state[6] ^= t;
  state[7] = rotl(state[7], 21);
  return result;
}

template <>
inline __host__ __device__ uint64_t next(xoshiro512plusplus_state& state) {
  const uint64_t result = rotl(state[0] + state[2], 17) + state[2];
  const uint64_t t = state[1] << 11;
  state[2] ^= state[0];
  state[5] ^= state[1];
  state[1] ^= state[2];
  state[7] ^= state[3];
  state[3] ^= state[4];
  state[4] ^= state[5];
  state[0] ^= state[6];
  state[6] ^= state[7];
  state[6] ^= t;
  state[7] = rotl(state[7], 21);
  return result;
}

template <>
inline __host__ __device__ uint64_t next(xoshiro512plus_state& state) {
  const uint64_t result = state[0] + state[2];
  const uint64_t t = state[1] << 11;
  state[2] ^= state[0];
  state[5] ^= state[1];
  state[1] ^= state[2];
  state[7] ^= state[3];
  state[3] ^= state[4];
  state[4] ^= state[5];
  state[0] ^= state[6];
  state[6] ^= state[7];
  state[6] ^= t;
  state[7] = rotl(state[7], 21);
  return result;
}

}
}

#endif
