#ifndef DUST_RANDOM_XOSHIRO128_HPP
#define DUST_RANDOM_XOSHIRO128_HPP

// 32-bit generators, state is 4 * uint32_t
//
//  xoshiro128**  | https://prng.di.unimi.it/xoshiro128starstar.c
//  xoshiro128++  | https://prng.di.unimi.it/xoshiro128plusplus.c
//  xoshiro128+   | https://prng.di.unimi.it/xoshiro128plus.c

namespace dust {
namespace random {

using xoshiro128starstar_state = xoshiro_state<uint32_t, 4, STARSTAR>;
using xoshiro128plusplus_state = xoshiro_state<uint32_t, 4, PLUSPLUS>;
using xoshiro128plus_state     = xoshiro_state<uint32_t, 4, PLUS>;

// Jump coefficients.
// * 2^64 calls to next
// * 2^64 non-overlapping subsequences
template <>
constexpr std::array<uint32_t, 4> jump_constants<uint32_t, 4, STARSTAR>() {
  return std::array<uint32_t, 4>{{
      0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b }};
}

template <>
constexpr std::array<uint32_t, 4> jump_constants<uint32_t, 4, PLUSPLUS>() {
  return jump_constants<uint32_t, 4, STARSTAR>();
}

template <>
constexpr std::array<uint32_t, 4> jump_constants<uint32_t, 4, PLUS>() {
  return jump_constants<uint32_t, 4, STARSTAR>();
}

// Long-jump coefficients.
// * 2^96 calls to next
// * 2^32 starting points, each with 2^32 subsequences
template <>
constexpr std::array<uint32_t, 4> long_jump_constants<uint32_t, 4, STARSTAR>() {
  return std::array<uint32_t, 4>{{
      0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662 }};
}

template <>
constexpr std::array<uint32_t, 4> long_jump_constants<uint32_t, 4, PLUSPLUS>() {
  return long_jump_constants<uint32_t, 4, STARSTAR>();
}

template <>
constexpr std::array<uint32_t, 4> long_jump_constants<uint32_t, 4, PLUS>() {
  return long_jump_constants<uint32_t, 4, STARSTAR>();
}

template <>
inline HOSTDEVICE uint32_t next(xoshiro128starstar_state& state) {
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
inline HOSTDEVICE uint32_t next(xoshiro128plusplus_state& state) {
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
inline HOSTDEVICE uint32_t next(xoshiro128plus_state& state) {
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
