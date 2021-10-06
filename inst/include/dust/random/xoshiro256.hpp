#ifndef DUST_RANDOM_XOSHIRO256_HPP
#define DUST_RANDOM_XOSHIRO256_HPP

// 64-bit Generators, state is 4 * uint64_t
//
// xoshiro256**  | https://prng.di.unimi.it/xoshiro256starstar.c
// xoshiro256++  | https://prng.di.unimi.it/xoshiro256plusplus.c
// xoshiro256+   | https://prng.di.unimi.it/xoshiro256plus.c

namespace dust {
namespace random {

using xoshiro256starstar_state = xoshiro_state<uint64_t, 4, STARSTAR>;
using xoshiro256plusplus_state = xoshiro_state<uint64_t, 4, PLUSPLUS>;
using xoshiro256plus_state     = xoshiro_state<uint64_t, 4, PLUS>;

// Jump coefficients.
// * 2^128 calls to next
// * 2^128 non-overlapping subsequences
template <>
constexpr std::array<uint64_t, 4> jump_constants<uint64_t, 4, STARSTAR>() {
  return std::array<uint64_t, 4>{{
      0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
      0xa9582618e03fc9aa, 0x39abdc4529b1661c }};
}

template <>
constexpr std::array<uint64_t, 4> jump_constants<uint64_t, 4, PLUSPLUS>() {
  return jump_constants<uint64_t, 4, STARSTAR>();
}

template <>
constexpr std::array<uint64_t, 4> jump_constants<uint64_t, 4, PLUS>() {
  return jump_constants<uint64_t, 4, STARSTAR>();
}

// Long-jump coefficients
// * 2^192 calls to next
// * 2^64 starting points, each with 2^64 subsequences
template <>
constexpr std::array<uint64_t, 4> long_jump_constants<uint64_t, 4, STARSTAR>() {
  return std::array<uint64_t, 4>{{
      0x76e15d3efefdcbbf, 0xc5004e441c522fb3,
      0x77710069854ee241, 0x39109bb02acbe635 }};
}

template <>
constexpr std::array<uint64_t, 4> long_jump_constants<uint64_t, 4, PLUSPLUS>() {
  return long_jump_constants<uint64_t, 4, STARSTAR>();
}

template <>
constexpr std::array<uint64_t, 4> long_jump_constants<uint64_t, 4, PLUS>() {
  return long_jump_constants<uint64_t, 4, STARSTAR>();
}

template <>
inline HOSTDEVICE uint64_t next(xoshiro256starstar_state& state) {
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
inline HOSTDEVICE uint64_t next(xoshiro256plusplus_state& state) {
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
inline HOSTDEVICE uint64_t next(xoshiro256plus_state& state) {
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
