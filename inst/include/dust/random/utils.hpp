#ifndef DUST_RANDOM_UTILS_HPP
#define DUST_RANDOM_UTILS_HPP

#include <dust/random/numeric.hpp>

namespace dust {
namespace random {

enum xoshiro_mode {STARSTAR, PLUSPLUS, PLUS};

// See for more background:
// https://github.com/mrc-ide/dust/issues/280
// https://mumble.net/~campbell/tmp/random_real.c
// https://doornik.com/research/randomdouble.pdf
template <typename T, typename U>
T int_to_real(U x);

#define TWOPOW32_INV (2.3283064e-10f)
#define TWOPOW32_INV_DOUBLE (2.3283064365386963e-10)
#define TWOPOW53_INV_DOUBLE (1.1102230246251565e-16)

template <>
inline HOSTDEVICE
double int_to_real(uint64_t x) {
  return (x >> 11) * TWOPOW53_INV_DOUBLE + (TWOPOW53_INV_DOUBLE / 2.0);
}

template <>
inline HOSTDEVICE
float int_to_real(uint64_t x) {
  uint32_t t = (uint32_t)(x >> 32);
  return t * TWOPOW32_INV + (TWOPOW32_INV / 2.0f);
}

template <>
inline HOSTDEVICE
float int_to_real(uint32_t x) {
  return x * TWOPOW32_INV + (TWOPOW32_INV / 2.0f);
}

inline HOSTDEVICE
uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

inline HOSTDEVICE
uint32_t rotl(const uint32_t x, int k) {
  return (x << k) | (x >> (32 - k));
}

// This is probably doable with sizeof(T) * 8 too?
template <typename T>
constexpr size_t bit_size();

template <>
inline HOST
constexpr size_t bit_size<uint32_t>() {
  return 32;
}

template <>
inline HOST
constexpr size_t bit_size<uint64_t>() {
  return 64;
}

template <typename T, size_t N, xoshiro_mode M>
std::array<T, N> jump_constants();

template <typename T, size_t N, xoshiro_mode M>
std::array<T, N> long_jump_constants();

inline HOST uint64_t splitmix64(uint64_t seed) {
  uint64_t z = (seed += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

}
}

#endif
