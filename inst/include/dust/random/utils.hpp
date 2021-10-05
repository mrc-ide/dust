#ifndef DUST_RANDOM_UTILS_HPP
#define DUST_RANDOM_UTILS_HPP

#include <dust/random/numeric.hpp>

namespace dust {
namespace random {

enum xoshiro_mode {STARSTAR, PLUSPLUS, PLUS};

template <typename T, typename U>
T int_to_real(U value);

template <>
inline HOSTDEVICE
double int_to_real(uint64_t value) {
  return double(value) / double(utils::uint64_max());
}

template <>
inline HOSTDEVICE
float int_to_real(uint64_t value) {
  return float(value) / float(utils::uint64_max());
}

template <>
inline HOSTDEVICE
float int_to_real(uint32_t value) {
  return float(value) / float(utils::uint32_max());
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
