#pragma once

#include <cstdint>
#include <limits>

namespace dust {
namespace random {
namespace utils {

template <typename T>
HOSTDEVICE T epsilon() {
#ifdef __CUDA_ARCH__
  return epsilon_nvcc<T>();
#else
  return std::numeric_limits<T>::epsilon();
#endif
}

inline HOSTDEVICE int integer_max() {
#ifdef __CUDA_ARCH__
  return INT_MAX;
#else
  return std::numeric_limits<int>::max();
#endif
}

inline HOSTDEVICE uint64_t uint64_max() {
#ifdef __CUDA_ARCH__
  return ULLONG_MAX;
#else
  return std::numeric_limits<uint64_t>::max();
#endif
}

inline HOSTDEVICE uint64_t uint32_max() {
#ifdef __CUDA_ARCH__
  return ULONG_MAX;
#else
  return std::numeric_limits<uint32_t>::max();
#endif
}

// We need this for the lgamma in rpois to work
#ifdef __NVCC__
template <typename real_t>
real_t lgamma_nvcc(real_t x);

template <>
inline DEVICE float lgamma_nvcc(float x) {
  return ::lgammaf(x);
}

template <>
inline DEVICE double lgamma_nvcc(double x) {
  return ::lgamma(x);
}
#endif

template <typename real_t>
HOSTDEVICE real_t lgamma(real_t x) {
#ifdef __CUDA_ARCH__
  return lgamma_nvcc(x);
#else
  static_assert(std::is_floating_point<real_t>::value,
                "lgamma should only be used with real types");
  return std::lgamma(x);
#endif
}

#ifdef __NVCC__
template <typename real_t>
real_t infinity_nvcc();

template <>
inline DEVICE float infinity_nvcc() {
  return HUGE_VALF;
}

template <>
inline DEVICE double infinity_nvcc() {
  return HUGE_VAL;
}
#endif

template <typename real_t>
HOSTDEVICE real_t infinity() {
#ifdef __CUDA_ARCH__
  return infinity_nvcc<real_t>();
#else
  return std::numeric_limits<real_t>::infinity();
#endif
}

}
}
}
