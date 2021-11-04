#ifndef DUST_RANDOM_NUMERIC_HPP
#define DUST_RANDOM_NUMERIC_HPP

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "dust/random/cuda.hpp"

namespace dust {
namespace random {
namespace utils {

#ifdef __NVCC__
template <typename T>
HOSTDEVICE T epsilon_nvcc();

template <>
inline DEVICE float epsilon_nvcc() {
  return FLT_EPSILON;
}

template <>
inline DEVICE double epsilon_nvcc() {
  return DBL_EPSILON;
}
#endif

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

// We need this for the lgamma in rpois to work
#ifdef __NVCC__
template <typename real_type>
real_type lgamma_nvcc(real_type x);

template <>
inline DEVICE float lgamma_nvcc(float x) {
  return ::lgammaf(x);
}

template <>
inline DEVICE double lgamma_nvcc(double x) {
  return ::lgamma(x);
}
#endif

template <typename real_type>
HOSTDEVICE real_type lgamma(real_type x) {
#ifdef __CUDA_ARCH__
  return lgamma_nvcc(x);
#else
  static_assert(std::is_floating_point<real_type>::value,
                "lgamma should only be used with real types");
  return std::lgamma(x);
#endif
}

#ifdef __NVCC__
template <typename real_type>
real_type infinity_nvcc();

template <>
inline DEVICE float infinity_nvcc() {
  return HUGE_VALF;
}

template <>
inline DEVICE double infinity_nvcc() {
  return HUGE_VAL;
}
#endif

template <typename real_type>
HOSTDEVICE real_type infinity() {
#ifdef __CUDA_ARCH__
  return infinity_nvcc<real_type>();
#else
  return std::numeric_limits<real_type>::infinity();
#endif
}

}
}
}

#endif
