#ifndef DUST_RANDOM_MATH_HPP
#define DUST_RANDOM_MATH_HPP

#include <cmath>
#include <limits>

#include "dust/random/cuda_compatibility.hpp"

// https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html
// https://stackoverflow.com/a/39409957

// There are really too many possible things that could go here,
// starting with a few that I need in mpoxspam to see if it makes any
// real difference.
namespace dust {
namespace math {

template <typename T>
__host__ __device__
T pow(T x, T y) {
  return std::pow(x, y);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float pow(float x, float y) {
  return ::powf(x, y);
}
#endif

template <typename T>
__host__ __device__
T exp(T x) {
  return std::exp(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float exp(float x) {
  return ::expf(x);
}
#endif

template <typename T>
__host__ __device__
T log(T x) {
  return std::log(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float log(float x) {
  return ::logf(x);
}
#endif

template <typename T>
__host__ __device__
T log1p(T x) {
  return std::log1p(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float log1p(float x) {
  return ::log1pf(x);
}
#endif

template <typename T>
__host__ __device__
T sqrt(T x) {
  return std::sqrt(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float sqrt(float x) {
  return ::sqrtf(x);
}
#endif

template <typename T>
__host__ __device__
T ceil(T x) {
  return std::ceil(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float ceil(float x) {
  return ::ceilf(x);
}
#endif

template <typename T>
__host__ __device__
T min(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
__device__
T max(T a, T b) {
  return a > b ? a : b;
}

template <typename real_type>
__host__ __device__ real_type lgamma(real_type x) {
  static_assert(std::is_floating_point<real_type>::value,
                "lgamma should only be used with real types");
  return std::lgamma(x);
}

#ifdef __CUDA_ARCH__
template <>
inline __device__ float lgamma(float x) {
  return ::lgammaf(x);
}

template <>
inline __device__ double lgamma(double x) {
  return ::lgamma(x);
}
#endif

template <typename real_type>
__host__ __device__
real_type lfactorial(int x) {
  return lgamma(static_cast<real_type>(x + 1));
}

template <typename T>
constexpr T epsilon = std::numeric_limits<T>::epsilon();

template <typename T>
constexpr T infinity = std::numeric_limits<T>::infinity();

template <typename T>
constexpr T max_value = std::numeric_limits<T>::max();

}
}

#endif
