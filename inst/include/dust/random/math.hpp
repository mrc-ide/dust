// Generated by scripts/update_dust_math - do not edit
#ifndef DUST_RANDOM_MATH_HPP
#define DUST_RANDOM_MATH_HPP

#include <cmath>
#include <limits>

#include "dust/random/cuda_compatibility.hpp"

// https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html
// https://stackoverflow.com/a/39409957

namespace dust {
namespace math {

// Automatically generated functions; see scripts/update_dust_math in
// the dust source repo
template <typename T>
__host__ __device__
T round(T x) {
  return std::round(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float round(float x) {
  return ::roundf(x);
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
T floor(T x) {
  return std::floor(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float floor(float x) {
  return ::floorf(x);
}
#endif

template <typename T>
__host__ __device__
T trunc(T x) {
  return std::trunc(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float trunc(float x) {
  return ::truncf(x);
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
T expm1(T x) {
  return std::expm1(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float expm1(float x) {
  return ::expm1f(x);
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
T log2(T x) {
  return std::log2(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float log2(float x) {
  return ::log2f(x);
}
#endif

template <typename T>
__host__ __device__
T log10(T x) {
  return std::log10(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float log10(float x) {
  return ::log10f(x);
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
T cos(T x) {
  return std::cos(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float cos(float x) {
  return ::cosf(x);
}
#endif

template <typename T>
__host__ __device__
T sin(T x) {
  return std::sin(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float sin(float x) {
  return ::sinf(x);
}
#endif

template <typename T>
__host__ __device__
T tan(T x) {
  return std::tan(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float tan(float x) {
  return ::tanf(x);
}
#endif

template <typename T>
__host__ __device__
T acos(T x) {
  return std::acos(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float acos(float x) {
  return ::acosf(x);
}
#endif

template <typename T>
__host__ __device__
T asin(T x) {
  return std::asin(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float asin(float x) {
  return ::asinf(x);
}
#endif

template <typename T>
__host__ __device__
T atan(T x) {
  return std::atan(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float atan(float x) {
  return ::atanf(x);
}
#endif

template <typename T>
__host__ __device__
T cosh(T x) {
  return std::cosh(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float cosh(float x) {
  return ::coshf(x);
}
#endif

template <typename T>
__host__ __device__
T sinh(T x) {
  return std::sinh(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float sinh(float x) {
  return ::sinhf(x);
}
#endif

template <typename T>
__host__ __device__
T tanh(T x) {
  return std::tanh(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float tanh(float x) {
  return ::tanhf(x);
}
#endif

template <typename T>
__host__ __device__
T acosh(T x) {
  return std::acosh(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float acosh(float x) {
  return ::acoshf(x);
}
#endif

template <typename T>
__host__ __device__
T asinh(T x) {
  return std::asinh(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float asinh(float x) {
  return ::asinhf(x);
}
#endif

template <typename T>
__host__ __device__
T atanh(T x) {
  return std::atanh(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float atanh(float x) {
  return ::atanhf(x);
}
#endif

template <typename T>
__host__ __device__
T atan2(T x, T y) {
  return std::atan2(x, y);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float atan2(float x, float y) {
  return ::atan2f(x, y);
}
#endif

// Functions written by hand because they don't generalise usefully

// Special beacuse we nee
template <typename T, typename U>
__host__ __device__
T pow(T x, U y) {
  return std::pow(x, y);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float pow(float x, float y) {
  return ::powf(x, y);
}

template <>
__device__
inline float pow(float x, int y) {
  // could possibly use fast power here (see binomial.hpp)
  return ::powf(x, static_cast<float>(y));
}
#endif

// Special because name does not follow pattern:
template <typename T>
__host__ __device__
T abs(T x) {
  return std::abs(x);
}

#ifdef __CUDA_ARCH__
template <>
__device__
inline float abs(float x) {
  return ::fabsf(x);
}
#endif

template <typename T>
__host__ __device__
T min(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
__host__ __device__
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

}
}

#endif
