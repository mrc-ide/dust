#ifndef DUST_UTILS_HPP
#define DUST_UTILS_HPP

#include <algorithm>
#include <numeric>
#include <vector>

namespace dust {
namespace utils {

// Translates index in y (full state) to index in y_selected
// Maps: 3 4 5 10 9 1 -> 1 2 3 5 4 0
template <typename T>
std::vector<size_t> sort_indexes(const T &v) {
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::vector<size_t> idx_offset = idx;

  // Find the idx order so that v[idx] is sort(v)
  // in the example this gives 5 0 1 2 4 3
  std::stable_sort(
    idx.begin(), idx.end(),
    [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  // Now sort these idx, as we want to these indices in the original order
  // in the example this gives 1 2 3 5 4 0
  std::stable_sort(
    idx_offset.begin(), idx_offset.end(),
    [&idx](size_t i1, size_t i2) { return idx[i1] < idx[i2]; });
  return idx_offset;
}

template <typename T, typename U, typename Enable = void>
size_t destride_copy(T dest, U& src, size_t at, size_t stride) {
  static_assert(!std::is_reference<T>::value,
                "destride_copy should only be used with reference types");
  size_t i;
  for (i = 0; at < src.size(); ++i, at += stride) {
    dest[i] = src[at];
  }
  return i;
}

template <typename T, typename U>
size_t stride_copy(T dest, U src, size_t at, size_t stride) {
  static_assert(!std::is_reference<T>::value,
                "stride_copy should only be used with reference types");
  dest[at] = src;
  return at + stride;
}

template <typename T, typename U>
size_t stride_copy(T dest, const std::vector<U>& src, size_t at,
                   size_t stride) {
  static_assert(!std::is_reference<T>::value,
                "stride_copy should only be used with reference types");
  for (size_t i = 0; i < src.size(); ++i, at += stride) {
    dest[at] = src[i];
  }
  return at;
}

template <typename T, typename U = T>
inline HOSTDEVICE T align_padding(const T offset, const U align) {
  T remainder = offset % align;
  return remainder ? align - remainder : 0;
}

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

// TODO: remove integer_max, epsilon, lgamma (random/numeric)

}
}

#endif
