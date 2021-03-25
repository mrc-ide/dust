#ifndef DUST_UTILS_HPP
#define DUST_UTILS_HPP

namespace dust {
namespace utils {

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

template <typename T>
T lgamma(T x);

// We need this for the lgamma in rpois to work
#ifdef __CUDA_ARCH__
template <>
inline DEVICE float lgamma<float>(float x) {
  return ::lgammaf(x);
}

template <>
inline DEVICE double lgamma<double>(double x) {
  return ::lgamma(x);
}

#else
template <typename T>
T lgamma(T x) {
  return std::lgamma(x);
}
#endif

}
}

#endif
