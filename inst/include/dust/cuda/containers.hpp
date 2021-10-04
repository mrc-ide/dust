#ifndef DUST_CUDA_CONTAINERS_HPP
#define DUST_CUDA_CONTAINERS_HPP

#include <cstdint>
#include <cstddef>
#include <cstring> // memcpy
#include <new>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <dust/cuda/cuda.hpp>

namespace dust {
namespace cuda {


}

// The class from before, which is a light wrapper around a pointer
// This can be used within a kernel with copying memory. There is no
// way of telling if the pointer has been freed or not, so this must
// have a lifecycle that is shorter than the calling function.
template <typename T>
class interleaved {
public:
  DEVICE interleaved(T* data, size_t offset, size_t stride) :
    data_(data + offset),
    stride_(stride) {
  }

  template <typename Container>
  DEVICE interleaved(Container& data, size_t offset, size_t stride) :
    interleaved(data.data(), offset, stride) {
  }

  DEVICE T& operator[](size_t i) {
    return data_[i * stride_];
  }

  DEVICE const T& operator[](size_t i) const {
    return data_[i * stride_];
  }

  DEVICE interleaved<T> operator+(size_t by) {
    return interleaved(data_ + by * stride_, 0, stride_);
  }

  DEVICE const interleaved<T> operator+(size_t by) const {
    return interleaved(data_ + by * stride_, 0, stride_);
  }

private:
  // TODO: these can be set as const.
  T* data_;
  size_t stride_;
};

}

#endif
