#ifndef DUST_CONTAINERS_HPP
#define DUST_CONTAINERS_HPP

#include <cstdint>
#include <cstddef>
#include <cstdlib> // malloc
#include <cstring> // memcpy
#include <stdexcept>
#include <sstream>
#include <vector>

namespace dust {

// TODO: move to device_array, get_array, set_array names
// TODO: document the vibe of the CUDA bits
// TODO: move to C++ standard exceptions
// TOOD: can we use RAII for the CPU version?
template <typename T>
class DeviceArray {
public:
  // Default constructor
  DeviceArray() : data_(nullptr), size_(0) {
  }

  // Constructor to allocate empty memory
  DeviceArray(const size_t size) : size_(size) {
    data_ = (T*) std::malloc(size_ * sizeof(T));
    if (!data_) {
      throw std::bad_alloc();
    }
    std::memset(data_, 0, size_ * sizeof(T));
  }

  // Constructor from vector
  DeviceArray(const std::vector<T>& data) : size_(data.size()) {
    data_ = (T*) std::malloc(size_ * sizeof(T));
    if (!data_) {
      throw std::bad_alloc();
    }
    std::memcpy(data_, data.data(), size_ * sizeof(T));
  }

  // Copy
  DeviceArray(const DeviceArray& other) : size_(other.size_) {
    std::memcpy(data_, other.data_, size_ * sizeof(T));
  }

  // Copy assign
  DeviceArray& operator=(const DeviceArray& other) {
    if (this != &other) {
      size_ = other.size_;
      std::free(data_);
      std::memcpy(data_, other.data_, size_ * sizeof(T));
    }
    return *this;
  }

  // Move
  DeviceArray(DeviceArray&& other) : data_(nullptr), size_(0) {
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }

  // Move assign
  DeviceArray& operator=(DeviceArray&& other) {
    if (this != &other) {
      std::free(data_);
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~DeviceArray() {
    std::free(data_);
  }

  void getArray(std::vector<T>& dst) const {
    if (dst.size() > size_) {
      std::stringstream msg;
      msg << "Tried device to host copy with device array (" << size_ <<
        ") shorter than host array (" << dst.size() << ")";
      throw std::invalid_argument(msg.str());
    }
    std::memcpy(dst.data(), data_, dst.size() * sizeof(T));
  }

  void setArray(const std::vector<T>& src) {
    if (src.size() > size_) {
      std::stringstream msg;
      msg << "Tried host to device copy with host array (" << src.size() <<
        ") longer than device array (" << size_ << ")";
      throw std::invalid_argument(msg.str());
    } else {
      size_ = src.size();
    }
    std::memcpy(data_, src.data(), size_ * sizeof(T));
  }

  T* data() {
    return data_;
  }

  size_t size() const {
    return size_;
  }
private:
  T* data_;
  size_t size_;
};

// The class from before, which is a light wrapper around a pointer
// This can be used within a kernel with copying memory. There is no
// way of telling if the pointer has been freed or not, so this must
// have a lifecycle that is shorter than the calling function.
template <typename T>
class interleaved {
public:
  interleaved(T* data, size_t offset, size_t stride) :
    data_(data + offset),
    stride_(stride) {
  }

  template <typename Container>
  interleaved(Container& data, size_t offset, size_t stride) :
    interleaved(data.data(), offset, stride) {
  }

  T& operator[](size_t i) {
    return data_[i * stride_];
  }

  const T& operator[](size_t i) const {
    return data_[i * stride_];
  }

  interleaved<T> operator+(size_t by) {
    return interleaved(data_ + by * stride_, stride_);
  }

  const interleaved<T> operator+(size_t by) const {
    return interleaved(data_ + by * stride_, stride_);
  }

private:
  // TODO: these can be set as const.
  T* data_;
  size_t stride_;
};

}

#endif
