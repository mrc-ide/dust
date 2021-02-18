#ifndef DUST_CONTAINERS_HPP
#define DUST_CONTAINERS_HPP

#include <cstdint>
#include <cstddef>
#include <cstring> // memcpy
#include <new>
#include <stdexcept>
#include <sstream>
#include <vector>

namespace dust {

template <typename T>
class device_array {
public:
  // Default constructor
  device_array() : data_(nullptr), size_(0) {
  }

  // Constructor to allocate empty memory
  device_array(const size_t size) : size_(size) {
    data_ = new T[size_];
    std::memset(data_, 0, size_ * sizeof(T));
  }

  // Constructor from vector
  device_array(const std::vector<T>& data) : size_(data.size()) {
    data_ = new T[size_];
    std::memcpy(data_, data.data(), size_ * sizeof(T));
  }

  // Copy
  device_array(const device_array& other) : size_(other.size_) {
    std::memcpy(data_, other.data_, size_ * sizeof(T));
  }

  // Copy assign
  device_array& operator=(const device_array& other) {
    if (this != &other) {
      size_ = other.size_;
      delete[] data_;
      // NOTE: the version in dustgpu lacked the allocation here. It's
      // very likely that we don't use this and we might replace body
      // with a static assert.
      data_ = new T[size_];
      std::memcpy(data_, other.data_, size_ * sizeof(T));
    }
    return *this;
  }

  // Move
  device_array(device_array&& other) : data_(nullptr), size_(0) {
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }

  // Move assign
  device_array& operator=(device_array&& other) {
    if (this != &other) {
      delete[] data_;
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~device_array() {
    delete[] data_;
  }

  void get_array(std::vector<T>& dst) const {
    std::memcpy(dst.data(), data_, dst.size() * sizeof(T));
  }

  void set_array(const std::vector<T>& src) {
    size_ = src.size();
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
    return interleaved(data_ + by * stride_, 0, stride_);
  }

  const interleaved<T> operator+(size_t by) const {
    return interleaved(data_ + by * stride_, 0, stride_);
  }

private:
  // TODO: these can be set as const.
  T* data_;
  size_t stride_;
};

}

#endif
