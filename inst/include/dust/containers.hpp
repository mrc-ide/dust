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

template <typename T>
class device_array {
public:
  // Default constructor
  device_array() : data_(nullptr), size_(0) {
  }

  // Constructor to allocate empty memory
  device_array(const size_t size) : size_(size) {
    data_ = (T*) std::malloc(size_ * sizeof(T));
    if (!data_) {
      // This is not tested (or easily testable without mocking) but
      // simple enough. This error will be caught by cpp11
      //
      // TODO: we might use `new` here which will throw automatically?
      throw std::bad_alloc(); // # nocov
    }
    std::memset(data_, 0, size_ * sizeof(T));
  }

  // Constructor from vector
  device_array(const std::vector<T>& data) : size_(data.size()) {
    data_ = (T*) std::malloc(size_ * sizeof(T));
    if (!data_) {
      throw std::bad_alloc();
    }
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
      std::free(data_);
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
      std::free(data_);
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~device_array() {
    std::free(data_);
  }

  void get_array(std::vector<T>& dst) const {
    // NOTE: there was error checking here making sure that dest.size() <= size_
    // but that's removed for now
    std::memcpy(dst.data(), data_, dst.size() * sizeof(T));
  }

  void set_array(const std::vector<T>& src) {
    // NOTE: there was error checking here making sure that src.size() == size_
    // but that's removed for now
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
