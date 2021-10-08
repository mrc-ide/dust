#ifndef DUST_CUDA_TYPES_HPP
#define DUST_CUDA_TYPES_HPP

#include <cstring>
#include <numeric>
#include <sstream>
#include <vector>

#include <dust/types.hpp>

#include <dust/cuda/filter_kernels.hpp>
#include <dust/cuda/utils.hpp>

#include <dust/random/numeric.hpp>

namespace dust {
namespace cuda {

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

template <typename T>
class device_array {
public:
  // Default constructor
  device_array() : data_(nullptr), size_(0) {
  }

  // Constructor to allocate empty memory
  device_array(const size_t size) : size_(size) {
#ifdef __NVCC__
    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
    CUDA_CALL(cudaMemset(data_, 0, size_ * sizeof(T)));
#else
    data_ = new T[size_];
    std::memset(data_, 0, size_ * sizeof(T));
#endif
  }

  // Constructor from vector
  device_array(const std::vector<T>& data) : size_(data.size()) {
#ifdef __NVCC__
    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
    CUDA_CALL(cudaMemcpy(data_, data.data(), size_ * sizeof(T),
                         cudaMemcpyDefault));
#else
    data_ = new T[size_];
    std::memcpy(data_, data.data(), size_ * sizeof(T));
#endif
  }

  // Copy
  device_array(const device_array& other) : size_(other.size_) {
#ifdef __NVCC__
    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
    CUDA_CALL(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                         cudaMemcpyDefault));
#else
    data_ = new T[size_];
    std::memcpy(data_, other.data_, size_ * sizeof(T));
#endif
  }

  // Copy assign
  device_array& operator=(const device_array& other) {
    if (this != &other) {
      size_ = other.size_;
#ifdef __NVCC__
      CUDA_CALL(cudaFree(data_));
      CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
      CUDA_CALL(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                           cudaMemcpyDefault));
#else
      delete[] data_;
      data_ = new T[size_];
      std::memcpy(data_, other.data_, size_ * sizeof(T));
#endif
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
#ifdef __NVCC__
      CUDA_CALL(cudaFree(data_));
#else
      delete[] data_;
#endif
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~device_array() {
#ifdef __NVCC__
    CUDA_CALL_NOTHROW(cudaFree(data_));
#else
    delete[] data_;
#endif
  }

  void get_array(std::vector<T>& dst, const bool async = false) const {
#ifdef __NVCC__
    if (async) {
      CUDA_CALL(cudaMemcpyAsync(dst.data(), data_, dst.size() * sizeof(T),
                          cudaMemcpyDefault));
    } else {
      CUDA_CALL(cudaMemcpy(dst.data(), data_, dst.size() * sizeof(T),
                          cudaMemcpyDefault));
    }
#else
    std::memcpy(dst.data(), data_, dst.size() * sizeof(T));
#endif
  }

  void get_array(T * dst, cuda_stream& stream, const bool async = false) const {
#ifdef __NVCC__
    if (async) {
      CUDA_CALL(cudaMemcpyAsync(dst, data_, size() * sizeof(T),
                          cudaMemcpyDefault, stream.stream()));
    } else {
      CUDA_CALL(cudaMemcpy(dst, data_, size() * sizeof(T),
                          cudaMemcpyDefault));
    }
#else
    std::memcpy(dst, data_, size() * sizeof(T));
#endif
  }

  // General method to set the device array, allowing src to be written
  // into the device data_ array starting at dst_offset
  void set_array(const T* src, const size_t src_size,
                 const size_t dst_offset, const bool async = false) {
#ifdef __NVCC__
    if (async) {
      CUDA_CALL(cudaMemcpyAsync(data_ + dst_offset, src,
                          src_size * sizeof(T), cudaMemcpyDefault));
    } else {
      CUDA_CALL(cudaMemcpy(data_ + dst_offset, src,
                          src_size * sizeof(T), cudaMemcpyDefault));
    }
#else
    std::memcpy(data_ + dst_offset, src, src_size * sizeof(T));
#endif
  }

  // Specialised form to set the device array, writing all of src into
  // the device data_
  void set_array(const std::vector<T>& src, const bool async = false) {
    size_ = src.size();
#ifdef __NVCC__
    if (async) {
      CUDA_CALL(cudaMemcpyAsync(data_, src.data(), size_ * sizeof(T),
                          cudaMemcpyDefault));
    } else {
      CUDA_CALL(cudaMemcpy(data_, src.data(), size_ * sizeof(T),
                          cudaMemcpyDefault));
    }
#else
    std::memcpy(data_, src.data(), size_ * sizeof(T));
#endif
  }

  void set_array(T * dst, cuda_stream& stream, const bool async = false) const {
#ifdef __NVCC__
    if (async) {
      CUDA_CALL(cudaMemcpyAsync(data_, dst, size() * sizeof(T),
                                cudaMemcpyDefault, stream.stream()));
    } else {
      CUDA_CALL(cudaMemcpy(data_, dst, size() * sizeof(T),
                           cudaMemcpyDefault));
    }
#else
    std::memcpy(data_, dst, size() * sizeof(T));
#endif
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

// Specialisation of the above for void* memory needed by some cub functions
// Construct once and use set_size() to modify
// Still using malloc/free instead of new and delete, as void type problematic
template <>
class device_array<void> {
public:
  // Default constructor
  device_array() : data_(nullptr), size_(0) {}
  // Constructor to allocate empty memory
  device_array(const size_t size) : size_(size) {
    if (size_ > 0) {
#ifdef __NVCC__
      CUDA_CALL(cudaMalloc((void**)&data_, size_));
#else
      data_ = (void*) std::malloc(size_);
      if (!data_) {
        throw std::bad_alloc();
      }
#endif
    }
  }
  ~device_array() {
#ifdef __NVCC__
    CUDA_CALL_NOTHROW(cudaFree(data_));
#else
    std::free(data_);
#endif
  }
  void set_size(size_t size) {
    size_ = size;
#ifdef __NVCC__
    CUDA_CALL(cudaFree(data_));
    if (size_ > 0) {
      CUDA_CALL(cudaMalloc((void**)&data_, size_));
    } else {
      data_ = nullptr;
    }
#else
    std::free(data_);
    if (size_ > 0) {
      data_ = (void*) std::malloc(size_);
      if (!data_) {
        throw std::bad_alloc();
      }
    } else {
      data_ = nullptr;
    }
#endif
  }
  void* data() {
    return data_;
  }
  size_t size() const {
    return size_;
  }

private:
  device_array ( const device_array<void> & ) = delete;
  device_array ( device_array<void> && ) = delete;

  void* data_;
  size_t size_;
};

template <typename real_type, typename rng_state_type>
struct device_state {
  void initialise(size_t n_particles, size_t n_state, size_t n_pars,
                  size_t n_shared_len_,
                  size_t n_internal_int, size_t n_internal_real,
                  size_t n_shared_int_, size_t n_shared_real_) {
    n_shared_len = n_shared_len_;
    n_shared_int = n_shared_int_;
    n_shared_real = n_shared_real_;
    const size_t n_rng = rng_state_type::size();
    y = device_array<real_type>(n_state * n_particles);
    y_next = device_array<real_type>(n_state * n_particles);
    internal_int = device_array<int>(n_internal_int * n_particles);
    internal_real = device_array<real_type>(n_internal_real * n_particles);
    shared_int = device_array<int>(n_shared_int * n_shared_len);
    shared_real = device_array<real_type>(n_shared_real * n_shared_len);
    rng = device_array<typename rng_state_type::int_type>(n_rng * n_particles);
    index = device_array<char>(n_state * n_particles);
    n_selected = device_array<int>(1);
    scatter_index = device_array<size_t>(n_particles);
    compare_res = device_array<real_type>(n_particles);
    resample_u = device_array<real_type>(n_pars);
    set_cub_tmp();
  }
  void swap() {
    std::swap(y, y_next);
  }
  void swap_selected() {
    std::swap(y_selected, y_selected_swap);
  }
  void set_cub_tmp() {
#ifdef __NVCC__
    // Free the array before running cub function below
    size_t tmp_bytes = 0;
    select_tmp.set_size(tmp_bytes);
    cub::DeviceSelect::Flagged(select_tmp.data(),
                               tmp_bytes,
                               y.data(),
                               index.data(),
                               y_selected.data(),
                               n_selected.data(),
                               y.size());
    select_tmp.set_size(tmp_bytes);
#endif
  }
  void set_device_index(const std::vector<size_t>& host_index,
                        const size_t n_particles,
                        const size_t n_state_full) {
    const size_t n_state = host_index.size();
    y_selected = device_array<real_type>(n_state * n_particles);
    y_selected_swap = device_array<real_type>(n_state * n_particles);

    // e.g. 4 particles with 3 states ABC stored on device as
    // [1_A, 2_A, 3_A, 4_A, 1_B, 2_B, 3_B, 4_B, 1_C, 2_C, 3_C, 4_C]
    // e.g. index [1, 3] would be
    // [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1] bool index on interleaved state
    // i.e. initialise to zero and copy 1 np times, at each offset given in
    // index
    std::vector<char> bool_idx(n_state_full * n_particles, 0);
    for (auto idx_pos = host_index.cbegin(); idx_pos != host_index.cend(); idx_pos++) {
      std::fill_n(bool_idx.begin() + (*idx_pos * n_particles), n_particles, 1);
    }
    index.set_array(bool_idx);

    std::vector<size_t> index_scatter = utils::sort_indexes(host_index);
    index_state_scatter = device_array<size_t>(n_state);
    index_state_scatter.set_array(index_scatter);

    set_cub_tmp();
  }

  size_t n_shared_len;
  size_t n_shared_int;
  size_t n_shared_real;
  device_array<real_type> y;
  device_array<real_type> y_next;
  device_array<real_type> y_selected;
  device_array<real_type> y_selected_swap;
  device_array<int> internal_int;
  device_array<real_type> internal_real;
  device_array<int> shared_int;
  device_array<real_type> shared_real;
  device_array<typename rng_state_type::int_type> rng;
  device_array<char> index;
  device_array<size_t> index_state_scatter;
  device_array<int> n_selected;
  device_array<void> select_tmp;
  device_array<size_t> scatter_index;
  device_array<real_type> compare_res;
  device_array<real_type> resample_u;
};

template <typename real_type>
struct device_scan_state {
  void initialise(const size_t n_particles,
                  device_array<real_type>& weights) {
    cum_weights = device_array<real_type>(n_particles);
    set_cub_tmp(weights);
  }

  void set_cub_tmp(device_array<real_type>& weights) {
#ifdef __NVCC__
    tmp_bytes = 0;
    scan_tmp.set_size(tmp_bytes);
    cub::DeviceScan::InclusiveSum(scan_tmp.data(),
                                  tmp_bytes,
                                  weights.data(),
                                  cum_weights.data(),
                                  cum_weights.size());
    scan_tmp.set_size(tmp_bytes);
#endif
  }

  size_t tmp_bytes;
  device_array<real_type> cum_weights;
  device_array<void> scan_tmp;
};

template <typename real_type>
class device_weights {
public:
  device_weights(const size_t n_particles, const size_t n_pars)
    : n_particles_(n_particles), n_pars_(n_pars),
      n_particles_each_(n_particles_ / n_pars_),
      exp_blockSize(128),
      exp_blockCount((n_particles_ + exp_blockSize - 1) / exp_blockSize),
      weight_blockSize(64),
      weight_blockCount((n_pars + weight_blockSize - 1) / weight_blockSize) {
  // Set up storage
  weights_ = device_array<real_type>(n_particles_);
  cum_weights_ = device_array<real_type>(n_particles_);
  weights_max_ = device_array<real_type>(n_pars_);
  log_likelihood_step_ = device_array<real_type>(n_pars_);

  pars_offsets_ = device_array<int>(n_pars_ + 1);
  std::vector<int> offsets(n_pars_ + 1);
  for (size_t i = 0; i < n_pars + 1; ++i) {
    offsets[i] = i * n_particles_each_;
  }
  pars_offsets_.set_array(offsets);

  // Allocate memory for cub
#ifdef __NVCC__
  if (n_pars_ > 1) {
    cub::DeviceSegmentedReduce::Max(max_tmp_.data(),
                                    max_tmp_bytes,
                                    weights_.data(),
                                    weights_max_.data(),
                                    n_pars_,
                                    pars_offsets_.data(),
                                    pars_offsets_.data() + 1);
  } else {
    cub::DeviceReduce::Max(max_tmp_.data(),
                            max_tmp_bytes,
                            weights_.data(),
                            weights_max_.data(),
                            n_particles_);
  }
  max_tmp_.set_size(max_tmp_bytes);

  if (n_pars_ > 1) {
    cub::DeviceSegmentedReduce::Sum(sum_tmp_.data(),
                                    sum_tmp_bytes,
                                    weights_.data(),
                                    log_likelihood_step_.data(),
                                    n_pars_,
                                    pars_offsets_.data(),
                                    pars_offsets_.data() + 1);
  } else {
    cub::DeviceReduce::Sum(sum_tmp_.data(),
                            sum_tmp_bytes,
                            weights_.data(),
                            log_likelihood_step_.data(),
                            n_particles_);
  }
  sum_tmp_.set_size(sum_tmp_bytes);
#endif
  }

  // CUDA version of log-sum-exp trick
  void scale_log_weights(device_array<real_type>& log_likelihood) {
#ifdef __NVCC__
    // Scale log-weights. First calculate the max
    if (n_pars_ > 1) {
      cub::DeviceSegmentedReduce::Max(max_tmp_.data(),
                                      max_tmp_bytes,
                                      weights_.data(),
                                      weights_max_.data(),
                                      n_pars_,
                                      pars_offsets_.data(),
                                      pars_offsets_.data() + 1,
                                      kernel_stream_.stream());
    } else {
      cub::DeviceReduce::Max(max_tmp_.data(),
                             max_tmp_bytes,
                             weights_.data(),
                             weights_max_.data(),
                             n_particles_,
                             kernel_stream_.stream());
    }
    kernel_stream_.sync();
    // Then exp
    dust::exp_weights<real_type><<<exp_blockCount,
                                exp_blockSize,
                                0,
                                kernel_stream_.stream()>>>(
      n_particles_,
      n_pars_,
      weights_.data(),
      weights_max_.data()
    );
    kernel_stream_.sync();
    // Then sum
    if (n_pars_ > 1) {
      cub::DeviceSegmentedReduce::Sum(sum_tmp_.data(),
                                      sum_tmp_bytes,
                                      weights_.data(),
                                      log_likelihood_step_.data(),
                                      n_pars_,
                                      pars_offsets_.data(),
                                      pars_offsets_.data() + 1,
                                      kernel_stream_.stream());
    } else {
      cub::DeviceReduce::Sum(sum_tmp_.data(),
                             sum_tmp_bytes,
                             weights_.data(),
                             log_likelihood_step_.data(),
                             n_particles_,
                             kernel_stream_.stream());
    }
    kernel_stream_.sync();
    // Finally log and add max
    dust::weight_log_likelihood<real_type><<<weight_blockCount,
                                             weight_blockSize,
                                             0,
                                             kernel_stream_.stream()>>>(
      n_pars_,
      n_particles_each_,
      log_likelihood.data(),
      log_likelihood_step_.data(),
      weights_max_.data()
    );
    kernel_stream_.sync();
#else
    std::vector<real_type> max_w(n_pars_,
                                 -dust::random::utils::infinity<real_type>());
    std::vector<real_type> host_w(n_particles_);
    weights_.get_array(host_w);
    for (size_t i = 0; i < n_pars_; ++i) {
      for (size_t j = 0; j < n_particles_each_; j++) {
        max_w[i] = std::max(host_w[i * n_particles_each_ + j], max_w[i]);
      }
    }
    weights_max_.set_array(max_w);
    dust::exp_weights<real_type>(
      n_particles_,
      n_pars_,
      weights_.data(),
      weights_max_.data()
    );
    std::vector<real_type> sum_w(n_pars_, 0);
    weights_.get_array(host_w);
    for (size_t i = 0; i < n_pars_; ++i) {
      for (size_t j = 0; j < n_particles_each_; j++) {
        sum_w[i] += host_w[i * n_particles_each_ + j];
      }
    }
    log_likelihood_step_.set_array(sum_w);
    dust::weight_log_likelihood<real_type>(
      n_pars_,
      n_particles_each_,
      log_likelihood.data(),
      log_likelihood_step_.data(),
      weights_max_.data()
    );
#endif
  }

  device_array<real_type>& weights() {
    return weights_;
  }

private:
  size_t n_particles_;
  size_t n_pars_;
  size_t n_particles_each_;

  const size_t exp_blockSize, exp_blockCount;
  const size_t weight_blockSize, weight_blockCount;

  size_t max_tmp_bytes, sum_tmp_bytes;
  device_array<real_type> weights_;
  device_array<real_type> cum_weights_;
  device_array<real_type> weights_max_;
  device_array<real_type> log_likelihood_step_;
  device_array<int> pars_offsets_;
  device_array<void> max_tmp_;
  device_array<void> sum_tmp_;

#ifdef __NVCC__
  cuda_stream kernel_stream_;
#endif
};

// We need to compute the size of space required for integers and
// reals on the device, per particle. Because we work on the
// requirement that every particle has the same dimension we pass an
// arbitrary set of shared parameters (really the first) to
// device_internal_size. The underlying model can overload this template
// for either real or int types and return the length of data
// required.
template <typename T>
size_t device_internal_int_size(typename dust::shared_ptr<T> shared) {
  return 0;
}

template <typename T>
size_t device_internal_real_size(typename dust::shared_ptr<T> shared) {
  return 0;
}

template <typename T>
size_t device_shared_int_size(typename dust::shared_ptr<T> shared) {
  return 0;
}

template <typename T>
size_t device_shared_real_size(typename dust::shared_ptr<T> shared) {
  return 0;
}

template <typename T>
void device_shared_copy(typename dust::shared_ptr<T> shared,
                        int * shared_int,
                        typename T::real_type * shared_real) {
}

template <typename T>
T* shared_copy(T* dest, const std::vector<T>& src) {
  memcpy(dest, src.data(), src.size() * sizeof(T));
  return dest + src.size();
}

template <typename T>
T* shared_copy(T* dest, const T src) {
  *dest = src;
  return dest + 1;
}

template <typename T>
struct device_ptrs {
  const int * shared_int;
  const typename T::real_type * shared_real;
  const typename T::data_type * data;
};

template <typename rng_state_type>
DEVICE
rng_state_type get_rng_state(const interleaved<typename rng_state_type::int_type>& full_rng_state) {
  rng_state_type rng_state;
  for (size_t i = 0; i < rng_state.size(); i++) {
    rng_state.state[i] = full_rng_state[i];
  }
  return rng_state;
}

// Write state into global memory
template <typename rng_state_type>
DEVICE
void put_rng_state(rng_state_type& rng_state,
                   interleaved<typename rng_state_type::int_type>& full_rng_state) {
  for (size_t i = 0; i < rng_state.size(); i++) {
    full_rng_state[i] = rng_state.state[i];
  }
}

}
}

#endif
