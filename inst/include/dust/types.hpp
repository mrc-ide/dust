#ifndef DUST_TYPES_HPP
#define DUST_TYPES_HPP

#include <numeric>
#include <sstream>
#include <vector>

#include <dust/filter_kernels.hpp>

namespace dust {

struct nothing {};
typedef nothing no_data;
typedef nothing no_internal;
typedef nothing no_shared;

// By default we do not support anything on the gpu. This name might
// change, but it does reflect our intent and it's likely that to work
// on a GPU the model will have to provide a number of things. If of
// those becomes a type (as with data, internal and shared) we could
// use the same approach as above.
template <typename T>
struct has_gpu_support : std::false_type {};

template <typename T>
using shared_ptr = std::shared_ptr<const typename T::shared_t>;

template <typename T>
struct pars_t {
  std::shared_ptr<const typename T::shared_t> shared;
  typename T::internal_t internal;

  pars_t(std::shared_ptr<const typename T::shared_t> shared_,
         typename T::internal_t internal_) :
    shared(shared_), internal(internal_) {
  }
  pars_t(typename T::shared_t shared_,
         typename T::internal_t internal_) :
    shared(std::make_shared<const typename T::shared_t>(shared_)),
    internal(internal_) {
  }
  pars_t(typename T::shared_t shared_) :
    pars_t(shared_, dust::nothing()) {
  }
  pars_t(typename T::internal_t internal_) :
    pars_t(dust::nothing(), internal_) {
  }
};

// Parameters for CUDA kernel launches
struct cuda_launch {
  size_t run_blockSize;
  size_t run_blockCount;
  size_t run_shared_size_bytes;
  bool run_L1;

  size_t compare_blockSize;
  size_t compare_blockCount;
  size_t compare_shared_size_bytes;
  bool compare_L1;

  size_t reorder_blockSize;
  size_t reorder_blockCount;

  size_t scatter_blockSize;
  size_t scatter_blockCount;

  size_t interval_blockSize;
  size_t interval_blockCount;
};

template <typename real_t>
struct device_state {
  void initialise(size_t n_particles, size_t n_state, size_t n_pars,
                  size_t n_shared_len_,
                  size_t n_internal_int, size_t n_internal_real,
                  size_t n_shared_int_, size_t n_shared_real_) {
    n_shared_len = n_shared_len_;
    n_shared_int = n_shared_int_;
    n_shared_real = n_shared_real_;
    const size_t n_rng = dust::rng_state_t<real_t>::size();
    y = dust::device_array<real_t>(n_state * n_particles);
    y_next = dust::device_array<real_t>(n_state * n_particles);
    y_selected = dust::device_array<real_t>(n_state * n_particles);
    internal_int = dust::device_array<int>(n_internal_int * n_particles);
    internal_real = dust::device_array<real_t>(n_internal_real * n_particles);
    shared_int = dust::device_array<int>(n_shared_int * n_shared_len);
    shared_real = dust::device_array<real_t>(n_shared_real * n_shared_len);
    rng = dust::device_array<uint64_t>(n_rng * n_particles);
    index = dust::device_array<char>(n_state * n_particles);
    n_selected = dust::device_array<int>(1);
    scatter_index = dust::device_array<size_t>(n_particles);
    compare_res = dust::device_array<real_t>(n_particles);
    resample_u = dust::device_array<real_t>(n_pars);
    set_cub_tmp();
  }
  void swap() {
    std::swap(y, y_next);
  }
  // TODO - use GPU templates
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

  size_t n_shared_len;
  size_t n_shared_int;
  size_t n_shared_real;
  dust::device_array<real_t> y;
  dust::device_array<real_t> y_next;
  dust::device_array<real_t> y_selected;
  dust::device_array<int> internal_int;
  dust::device_array<real_t> internal_real;
  dust::device_array<int> shared_int;
  dust::device_array<real_t> shared_real;
  dust::device_array<uint64_t> rng;
  dust::device_array<char> index;
  dust::device_array<int> n_selected;
  dust::device_array<void> select_tmp;
  dust::device_array<size_t> scatter_index;
  dust::device_array<real_t> compare_res;
  dust::device_array<real_t> resample_u;
};

template <typename real_t>
struct device_scan_state {
  void initialise(const size_t n_particles,
                  dust::device_array<real_t>& weights) {
    cum_weights = dust::device_array<real_t>(n_particles);
    set_cub_tmp(weights);
  }

  void set_cub_tmp(dust::device_array<real_t>& weights) {
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
  dust::device_array<real_t> cum_weights;
  dust::device_array<void> scan_tmp;
};

template <typename real_t>
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
  weights_ = dust::device_array<real_t>(n_particles_);
  cum_weights_ = dust::device_array<real_t>(n_particles_);
  weights_max_ = dust::device_array<real_t>(n_pars_);
  log_likelihood_step_ = dust::device_array<real_t>(n_pars_);

  pars_offsets_ = dust::device_array<int>(n_pars_ + 1);
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
  void scale_log_weights(dust::device_array<real_t>& log_likelihood) {
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
    dust::exp_weights<real_t><<<exp_blockCount,
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
    dust::weight_log_likelihood<real_t><<<weight_blockCount,
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
    std::vector<real_t> max_w(n_pars_, -dust::utils::infinity<real_t>());
    std::vector<real_t> host_w(n_particles_);
    weights_.get_array(host_w);
    for (size_t i = 0; i < n_pars_; ++i) {
      for (size_t j = 0; j < n_particles_each_; j++) {
        max_w[i] = std::max(host_w[i * n_particles_each_ + j], max_w[i]);
      }
    }
    weights_max_.set_array(max_w);
    dust::exp_weights<real_t>(
      n_particles_,
      n_pars_,
      weights_.data(),
      weights_max_.data()
    );
    std::vector<real_t> sum_w(n_pars_, 0);
    weights_.get_array(host_w);
    for (size_t i = 0; i < n_pars_; ++i) {
      for (size_t j = 0; j < n_particles_each_; j++) {
        sum_w[i] += host_w[i * n_particles_each_ + j];
      }
    }
    log_likelihood_step_.set_array(sum_w);
    dust::weight_log_likelihood<real_t>(
      n_pars_,
      n_particles_each_,
      log_likelihood.data(),
      log_likelihood_step_.data(),
      weights_max_.data()
    );
#endif
  }

  dust::device_array<real_t>& weights() {
    return weights_;
  }

private:
  size_t n_particles_;
  size_t n_pars_;
  size_t n_particles_each_;

  const size_t exp_blockSize, exp_blockCount;
  const size_t weight_blockSize, weight_blockCount;

  size_t max_tmp_bytes, sum_tmp_bytes;
  dust::device_array<real_t> weights_;
  dust::device_array<real_t> cum_weights_;
  dust::device_array<real_t> weights_max_;
  dust::device_array<real_t> log_likelihood_step_;
  dust::device_array<int> pars_offsets_;
  dust::device_array<void> max_tmp_;
  dust::device_array<void> sum_tmp_;

#ifdef __NVCC__
  dust::cuda::cuda_stream kernel_stream_;
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
                        typename T::real_t * shared_real) {
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
  const typename T::real_t * shared_real;
  const typename T::data_t * data;
};

class openmp_errors {
public:
  openmp_errors() : openmp_errors(0) {
  }
  openmp_errors(size_t len) :
    count(0), err(len), seen(len) {
  }

  void reset() {
    count = 0;
    std::fill(seen.begin(), seen.end(), false);
    std::fill(err.begin(), err.end(), "");
  }

  bool unresolved() const {
    return count > 0;
  }

  template <typename T>
  void capture(const T& e, size_t i) {
    err[i] = e.what();
    seen[i] = true;
  }

  void report(size_t n_max = 4) {
    count = std::accumulate(std::begin(seen), std::end(seen), 0);
    if (count == 0) {
      return;
    }

    std::stringstream msg;
    msg << count << " particles reported errors.";

    const size_t n_report = std::min(n_max, count);
    for (size_t i = 0, j = 0; i < seen.size() && j < n_report; ++i) {
      if (seen[i]) {
        msg << std::endl << "  - " << i + 1 << ": " << err[i];
        ++j;
      }
    }
    if (n_report < count) {
      msg << std::endl << "  - (and " << (count - n_report) << " more)";
    }

    throw std::runtime_error(msg.str());
  }

private:
  size_t count;
  std::vector<std::string> err;
  std::vector<bool> seen;
};

}

#endif
