#ifndef DUST_GPU_FILTER_STATE_HPP
#define DUST_GPU_FILTER_STATE_HPP

#include "dust/filter_state.hpp"
#include "dust/gpu/device_state.hpp"

namespace dust {
namespace filter {

template <typename real_type>
class filter_trajectories_device : public filter_trajectories_host<real_type> {
public:
  filter_trajectories_device() : page_locked(false) {
  }

#ifdef __NVCC__
  ~filter_trajectories_device() {
    pageable();
  }
#endif

  void resize(size_t n_state, size_t n_particles, size_t n_data) {
    pageable();
    this->n_state_ = n_state;
    this->n_particles_ = n_particles;
    this->n_data_ = n_data;
    this->offset_ = 0;

    assert_has_storage(this->n_state_ , this->n_particles_, this->n_data_ + 1);
    this->history_value.resize(this->n_state_ * this->n_particles_ * (this->n_data_ + 1));
    this->history_order.resize(this->n_particles_ * (this->n_data_ + 1));
    for (size_t i = 0; i < this->n_particles_; ++i) {
      this->history_order[i] = i;
    }

    history_value_swap = dust::gpu::device_array<real_type>(this->n_state_ * this->n_particles_);
    history_order_swap = dust::gpu::device_array<size_t>(this->n_particles_);

#ifdef __NVCC__
    // Page lock memory on host
    CUDA_CALL(cudaHostRegister(this->history_value.data(),
                               this->history_value.size() * sizeof(real_type),
                               cudaHostRegisterDefault));
    CUDA_CALL(cudaHostRegister(this->history_order.data(),
                               this->history_order.size() * sizeof(real_type),
                               cudaHostRegisterDefault));
#endif
    page_locked = true;
  }

  size_t value_offset() {
    return this->offset_ * this->n_state_ * this->n_particles_;
  }

  size_t order_offset() {
    return this->offset_ * this->n_particles_;
  }

  void store_values(dust::gpu::device_array<real_type>& state) {
    host_memory_stream_.sync();
    state.get_array(history_value_swap.data(), device_memory_stream_, true);
    device_memory_stream_.sync();
    history_value_swap.get_array(this->history_value.data() + value_offset(),
                                 host_memory_stream_, true);
  }

  void store_order(dust::gpu::device_array<size_t>& kappa) {
    host_memory_stream_.sync();
    kappa.get_array(history_order_swap.data(), device_memory_stream_, true);
    device_memory_stream_.sync();
    history_order_swap.get_array(this->history_order.data() + order_offset(),
                                 host_memory_stream_, true);
  }

  template <typename Iterator>
  void history(Iterator ret) const {
    std::vector<real_type> host_history = destride_history();
    this->particle_ancestry(ret, host_history.cbegin(), this->history_order.cbegin());
  }

private:
  filter_trajectories_device ( const filter_trajectories_device & ) = delete;
  filter_trajectories_device ( filter_trajectories_device && ) = delete;

  dust::gpu::device_array<real_type> history_value_swap;
  dust::gpu::device_array<size_t> history_order_swap;

  dust::gpu::cuda_stream device_memory_stream_;
  dust::gpu::cuda_stream host_memory_stream_;

  bool page_locked;

  std::vector<real_type> destride_history() const {
    std::vector<real_type> blocked_history(this->size());
    // Destride and copy into iterator
    // TODO openmp here?
    for (size_t i = 0; i < this->n_data_ + 1; ++i) {
      for (size_t j = 0; j < this->n_particles_; ++j) {
        for (size_t k = 0; k < this->n_state_; ++k) {
          blocked_history[i * (this->n_particles_ * this->n_state_) +
                          j * this->n_state_ +
                          k] = this->history_value[i * (this->n_particles_ * this->n_state_) +
                                                   j +
                                                   k * (this->n_particles_)];
        }
      }
    }
    return blocked_history;
  }

  void pageable() {
#ifdef __NVCC__
    // Make memory pageable again
    if (page_locked) {
      CUDA_CALL_NOTHROW(cudaHostUnregister(this->history_value.data()));
      CUDA_CALL_NOTHROW(cudaHostUnregister(this->history_order.data()));
    }
#endif
    page_locked = false;
  }
};

template <typename real_type>
class filter_snapshots_device : public filter_snapshots_host<real_type> {
public:
  filter_snapshots_device() : page_locked(false) {
  }

#ifdef __NVCC__
  ~filter_snapshots_device() {
    pageable();
  }
#endif

  void resize(size_t n_state, size_t n_particles, std::vector<size_t> steps) {
    pageable();
    this->n_state_ = n_state;
    this->n_particles_ = n_particles;
    this->n_steps_ = steps.size();
    this->offset_ = 0;
    this->steps_ = steps;

    assert_has_storage(this->n_state_ , this->n_particles_, this->n_steps_);
    this->state_.resize(this->n_state_ * this->n_particles_ * this->n_steps_);

    state_swap = dust::gpu::device_array<real_type>(this->n_state_ * this->n_particles_);

#ifdef __NVCC__
    // Page lock memory on host
    CUDA_CALL(cudaHostRegister(this->state_.data(),
                               this->state_.size() * sizeof(real_type),
                               cudaHostRegisterDefault));
#endif
    page_locked = true;
  }

  size_t value_offset() {
    return this->offset_ * this->n_state_ * this->n_particles_;
  }

  void store(dust::gpu::device_array<real_type>& state) {
    host_memory_stream_.sync();
    state.get_array(state_swap.data(), device_memory_stream_, true);
    device_memory_stream_.sync();
    state_swap.get_array(this->state_.data() + value_offset(),
                         host_memory_stream_, true);
  }

  template <typename Iterator>
  void history(Iterator dest) const {
    // Destride and copy into iterator
    // TODO: openmp here? collapse(2)
    for (size_t i = 0; i < this->n_steps_; ++i) {
      for (size_t j = 0; j < this->n_particles_; ++j) {
        for (size_t k = 0; k < this->n_state_; ++k) {
          *(dest +
            i * (this->n_particles_ * this->n_state_) +
            j * this->n_state_ +
            k) = this->state_[i * (this->n_particles_ * this->n_state_) +
                              j +
                              k * (this->n_particles_)];
        }
      }
    }
  }

private:
  filter_snapshots_device ( const filter_snapshots_device & ) = delete;
  filter_snapshots_device ( filter_snapshots_device && ) = delete;

  dust::gpu::device_array<real_type> state_swap;

  dust::gpu::cuda_stream device_memory_stream_;
  dust::gpu::cuda_stream host_memory_stream_;

  bool page_locked;

  void pageable() {
#ifdef __NVCC__
    // Make memory pageable again
    if (page_locked) {
      CUDA_CALL_NOTHROW(cudaHostUnregister(this->state_.data()));
    }
#endif
    page_locked = false;
  }
};

template <typename real_type>
struct filter_state_device {
  filter_trajectories_device<real_type> trajectories;
  filter_snapshots_device<real_type> snapshots;
};

}
}

#endif
