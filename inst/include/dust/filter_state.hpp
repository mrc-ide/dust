#ifndef DUST_FILTER_STATE_HPP
#define DUST_FILTER_STATE_HPP

#include <cstddef>

namespace dust {
namespace filter {

template <typename real_t>
class filter_trajectories_host {
public:
  filter_trajectories_host() {
  }

  void resize(size_t n_state, size_t n_particles, size_t n_data) {
    n_state_ = n_state;
    n_particles_ = n_particles;
    n_data_ = n_data;
    offset_ = 0;
    history_value.resize(n_state_ * n_particles_ * (n_data_ + 1));
    history_order.resize(n_particles_ * (n_data_ + 1));
    for (size_t i = 0; i < n_particles_; ++i) {
      history_order[i] = i;
    }
  }

  size_t size() const {
    return history_value.size();;
  }

  void advance() {
    offset_++;
  }

  typename std::vector<real_t>::iterator value_iterator() {
    return history_value.begin() + offset_ * n_state_ * n_particles_;
  }

  typename std::vector<size_t>::iterator order_iterator() {
    return history_order.begin() + offset_ * n_particles_;
  }

  std::vector<real_t> history() const {
    std::vector<real_t> ret(size());
    history(ret.begin());
    return ret;
  }

  // This is a particularly unpleasant bit of bookkeeping and is
  // adapted from mcstate (see the helper files in tests for a
  // translation of the the code). As we proceed we store the values
  // of particles *before* resampling and then we store the index used
  // in resampling. We do not resample all the history at each
  // resample as that is prohibitively expensive.
  //
  // So to output sensible history we start with a particle and we
  // look to see where it "came from" in the previous step
  // (history_index) and propagate this backward in time to
  // reconstruct what is in effect a multifurcating tree.
  // This is analogous to the particle ancestor concept in the
  // particle filter literature.
  //
  // It's possible we could do this more efficiently for some subset
  // of particles too (give me the history of just one particle) by
  // breaking the function before the loop over 'k'.
  //
  // Note that we treat history_order and history_value as read-only
  // though this process so one could safely call this multiple times.
  template <typename OutIt, typename RealIt, typename IntIt>
  void particle_ancestry(OutIt ret,
                         const RealIt value_begin,
                         const IntIt order_begin) const {
    std::vector<size_t> index_particle(n_particles_);
    for (size_t i = 0; i < n_particles_; ++i) {
      index_particle[i] = i;
    }
    for (size_t k = 0; k < n_data_ + 1; ++k) {
      size_t i = n_data_ - k;
      auto const it_order = order_begin + i * n_particles_;
      auto const it_value = value_begin + i * n_state_ * n_particles_;
      auto it_ret = ret + i * n_state_ * n_particles_;
      for (size_t j = 0; j < n_particles_; ++j) {
        const size_t idx = *(it_order + index_particle[j]);
        index_particle[j] = idx;
        std::copy_n(it_value + idx * n_state_, n_state_,
                    it_ret + j * n_state_);
      }
    }
  }

  template <typename Iterator>
  void history(Iterator ret) const {
    particle_ancestry(ret, history_value.cbegin(), history_order.cbegin());
  }

protected:
  size_t n_state_;
  size_t n_particles_;
  size_t n_data_;
  size_t offset_;

  std::vector<real_t> history_value;
  std::vector<size_t> history_order;
};

template <typename real_t>
class filter_trajectories_device : public filter_trajectories_host<real_t> {
public:
  filter_trajectories_device() {
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
    this->history_value.resize(this->n_state_ * this->n_particles_ * (this->n_data_ + 1));
    this->history_order.resize(this->n_particles_ * (this->n_data_ + 1));
    for (size_t i = 0; i < this->n_particles_; ++i) {
      this->history_order[i] = i;
    }

    history_value_swap = dust::device_array<real_t>(this->n_state_ * this->n_particles_);
    history_order_swap = dust::device_array<size_t>(this->n_particles_);

#ifdef __NVCC__
    // Page lock memory on host
    CUDA_CALL(cudaHostRegister(this->history_value.data(),
                               this->history_value.size() * sizeof(real_t),
                               cudaHostRegisterDefault));
    CUDA_CALL(cudaHostRegister(this->history_order.data(),
                               this->history_order.size() * sizeof(real_t),
                               cudaHostRegisterDefault));
#endif
  }

  size_t value_offset() {
    return this->offset_ * this->n_state_ * this->n_particles_;
  }

  size_t order_offset() {
    return this->offset_ * this->n_particles_;
  }

  void store_values(dust::device_array<real_t>& state) {
    host_memory_stream_.sync();
    state.get_array(history_value_swap.data(), device_memory_stream_, true);
    device_memory_stream_.sync();
    history_value_swap.get_array(this->history_value.data() + value_offset(),
                                 host_memory_stream_, true);
  }

  void store_order(dust::device_array<size_t>& kappa) {
    host_memory_stream_.sync();
    kappa.get_array(history_order_swap.data(), device_memory_stream_, true);
    device_memory_stream_.sync();
    history_order_swap.get_array(this->history_order.data() + order_offset(),
                                 host_memory_stream_, true);
  }

  template <typename Iterator>
  void history(Iterator ret) const {
    std::vector<real_t> host_history = destride_history();
    this->particle_ancestry(ret, host_history.cbegin(), this->history_order.cbegin());
  }

private:
  filter_trajectories_device ( const filter_trajectories_device & ) = delete;
  filter_trajectories_device ( filter_trajectories_device && ) = delete;

  dust::device_array<real_t> history_value_swap;
  dust::device_array<size_t> history_order_swap;

  dust::cuda::cuda_stream device_memory_stream_;
  dust::cuda::cuda_stream host_memory_stream_;

  std::vector<real_t> destride_history() const {
    std::vector<real_t> blocked_history(size());
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
    CUDA_CALL_NOTHROW(cudaHostUnregister(this->history_value.data()));
    CUDA_CALL_NOTHROW(cudaHostUnregister(this->history_order.data()));
#endif
  }
};

template <typename real_t>
class filter_snapshots_host {
public:
  filter_snapshots_host() {
  }

  void resize(size_t n_state, size_t n_particles, std::vector<size_t> steps) {
    n_state_ = n_state;
    n_particles_ = n_particles;
    n_steps_ = steps.size();
    offset_ = 0;
    steps_ = steps;
    state_.resize(n_state_ * n_particles_ * n_steps_);
  }

  bool is_snapshot_step(size_t step) {
    return offset_ < n_steps_ && steps_[offset_] == step;
  }

  void advance() {
    offset_++;
  }

  size_t size() const {
    return state_.size();
  }

  typename std::vector<real_t>::iterator value_iterator() {
    return state_.begin() + offset_ * n_state_ * n_particles_;
  }

  template <typename Iterator>
  void history(Iterator dest) const {
    std::copy(state_.begin(), state_.end(), dest);
  }

protected:
  size_t n_state_;
  size_t n_particles_;
  size_t n_steps_;
  size_t offset_;
  std::vector<size_t> steps_;
  std::vector<real_t> state_;
};

template <typename real_t>
class filter_snapshots_device : public filter_snapshots_host<real_t> {
public:
  filter_snapshots_device() {
  }

  void resize(size_t n_state, size_t n_particles, std::vector<size_t> steps) {
    pageable();
    this->n_state_ = n_state;
    this->n_particles_ = n_particles;
    this->n_steps_ = steps.size();
    this->offset_ = 0;
    this->steps_ = steps;
    this->state_.resize(this->n_state_ * this->n_particles_ * this->n_steps_);

    state_swap = dust::device_array<real_t>(this->n_state_ * this->n_particles_);

#ifdef __NVCC__
    // Page lock memory on host
    CUDA_CALL(cudaHostRegister(this->state_.data(),
                               this->state_.size() * sizeof(real_t),
                               cudaHostRegisterDefault));
#else
  }

  size_t value_offset() {
    return this->offset_ * this->n_state_ * this->n_particles_;
  }

  void store(dust::device_array<real_t>& state) {
    host_memory_stream_.sync();
    state.get_array(state_swap.data(), device_memory_stream_, true);
    device_memory_stream_.sync();
    state_swap.get_array(this->state_.data() + value_offset(),
                         host_memory_stream_, true);
  }

  template <typename Iterator>
  void history(Iterator dest) const {
    // Copy from D->H
    std::vector<real_t> state_host(size());
    state_.get_array(state_host);
    // Destride and copy into iterator
    // TODO: openmp here? collapse(2)
    for (size_t i = 0; i < this->n_steps_; ++i) {
      for (size_t j = 0; j < this->n_particles_; ++j) {
        for (size_t k = 0; k < this->n_state_; ++k) {
          *(dest +
            i * (this->n_particles_ * this->n_state_) +
            j * this->n_state_ +
            k) = state_host[i * (this->n_particles_ * this->n_state_) +
                            j +
                            k * (this->n_particles_)];
        }
      }
    }
  }

private:
  filter_snapshots_device ( const filter_snapshots_device & ) = delete;
  filter_snapshots_device ( filter_snapshots_device && ) = delete;

  dust::device_array<real_t> state_swap;

  dust::cuda::cuda_stream device_memory_stream_;
  dust::cuda::cuda_stream host_memory_stream_;

  void pageable() {
#ifdef __NVCC__
    // Make memory pageable again
    CUDA_CALL_NOTHROW(cudaHostUnregister(this->state_.data()));
#endif
  }

};

template <typename real_t>
struct filter_state_host {
  filter_trajectories_host<real_t> trajectories;
  filter_snapshots_host<real_t> snapshots;
};

template <typename real_t>
struct filter_state_device {
  filter_trajectories_device<real_t> trajectories;
  filter_snapshots_device<real_t> snapshots;
};

}
}

#endif
