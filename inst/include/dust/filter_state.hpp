#ifndef DUST_FILTER_STATE_HPP
#define DUST_FILTER_STATE_HPP

#include <cstddef>

namespace dust {
namespace filter {

template <typename real_t>
class filter_trajectories {
public:
  filter_trajectories() {
  }

  size_t size() const {
    return n_state_ * n_particles_ * (n_data_ + 1);
  }

  void advance() {
    offset_++;
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

protected:
  size_t n_state_;
  size_t n_particles_;
  size_t n_data_;
  size_t offset_;
};

// The first issue is that we need real names for these things. One is
// the indexed state that we store over all time - these are
// "trajectories". The other all state at a few times - these are
// "snapshots"
template <typename real_t>
class filter_trajectories_host : public filter_trajectories<real_t> {
public:
  filter_trajectories_host() {
  }

  void resize(size_t n_state, size_t n_particles, size_t n_data) {
    this->n_state_ = n_state;
    this->n_particles_ = n_particles;
    this->n_data_ = n_data;
    this->offset_ = 0;
    history_value.resize(this->n_state_ * this->n_particles_ * (this->n_data_ + 1));
    history_order.resize(this->n_particles_ * (this->n_data_ + 1));
    for (size_t i = 0; i < this->n_particles_; ++i) {
      history_order[i] = i;
    }
  }

  size_t size() const {
    return history_value.size();
  }

  typename std::vector<real_t>::iterator value_iterator() {
    return history_value.begin() + this->offset_ * this->n_state_ * this->n_particles_;
  }

  typename std::vector<size_t>::iterator order_iterator() {
    return history_order.begin() + this->offset_ * this->n_particles_;
  }

  template <typename Iterator>
  void history(Iterator ret) const {
    particle_ancestry(ret, history_value.cbegin(), history_order.cbegin());
  }

private:
  std::vector<real_t> history_value;
  std::vector<size_t> history_order;
};

template <typename real_t>
class filter_trajectories_device : public filter_trajectories<real_t> {
public:
  filter_trajectories_device() {
  }

  void resize(size_t n_state, size_t n_particles, size_t n_data) {
    this->n_state_ = n_state;
    this->n_particles_ = n_particles;
    this->n_data_ = n_data;
    this->offset_ = 0;
    history_value = dust::device_array<real_t>(this->n_state_ * this->n_particles_ * (this->n_data_ + 1));
    history_order = dust::device_array<size_t>(this->n_particles_ * (this->n_data_ + 1));
    std::vector<size_t> index_init(this->n_particles_);
    std::iota(index_init.begin(), index_init.end(), 0);
    history_order.set_array(index_init);
  }

  size_t size() const {
    return history_value.size();
  }

  size_t value_offset() {
    return this->offset_ * this->n_state_ * this->n_particles_;
  }

  size_t order_offset() {
    return this->offset_ * this->n_particles_;
  }

  dust::device_array<real_t> &values() {
    return history_value;
  }

  dust::device_array<size_t> &order() {
    return history_order;
  }

  template <typename Iterator>
  void history(Iterator ret) const {
    std::vector<real_t> host_history = destride_history();
    std::vector<size_t> host_order(this->n_particles_ * (this->n_data_ + 1));
    history_order.get_array(host_order);
    particle_ancestry(ret, host_history.cbegin(), host_order.cbegin());
  }

private:
  dust::device_array<real_t> history_value;
  dust::device_array<size_t> history_order;

  std::vector<real_t> destride_history() const {
    // Copy H->D
    std::vector<real_t> history_host(size());
    std::vector<real_t> destride_history(size());
    history_value.get_array(history_host);
    // Destride and copy into iterator
    // TODO openmp here?
    for (size_t i = 0; i < this->n_data_ + 1; ++i) {
      for (size_t j = 0; j < this->n_particles_; ++j) {
        for (size_t k = 0; k < this->n_state_; ++k) {
          destride_history[i * (this->n_particles_ * this->n_state_) +
                           j * this->n_state_ +
                           k] = history_host[i * (this->n_particles_ * this->n_state_) +
                                             j +
                                             k * (this->n_particles_)];
        }
      }
    }
    return destride_history;
  }
};

template <typename real_t>
class filter_snapshots {
public:
  filter_snapshots() {
  }

  bool is_snapshot_step(size_t step) {
    return offset_ < n_steps_ && steps_[offset_] == step;
  }

  void advance() {
    offset_++;
  }

protected:
  size_t n_state_;
  size_t n_particles_;
  size_t n_steps_;
  size_t offset_;
  std::vector<size_t> steps_;
};

template <typename real_t>
class filter_snapshots_host : public filter_snapshots<real_t> {
public:
  filter_snapshots_host() {
  }

  void resize(size_t n_state, size_t n_particles, std::vector<size_t> steps) {
    this->n_state_ = n_state;
    this->n_particles_ = n_particles;
    this->n_steps_ = steps.size();
    this->offset_ = 0;
    this->steps_ = steps;
    state_.resize(this->n_state_ * this->n_particles_ * this->n_steps_);
  }

  size_t size() const {
    return state_.size();
  }

  typename std::vector<real_t>::iterator value_iterator() {
    return state_.begin() + this->offset_ * this->n_state_ * this->n_particles_;
  }

  template <typename Iterator>
  void history(Iterator dest) const {
    std::copy(state_.begin(), state_.end(), dest);
  }

private:
  std::vector<real_t> state_;
};

template <typename real_t>
class filter_snapshots_device : public filter_snapshots<real_t> {
public:
  filter_snapshots_device() {
  }

  void resize(size_t n_state, size_t n_particles, std::vector<size_t> steps) {
    this->n_state_ = n_state;
    this->n_particles_ = n_particles;
    this->n_steps_ = steps.size();
    this->offset_ = 0;
    this->steps_ = steps;
    state_ = dust::device_array<real_t>(this->n_state_ * this->n_particles_ * this->n_steps_);
  }

  size_t size() const {
    return state_.size();
  }

  dust::device_array<real_t> &state() {
    return state_;
  }

  size_t value_offset() {
    return this->offset_ * this->n_state_ * this->n_particles_;
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
  dust::device_array<real_t> state_;
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
