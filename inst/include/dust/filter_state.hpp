#ifndef DUST_FILTER_STATE_HPP
#define DUST_FILTER_STATE_HPP

#include <cstddef>

namespace dust {
namespace filter {

// The first issue is that we need real names for these things. One is
// the indexed state that we store over all time - these are
// "trajectories". The other all state at a few times - these are
// "snapshots"
template <typename real_t>
class filter_trajectories {
public:
  filter_trajectories() {
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
  template <typename Iterator>
  void history(Iterator ret) const {
    std::vector<size_t> index_particle(n_particles_);
    for (size_t i = 0; i < n_particles_; ++i) {
      index_particle[i] = i;
    }
    for (size_t k = 0; k < n_data_ + 1; ++k) {
      size_t i = n_data_ - k;
      auto const it_order = history_order.begin() + i * n_particles_;
      auto const it_value = history_value.begin() + i * n_state_ * n_particles_;
      auto it_ret = ret + i * n_state_ * n_particles_;
      for (size_t j = 0; j < n_particles_; ++j) {
        const size_t idx = *(it_order + index_particle[j]);
        index_particle[j] = idx;
        std::copy_n(it_value + idx * n_state_, n_state_,
                    it_ret + j * n_state_);
      }
    }
  }

  size_t size() const {
    return history_value.size();
  }

  void advance() {
    offset_++;
  }

private:
  size_t n_state_;
  size_t n_particles_;
  size_t n_data_;
  size_t offset_;
  size_t len_;
  std::vector<real_t> history_value;
  std::vector<size_t> history_order;
  std::vector<real_t> state_value;
};

template <typename real_t>
class filter_trajectories_device {
public:
  filter_trajectories_device() {
  }

  void resize(size_t n_state, size_t n_particles, size_t n_data) {
    n_state_ = n_state;
    n_particles_ = n_particles;
    n_data_ = n_data;
    offset_ = 0;
    history_value = dust::device_array<real_t>(n_state_ * n_particles_ * (n_data_ + 1));
    history_order = dust::device_array<size_t>(n_particles_ * (n_data_ + 1));
    std::vector<size_t> index_init(n_particles_);
    std::iota(index_init.begin(), index_init.end(), 0);
    history_order.set_array(index_init);
  }

  size_t value_offset() {
    return offset_ * n_state_ * n_particles_;
  }

  size_t order_offset() {
    return offset_ * n_particles_;
  }

  dust::device_array<real_t> &values() {
    return history_value;
  }

  dust::device_array<size_t> &order() {
    return history_order;
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

  // TODO: rewrite this to deal with interleaved state
  // TODO: may be best to write this as a kernel instead
  template <typename Iterator>
  void history(Iterator ret) const {
    throw std::runtime_error("GPU support not yet implemented for history");  }

  size_t size() const {
    return history_value.size();
  }

  void advance() {
    offset_++;
  }

private:
  size_t n_state_;
  size_t n_particles_;
  size_t n_data_;
  size_t offset_;
  size_t len_;
  dust::device_array<real_t> history_value;
  dust::device_array<size_t> history_order;
};

template <typename real_t>
class filter_snapshots {
public:
  filter_snapshots() {
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

  typename std::vector<real_t>::iterator value_iterator() {
    return state_.begin() + offset_ * n_state_ * n_particles_;
  }

  void advance() {
    offset_++;
  }

  size_t size() const {
    return state_.size();
  }

  template <typename Iterator>
  void history(Iterator dest) const {
    throw std::runtime_error("GPU support not yet implemented for snapshots");
  }

private:
  size_t n_state_;
  size_t n_particles_;
  size_t n_steps_;
  size_t offset_;
  std::vector<size_t> steps_;
  std::vector<real_t> state_;
};

template <typename real_t>
class filter_snapshots_device {
public:
  filter_snapshots_device() {
  }

  void resize(size_t n_state, size_t n_particles, std::vector<size_t> steps) {
    n_state_ = n_state;
    n_particles_ = n_particles;
    n_steps_ = steps.size();
    offset_ = 0;
    steps_ = steps;
    state_ = dust::device_array<real_t>(n_state_ * n_particles_ * n_steps_);
  }

  bool is_snapshot_step(size_t step) {
    return offset_ < n_steps_ && steps_[offset_] == step;
  }

  dust::device_array<real_t> &state() {
    return state_;
  }

  size_t value_offset() {
    return offset_ * n_state_ * n_particles_;
  }

  void advance() {
    offset_++;
  }

  size_t size() const {
    return state_.size();
  }

  // TODO: deinterleave this when it is written out (see dust refresh_state)
  // or write a kernel to do so and then just use a memcpy
  template <typename Iterator>
  void history(Iterator dest) const {
    std::copy(state_.begin(), state_.end(), dest);
  }

private:
  size_t n_state_;
  size_t n_particles_;
  size_t n_steps_;
  size_t offset_;
  std::vector<size_t> steps_;
  dust::device_array<real_t> state_;
};

template <typename real_t>
struct filter_state {
  filter_trajectories<real_t> trajectories;
  filter_snapshots<real_t> snapshots;
};

template <typename real_t>
struct filter_state_device {
  filter_trajectories_device<real_t> trajectories;
  filter_snapshots_device<real_t> snapshots;
};

}
}

#endif
