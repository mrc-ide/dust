#ifndef DUST_FILTER_STATE_HPP
#define DUST_FILTER_STATE_HPP

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace dust {
namespace filter {

inline void assert_has_storage(size_t n_state, size_t n_particles,
                               size_t n_times) {
  if (n_state == 0 || n_particles == 0 || n_times == 0) {
    throw std::runtime_error("Invalid size (zero) for filter state"); // #nocov
  }
}

template <typename real_type, typename time_type>
class filter_snapshots_host {
public:
  filter_snapshots_host() {
  }

  void resize(size_t n_state, size_t n_particles, std::vector<time_type> times) {
    n_state_ = n_state;
    n_particles_ = n_particles;
    n_times_ = times.size();
    offset_ = 0;
    times_ = times;

    assert_has_storage(n_state_ , n_particles_, n_times_);
    state_.resize(n_state_ * n_particles_ * n_times_);
  }

  bool is_snapshot_time(time_type time) {
    return offset_ < n_times_ && times_[offset_] == time;
  }

  void advance() {
    offset_++;
  }

  size_t size() const {
    return state_.size();
  }

  typename std::vector<real_type>::iterator value_iterator() {
    return state_.begin() + offset_ * n_state_ * n_particles_;
  }

  template <typename Iterator>
  void history(Iterator dest) const {
    std::copy(state_.begin(), state_.end(), dest);
  }

protected:
  size_t n_state_;
  size_t n_particles_;
  size_t n_times_;
  size_t offset_;
  std::vector<time_type> times_;
  std::vector<real_type> state_;
};

template <typename real_type>
class filter_trajectories_host {
public:
  filter_trajectories_host() {
  }

  void resize(size_t n_state, size_t n_particles, size_t n_pars,
              size_t n_data) {
    n_state_ = n_state;
    n_particles_ = n_particles;
    n_pars_ = n_pars;
    n_data_ = n_data;
    offset_ = 0;

    assert_has_storage(n_state_, n_particles_, n_data_ + 1);
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

  typename std::vector<real_type>::iterator value_iterator() {
    return history_value.begin() + offset_ * n_state_ * n_particles_;
  }

  typename std::vector<size_t>::iterator order_iterator() {
    return history_order.begin() + offset_ * n_particles_;
  }

  std::vector<real_type> history() const {
    std::vector<real_type> ret(size());
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
  // look to see where it "came from" in the previous time step
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
    const size_t n_state_particles = n_state_ * n_particles_;
    if (n_particles_ == n_pars_) {
      // No reordering possible, just copy straight through:
      std::copy_n(value_begin, offset_ * n_state_particles, ret);
    } else {
      std::vector<size_t> index_particle(n_particles_);
      for (size_t i = 0; i < n_particles_; ++i) {
        index_particle[i] = i;
      }

      for (size_t k = 0; k < offset_; ++k) {
        size_t i = offset_ - k - 1;
        auto const it_order = order_begin + i * n_particles_;
        auto const it_value = value_begin + i * n_state_particles;
        auto it_ret = ret + i * n_state_particles;
        for (size_t j = 0; j < n_particles_; ++j) {
          std::copy_n(it_value + index_particle[j] * n_state_,
                      n_state_,
                      it_ret + j * n_state_);
          index_particle[j] = *(it_order + index_particle[j]);
        }
      }
    }

    // In the case where we've not filled all data, the remaining
    // memory might be uninitialised.  This ensures that it everything
    // we return has been set by us.  We will ignore this data anyway,
    // but should not return -1e-308 junk.
    if (offset_ < n_data_) {
      std::fill(ret + offset_ * n_state_particles,
                ret + (n_data_ + 1) * n_state_particles,
                0);
    }
  }

  template <typename Iterator>
  void history(Iterator ret) const {
    particle_ancestry(ret, history_value.cbegin(), history_order.cbegin());
  }

protected:
  size_t n_state_;
  size_t n_particles_;
  size_t n_pars_;
  size_t n_data_;
  size_t offset_;

  std::vector<real_type> history_value;
  std::vector<size_t> history_order;
};

template <typename real_type, typename time_type>
struct filter_state_host {
  filter_trajectories_host<real_type> trajectories;
  filter_snapshots_host<real_type, time_type> snapshots;
};

}
}

#endif
