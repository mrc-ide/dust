#ifndef DUST_PARTICLE_HPP
#define DUST_PARTICLE_HPP

#include <vector>

#include "dust/types.hpp"

namespace dust {

template <typename T>
class particle {
public:
  using pars_type = dust::pars_type<T>;
  using real_type = typename T::real_type;
  using data_type = typename T::data_type;
  using rng_state_type = typename T::rng_state_type;

  particle(pars_type pars, size_t time) :
    model_(pars),
    time_(time),
    y_(model_.initial(time_)),
    y_swap_(model_.size()) {
  }

  void run(const size_t time_end, rng_state_type& rng_state) {
    while (time_ < time_end) {
      model_.update(time_, y_.data(), rng_state, y_swap_.data());
      time_++;
      std::swap(y_, y_swap_);
    }
  }

  void state(const std::vector<size_t>& index,
             typename std::vector<real_type>::iterator end_state) const {
    for (size_t i = 0; i < index.size(); ++i) {
      *(end_state + i) = y_[index[i]];
    }
  }

  void state_full(typename std::vector<real_type>::iterator end_state) const {
    for (size_t i = 0; i < y_.size(); ++i) {
      *(end_state + i) = y_[i];
    }
  }

  size_t size() const {
    return y_.size();
  }

  size_t time() const {
    return time_;
  }

  void swap() {
    std::swap(y_, y_swap_);
  }

  void set_time(const size_t time) {
    time_ = time;
  }

  void set_state(const particle<T>& other) {
    y_swap_ = other.y_;
  }

  void set_pars(const particle<T>& other, bool set_state) {
    model_ = other.model_;
    time_ = other.time_;
    if (set_state) {
      y_ = model_.initial(time_);
    }
  }

  void set_state(typename std::vector<real_type>::const_iterator state) {
    for (size_t i = 0; i < y_.size(); ++i, ++state) {
      y_[i] = *state;
    }
  }

  real_type compare_data(const data_type& data, rng_state_type& rng_state) {
    return model_.compare_data(y_.data(), data, rng_state);
  }

private:
  T model_;
  size_t time_;

  std::vector<real_type> y_;
  std::vector<real_type> y_swap_;
};

}

#endif
