#ifndef DUST_PARTICLE_HPP
#define DUST_PARTICLE_HPP

#include <vector>

#include "dust/types.hpp"

namespace dust {

template <typename T>
class particle {
public:
  using pars_type = dust::pars_type<T>;
  using time_type = size_t;
  using real_type = typename T::real_type;
  using data_type = typename T::data_type;
  using rng_state_type = typename T::rng_state_type;

  particle(pars_type pars, time_type time, rng_state_type& rng_state) :
    model_(pars),
    time_(time),
    y_(model_.initial(time_, rng_state)),
    y_swap_(model_.size()) {
  }

  void run(const time_type time_end, rng_state_type& rng_state) {
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

  time_type time() const {
    return time_;
  }

  void swap() {
    std::swap(y_, y_swap_);
  }

  void set_time(const time_type time) {
    time_ = time;
  }

  void set_state(const particle<T>& other) {
    y_swap_ = other.y_;
  }

  void set_pars(const pars_type pars, const time_type time, bool set_state,
                rng_state_type& rng_state) {
    const auto m = T(pars);
    if (m.size() != size()) {
      std::stringstream msg;
      msg << "'pars' created inconsistent state size: " <<
        "expected length " << size() << " but created length " <<
        m.size();
      throw std::invalid_argument(msg.str());
    }
    model_ = m;
    time_ = time;
    if (set_state) {
      y_ = model_.initial(time_, rng_state);
    }
  }

  void set_state(typename std::vector<real_type>::const_iterator state) {
    for (size_t i = 0; i < y_.size(); ++i, ++state) {
      y_[i] = *state;
    }
  }

  void set_state(typename std::vector<real_type>::const_iterator state,
                 const std::vector<size_t>& index) {
    for (size_t i = 0; i < index.size(); ++i, ++state) {
      y_[index[i]] = *state;
    }
  }

  real_type compare_data(const data_type& data, rng_state_type& rng_state) {
    return model_.compare_data(y_.data(), data, rng_state);
  }

private:
  T model_;
  time_type time_;

  std::vector<real_type> y_;
  std::vector<real_type> y_swap_;
};

}

#endif
