#ifndef DUST_PARTICLE_HPP
#define DUST_PARTICLE_HPP

namespace dust {

template <typename T>
class Particle {
public:
  typedef dust::pars_t<T> pars_t;
  typedef typename T::real_t real_t;
  typedef typename T::data_t data_t;
  typedef typename T::rng_state_type rng_state_type;

  Particle(pars_t pars, size_t step) :
    model_(pars),
    step_(step),
    y_(model_.initial(step_)),
    y_swap_(model_.size()) {
  }

  void run(const size_t step_end, rng_state_type& rng_state) {
    while (step_ < step_end) {
      model_.update(step_, y_.data(), rng_state, y_swap_.data());
      step_++;
      std::swap(y_, y_swap_);
    }
  }

  void state(const std::vector<size_t>& index,
             typename std::vector<real_t>::iterator end_state) const {
    for (size_t i = 0; i < index.size(); ++i) {
      *(end_state + i) = y_[index[i]];
    }
  }

  void state_full(typename std::vector<real_t>::iterator end_state) const {
    for (size_t i = 0; i < y_.size(); ++i) {
      *(end_state + i) = y_[i];
    }
  }

  size_t size() const {
    return y_.size();
  }

  size_t step() const {
    return step_;
  }

  void swap() {
    std::swap(y_, y_swap_);
  }

  void set_step(const size_t step) {
    step_ = step;
  }

  void set_state(const Particle<T>& other) {
    y_swap_ = other.y_;
  }

  void set_pars(const Particle<T>& other, bool set_state) {
    model_ = other.model_;
    step_ = other.step_;
    if (set_state) {
      y_ = model_.initial(step_);
    }
  }

  void set_state(typename std::vector<real_t>::const_iterator state) {
    for (size_t i = 0; i < y_.size(); ++i, ++state) {
      y_[i] = *state;
    }
  }

  real_t compare_data(const data_t& data, rng_state_type& rng_state) {
    return model_.compare_data(y_.data(), data, rng_state);
  }

private:
  T model_;
  size_t step_;

  std::vector<real_t> y_;
  std::vector<real_t> y_swap_;
};

}

#endif
