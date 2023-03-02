#ifndef DUST_ODE_SOLVER_HPP
#define DUST_ODE_SOLVER_HPP

#include <limits>
#include <stdexcept>
#include <vector>

#include "dust/ode/control.hpp"
#include "dust/ode/statistics.hpp"
#include "dust/ode/stepper.hpp"

namespace dust {
namespace ode {

template <typename Model>
class solver {
public:
  using rng_state_type = typename Model::rng_state_type;
  using real_type = typename Model::real_type;

  solver(Model m, real_type t, control<real_type> ctl) : t_(t),
                                                         ctl_(ctl),
                                                         last_error_(0),
                                                         stepper_(m, t),
                                                         n_variables_(m.n_variables()),
                                                         n_output_(m.n_output()) {
    statistics_.reset();
    set_initial_step_size();
  }

  real_type time() const {
    return t_;
  }

  size_t n_variables() const {
    return n_variables_;
  }

  size_t n_output() const {
    return n_output_;
  }

  control<real_type> ctl() const {
    return ctl_;
  }

  real_type step(real_type tcrit) {
    bool success = false;
    bool reject = false;
    const real_type fac_old = std::max(last_error_, static_cast<real_type>(1e-4));

    real_type h = h_;
    while (!success) {
      if (statistics_.n_steps > ctl_.max_steps) {
        throw std::runtime_error("too many steps");
      }
      if (h < ctl_.step_size_min) {
        throw std::runtime_error("step too small");
      }
      if (h <= std::abs(t_) * std::numeric_limits<real_type>::epsilon()) {
        throw std::runtime_error("step size vanished");
      }
      if (t_ + h > tcrit) {
        h = tcrit - t_;
      }

      // Carry out the step
      stepper_.step(t_, h);
      statistics_.n_steps++;

      // Error estimation
      const auto err = stepper_.error(ctl_.atol, ctl_.rtol);

      const real_type fac11 = std::pow(err, ctl_.constant);
      const real_type facc1 = 1.0 / ctl_.factor_min;
      const real_type facc2 = 1.0 / ctl_.factor_max;

      if (err <= 1) {
        success = true;
        statistics_.n_steps_accepted++;
        stepper_.step_complete(t_, h);
        real_type fac = fac11 / std::pow(fac_old, ctl_.beta);
        fac = utils::clamp(fac / ctl_.factor_safe, facc2, facc1);
        const real_type h_new = h / fac;

        t_ += h;
        if (ctl_.debug_record_step_times) {
          statistics_.step_times.push_back(t_);
        }
        if (reject) {
          h_ = std::min(h_new, h);
        } else {
          h_ = std::min(h_new, ctl_.step_size_max);
        }
        last_error_ = err;
      } else {
        reject = true;
        if (statistics_.n_steps_accepted >= 1) {
          statistics_.n_steps_rejected++;
        }
        h /= std::min(facc1, fac11 / ctl_.factor_safe);
      }
    }
    return t_;
  }

  void solve(real_type t_end, rng_state_type& rng_state) {
    // TODO: we can tidy this bit of bookkeeping up later once it's
    // correct; it should be possible to hold a pointer to where we
    // are, and update it when doing set_time()
    const auto end = stochastic_schedule_.end();
    auto it = std::lower_bound(stochastic_schedule_.begin(), end, t_);
    while (t_ < t_end) {
      if (it != end && *it == t_) {
        stepper_.update_stochastic(t_, rng_state);
        ++it;
      }
      const real_type t_next = it == end ? t_end : std::min(*it, t_end);
      step(t_next);
    }
  }

  void set_stochastic_schedule(const std::vector<real_type>& time) {
    stochastic_schedule_ = time;
  }

  void set_time(real_type t) {
    if (t != t_) {
      statistics_.reset();
      t_ = t;
    }
  }

  void set_initial_step_size() {
    h_ = stepper_.init_step_size(t_, ctl_);
  }

  void set_state(typename std::vector<real_type>::const_iterator state) {
    stepper_.set_state(state);
  }

  void set_state(typename std::vector<real_type>::const_iterator state,
                 const std::vector<size_t>& index) {
    stepper_.set_state(state, index);
  }

  void set_state(const std::vector<real_type> &state) {
    set_state(state.begin());
  }

  void set_state(const std::vector<real_type> &state,
                 const std::vector<size_t>& index) {
    set_state(state.begin(), index);
  }

  void set_state(const solver<Model>& other) {
    stepper_.set_state(other.stepper_);
    h_swap_ = other.h_;
    last_error_swap_ = other.last_error_;
    statistics_swap_ = other.statistics_;
  }

  void swap() {
    stepper_.swap();
    h_ = h_swap_;
    last_error_ = last_error_swap_;
    statistics_ = statistics_swap_;
  }

  void set_model(Model m, bool set_initial_state) {
    if (set_initial_state) {
      stepper_.set_model(m, t_);
    } else {
      stepper_.set_model(m);
    }
  }

  size_t n_variables() {
    return n_variables_;
  }

  void state(const std::vector<size_t>& index,
             typename std::vector<real_type>::iterator end_state) {
    stepper_.state(t_, index, end_state);
  }

  void state(typename std::vector<real_type>::iterator end_state) {
    stepper_.state(t_, end_state);
  }

  real_type compare_data(const typename Model::data_type& data, rng_state_type& rng_state) {
    return stepper_.compare_data(data, rng_state);
  }

  std::vector<size_t>::iterator
  get_statistics(std::vector<size_t>::iterator all_statistics) const {
    all_statistics[0] = statistics_.n_steps;
    all_statistics[1] = statistics_.n_steps_accepted;
    all_statistics[2] = statistics_.n_steps_rejected;
    return all_statistics + 3;
  }

  std::vector<real_type> debug_step_times() {
    return statistics_.step_times;
  }

private:
  real_type t_;
  real_type h_;
  control<real_type> ctl_;
  statistics<real_type> statistics_;
  real_type last_error_;
  stepper<Model> stepper_;
  size_t n_variables_;
  size_t n_output_;
  real_type h_swap_;
  real_type last_error_swap_;
  statistics<real_type> statistics_swap_;
  std::vector<real_type> stochastic_schedule_;
};

}
}

#endif
