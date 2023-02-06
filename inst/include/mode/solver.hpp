#pragma once

#include <vector>

#include "control.hpp"
#include "stats.hpp"
#include "stepper.hpp"

namespace mode {
template<typename Model>
class solver {
private:
  double t_;
  double h_;
  control ctl_;
  stats stats_;
  double last_error_;
  stepper<Model> stepper_;
  size_t n_variables_;
  size_t n_output_;
  double h_swap_;
  double last_error_swap_;
  stats stats_swap_;
  std::vector<double> stochastic_schedule_;

public:
  using rng_state_type = typename Model::rng_state_type;

  solver(Model m,
         double t,
         std::vector<double> y,
         control ctl) : t_(t),
                        ctl_(ctl),
                        last_error_(0),
                        stepper_(m),
                        n_variables_(m.n_variables()),
                        n_output_(m.n_output()) {
    stats_.reset();
    set_state(y);
    initialise();
    set_initial_step_size();
  }

  double time() const {
    return t_;
  }

  control ctl() const {
    return ctl_;
  }

  double step(double tcrit) {
    bool success = false;
    bool reject = false;
    const double fac_old = std::max(last_error_, 1e-4);

    double h = h_;
    while (!success) {
      if (stats_.n_steps > ctl_.max_steps) {
        throw std::runtime_error("too many steps");
      }
      if (h < ctl_.step_size_min) {
        throw std::runtime_error("step too small");
      }
      if (h <= std::abs(t_) * std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("step size vanished");
      }
      if (t_ + h > tcrit) {
        h = tcrit - t_;
      }

      // Carry out the step
      stepper_.step(t_, h);
      stats_.n_steps++;

      // Error estimation
      const auto err = stepper_.error(ctl_.atol, ctl_.rtol);

      const double fac11 = std::pow(err, ctl_.constant);
      const double facc1 = 1.0 / ctl_.factor_min;
      const double facc2 = 1.0 / ctl_.factor_max;

      if (err <= 1) {
        success = true;
        stats_.n_steps_accepted++;
        stepper_.step_complete(t_, h);
        double fac = fac11 / std::pow(fac_old, ctl_.beta);
        fac = clamp(fac / ctl_.factor_safe,
                    facc2, facc1);
        const double h_new = h / fac;

        t_ += h;
        if (ctl_.debug_record_step_times) {
          stats_.step_times.push_back(t_);
        }
        if (reject) {
          h_ = std::min(h_new, h);
        } else {
          h_ = std::min(h_new, ctl_.step_size_max);
        }
        last_error_ = err;
      } else {
        reject = true;
        if (stats_.n_steps_accepted >= 1) {
          stats_.n_steps_rejected++;
        }
        h /= std::min(facc1, fac11 / ctl_.factor_safe);
      }
    }
    return t_;
  }

  void solve(double t_end, rng_state_type& rng_state) {
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
      const double t_next = it == end ? t_end : std::min(*it, t_end);
      step(t_next);
    }
  }

  void set_stochastic_schedule(const std::vector<double>& time) {
    stochastic_schedule_ = time;
  }

  void set_time(double t) {
    if (t != t_) {
      stats_.reset();
      t_ = t;
    }
  }

  void set_initial_step_size() {
    h_ = stepper_.init_step_size(t_, ctl_);
  }

  void set_state(std::vector<double>::const_iterator state) {
    stepper_.set_state(state);
  }

  void set_state(std::vector<double>::const_iterator state,
                 const std::vector<size_t>& index) {
    stepper_.set_state(state, index);
  }

  void initialise() {
    stepper_.initialise(t_);
  }

  void set_state(const std::vector<double> &state) {
    set_state(state.begin());
  }

  void set_state(const std::vector<double> &state,
                 const std::vector<size_t>& index) {
    set_state(state.begin(), index);
  }

  void set_state(const solver<Model>& other) {
    stepper_.set_state(other.stepper_);
    h_swap_ = other.h_;
    last_error_swap_ = other.last_error_;
    stats_swap_ = other.stats_;
  }

  void swap() {
    stepper_.swap();
    h_ = h_swap_;
    last_error_ = last_error_swap_;
    stats_ = stats_swap_;
  }

  void set_model(Model m) {
    stepper_.set_model(m);
  }

  size_t n_variables() {
    return n_variables_;
  }

  void state(const std::vector<size_t>& index,
        std::vector<double>::iterator end_state) {
    stepper_.state(t_, index, end_state);
  }

  void state(std::vector<double>::iterator end_state) {
    stepper_.state(t_, end_state);
  }

  std::vector<size_t>::iterator
  statistics(std::vector<size_t>::iterator all_stats) const {
    all_stats[0] = stats_.n_steps;
    all_stats[1] = stats_.n_steps_accepted;
    all_stats[2] = stats_.n_steps_rejected;
    return all_stats + 3;
  }

  std::vector<double> debug_step_times() {
    return stats_.step_times;
  }
};
}
