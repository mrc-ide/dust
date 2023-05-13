#ifndef DUST_ODE_DUST_ODE_HPP
#define DUST_ODE_DUST_ODE_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include <map>
#include <stdexcept>

#include "dust/filter_state.hpp"
#include "dust/filter_tools.hpp"
#include "dust/ode/solver.hpp"
#include "dust/random/random.hpp"
#include "dust/types.hpp"
#include "dust/utils.hpp"

namespace dust {

template <typename T>
class dust_ode {
public:
  using model_type = T;
  using real_type = typename T::real_type;
  using time_type = real_type;
  using data_type = typename T::data_type;
  using pars_type = dust::pars_type<T>;
  using rng_state_type = typename T::rng_state_type;
  using rng_int_type = typename rng_state_type::int_type;
  using filter_state_type = dust::filter::filter_state_host<real_type, time_type>;

  dust_ode(const pars_type &pars, const time_type time,
           const size_t n_particles, const size_t n_threads,
           const ode::control<real_type> ctl, const std::vector<rng_int_type>& seed,
           bool deterministic)
    : n_pars_(0),
      n_particles_each_(n_particles),
      n_particles_total_(n_particles),
      pars_are_shared_(true),
      n_threads_(n_threads),
      rng_(n_particles_total_ + 1, seed, deterministic), // +1 for filter
      errors_(n_particles),
      control_(ctl) {
    initialise(pars, time, true);
    initialise_index();
    shape_ = {n_particles};
  }

  dust_ode(const std::vector<pars_type>& pars, const time_type time,
           const size_t n_particles, const size_t n_threads,
           const ode::control<real_type> ctl, const std::vector<rng_int_type>& seed,
           bool deterministic,
           const std::vector<size_t>& shape)
    : n_pars_(pars.size()),
      n_particles_each_(n_particles == 0 ? 1 : n_particles),
      n_particles_total_(n_particles_each_ * pars.size()),
      pars_are_shared_(n_particles != 0),
      n_threads_(n_threads),
      rng_(n_particles_total_ + 1, seed, deterministic),  // +1 for filter
      errors_(n_particles_total_),
      control_(ctl) {
    initialise(pars, time, true);
    initialise_index();
    // constructing the shape here is harder than above.
    if (n_particles > 0) {
      shape_.push_back(n_particles);
    }
    for (auto i : shape) {
      shape_.push_back(i);
    }
  }

  // This is called exactly once, for pulling out debug step times;
  // can we avoid that? Sniff length of the return value perhaps?
  ode::control<real_type> ctl() {
    return control_;
  }

  size_t n_particles() const {
    return n_particles_total_;
  }

  size_t n_state_full() const {
    return solver_[0].n_variables() + solver_[0].n_output();
  }

  size_t n_state() const {
    return index_.size();
  }

  size_t n_variables() const {
    return solver_[0].n_variables();
  }

  // Until we support multiple parameter sets, this is always zero
  // (i.e., what dust uses when pars_multi = FALSE)
  size_t n_pars() const {
    return n_pars_;
  }

  size_t n_pars_effective() const {
    return n_pars_ == 0 ? 1 : n_pars_;
  }

  size_t pars_are_shared() const {
    return pars_are_shared_;
  }

  size_t n_data() const {
    return data_.size();
  }

  const std::map<time_type, std::vector<data_type>>& data() const {
    return data_;
  }

  void set_index(const std::vector<size_t>& index) {
    index_ = index;
  }

  void set_n_threads(size_t n_threads) {
    n_threads_ = n_threads;
  }

  bool deterministic() const {
    return rng_.deterministic();
  }

  void set_stochastic_schedule(const std::vector<time_type>& time) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      solver_[i].set_stochastic_schedule(time);
    }
  }

  void initialise_index() {
    const size_t n = n_state_full();
    index_.clear();
    index_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      index_.push_back(i);
    }
  }

  time_type time() {
    return solver_[0].time();
  }

  const std::vector<size_t>& shape() const {
    return shape_;
  }

  void run(time_type time_end) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      try {
        solver_[i].solve(time_end, rng_.state(i));
      } catch (std::exception const& e) {
        errors_.capture(e, i);
      }
    }
    errors_.report();
  }

  std::vector<real_type> simulate(const std::vector<time_type>& time_end) {
    const size_t n_time = time_end.size();
    std::vector<real_type> ret(n_particles() * n_state() * n_time);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      try {
        for (size_t t = 0; t < n_time; ++t) {
          solver_[i].solve(time_end[t], rng_.state(i));
          size_t offset = t * n_state() * n_particles() + i * n_state();
          solver_[i].state(index_, ret.begin() + offset);
        }
      } catch (std::exception const& e) {
        errors_.capture(e, i);
      }
    }
    errors_.report();
    return ret;
  }

  void state_full(std::vector<real_type> &end_state) {
    state_full(end_state.begin());
  }

  void state_full(typename std::vector<real_type>::iterator end_state) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      solver_[i].state(end_state + i * n_state_full());
    }
  }

  void state(typename std::vector<real_type>::iterator end_state) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      solver_[i].state(index_, end_state + i * n_state());
    }
  }

  void state(std::vector<real_type> &end_state) {
    state(end_state.begin());
  }

  void state(const std::vector<size_t>& index,
             std::vector<real_type> &end_state) {
    auto it = end_state.begin();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      solver_[i].state(index, it + i * index.size());
    }
  }

  void set_time(time_type time) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      solver_[i].set_time(time);
    }
  }

  void set_state(const std::vector<real_type>& state,
                 const std::vector<size_t>& index) {
    const size_t n_particles = solver_.size();
    const bool use_index = index.size() > 0;
    const size_t n_state = use_index ? index.size() : n_variables();
    const bool individual = state.size() == n_state * n_particles;
    const size_t n = individual ? 1 : n_particles_each_;
    auto it = state.begin();

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      const auto it_i = it + (i / n) * n_state;
      if (use_index) {
        solver_[i].set_state(it_i, index);
      } else {
        solver_[i].set_state(it_i);
      }
    }
  }

  void reset_step_size() {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      solver_[i].set_initial_step_size();
    }
  }

  void set_pars(const pars_type& pars, bool set_initial_state) {
    initialise(pars, time(), set_initial_state);
  }

  void set_pars(const std::vector<pars_type>& pars, bool set_initial_state) {
    initialise(pars, time(), set_initial_state);
  }

  void reorder(const std::vector<size_t>& index) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      size_t j = index[i];
      solver_[i].set_state(solver_[j]);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      solver_[i].swap();
    }
  }

  std::vector<size_t> resample(const std::vector<real_type>& weights) {
    std::vector<size_t> index(n_particles());
    resample(weights, index);
    return index;
  }

  void resample(const std::vector<real_type>& weights,
                std::vector<size_t>& index) {
    dust::filter::resample_index(weights, n_pars_, n_particles_each_, n_threads_,
                                 index, rng_.state(n_particles_total_));
    reorder(index);
  }

  void statistics(std::vector<size_t> &all_statistics) {
    auto it = all_statistics.begin();
    // this is hard to make parallel safe without doing
    //
    // solver_i[i].statistics(it + i * 3);
    //
    // which requires knowing that we always have three statistics
    // (though we do rely on this in r/dust.hpp::dust_ode_statistics)
    for (size_t i = 0; i < solver_.size(); ++i) {
      it = solver_[i].get_statistics(it);
    }
  }

  std::vector<std::vector<time_type>> debug_step_times() {
    std::vector<std::vector<time_type>> ret(solver_.size());
    // This could be in parallel safely
    for (size_t i = 0; i < solver_.size(); ++i) {
      ret[i] = solver_[i].debug_step_times();
    }
    return ret;
  }

  void check_errors() {
    if (errors_.unresolved()) {
      throw std::runtime_error("Errors pending; reset required");
    }
  }

  void reset_errors() {
    errors_.reset();
  }

  std::vector<typename rng_state_type::int_type> rng_state() {
    return rng_.export_state();
  }

  void set_rng_state(const std::vector<typename rng_state_type::int_type>& rng_state) {
    rng_.import_state(rng_state);
  }

  void set_data(std::map<time_type, std::vector<data_type>> data,
                bool data_is_shared) {
    data_ = data;
    data_is_shared_ = data_is_shared;
  }

  std::vector<real_type> compare_data() {
    std::vector<real_type> res;
    auto d = data_.find(time());
    if (d != data_.end()) {
      res.resize(solver_.size());
      compare_data(res, d->second);
    }
    return res;
  }

  void compare_data(std::vector<real_type>& res, const std::vector<data_type>& data) {
    const size_t np = solver_.size() / n_pars_effective();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      const size_t j = data_is_shared_ ? 0 : i / np;
      res[i] = solver_[i].compare_data(data[j], rng_.state(i));
    }
  }

private:
  // delete move and copy to avoid accidentally using them
  dust_ode(const dust_ode &) = delete;
  dust_ode(dust_ode &&) = delete;

  const size_t n_pars_; // 0 in the "single" case, >=1 otherwise
  const size_t n_particles_each_; // Particles per parameter set
  const size_t n_particles_total_; // Total number of particles
  const bool pars_are_shared_; // Does the n_particles dimension exist in shape?
  std::vector<size_t> shape_; // shape of output
  size_t n_threads_;
  dust::random::prng<rng_state_type> rng_;
  std::map<time_type, std::vector<data_type>> data_;
  bool data_is_shared_;
  dust::utils::openmp_errors errors_;

  std::vector<size_t> index_;
  std::vector<dust::ode::solver<model_type>> solver_;
  ode::control<real_type> control_;

  void initialise(const pars_type& pars, const time_type time, bool set_state) {
    const auto m = model_type(pars);
    if (solver_.empty()) {
      solver_.reserve(n_particles_total_);
      for (size_t i = 0 ; i < n_particles_total_; ++i) {
        solver_.push_back(dust::ode::solver<model_type>(m, time, control_,
                                                        rng_.state(i)));
      }
    } else {
      errors_.reset();
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < n_particles_total_; ++i) {
        try {
          solver_[i].set_model(m, set_state, rng_.state(i));
        } catch (std::exception const& e) {
          errors_.capture(e, i);
        }
      }
      errors_.report(true);
    }
    reset_errors();
  }

  void initialise(const std::vector<pars_type>& pars, const size_t time,
                  bool set_state) {
    std::vector<model_type> m;
    for (size_t i = 0; i < n_pars_; ++i) {
      m.push_back(model_type(pars[i]));
    }
    if (solver_.empty()) {
      for (size_t i = 0; i < n_particles_total_; ++i) {
        const size_t j = i / n_particles_each_;
        solver_.push_back(dust::ode::solver<model_type>(m[j], time, control_,
                                                        rng_.state(i)));
      }
    } else {
      errors_.reset();
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < n_particles_total_; ++i) {
        try {
          const size_t j = i / n_particles_each_;
          solver_[i].set_model(m[j], set_state, rng_.state(i));
        } catch (std::exception const& e) {
          errors_.capture(e, i);
        }
      }
      errors_.report(true);
    }
    reset_errors();
  }
};

}

#endif
