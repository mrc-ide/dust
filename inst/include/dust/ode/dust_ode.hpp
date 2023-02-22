#ifndef DUST_ODE_DUST_ODE_HPP
#define DUST_ODE_DUST_ODE_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdexcept>

#include "dust/random/random.hpp"
#include "dust/types.hpp"
#include "dust/utils.hpp"
#include "dust/ode/solver.hpp"

namespace dust {

template <typename T>
class dust_ode {
public:
  using model_type = T;
  using time_type = double;
  using real_type = typename T::real_type;
  using data_type = typename T::data_type;
  using pars_type = dust::pars_type<T>;
  using rng_state_type = typename T::rng_state_type;
  using rng_int_type = typename rng_state_type::int_type;

  dust_ode(const pars_type &pars, const double time,
           const size_t n_particles, const size_t n_threads,
           const ode::control ctl, const std::vector<rng_int_type>& seed)
    : n_pars_(0),
      n_particles_each_(n_particles),
      n_particles_total_(n_particles),
      pars_are_shared_(true),
      n_threads_(n_threads),
      rng_(n_particles_total_ + 1, seed, false), // +1 for filter
      errors_(n_particles),
      control_(ctl) {
    initialise(pars, time, true);
    initialise_index();
    shape_ = {n_particles};
  }

  // This is called exactly once, for pulling out debug step times;
  // can we avoid that? Sniff length of the return value perhaps?
  ode::control ctl() {
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

  void set_index(const std::vector<size_t>& index) {
    index_ = index;
  }

  void set_n_threads(size_t n_threads) {
    n_threads_ = n_threads;
  }

  void set_stochastic_schedule(const std::vector<double>& time) {
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

  double time() {
    return solver_[0].time();
  }

  const std::vector<size_t>& shape() const {
    return shape_;
  }

  void run(double time_end) {
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

  std::vector<double> simulate(const std::vector<double>& time_end) {
    const size_t n_time = time_end.size();
    std::vector<double> ret(n_particles() * n_state() * n_time);

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

  void state_full(std::vector<double> &end_state) {
    auto it = end_state.begin();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      solver_[i].state(it + i * n_state_full());
    }
  }

  void state(std::vector<double> &end_state) {
    auto it = end_state.begin();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      solver_[i].state(index_, it + i * n_state());
    }
  }

  void state(const std::vector<size_t>& index,
             std::vector<double> &end_state) {
    auto it = end_state.begin();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      solver_[i].state(index, it + i * index.size());
    }
  }

  void set_time(double time) {
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

  void initialise_solver(bool reset_step_size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < solver_.size(); ++i) {
      if (reset_step_size) {
        solver_[i].set_initial_step_size();
      }
      solver_[i].initialise();
    }
  }

  void set_pars(const pars_type& pars, bool set_initial_state) {
    initialise(pars, time(), set_initial_state);
  }

  void set_pars(const std::vector<pars_type>& pars, bool set_initial_state) {
    throw std::runtime_error("Multiparameter setting not yet supported");
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
    initialise_solver(false);
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

  std::vector<std::vector<double>> debug_step_times() {
    std::vector<std::vector<double>> ret(solver_.size());
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

  // needs proper implementation after merge with dust
  std::vector<size_t> resample(std::vector<real_type>& weights) {
    std::runtime_error("resample not supported");
    return std::vector<size_t>();
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
  // data_
  // data_is_shared_
  dust::utils::openmp_errors errors_;

  std::vector<size_t> index_;
  std::vector<dust::ode::solver<model_type>> solver_;
  ode::control control_;

  void initialise(const pars_type& pars, const double time, bool set_state) {
    const bool first_time = solver_.empty();
    const size_t n = first_time ? 0 : n_state_full();
    const auto m = model_type(pars);

    const auto m_size = m.n_variables() + m.n_output();
    if (n > 0 && m_size != n) {
      std::stringstream msg;
      msg << "'pars' created inconsistent state size: " <<
        "expected length " << n << " but created length " <<
        m_size;
      throw std::invalid_argument(msg.str());
    }

    if (first_time) {
      solver_.reserve(n_particles_total_);
      for (size_t i = 0 ; i < n_particles_total_; ++i) {
        solver_.push_back(dust::ode::solver<model_type>(m, time, control_));
      }
      // shared_ = {pars.shared};
    } else {
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < n_particles_total_; ++i) {
        solver_[i].set_model(m, set_state);
      }
      // shared_[0] = pars.shared;
    }
    reset_errors();
  }
};

}

#endif
