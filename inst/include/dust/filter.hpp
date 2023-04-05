#ifndef DUST_FILTER_HPP
#define DUST_FILTER_HPP

#include "dust/filter_state.hpp"
#include "dust/filter_tools.hpp"

namespace dust {
namespace filter {

template <typename T>
std::vector<typename T::real_type>
filter(T * obj,
       typename T::time_type time_end,
       filter_state_host<typename T::real_type, typename T::time_type>& state,
       bool save_trajectories,
       std::vector<typename T::time_type> time_snapshot,
       const std::vector<typename T::real_type>& min_log_likelihood) {
  using real_type = typename T::real_type;

  const size_t n_particles = obj->n_particles();
  const size_t n_data = obj->n_data();
  const size_t n_pars = obj->n_pars_effective();
  const size_t n_particles_each = n_particles / n_pars;
  std::vector<real_type> log_likelihood(n_pars);
  std::vector<real_type> weights(n_particles);
  std::vector<size_t> kappa(n_particles);

  if (save_trajectories) {
    state.trajectories.resize(obj->n_state(), n_particles, n_pars, n_data);

    // On the first time we save the initial conditions; that is
    // whenever `time_end` falls before the first data point (rhs here
    // is just the time that the first data point ends at).
    const auto time_first_data = obj->data().begin()->first;
    if (obj->time() <= time_first_data) {
      obj->state(state.trajectories.value_iterator());
    }

    state.trajectories.advance();
  }

  bool save_snapshots = false;
  if (time_snapshot.size() > 0) {
    save_snapshots = true;
    state.snapshots.resize(obj->n_state_full(), n_particles, time_snapshot);
  }

  const bool is_stochastic = !obj->deterministic();
  auto d = obj->data().cbegin();
  const auto d_end = obj->data().cend();

  while (d->first <= obj->time() && d != d_end) {
    d++;
    state.trajectories.advance();
  }

  for (; d != d_end && obj->time() < time_end; ++d) {
    const auto time = d->first;
    obj->run(time);
    obj->compare_data(weights, d->second);

    // TODO: we should cope better with the case where all weights
    // are 0; I think that is the behaviour in the model (or rather
    // the case where there is no data and so we do not resample)
    //
    // TODO: we should cope better with the case where one filter
    // has become impossible but others continue, but that's hard!
    auto wi = weights.begin();
    for (size_t i = 0; i < n_pars; ++i) {
      log_likelihood[i] += scale_log_weights<real_type>(wi, n_particles_each);
      wi += n_particles_each;
    }

    if (early_exit(log_likelihood, min_log_likelihood)) {
      std::fill(log_likelihood.begin(),
                log_likelihood.end(),
                -std::numeric_limits<real_type>::infinity());
      obj->set_time(time_end);
      return log_likelihood;
    }

    if (is_stochastic) {
      obj->resample(weights, kappa);
    }

    if (save_trajectories) {
      if (is_stochastic) {
        std::copy(kappa.begin(), kappa.end(),
                  state.trajectories.order_iterator());
      }
      obj->state(state.trajectories.value_iterator());
      state.trajectories.advance();
    }

    if (save_snapshots && state.snapshots.is_snapshot_time(time)) {
      obj->state_full(state.snapshots.value_iterator());
      state.snapshots.advance();
    }
  }

  return log_likelihood;
}

}
}

#endif
