#ifndef DUST_FILTER_HPP
#define DUST_FILTER_HPP

#include "dust/filter_state.hpp"
#include "dust/filter_tools.hpp"

namespace dust {
namespace filter {

template <typename T>
std::vector<typename T::real_type>
filter(T * obj,
       size_t step_end,
       filter_state_host<typename T::real_type>& state,
       bool save_trajectories,
       std::vector<size_t> step_snapshot) {
  if (obj->deterministic()) {
    return deterministic(obj, step_end, state,
                         save_trajectories, step_snapshot);
  } else {
    return stochastic(obj, step_end, state,
                      save_trajectories, step_snapshot);
  }
}


template <typename T>
std::vector<typename T::real_type>
stochastic(T * obj,
           size_t step_end,
           filter_state_host<typename T::real_type>& state,
           bool save_trajectories,
           std::vector<size_t> step_snapshot) {
  using real_type = typename T::real_type;

  const size_t n_particles = obj->n_particles();
  const size_t n_data = obj->n_data();
  const size_t n_pars = obj->n_pars_effective();
  const size_t n_particles_each = n_particles / n_pars;
  std::vector<real_type> log_likelihood(n_pars);
  std::vector<real_type> log_likelihood_step(n_pars);
  std::vector<real_type> weights(n_particles);
  std::vector<size_t> kappa(n_particles);

  if (save_trajectories) {
    state.trajectories.resize(obj->n_state(), n_particles, n_data);

    if (obj->step() <= obj->data().begin()->first) {
      obj->state(state.trajectories.value_iterator());
    }

    state.trajectories.advance();
  }

  bool save_snapshots = false;
  if (step_snapshot.size() > 0) {
    save_snapshots = true;
    state.snapshots.resize(obj->n_state_full(), n_particles, step_snapshot);
  }

  auto d = obj->data().cbegin();
  const auto d_end = obj->data().cend();

  while (d->first <= obj->step() && d != d_end) {
    d++;
    state.trajectories.advance();
  }

  for (; d != d_end; ++d) {
    const auto step = d->first;
    obj->run(step);
    obj->compare_data(weights, d->second);

    // TODO: we should cope better with the case where all weights
    // are 0; I think that is the behaviour in the model (or rather
    // the case where there is no data and so we do not resample)
    //
    // TODO: we should cope better with the case where one filter
    // has become impossible but others continue, but that's hard!
    auto wi = weights.begin();
    for (size_t i = 0; i < n_pars; ++i) {
      log_likelihood_step[i] =
        scale_log_weights<real_type>(wi, n_particles_each);
      log_likelihood[i] += log_likelihood_step[i];
      wi += n_particles_each;
    }

    obj->resample(weights, kappa);

    if (save_trajectories) {
      obj->state(state.trajectories.value_iterator());
      std::copy(kappa.begin(), kappa.end(),
                state.trajectories.order_iterator());
      state.trajectories.advance();
    }

    if (save_snapshots && state.snapshots.is_snapshot_step(step)) {
      obj->state_full(state.snapshots.value_iterator());
      state.snapshots.advance();
    }

    if (step >= step_end) {
      break;
    }
  }

  return log_likelihood;
}


template <typename T>
std::vector<typename T::real_type>
deterministic(T * obj,
              size_t step_end,
              filter_state_host<typename T::real_type>& state,
              bool save_trajectories,
              std::vector<size_t> step_snapshot) {
  using real_type = typename T::real_type;

  const size_t n_data = obj->n_data();
  const size_t n_pars = obj->n_pars_effective();
  std::vector<real_type> log_likelihood(n_pars);
  std::vector<real_type> log_likelihood_step(n_pars);
  std::vector<real_type> weights(n_pars);

  if (save_trajectories) {
    state.trajectories.resize(obj->n_state(), n_pars, n_data);

    if (obj->step() <= obj->data().begin()->first) {
      obj->state(state.trajectories.value_iterator());
    }

    state.trajectories.advance();
  }

  bool save_snapshots = false;
  if (step_snapshot.size() > 0) {
    save_snapshots = true;
    state.snapshots.resize(obj->n_state_full(), n_pars, step_snapshot);
  }

  auto d = obj->data().cbegin();
  const auto d_end = obj->data().cend();

  while (d->first <= obj->step() && d != d_end) {
    d++;
    state.trajectories.advance();
  }

  for (; d != d_end; ++d) {
    const auto step = d->first;
    obj->run(step);
    obj->compare_data(weights, d->second);

    for (size_t i = 0; i < n_pars; ++i) {
      log_likelihood[i] += weights[i];
    }

    if (save_trajectories) {
      obj->state(state.trajectories.value_iterator());
      state.trajectories.advance();
    }

    if (save_snapshots && state.snapshots.is_snapshot_step(step)) {
      obj->state_full(state.snapshots.value_iterator());
      state.snapshots.advance();
    }

    if (step >= step_end) {
      break;
    }
  }

  return log_likelihood;
}


}
}

#endif
