#ifndef DUST_FILTER_HPP
#define DUST_FILTER_HPP

#include <dust/filter_state.hpp>
#include <dust/filter_tools.hpp>

namespace dust {
namespace filter {

template <typename T>
std::vector<typename T::real_t> filter(Dust<T> * obj,
                                       filter_state<typename T::real_t>& state,
                                       bool save_trajectories,
                                       std::vector<size_t> step_snapshot) {
  typedef typename T::real_t real_t;

  const size_t n_particles = obj->n_particles();
  const size_t n_data = obj->n_data();
  const size_t n_pars = obj->n_pars_effective();
  std::vector<real_t> log_likelihood(n_pars);
  std::vector<real_t> log_likelihood_step(n_pars);
  std::vector<real_t> weights(n_particles);
  std::vector<size_t> kappa(n_particles);

  if (save_trajectories) {
    state.trajectories.resize(obj->n_state(), n_particles, n_data);
    obj->state(state.trajectories.value_iterator());
    state.trajectories.advance();
  }

  state.snapshots.resize(obj->n_state_full(), n_particles, step_snapshot);

  for (auto & d : obj->data()) {
    obj->run(d.first);
    obj->compare_data(weights, d.second);

    // TODO: we should cope better with the case where all weights
    // are 0; I think that is the behaviour in the model (or rather
    // the case where there is no data and so we do not resample)
    //
    // TODO: we should cope better with the case where one filter
    // has become impossible but others continue, but that's hard!
    auto wi = weights.begin();
    for (size_t i = 0; i < n_pars; ++i) {
      log_likelihood_step[i] = scale_log_weights<real_t>(wi, n_particles_each);
      log_likelihood[i] += log_likelihood_step[i];
      wi += n_particles_each;
    }

    // We could move this below if wanted but we'd have to rewrite
    // the re-sort algorithm; that would be worth doing I think
    // https://github.com/mrc-ide/dust/issues/202
    if (save_trajectories) {
      obj->state(state.trajectories.value_iterator());
    }

    obj->resample(weights, kappa);

    if (save_trajectories) {
      std::copy(kappa.begin(), kappa.end(),
                state.trajectories.order_iterator());
      state.trajectories.advance();
    }

    if (state.snapshots.is_snapshot_step(d.first)) {
      obj->state_full(state.snapshots.value_iterator());
      state.snapshots.advance();
    }
  }

  return log_likelihood;
}


template <typename T>
std::vector<typename T::real_t> filter_device(Dust<T> * obj,
                                       filter_state_device<typename T::real_t>& state,
                                       bool save_trajectories,
                                       std::vector<size_t> step_snapshot) {
  typedef typename T::real_t real_t;

  const size_t n_particles = obj->n_particles();
  const size_t n_data = obj->n_data();
  const size_t n_pars = obj->n_pars_effective();
  const size_t n_particles_each = n_particles / n_pars;
  dust::device_array<real_t> log_likelihood(n_pars);
  dust::device_weights<real_t> weights(n_particles, n_pars);
  dust::device_scan_state<real_t> scan;
  scan.initialise(n_particles, weights);

  if (save_trajectories) {
    state.trajectories.resize(obj->n_state(), n_particles, n_data);
    obj->state(state.trajectories.values(), state.trajectories.value_offset());
    state.trajectories.advance();
  }

  state.snapshots.resize(obj->n_state_full(), n_particles, step_snapshot);

  for (auto & d : obj->data_offsets()) {
    // MODEL UPDATE
    obj->run_device(d.first);

    // SAVE HISTORY (async)
    if (save_trajectories) {
      obj->state(state.trajectories.values(),
                 state.trajectories.value_offset(),
                 true);
    }

    // COMPARISON FUNCTION
    obj->compare_data_device(weights, d.second);

    // SCALE WEIGHTS
    weights.scale_weights(log_likelihood);

    // RESAMPLE
    // Normalise the weights and calculate cumulative sum for resample
    obj->resample_device(weights.weights(), scan);

    // SAVE HISTORY ORDER
    if (save_trajectories) {
      state.trajectories.order().set_array(obj->kappa().data(), n_particles,
        state.trajectories.order_offset());
      state.trajectories.advance();
    }

    // SAVE SNAPSHOT
    if (state.snapshots.is_snapshot_step(d.first)) {
      obj->state_full(state.snapshots.state(), state.snapshots.value_offset());
      state.snapshots.advance();
    }
  }

  // Copy likelihoods back to host
  std::vector<real_t> ll_host(n_pars);
  log_likelihood.get_array(ll_host);
  return ll_host;
}

}
}

#endif
