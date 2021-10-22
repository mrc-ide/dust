#ifndef DUST_FILTER_HPP
#define DUST_FILTER_HPP

#include <dust/filter_state.hpp>
#include <dust/filter_tools.hpp>

namespace dust {
namespace filter {

// Host version
template <typename T>
std::vector<typename T::real_type>
filter(Dust<T> * obj,
       filter_state_host<typename T::real_type>& state,
       bool save_trajectories,
       std::vector<size_t> step_snapshot) {
  typedef typename T::real_type real_type;

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
    obj->state(state.trajectories.value_iterator());
    state.trajectories.advance();
  }

  bool save_snapshots = false;
  if (step_snapshot.size() > 0) {
    save_snapshots = true;
    state.snapshots.resize(obj->n_state_full(), n_particles, step_snapshot);
  }

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
      log_likelihood_step[i] = scale_log_weights<real_type>(wi, n_particles_each);
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

    if (save_snapshots && state.snapshots.is_snapshot_step(d.first)) {
      obj->state_full(state.snapshots.value_iterator());
      state.snapshots.advance();
    }
  }

  return log_likelihood;
}

template <typename T>
typename std::enable_if<dust::has_gpu_support<T>::value, std::vector<typename T::real_type>>::type
filter(DustDevice<T> * obj,
       filter_state_device<typename T::real_type>& state,
       bool save_trajectories,
       std::vector<size_t> step_snapshot) {
  typedef typename T::real_type real_type;

  const size_t n_particles = obj->n_particles();
  const size_t n_data = obj->n_data();
  const size_t n_pars = obj->n_pars_effective();

  std::vector<real_type> ll_host(n_pars, 0);
  dust::cuda::device_array<real_type> log_likelihood(ll_host);
  dust::cuda::device_weights<real_type> weights(n_particles, n_pars);
  dust::cuda::device_scan_state<real_type> scan;
  scan.initialise(n_particles, weights.weights());

  if (save_trajectories) {
    state.trajectories.resize(obj->n_state(), n_particles, n_data);
    state.trajectories.store_values(obj->device_state_selected());
    state.trajectories.advance();
  }

  bool save_snapshots = false;
  if (step_snapshot.size() > 0) {
    save_snapshots = true;
    state.snapshots.resize(obj->n_state_full(), n_particles, step_snapshot);
  }

  for (auto & d : obj->data_offsets()) {
    // MODEL UPDATE
    obj->run(d.first);

    // SAVE HISTORY (async)
    if (save_trajectories) {
      state.trajectories.store_values(obj->device_state_selected());
    }

    // COMPARISON FUNCTION
    obj->compare_data(weights.weights(), d.second);

    // SCALE WEIGHTS
    weights.scale_log_weights(log_likelihood);

    // RESAMPLE
    // Normalise the weights and calculate cumulative sum for resample
    obj->resample(weights.weights(), scan);

    // SAVE HISTORY ORDER
    if (save_trajectories) {
      state.trajectories.store_order(obj->kappa());
      state.trajectories.advance();
    }

    // SAVE SNAPSHOT
    if (save_snapshots && state.snapshots.is_snapshot_step(d.first)) {
      state.snapshots.store(obj->device_state_full());
      state.snapshots.advance();
    }
  }

  // Copy likelihoods back to host (this will sync everything)
  log_likelihood.get_array(ll_host);
  return ll_host;
}

}
}

#endif
