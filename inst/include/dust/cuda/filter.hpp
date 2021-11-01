#ifndef DUST_CUDA_FILTER_HPP
#define DUST_CUDA_FILTER_HPP

#include <dust/filter_state.hpp>
#include <dust/filter_tools.hpp>
#include <dust/cuda/filter_state.hpp>

namespace dust {
namespace filter {

template <typename T>
std::vector<typename T::real_type>
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

  for (auto & d : obj->data()) {
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
