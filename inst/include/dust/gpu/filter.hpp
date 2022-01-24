#ifndef DUST_GPU_FILTER_HPP
#define DUST_GPU_FILTER_HPP

#include "dust/gpu/filter_state.hpp"
#include "dust/filter_state.hpp"
#include "dust/filter_tools.hpp"

namespace dust {
namespace filter {

template <typename T>
std::vector<typename T::real_type>
filter(T * obj,
       size_t step_end,
       filter_state_device<typename T::real_type>& state,
       bool save_trajectories,
       std::vector<size_t> step_snapshot) {
  using real_type = typename T::real_type;

  const size_t n_particles = obj->n_particles();
  const size_t n_data = obj->n_data();
  const size_t n_pars = obj->n_pars_effective();

  std::vector<real_type> ll_host(n_pars, 0);
  dust::gpu::device_array<real_type> log_likelihood(ll_host);
  dust::gpu::device_weights<real_type> weights(n_particles, n_pars);
  dust::gpu::device_scan_state<real_type> scan;
  scan.initialise(n_particles, weights.weights());

  if (save_trajectories) {
    state.trajectories.resize(obj->n_state(), n_particles, n_data);

    if (obj->step() <= obj->data().begin()->first) {
      state.trajectories.store_values(obj->device_state_selected());
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
    // MODEL UPDATE
    obj->run(step);

    // COMPARISON FUNCTION
    obj->compare_data(weights.weights(), d->second);

    // SCALE WEIGHTS
    weights.scale_log_weights(log_likelihood);

    // RESAMPLE
    // Normalise the weights and calculate cumulative sum for resample
    obj->resample(weights.weights(), scan);

    // SAVE HISTORY (async)
    if (save_trajectories) {
      state.trajectories.store_values(obj->device_state_selected());
      state.trajectories.store_order(obj->filter_kappa());
      state.trajectories.advance();
    }

    // SAVE SNAPSHOT
    if (save_snapshots && state.snapshots.is_snapshot_step(step)) {
      state.snapshots.store(obj->device_state_full());
      state.snapshots.advance();
    }

    if (step >= step_end) {
      break;
    }
  }

  // Copy likelihoods back to host (this will sync everything)
  log_likelihood.get_array(ll_host);
  return ll_host;
}

}
}

#endif
