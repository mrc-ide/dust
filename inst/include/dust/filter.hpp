#ifndef DUST_FILTER_HPP
#define DUST_FILTER_HPP

#include <dust/filter_state.hpp>
#include <dust/filter_tools.hpp>
#include <dust/prefix_scan.cuh>

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
  const size_t n_particles_each = n_particles / n_pars;
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
  dust::device_array<real_t> log_likelihood_step(n_pars);

  // Set up storage
  dust::device_array<real_t> weights(n_particles);
  dust::device_array<real_t> cum_weights(n_particles);
  dust::device_array<real_t> weights_max(n_pars);
  dust::device_array<int> pars_offsets(n_pars + 1);
  dust::device_array<void> max_tmp, sum_tmp;
  std::vector<int> offsets(n_pars + 1);
  for (size_t i = 0; i < n_pars + 1; ++i) {
    offsets[i] = i * n_particles_each;
  }
  pars_offsets.set_array(offsets);

#ifdef __NVCC__
  // Allocate memory for cub
  size_t max_tmp_bytes = 0;
  cub::DeviceSegmentedReduce::Max(max_tmp.data(),
                                  max_tmp_bytes,
                                  weights.data(),
                                  weights_max.data(),
                                  n_pars,
                                  pars_offsets.data(),
                                  pars_offsets.data() + 1);
  max_tmp.set_size(max_tmp_bytes);
  size_t sum_tmp_bytes = 0;
  cub::DeviceSegmentedReduce::Max(sum_tmp.data(),
                                  sum_tmp_bytes,
                                  weights.data(),
                                  weights_max.data(),
                                  n_pars,
                                  pars_offsets.data(),
                                  pars_offsets.data() + 1);
  sum_tmp.set_size(max_tmp_bytes);

  const size_t exp_blockSize = 32;
  const size_t exp_blockCount = (n_particles + blockexp_blockSizeSize - 1) / exp_blockSize;
  const size_t weight_blockSize = 32;
  const size_t weight_blockCount = (n_pars + weight_blockSize - 1) / weight_blockSize;
  const size_t normalise_blockSize = 128;
  const size_t normalise_blockCount = (n_particles + normalise_blockSize - 1) / normalise_blockSize;
#endif

  if (save_trajectories) {
    state.trajectories.resize(obj->n_state(), n_particles, n_data);
    obj->state(state.trajectories.values(), state.trajectories.value_offset());
    state.trajectories.advance();
  }

  state.snapshots.resize(obj->n_state_full(), n_particles, step_snapshot);

  for (auto & d : obj->data_offsets()) {
    // MODEL UPDATE
    obj->run_device(d.first);

    // COMPARISON FUNCTION
    obj->compare_data_device(weights, d.second);

    // SCALE WEIGHTS
#ifdef __NVCC__
    // Scale log-weights. First calculate the max
    cub::DeviceSegmentedReduce::Max(max_tmp.data(),
                                  max_tmp_bytes,
                                  weights.data(),
                                  log_likelihood_step.data(),
                                  n_pars,
                                  pars_offsets.data(),
                                  pars_offsets.data() + 1);
    // Then exp
    dust::exp_weights<real_t><<<exp_blockCount, exp_blockSize>>>(
      n_particles,
      n_pars,
      weights.data(),
      weights_max.data()
    );
    CUDA_CALL(cudaDeviceSynchronize());
    // Then sum
    cub::DeviceSegmentedReduce::Sum(sum_tmp.data(),
                                  sum_tmp_bytes,
                                  weights.data(),
                                  log_likelihood_step.data(),
                                  n_pars,
                                  pars_offsets.data(),
                                  pars_offsets.data() + 1);
    // Finally log and add max
    dust::weight_log_likelihood<real_t><<<weight_blockCount, weight_blockSize>>>(
      n_pars,
      n_particles_each,
      log_likelihood.data(),
      log_likelihood_step.data(),
      weights_max.data()
    );
    CUDA_CALL(cudaDeviceSynchronize());
#else
    std::vector<real_t> max_w(n_pars, -dust::utils::infinity<real_t>());
    std::vector<real_t> host_w(n_particles);
    weights.get_array(host_w);
    for (size_t i = 0; i < n_pars; ++i) {
      for (size_t j = 0; j < n_particles_each; j++) {
        max_w[i] = std::max(host_w[i * n_particles_each + j], max_w[i]);
      }
    }
    weights_max.set_array(max_w);
    dust::exp_weights<real_t>(
      n_particles,
      n_pars,
      weights.data(),
      weights_max.data()
    );
    dust::weight_log_likelihood<real_t>(
      n_pars,
      n_particles_each,
      log_likelihood.data(),
      log_likelihood_step.data(),
      weights_max.data()
    );
#endif

    // SAVE HISTORY
    if (save_trajectories) {
      obj->state(state.trajectories.values(), state.trajectories.value_offset());
    }

    // RESAMPLE
    // Normalise the weights and calculate cumulative sum for resample
#ifdef __NVCC__
    dust::normalise_scan<real_t><<<normalise_blockSize, normalise_blockCount>>>(
      log_likelihood_step.data(),
      weights.data(),
      cum_weights.data(),
      n_particles, n_pars
    );
    CUDA_CALL(cudaDeviceSynchronize());
    // Cumulative sum
    using BlockReduceInt = cub::BlockReduce<real_t, scan_block_size>;
    using BlockReduceIntStorage = typename BlockReduceInt::TempStorage;
    size_t shared_size = sizeof(BlockReduceIntStorage);
    dust::prefix_scan<real_t><<<scan_block_size, n_pars, shared_size>>>(
      cum_weights.data(),
      n_particles,
      n_pars
    );
    CUDA_CALL(cudaDeviceSynchronize());
#else
    dust::normalise_scan<real_t>(
      log_likelihood_step.data(),
      weights.data(),
      cum_weights.data(),
      n_particles, n_pars
    );
    weights.get_array(host_w);
    std::vector<real_t> host_cum_weights(n_particles);
    for (size_t i = 0; i < n_particles; ++i) {
      real_t prev_weight;
      if (i % n_particles_each == 0) {
        prev_weight = 0;
      } else {
        prev_weight = host_cum_weights[i - 1];
      }
      host_cum_weights[i] = prev_weight + host_w[i];
    }
    cum_weights.set_array(host_w);
#endif
    obj->resample_device(cum_weights);

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
