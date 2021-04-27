#ifndef DUST_FILTER_TOOLS_HPP
#define DUST_FILTER_TOOLS_HPP

#include <algorithm>
#include <numeric>

#include <dust/types.hpp>
#include <dust/kernels.hpp>

namespace dust {
namespace filter {

template <typename real_t>
void resample_weight(typename std::vector<real_t>::const_iterator w,
                     size_t n, real_t u, size_t offset,
                     typename std::vector<size_t>::iterator idx) {
  const real_t tot = std::accumulate(w, w + n, static_cast<real_t>(0));
  real_t ww = 0.0, uu = tot * u / n, du = tot / n;

  size_t j = offset;
  for (size_t i = 0; i < n; ++i) {
    while (ww < uu) {
      ww += *w;
      ++w;
      ++j;
    }
    uu += du;
    *idx = j == 0 ? 0 : j - 1;
    ++idx;
  }
}

template <typename real_t>
void run_device_resample(const size_t n_particles,
                         const size_t n_pars,
                         const size_t n_state,
                         rng_state_t<real_t>& resample_rng,
                         dust::device_state<real_t>& device_state,
                         dust::device_array<real_t>& weights,
                         dust::device_scan_state<real_t>& scan) {
#ifdef __NVCC__
    // Cumulative sum
    // Note this is over all parameters, this is fixed with a
    // subtraction in the interval kernel
    cub::DeviceScan::InclusiveSum(scan.scan_tmp.data(),
                                  scan.tmp_bytes,
                                  weights.data(),
                                  scan.cum_weights.data(),
                                  scan.cum_weights.size());
    // Don't sync yet, as this can run while the u draws are made and copied
    // to the device
#else
    std::vector<real_t> host_w(weights.size());
    std::vector<real_t> host_cum_weights(weights.size());
    weights.get_array(host_w);
    host_cum_weights[0] = host_w[0];
    for (size_t i = 1; i < n_particles; ++i) {
      host_cum_weights[i] = host_cum_weights[i - 1] + host_w[i];
    }
    scan.cum_weights.set_array(host_cum_weights);
#endif

    // Generate random numbers for each parameter set
    std::vector<real_t> shuffle_draws(n_pars);
    for (size_t i = 0; i < n_pars; ++i) {
      shuffle_draws[i] = dust::unif_rand(resample_rng);
    }
    // Copying this also syncs the prefix scan
    device_state.resample_u.set_array(shuffle_draws);

    // Generate the scatter indices
#ifdef __NVCC__
    const size_t interval_blockSize = 128;
    const size_t interval_blockCount =
        (n_particles + interval_blockSize - 1) / interval_blockSize;
    dust::find_intervals<real_t><<<interval_blockCount, interval_blockSize>>>(
      scan.cum_weights.data(),
      n_particles,
      n_pars,
      device_state.scatter_index.data(),
      device_state.resample_u.data()
    );
    CUDA_CALL(cudaDeviceSynchronize());
#else
    dust::find_intervals<real_t>(
      scan.cum_weights.data(),
      n_particles,
      n_pars,
      device_state_.scatter_index.data(),
      device_state_.resample_u.data()
    );
#endif

    // Shuffle the particles
#ifdef __NVCC__
    const size_t scatter_blockSize = 64;
    const size_t scatter_blockCount =
        (n_particles * n_state + scatter_blockSize - 1) / scatter_blockSize;
    dust::scatter_device<real_t><<<scatter_blockCount, scatter_blockSize>>>(
        device_state.scatter_index.data(),
        device_state.y.data(),
        device_state.y_next.data(),
        n_state,
        n_particles;
    CUDA_CALL(cudaDeviceSynchronize());
#else
    dust::scatter_device<real_t>(
        device_state.scatter_index.data(),
        device_state.y.data(),
        device_state.y_next.data(),
        n_state,
        n_particles;
#endif
    device_state_.swap();
}

// Given some vector of log probabilities 'w' we want to compute a
// vector of numbers such that their ratio equals the exponential of
// their difference, along with the log average value of the numbers.
//
// We can't just do exp(w) because most of the numbers are impossibly
// small. Instead we scale them so that the largest value of exp(w)
// will be 1 and this preserves the relative probabilities because all
// numbers are multiplied by the same value.
//
// Returns scaled weights by modifying 'w' and returns the single
// value of the average log weight.
template <typename real_t>
real_t scale_log_weights(typename std::vector<real_t>::iterator w, size_t n) {
  real_t max_w = -std::numeric_limits<real_t>::infinity();
  auto wi = w;
  for (size_t i = 0; i < n; ++i, ++wi) {
    if (std::isnan(*wi)) {
      *wi = -std::numeric_limits<real_t>::infinity();
    } else {
      max_w = std::max(max_w, *wi);
    }
  }
  real_t tot = 0.0;
  wi = w;
  for (size_t i = 0; i < n; ++i, ++wi) {
    *wi = std::exp(*wi - max_w);
    tot += *wi;
  }
  return std::log(tot / n) + max_w;
}



}
}

#endif
