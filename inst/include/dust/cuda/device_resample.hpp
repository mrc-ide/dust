#ifndef DUST_CUDA_DEVICE_RESAMPLE_HPP
#define DUST_CUDA_DEVICE_RESAMPLE_HPP

#include "dust/random/random.hpp"
#include "dust/cuda/launch_control.hpp"
#include "dust/cuda/kernels.hpp"

namespace dust {

namespace filter {

template <typename real_type, typename rng_state_type>
void run_device_resample(const size_t n_particles,
                         const size_t n_pars,
                         const size_t n_state,
                         const dust::cuda::launch_control_dust& cuda_pars,
                         dust::cuda::cuda_stream& kernel_stream,
                         dust::cuda::cuda_stream& resample_stream,
                         rng_state_type& resample_rng,
                         dust::cuda::device_state<real_type, rng_state_type>& device_state,
                         dust::cuda::device_array<real_type>& weights,
                         dust::cuda::device_scan_state<real_type>& scan) {
#ifdef __NVCC__
    // Cumulative sum
    // Note this is over all parameters, this is fixed with a
    // subtraction in the interval kernel
    cub::DeviceScan::InclusiveSum(scan.scan_tmp.data(),
                                  scan.tmp_bytes,
                                  weights.data(),
                                  scan.cum_weights.data(),
                                  scan.cum_weights.size(),
                                  kernel_stream.stream());
    // Don't sync yet, as this can run while the u draws are made and copied
    // to the device
#else
    std::vector<real_type> host_w(weights.size());
    std::vector<real_type> host_cum_weights(weights.size());
    weights.get_array(host_w);
    host_cum_weights[0] = host_w[0];
    for (size_t i = 1; i < n_particles; ++i) {
      host_cum_weights[i] = host_cum_weights[i - 1] + host_w[i];
    }
    scan.cum_weights.set_array(host_cum_weights);
#endif

    // Generate random numbers for each parameter set
    std::vector<real_type> shuffle_draws(n_pars);
    for (size_t i = 0; i < n_pars; ++i) {
      shuffle_draws[i] = dust::random::uniform<real_type>(resample_rng, 0, 1);
    }
    const bool async_copy = true;
    device_state.resample_u.set_array(shuffle_draws.data(),
                                      resample_stream, async_copy);

    // Now sync the streams
    kernel_stream.sync();
    resample_stream.sync();

    // Generate the scatter indices
#ifdef __NVCC__
    dust::cuda::find_intervals<real_type><<<cuda_pars.interval.block_count,
                                         cuda_pars.interval.block_size,
                                         0,
                                         kernel_stream.stream()>>>(
      scan.cum_weights.data(),
      n_particles,
      n_pars,
      device_state.scatter_index.data(),
      device_state.resample_u.data()
    );
    kernel_stream.sync();
#else
    dust::cuda::find_intervals<real_type>(
      scan.cum_weights.data(),
      n_particles,
      n_pars,
      device_state.scatter_index.data(),
      device_state.resample_u.data()
    );
#endif

    // Shuffle the particles
    const bool select_kernel = false;
#ifdef __NVCC__
    dust::cuda::scatter_device<real_type><<<cuda_pars.scatter.block_count,
                                   cuda_pars.scatter.block_size,
                                   0,
                                   kernel_stream.stream()>>>(
        device_state.scatter_index.data(),
        device_state.y.data(),
        device_state.y_next.data(),
        n_state,
        n_particles,
        select_kernel);
    kernel_stream.sync();
#else
    dust::cuda::scatter_device<real_type>(
        device_state.scatter_index.data(),
        device_state.y.data(),
        device_state.y_next.data(),
        n_state,
        n_particles,
        select_kernel);
#endif
    device_state.swap();
}

}
}

#endif
