#ifndef DUST_FILTER_KERNELS_HPP
#define DUST_FILTER_KERNELS_HPP

namespace dust {

template <typename real_t>
KERNEL void exp_weights(const size_t n_particles,
                        const size_t n_pars,
                        real_t * weights,
                        const real_t * max_weights) {
  const size_t n_particles_each = n_particles / n_pars;
#ifdef __NVCC__
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_particles;
       i += blockDim.x * gridDim.x) {
#else
  for (size_t i = 0; i < n_particles; ++i) {
#endif
    const size_t pars_idx = i / n_particles_each;
    weights[i] = std::exp(weights[i] - max_weights[pars_idx]);
  }
}

template <typename real_t>
KERNEL void weight_log_likelihood(const size_t n_pars,
                                  const size_t n_particles_each,
                                  real_t * ll,
                                  real_t * ll_step,
                                  const real_t * max_weights) {
#ifdef __NVCC__
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_pars;
       i += blockDim.x * gridDim.x) {
#else
  for (size_t i = 0; i < n_pars; ++i) {
#endif
    real_t ll_scaled = std::log(ll_step[i] / n_particles_each) + max_weights[i];
    ll_step[i] = ll_scaled;
    ll[i] += ll_scaled;
  }
}

}

#endif