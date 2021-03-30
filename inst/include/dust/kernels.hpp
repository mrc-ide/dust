#ifndef DUST_KERNELS_HPP
#define DUST_KERNELS_HPP

#include <assert.h>
#include <dust/utils.hpp>
#include <dust/device_state.cuh>

namespace dust {

// This is the main model update, will be defined by the model code
// (see inst/examples/variable.cpp for an example). This is unique
// within the file in that we expect that the user will specialise it.
template <typename T>
DEVICE void update_device(size_t step,
                   const dust::interleaved<typename T::real_t> state,
                   dust::interleaved<int> internal_int,
                   dust::interleaved<typename T::real_t> internal_real,
                   const int * shared_int,
                   const typename T::real_t * shared_real,
                   dust::rng_state_t<typename T::real_t>& rng_state,
                   dust::interleaved<typename T::real_t> state_next);

template <typename T>
DEVICE typename T::real_t compare_device(
                   const dust::interleaved<typename T::real_t> state,
                   const typename T::data_t * data,
                   dust::interleaved<int> internal_int,
                   dust::interleaved<typename T::real_t> internal_real,
                   const int * shared_int,
                   const typename T::real_t * shared_real,
                   dust::rng_state_t<typename T::real_t>& rng_state);

// __global__ for shuffling particles
template<typename real_t>
KERNEL void scatter_device(const size_t* index,
                           real_t* state,
                           real_t* scatter_state,
                           const size_t n_state,
                           const size_t n_particles) {
  // e.g. 4 particles with 3 states ABC stored on device as
  // [1_A, 2_A, 3_A, 4_A, 1_B, 2_B, 3_B, 4_B, 1_C, 2_C, 3_C, 4_C]
  // e.g. index [3, 1, 3, 2] with would be
  // [3_A, 1_A, 3_A, 2_A, 3_B, 1_B, 3_B, 2_B, 3_C, 1_C, 3_C, 2_C]
  // interleaved, i.e. input repeated n_state_full times, plus a strided
  // offset
  // [3, 1, 3, 2, 3 + 4, 1 + 4, 3 + 4, 2 + 4, 3 + 8, 1 + 8, 3 + 8, 2 + 8]
  // [3, 1, 3, 2, 7, 5, 7, 6, 11, 9, 11, 10]
  int state_size = n_state * n_particles;
#ifdef __NVCC__
  // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < state_size;
       i += blockDim.x * gridDim.x) {
#else
  for (int i = 0; i < state_size; ++i) {
#endif
    const int scatter_index = index[i % n_particles] + (i / n_particles) * n_particles;
    scatter_state[i] = state[scatter_index];
  }
}

template <typename T>
KERNEL void run_particles(size_t step_start,
                          size_t step_end,
                          size_t n_particles,
                          size_t n_pars,
                          typename T::real_t * state,
                          typename T::real_t * state_next,
                          int * internal_int,
                          typename T::real_t * internal_real,
                          size_t n_shared_int, size_t n_shared_real,
                          const int * shared_int,
                          const typename T::real_t * shared_real,
                          uint64_t * rng_state,
                          bool use_shared_L1) {
  typedef typename T::real_t real_t;
  const size_t n_particles_each = n_particles / n_pars;

#ifdef __CUDA_ARCH__
  const int block_per_pars = (n_particles_each + blockDim.x - 1) / blockDim.x;
  const int j = blockIdx.x / block_per_pars;
  dust::device_ptrs<T> shared_state = dust::load_shared_state<T>(j,
                                                  n_shared_int,
                                                  n_shared_real,
                                                  shared_int,
                                                  shared_real,
                                                  nullptr,
                                                  use_shared_L1);

  int i, max_i;
  if (use_shared_L1) {
    // Pick particle index based on block, don't process if off the end
    i = j * n_particles_each + (blockIdx.x % block_per_pars) * blockDim.x +
      threadIdx.x;
    max_i = n_particles_each * (j + 1);
  } else {
    // Otherwise CUDA thread number = particle
    i = blockIdx.x * blockDim.x + threadIdx.x;
    max_i = n_particles;
  }

  if (i < max_i) {
#else
  // omp here
  for (size_t i = 0; i < n_particles; ++i) {
    const int j = i / n_particles_each;
    dust::device_ptrs<T> shared_state = dust::load_shared_state<T>(j,
                                                  n_shared_int,
                                                  n_shared_real,
                                                  shared_int,
                                                  shared_real,
                                                  nullptr,
                                                  false);
#endif
    dust::interleaved<real_t> p_state(state, i, n_particles);
    dust::interleaved<real_t> p_state_next(state_next, i, n_particles);
    dust::interleaved<int> p_internal_int(internal_int, i, n_particles);
    dust::interleaved<real_t> p_internal_real(internal_real, i, n_particles);
    dust::interleaved<uint64_t> p_rng(rng_state, i, n_particles);

    dust::rng_state_t<real_t> rng_block = dust::get_rng_state<real_t>(p_rng);
    for (size_t step = step_start; step < step_end; ++step) {
      update_device<T>(step,
                       p_state,
                       p_internal_int,
                       p_internal_real,
                       shared_state.shared_int,
                       shared_state.shared_real,
                       rng_block,
                       p_state_next);
#ifdef __CUDA_ARCH__
      __syncwarp();
#endif
      dust::interleaved<real_t> tmp = p_state;
      p_state = p_state_next;
      p_state_next = tmp;
    }
    dust::put_rng_state(rng_block, p_rng);
  }
}

template <typename T>
KERNEL void compare_particles(size_t n_particles,
                              size_t n_pars,
                              typename T::real_t * state,
                              typename T::real_t * weights,
                              int * internal_int,
                              typename T::real_t * internal_real,
                              size_t n_shared_int,
                              size_t n_shared_real,
                              const int * shared_int,
                              const typename T::real_t * shared_real,
                              const typename T::data_t * data,
                              uint64_t * rng_state,
                              bool use_shared_L1) {
  // This setup is mostly shared with run_particles
  typedef typename T::real_t real_t;
  const size_t n_particles_each = n_particles / n_pars;

#ifdef __CUDA_ARCH__
  const int block_per_pars = (n_particles_each + blockDim.x - 1) / blockDim.x;
  const int j = blockIdx.x / block_per_pars;
  dust::device_ptrs<T> shared_state = dust::load_shared_state<T>(j,
                                                  n_shared_int,
                                                  n_shared_real,
                                                  shared_int,
                                                  shared_real,
                                                  data,
                                                  use_shared_L1);

  int i, max_i;
  if (use_shared_L1) {
    // Pick particle index based on block, don't process if off the end
    i = j * n_particles_each + (blockIdx.x % block_per_pars) * blockDim.x +
      threadIdx.x;
    max_i = n_particles_each * (j + 1);
  } else {
    // Otherwise CUDA thread number = particle
    i = blockIdx.x * blockDim.x + threadIdx.x;
    max_i = n_particles;
  }

  if (i < max_i) {
#else
  // omp here
  for (size_t i = 0; i < n_particles; ++i) {
    const int j = i / n_particles_each;
    dust::device_ptrs<T> shared_state = dust::load_shared_state<T>(j,
                                                  n_shared_int,
                                                  n_shared_real,
                                                  shared_int,
                                                  shared_real,
                                                  data,
                                                  use_shared_L1);
#endif
    dust::interleaved<real_t> p_state(state, i, n_particles);
    dust::interleaved<int> p_internal_int(internal_int, i, n_particles);
    dust::interleaved<real_t> p_internal_real(internal_real, i, n_particles);
    dust::interleaved<uint64_t> p_rng(rng_state, i, n_particles);
    dust::rng_state_t<real_t> rng_block = dust::get_rng_state<real_t>(p_rng);

    weights[i] = compare_device<T>(p_state,
                                   shared_state.data,
                                   p_internal_int,
                                   p_internal_real,
                                   shared_state.shared_int,
                                   shared_state.shared_real,
                                   rng_block);
#ifdef __CUDA_ARCH__
    // Branching unlikely in compare_device, but just in case
    __syncwarp();
#endif
    dust::put_rng_state(rng_block, p_rng);
  }
}

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
  for (int i = 0; i < n_particles; ++i) {
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
  for (int i = 0; i < n_pars; ++i) {
#endif
    real_t ll_scaled = std::log(ll_step[i] / n_particles_each) + max_weights[i];
    ll_step[i] = ll_scaled;
    ll[i] += ll_scaled;
  }
}

// Likely not particularly CUDA friendly, but will do for now
// (better alternative would be merge, as both lists sorted)
template <typename T>
DEVICE size_t binary_interval_search(const T * array,
                            const size_t array_len,
                            const T search) {
  size_t l_pivot = 0;
  size_t r_pivot = array_len;
  while (l_pivot < r_pivot) {
    const size_t m = std::floor((l_pivot + r_pivot) / 2);
    if (array[m] < search) {
      l_pivot = m + 1;
    } else {
      r_pivot = m;
    }
  }
#ifdef __CUDA_ARCH__
  __syncwarp();
#endif
  return l_pivot;
}

// index = findInterval(u, cum_weights)
template <typename real_t>
KERNEL void find_intervals(const real_t * cum_weights,
                           const size_t n_particles, const size_t n_pars,
                           size_t * index, real_t * u) {
  const size_t n_particles_each = n_particles / n_pars;
#ifdef __NVCC__
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_particles;
       i += blockDim.x * gridDim.x) {
#else
  for (int i = 0; i < n_particles; ++i) {
#endif
    const int par_idx = i / n_particles_each;
    real_t u_particle = 1 / static_cast<real_t>(n_particles_each) *
                        (u[par_idx] + i % n_particles_each);
    index[i] = binary_interval_search(
      cum_weights + par_idx * n_particles_each,
      n_particles_each, u_particle) + i % n_particles_each;
#ifdef __NVCC__
  }
#else
  }
#endif
}

// cum_weights = weights / sum(weights)
template <typename real_t>
KERNEL void normalise_scan(const real_t * weight_sum, const real_t * weights,
                           real_t * cum_weights,
                           const size_t n_particles, const size_t n_pars) {
  const size_t n_particles_each = n_particles / n_pars;
#ifdef __NVCC__
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_particles;
       i += blockDim.x * gridDim.x) {
#else
  for (int i = 0; i < n_particles; ++i) {
#endif
    cum_weights[i] = weights[i] / weight_sum[i / n_particles_each];
  }
}

}

#endif
