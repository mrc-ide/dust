#ifndef DUST_GPU_KERNELS_HPP
#define DUST_GPU_KERNELS_HPP

#include "dust/gpu/device_state.hpp"
#include "dust/utils.hpp"

namespace dust {
namespace gpu {

// This is the main model update, will be defined by the model code
// (see inst/examples/variable.cpp for an example). This is unique
// within the file in that we expect that the user will specialise it.
template <typename T>
__device__
void update_gpu(size_t time,
                const interleaved<typename T::real_type> state,
                interleaved<int> internal_int,
                interleaved<typename T::real_type> internal_real,
                const int * shared_int,
                const typename T::real_type * shared_real,
                typename T::rng_state_type& rng_state,
                interleaved<typename T::real_type> state_next);

template <typename T>
__device__
typename T::real_type compare_gpu(
                   const interleaved<typename T::real_type> state,
                   const typename T::data_type& data,
                   interleaved<int> internal_int,
                   interleaved<typename T::real_type> internal_real,
                   const int * shared_int,
                   const typename T::real_type * shared_real,
                   typename T::rng_state_type& rng_state);

// __global__ for shuffling particles
template <typename real_type>
__global__
void scatter_device(const size_t* index,
                    real_type* state,
                    real_type* scatter_state,
                    const size_t n_state,
                    const size_t n_particles,
                    bool selected) {
  int state_size = n_state * n_particles;
#ifdef __NVCC__
  // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < state_size;
       i += blockDim.x * gridDim.x) {
#else
  for (int i = 0; i < state_size; ++i) {
#endif
    int scatter_index;
    if (selected) {
      // Scattering n_state with index
      // e.g. 3, 2, 4
      // 2*np, 2*np + 1, ..., 2*np + (np - 1), 1*np,
      scatter_index = index[i / n_particles] * n_particles + (i % n_particles);
    } else {
      // Scattering n_state_full with index
      // e.g. 4 particles with 3 states ABC stored on device as
      // [1_A, 2_A, 3_A, 4_A, 1_B, 2_B, 3_B, 4_B, 1_C, 2_C, 3_C, 4_C]
      // e.g. index [3, 1, 3, 2] with would be
      // [3_A, 1_A, 3_A, 2_A, 3_B, 1_B, 3_B, 2_B, 3_C, 1_C, 3_C, 2_C]
      // interleaved, i.e. input repeated n_state_full times, plus a strided
      // offset
      // [3, 1, 3, 2, 3 + 4, 1 + 4, 3 + 4, 2 + 4, 3 + 8, 1 + 8, 3 + 8, 2 + 8]
      // [3, 1, 3, 2, 7, 5, 7, 6, 11, 9, 11, 10]
      scatter_index = index[i % n_particles] + (i / n_particles) * n_particles;
    }
    scatter_state[i] = state[scatter_index];
  }
}

template <typename T>
__global__
void run_particles(size_t time_start,
                   size_t time_end,
                   size_t n_particles,
                   size_t n_pars,
                   typename T::real_type * state,
                   typename T::real_type * state_next,
                   int * internal_int,
                   typename T::real_type * internal_real,
                   size_t n_shared_int, size_t n_shared_real,
                   const int * shared_int,
                   const typename T::real_type * shared_real,
                   typename T::rng_state_type::int_type * rng_state,
                   bool use_shared_int,
                   bool use_shared_real) {
  using real_type = typename T::real_type;
  using rng_state_type = typename T::rng_state_type;
  using rng_int_type = typename rng_state_type::int_type;
  const size_t n_particles_each = n_particles / n_pars;
  const auto data = nullptr;
  const bool data_is_shared = false;

#ifdef __CUDA_ARCH__
  const int block_per_pars = (n_particles_each + blockDim.x - 1) / blockDim.x;
  int j;
  if (use_shared_int || use_shared_real) {
    j = blockIdx.x / block_per_pars;
  } else {
    j = (blockIdx.x * blockDim.x + threadIdx.x) / n_particles_each;
  }
  device_ptrs<T> shared_state =
    load_shared_state<T>(j,
                         n_shared_int,
                         n_shared_real,
                         shared_int,
                         shared_real,
                         data,             // nullptr
                         use_shared_int,
                         use_shared_real,
                         data_is_shared);  // false

  int i, max_i;
  if (use_shared_int || use_shared_real) {
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
    device_ptrs<T> shared_state =
      load_shared_state<T>(j,
                           n_shared_int,
                           n_shared_real,
                           shared_int,
                           shared_real,
                           data,             // nullptr
                           use_shared_int,   // ignored
                           use_shared_real,  // ignored
                           data_is_shared);  // false
#endif
    interleaved<real_type> p_state(state, i, n_particles);
    interleaved<real_type> p_state_next(state_next, i, n_particles);
    interleaved<int> p_internal_int(internal_int, i, n_particles);
    interleaved<real_type> p_internal_real(internal_real, i, n_particles);
    interleaved<rng_int_type> p_rng(rng_state, i, n_particles);

    rng_state_type rng_block = get_rng_state<rng_state_type>(p_rng);
    for (size_t time = time_start; time < time_end; ++time) {
      update_gpu<T>(time,
                    p_state,
                    p_internal_int,
                    p_internal_real,
                    shared_state.shared_int,
                    shared_state.shared_real,
                    rng_block,
                    p_state_next);
      SYNCWARP

      interleaved<real_type> tmp = p_state;
      p_state = p_state_next;
      p_state_next = tmp;
    }
    put_rng_state(rng_block, p_rng);
  }
}


// NOTE: there's an unfortunate overloading here where
// "data_is_shared" refers to data being shared across parameters,
// while use_shared_{int,real} refers to whether int and real
// parameters should be stored in shared memory.
template <typename T>
__global__
  void compare_particles(size_t n_particles,
                         size_t n_pars,
                         typename T::real_type * state,
                         typename T::real_type * weights,
                         int * internal_int,
                         typename T::real_type * internal_real,
                         size_t n_shared_int,
                         size_t n_shared_real,
                         const int * shared_int,
                         const typename T::real_type * shared_real,
                         const typename T::data_type * data,
                         typename T::rng_state_type::int_type * rng_state,
                         bool use_shared_int,
                         bool use_shared_real,
                         bool data_is_shared) {
  // This setup is mostly shared with run_particles
  using real_type = typename T::real_type;
  using rng_state_type = typename T::rng_state_type;
  using rng_int_type = typename rng_state_type::int_type;
  const size_t n_particles_each = n_particles / n_pars;

#ifdef __CUDA_ARCH__
  const int block_per_pars = (n_particles_each + blockDim.x - 1) / blockDim.x;
  int j;
  if (use_shared_int || use_shared_real) {
    j = blockIdx.x / block_per_pars;
  } else {
    j = (blockIdx.x * blockDim.x + threadIdx.x) / n_particles_each;
  }
  device_ptrs<T> shared_state =
    load_shared_state<T>(j,
                         n_shared_int,
                         n_shared_real,
                         shared_int,
                         shared_real,
                         data,
                         use_shared_int,
                         use_shared_real,
                         data_is_shared);

  // Particle index i, and max index to process in the block
  int i, max_i;
  if (use_shared_int || use_shared_real) {
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
    device_ptrs<T> shared_state =
      load_shared_state<T>(j,
                           n_shared_int,
                           n_shared_real,
                           shared_int,
                           shared_real,
                           data,
                           use_shared_int,
                           use_shared_real,
                           data_is_shared);
#endif
    interleaved<real_type> p_state(state, i, n_particles);
    interleaved<int> p_internal_int(internal_int, i, n_particles);
    interleaved<real_type> p_internal_real(internal_real, i, n_particles);
    interleaved<rng_int_type> p_rng(rng_state, i, n_particles);
    rng_state_type rng_block = get_rng_state<rng_state_type>(p_rng);

    weights[i] = compare_gpu<T>(p_state,
                                *shared_state.data,
                                p_internal_int,
                                p_internal_real,
                                shared_state.shared_int,
                                shared_state.shared_real,
                                rng_block);
    SYNCWARP
    put_rng_state(rng_block, p_rng);
  }
}

// Likely not particularly CUDA friendly, but will do for now
// (better alternative would be merge, as both lists sorted)
template <typename T>
__device__ size_t binary_interval_search(const T * array,
                                         const size_t array_len,
                                         const T search,
                                         const T offset) {
  size_t l_pivot = 0;
  size_t r_pivot = array_len;
  while (l_pivot < r_pivot) {
    const size_t m = std::floor((l_pivot + r_pivot) / static_cast<T>(2.0));
    if ((array[m] - offset) < search) {
      l_pivot = m + 1;
    } else {
      r_pivot = m;
    }
  }
  SYNCWARP
  return l_pivot;
}

// index = findInterval(u, cumsum(weights / sum(weights)))
// same as
// index = findInterval(u * cumsum(weights)[n], cumsum(weights))
template <typename real_type>
__global__
void find_intervals(const real_type * cum_weights,
                    const size_t n_particles, const size_t n_pars,
                    size_t * index, const real_type * u) {
  const size_t n_particles_each = n_particles / n_pars;
#ifdef __NVCC__
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_particles;
       i += blockDim.x * gridDim.x) {
#else
  for (size_t i = 0; i < n_particles; ++i) {
#endif
    const int par_idx = i / n_particles_each;
    real_type start_val = par_idx > 0 ? cum_weights[par_idx * n_particles_each - 1] : 0;
    real_type normalising_constant =
      cum_weights[(par_idx + 1) * n_particles_each - 1] - start_val;
    if (normalising_constant == 0) {
      index[i] = i;
    } else {
      real_type u_particle = normalising_constant /
                          static_cast<real_type>(n_particles_each) *
                          (u[par_idx] + i % n_particles_each);
      index[i] = binary_interval_search(
        cum_weights + par_idx * n_particles_each,
        n_particles_each, u_particle, start_val) + par_idx * n_particles_each;
    }
#ifdef __NVCC__
  }
#else
  }
#endif
}

}
}

#endif
