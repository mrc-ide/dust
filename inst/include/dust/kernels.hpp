#ifndef DUST_KERNELS_HPP
#define DUST_KERNELS_HPP

#include <assert.h>

// This is the main model update, will be defined by the model code
// (see inst/examples/variable.cpp for an example)
template <typename T>
DEVICE void update_device(size_t step,
                   const dust::interleaved<typename T::real_t> state,
                   dust::interleaved<int> internal_int,
                   dust::interleaved<typename T::real_t> internal_real,
                   const int * shared_int,
                   const typename T::real_t * shared_real,
                   dust::rng_state_t<typename T::real_t>& rng_state,
                   dust::interleaved<typename T::real_t> state_next);

// __global__ for shuffling particles
template<typename real_t>
KERNEL void scatter_device(int* scatter_index,
                           real_t* state,
                           real_t* scatter_state,
                           size_t state_size) {
#ifdef __NVCC__
  // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < state_size;
       i += blockDim.x * gridDim.x) {
#else
  for (size_t i = 0; i < state_size; ++i) {
#endif
    scatter_state[i] = state[scatter_index[i]];
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
  // Particle index i, and max index to process in the block
  int i, max_i;

  // Get pars index j, and start address in shared space
  const int block_per_pars = (n_particles_each + blockDim.x - 1) / blockDim.x;
  const int j = blockIdx.x / block_per_pars;
  const int * p_shared_int = shared_int + j * n_shared_int;
  const real_t * p_shared_real = shared_real + j * n_shared_real;

  // If we're using it, use the first warp in the block to load the shared pars
  // into __shared__ L1
  extern __shared__ int shared_block[];
  auto block = cooperative_groups::this_thread_block();
  if (use_shared_L1) {
    int * shared_block_int = shared_block;
    shared_mem_cpy(block, shared_block_int, p_shared_int, n_shared_int);
    p_shared_int = shared_block_int;

    // Must only have a single __shared__ definition, cast to use different
    // types within it
    // Furthermore, writing must be aligned to the word length (may be an issue
    // with int and real, as odd n_shared_int leaves pointer in the middle of an
    // 8-byte word)
    assert(sizeof(real_t) > sizeof(int));
    size_t real_ptr_start = n_shared_int +
      align_padding(n_shared_int * sizeof(int), sizeof(real_t)) / sizeof(int);
    real_t * shared_block_real = (real_t*)&shared_block[real_ptr_start];
    shared_mem_cpy(block, shared_block_real, p_shared_real, n_shared_real);
    p_shared_real = shared_block_real;

    // Pick particle index based on block, don't process if off the end
    i = j * n_particles_each + (blockIdx.x % block_per_pars) * blockDim.x +
      threadIdx.x;
    max_i = n_particles_each * (j + 1);
  } else {
    // Otherwise CUDA thread number = particle
    i = blockIdx.x * blockDim.x + threadIdx.x;
    max_i = n_particles;
  }

  // Required to sync loads into L1 cache
  shared_mem_wait(block);

  if (i < max_i) {
#else
  // omp here
  for (size_t i = 0; i < n_particles; ++i) {
    const int j = i / n_particles_each;
    const int * p_shared_int = shared_int + j * n_shared_int;
    const real_t * p_shared_real = shared_real + j * n_shared_real;
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
                       p_shared_int,
                       p_shared_real,
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
#ifdef __CUDA_ARCH__
    block.sync();
#endif
  }
}

#endif
