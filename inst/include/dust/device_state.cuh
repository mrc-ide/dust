// -*- c++ -*-
#ifndef DUST_DEVICE_STATE_CUH
#define DUST_DEVICE_STATE_CUH

#include <dust/cuda.cuh>
#include <dust/types.hpp>

template <typename T>
DEVICE dust::device_ptrs<typename T::real_t, typename T::data_t>
load_shared_state(size_t n_particles,
                  size_t n_pars,
                  size_t n_shared_int, size_t n_shared_real,
                  const int * shared_int,
                  const typename T::real_t * shared_real,
                  const typename T::data_t * data
                  bool use_shared_L1) {
  typedef typename T::real_t real_t;
  typedef typename T::data_t data_t;
  const size_t n_particles_each = n_particles / n_pars;
#ifdef __NVCC__
  // Particle index i, and max index to process in the block
  dust::device_ptrs<T> ptrs;

  // Get pars index j, and start address in shared space
  const int block_per_pars = (n_particles_each + blockDim.x - 1) / blockDim.x;
  const int j = blockIdx.x / block_per_pars;
  ptrs.shared_int = shared_int + j * n_shared_int;
  ptrs.shared_real = shared_real + j * n_shared_real;
  ptrs.shared_data = shared_data + j;

  // If we're using it, use the first warp in the block to load the shared pars
  // into __shared__ L1
  extern __shared__ int shared_block[];
  auto block = cooperative_groups::this_thread_block();
  static_assert(sizeof(real_t) >= sizeof(int),
                "real_t and int shared memory not alignable");
  if (use_shared_L1) {
    int * shared_block_int = shared_block;
    dust::cuda::shared_mem_cpy(block, shared_block_int, ptrs.shared_int,
                               n_shared_int);
    ptrs.shared_int = shared_block_int;

    // Must only have a single __shared__ definition, cast to use different
    // types within it
    // Furthermore, writing must be aligned to the word length (may be an issue
    // with int and real, as odd n_shared_int leaves pointer in the middle of an
    // 8-byte word)
    size_t real_ptr_start = n_shared_int +
      dust::utils::align_padding(n_shared_int * sizeof(int), sizeof(real_t)) / sizeof(int);
    real_t * shared_block_real = (real_t*)&shared_block[real_ptr_start];
    dust::cuda::shared_mem_cpy(block, shared_block_real, ptrs.shared_real,
                               n_shared_real);
    ptrs.shared_real = shared_block_real;

    // Copy data in
    if (sizeof(data_t) > 0) {
      size_t data_ptr_start = n_shared_real +
        dust::utils::align_padding(n_shared_real * sizeof(real_t), 16) / sizeof(real_t);
      data_t * shared_block_data = (data_t*)&shared_block[data_ptr_start];
      dust::cuda::shared_mem_cpy(block, shared_block_data, ptrs.data, 1);
      ptrs.data = shared_block_data;
    } else {
      ptrs.data = nullptr;
    }
  }

  // Required to sync loads into L1 cache
  dust::cuda::shared_mem_wait(block);

  return(ptrs);
}