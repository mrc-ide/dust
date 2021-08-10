#ifndef DUST_CUDA_DEVICE_STATE_CUH
#define DUST_CUDA_DEVICE_STATE_CUH

#include <dust/cuda/cuda.hpp>
#include <dust/cuda/types.hpp>

namespace dust {

// This function loads the shared state (shared_int and shared_real) for a
// single set of parameters into __shared__ memory. If data is set, provided
// it will also be loaded.
// Returns a struct of pointers to the start of shared_int, shared_real and
// data in shared memory.

// Each parameter set must be run in a single CUDA block for this to work
// correctly. If this is not possible (due to being larger than the size of
// L1 cache, or fewer than blockSize particles per parameter) set
// use_shared_L1 = false, which will not load, and simply return pointers
// to the correct start of the shared state in global memory.
// This is also the default behaviour for non-NVCC compiled code through
// this function (which does not have __shared__ memory).
template <typename T>
DEVICE dust::device_ptrs<T> load_shared_state(const int pars_idx,
                                              const size_t n_shared_int,
                                              const size_t n_shared_real,
                                              const int * shared_int,
                                              const typename T::real_t * shared_real,
                                              const typename T::data_t * data,
                                              bool use_shared_int,
                                              bool use_shared_real) {
  dust::device_ptrs<T> ptrs;

  // Get start address in shared space
  ptrs.shared_int = shared_int + pars_idx * n_shared_int;
  ptrs.shared_real = shared_real + pars_idx * n_shared_real;
  ptrs.data = data + pars_idx;

#ifdef __NVCC__
  typedef typename T::real_t real_t;
  typedef typename T::data_t data_t;

  // If we're using it, use the first warp in the block to load the shared pars
  // into __shared__ L1
  extern __shared__ int shared_block[];
  auto block = cooperative_groups::this_thread_block();
  static_assert(sizeof(real_t) >= sizeof(int),
                "real_t and int shared memory not alignable");
  if (use_shared_int) {
    int * shared_block_int = shared_block;
    dust::cuda::shared_mem_cpy(block, shared_block_int, ptrs.shared_int,
                               n_shared_int);
    ptrs.shared_int = shared_block_int;
  }

  if (use_shared_real) {
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

    // Copy data in, which is aligned to 16-bytes
    if (data != nullptr && sizeof(data_t) > 0) {
      size_t data_ptr_start = n_shared_real +
        dust::utils::align_padding(real_ptr_start * sizeof(int) +
                                   n_shared_real * sizeof(real_t),
                                   16) / sizeof(real_t);
      data_t * shared_block_data = (data_t*)&shared_block_real[data_ptr_start];
      dust::cuda::shared_mem_cpy(block, shared_block_data, ptrs.data, 1);
      ptrs.data = shared_block_data;
    } else {
      ptrs.data = nullptr;
    }
  }

  // Required to sync loads into L1 cache
  dust::cuda::shared_mem_wait(block);
#endif

  return ptrs;
}

}

#endif
