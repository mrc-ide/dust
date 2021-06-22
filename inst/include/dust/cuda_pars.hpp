#include <dust/types.hpp>

namespace dust {

template <typename real_t>
cuda_launch set_cuda_pars(int device_id,
                          size_t n_particles,
                          size_t n_particles_each,
                          size_t n_state,
                          size_t n_state_full,
                          size_t n_shared_int,
                          size_t n_shared_real,
                          size_t data_size,
                          size_t shared_size) {
  if (device_id < 0) {
    // this forces default initialisation of all struct elements:
    return cuda_launch{};
  }

  cuda_launch cuda_pars;

  // Alternatively we take n_pars here?
  const size_t n_pars_effective = n_particles / n_particles_each;
  const int warp_size = dust::cuda::warp_size;

  size_t warp_blockSize =
    warp_size * (n_particles_each + warp_size - 1) / warp_size;

  // Run kernel
  size_t n_shared_int_effective = n_shared_int +
    dust::utils::align_padding(n_shared_int * sizeof(int),
                               sizeof(real_t)) / sizeof(int);
  size_t shared_size_int_bytes = n_shared_int_effective * sizeof(int);
  size_t shared_size_both_bytes =
    shared_size_int_bytes + n_shared_real * sizeof(real_t);

  if (n_particles_each < warp_size) {
    cuda_pars.run_L1_int = false;
    cuda_pars.run_L1_real = false;
  } else {
    cuda_pars.run_L1_int = shared_size_int_bytes <= shared_size;
    cuda_pars.run_L1_real = shared_size_both_bytes <= shared_size;
  }

  cuda_pars.run_shared_size_bytes =
    cuda_pars.run_L1_int * n_shared_int_effective * sizeof(int) +
    cuda_pars.run_L1_real * n_shared_real * sizeof(real_t);

  cuda_pars.run_blockSize = 128;
  if (cuda_pars.run_L1_int || cuda_pars.run_L1_real) {
    // Either (or both) int and real will fit into shared (L1
    // cache), each block runs a pars set. Each pars set has enough
    // blocks to run all of its particles, the final block may have
    // some threads that don't do anything (hang off the end)
    cuda_pars.run_blockSize =
      std::min(cuda_pars.run_blockSize, warp_blockSize);
    cuda_pars.run_blockCount =
      n_pars_effective *
      (n_particles_each + cuda_pars.run_blockSize - 1) /
      cuda_pars.run_blockSize;
  } else {
    // If not enough particles per pars to make a whole block use
    // shared, or if shared_t too big for L1, turn it off, and run
    // in 'classic' mode where each particle is totally independent
    // cuda_pars.run_blockSize = 128;
    cuda_pars.run_blockCount =
      (n_particles + cuda_pars.run_blockSize - 1) /
      cuda_pars.run_blockSize;
  }

  // Compare kernel
  cuda_pars.compare_blockSize = 128;
  cuda_pars.compare_L1_int = true;
  cuda_pars.compare_L1_real = true;
  // Compare uses data_t too, with real aligned to 16-bytes, so has a larger
  // shared memory requirement
  size_t n_shared_real_effective = n_shared_real +
    dust::utils::align_padding(n_shared_int_effective * sizeof(int) +
                               n_shared_real * sizeof(real_t),
                               16) / sizeof(real_t);
  cuda_pars.compare_shared_size_bytes = n_shared_int_effective * sizeof(int) +
    n_shared_real_effective * sizeof(real_t) +
    data_size;
  if (n_particles_each < warp_size || cuda_pars.compare_shared_size_bytes > shared_size) {
    // If not enough particles per pars to make a whole block use
    // shared, or if shared_t too big for L1, turn it off, and run
    // in 'classic' mode where each particle is totally independent
    cuda_pars.compare_L1_int = false;
    cuda_pars.compare_L1_real = false;
    cuda_pars.compare_shared_size_bytes = 0;
    cuda_pars.compare_blockCount = (n_particles + cuda_pars.compare_blockSize - 1) / cuda_pars.compare_blockSize;
  } else {
    // If it's possible to make blocks with shared_t in L1 cache,
    // each block runs a pars set. Each pars set has enough blocks
    // to run all of its particles, the final block may have some
    // threads that don't do anything (hang off the end)
    // This is nocov as it requires __shared__ to exist (so shared_size > 0)
    cuda_pars.compare_blockSize =                              // #nocov start
      std::min(cuda_pars.compare_blockSize, warp_blockSize);
    cuda_pars.compare_blockCount =
      n_pars_effective *
      (n_particles_each + cuda_pars.compare_blockSize - 1) /
      cuda_pars.compare_blockSize;                             // #nocov end
  }

  // Reorder kernel
  cuda_pars.reorder_blockSize = 128;
  cuda_pars.reorder_blockCount =
    (n_particles * n_state_full + cuda_pars.reorder_blockSize - 1) / cuda_pars.reorder_blockSize;

  // Scatter kernels
  cuda_pars.scatter_blockSize = 64;
  cuda_pars.scatter_blockCount =
    (n_particles * n_state_full + cuda_pars.scatter_blockSize - 1) / cuda_pars.scatter_blockSize;

  cuda_pars.index_scatter_blockSize = 64;
  cuda_pars.index_scatter_blockCount =
    (n_particles * n_state + cuda_pars.index_scatter_blockSize - 1) / cuda_pars.index_scatter_blockSize;

  // Interval kernel
  cuda_pars.interval_blockSize = 128;
  cuda_pars.interval_blockCount =
    (n_particles + cuda_pars.interval_blockSize - 1) / cuda_pars.interval_blockSize;

  return cuda_pars;
}

}
