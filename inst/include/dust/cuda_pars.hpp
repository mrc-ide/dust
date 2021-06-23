#include <dust/types.hpp>

namespace dust {

inline void set_block_size(cuda_launch_control &ctrl,
                           size_t n, size_t block_size) {
  ctrl.block_size = block_size;
  ctrl.block_count = (n + block_size - 1) / block_size;
}

// Run / Compare kernel
//
// For the compare version we must also cope with the size of the data
// element too.
inline cuda_launch_control launch_control_model(size_t n_particles,
                                                size_t n_particles_each,
                                                size_t n_shared_int,
                                                size_t n_shared_real,
                                                size_t real_size,
                                                size_t data_size,
                                                size_t shared_size) {
  const size_t int_size = sizeof(int);

  const size_t n_pars_effective = n_particles / n_particles_each;
  const int warp_size = dust::cuda::warp_size;
  const size_t warp_block_size =
    warp_size * (n_particles_each + warp_size - 1) / warp_size;
  const size_t n_shared_int_effective = n_shared_int +
    dust::utils::align_padding(n_shared_int * int_size,
                               real_size) / int_size;
  const size_t shared_size_int_bytes = n_shared_int_effective * int_size;

  const size_t real_align = data_size == 0 ? real_size : 16;
  const size_t n_shared_real_effective = n_shared_real +
    dust::utils::align_padding(shared_size_int_bytes +
                               n_shared_real * real_size,
                               real_align) / real_size;

  const size_t shared_size_both_bytes =
    shared_size_int_bytes +
    n_shared_real_effective * real_size +
    data_size;

  cuda_launch_control ret;
  if (n_particles_each < warp_size) {
    ret.shared_int = false;
    ret.shared_real = false;
    ret.shared_size_bytes = 0;
  } else {
    ret.shared_int = shared_size_int_bytes <= shared_size;
    ret.shared_real = shared_size_both_bytes <= shared_size;
    if (ret.shared_real) {
      ret.shared_size_bytes = shared_size_both_bytes;
    } else if (ret.shared_int) {
      ret.shared_size_bytes = shared_size_int_bytes;
    } else {
      ret.shared_size_bytes = 0;
    }
  }

  if (ret.shared_int || ret.shared_real) {
    // Either (or both) int and real will fit into shared (L1
    // cache), each block runs a pars set. Each pars set has enough
    // blocks to run all of its particles, the final block may have
    // some threads that don't do anything (hang off the end)
    //
    // TODO: This is the only bit of block size calculation that is
    // different to set_block_size, John can you expand why?
    ret.block_size = std::min(static_cast<size_t>(128), warp_block_size);
    ret.block_count =
      n_pars_effective * (n_particles_each + ret.block_size - 1) /
      ret.block_size;
  } else {
    // If not enough particles per pars to make a whole block use
    // shared, or if shared_t too big for L1, turn it off, and run
    // in 'classic' mode where each particle is totally independent
    // ret.block_size = 128;
    set_block_size(ret, n_particles, 128);
  }

  return ret;
}


// TODO: rename!
inline cuda_launch_dust set_cuda_pars(int device_id,
                                      size_t n_particles,
                                      size_t n_particles_each,
                                      size_t n_state,
                                      size_t n_state_full,
                                      size_t n_shared_int,
                                      size_t n_shared_real,
                                      size_t real_size,
                                      size_t data_size,
                                      size_t shared_size) {
  cuda_launch_dust cuda_pars{};

  if (device_id < 0) {
    return cuda_pars;
  }

  cuda_pars.run = launch_control_model(n_particles, n_particles_each,
                                       n_shared_int, n_shared_real,
                                       real_size, 0, shared_size);
  cuda_pars.compare = launch_control_model(n_particles, n_particles_each,
                                           n_shared_int, n_shared_real,
                                           real_size, data_size, shared_size);

  set_block_size(cuda_pars.reorder, n_particles * n_state_full, 128);
  set_block_size(cuda_pars.scatter, n_particles * n_state_full, 64);
  set_block_size(cuda_pars.index_scatter, n_particles * n_state, 64);
  set_block_size(cuda_pars.interval, n_particles, 128);

  return cuda_pars;
}

}
