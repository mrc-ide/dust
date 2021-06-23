#ifndef DUST_CUDA_LAUNCH_CONTROL_HPP
#define DUST_CUDA_LAUNCH_CONTROL_HPP

#include <dust/types.hpp>

namespace dust {

namespace cuda {

struct launch_control {
  size_t block_size;
  size_t block_count;
  size_t shared_size_bytes;
  bool shared_int;
  bool shared_real;
};

class launch_control_dust {
public:
  launch_control_dust(int device_id,
                      size_t n_particles, size_t n_particles_each,
                      size_t n_state, size_t n_state_full,
                      size_t n_shared_int, size_t n_shared_real,
                      size_t real_size, size_t data_size,
                      size_t shared_size);
  launch_control_dust();
  launch_control run;
  launch_control compare;
  launch_control reorder;
  launch_control scatter;
  launch_control index_scatter;
  launch_control interval;
};

inline void set_block_size(launch_control &ctrl, size_t block_size, size_t n) {
  ctrl.block_size = block_size;
  ctrl.block_count = (n + block_size - 1) / block_size;
}

inline launch_control launch_control_model(size_t n_particles,
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

  launch_control ret;
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
    set_block_size(ret, 128, n_particles);
  }

  return ret;
}


inline launch_control launch_control_simple(size_t block_size, size_t n) {
  launch_control ret{};
  set_block_size(ret, block_size, n);
  return ret;
}


inline launch_control_dust::launch_control_dust() {
  run = launch_control{};
  compare = launch_control{};
  reorder = launch_control{};
  scatter = launch_control{};
  index_scatter = launch_control{};
  interval = launch_control{};
}


inline launch_control_dust::launch_control_dust(int device_id,
                                                size_t n_particles,
                                                size_t n_particles_each,
                                                size_t n_state,
                                                size_t n_state_full,
                                                size_t n_shared_int,
                                                size_t n_shared_real,
                                                size_t real_size,
                                                size_t data_size,
                                                size_t shared_size) {
  if (device_id < 0) {
    run = launch_control{};
    compare = launch_control{};
    reorder = launch_control{};
    scatter = launch_control{};
    index_scatter = launch_control{};
    interval = launch_control{};
  } else {
    run = launch_control_model(n_particles, n_particles_each,
                                   n_shared_int, n_shared_real,
                                   real_size, 0, shared_size);
    compare = launch_control_model(n_particles, n_particles_each,
                                       n_shared_int, n_shared_real,
                                       real_size, data_size, shared_size);

    reorder       = launch_control_simple(128, n_particles * n_state_full);
    scatter       = launch_control_simple( 64, n_particles * n_state_full);
    index_scatter = launch_control_simple( 64, n_particles * n_state);
    interval      = launch_control_simple(128, n_particles);
  }
}

}

}

#endif
