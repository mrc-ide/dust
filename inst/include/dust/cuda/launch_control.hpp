#ifndef DUST_CUDA_LAUNCH_CONTROL_HPP
#define DUST_CUDA_LAUNCH_CONTROL_HPP

#include <dust/cuda/types.hpp>
#include <dust/cuda/utils.hpp>

namespace dust {
namespace cuda {

struct launch_control {
  size_t block_size;
  size_t block_count;
  size_t shared_size_bytes;
  bool shared_int;
  bool shared_real;
};


inline size_t device_shared_size(int device_id) {
  int size = 0;
#ifdef __NVCC__
  if (device_id >= 0) {
    CUDA_CALL(cudaDeviceGetAttribute(&size,
                                     cudaDevAttrMaxSharedMemoryPerBlock,
                                     device_id));
  }
#endif
  return static_cast<size_t>(size);
}


// Tunable bits exposed to the front end
class device_config {
public:
  device_config(int device_id, int run_block_size) :
    device_id_(device_id),
    run_block_size_(run_block_size),
    shared_size_(device_shared_size(device_id_)){
#ifdef __NVCC__
    if (device_id_ >= 0) {
      CUDA_CALL(cudaSetDevice(device_id_));
      CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    }
#endif
  }

  const int device_id_;
  const size_t run_block_size_;
  size_t shared_size_;
};


inline void cuda_profiler_start(const device_config& config) {
#ifdef DUST_USING_CUDA_PROFILER
  if (config.enabled_) {
      CUDA_CALL(cudaProfilerStart());
  }
#endif
}

inline void cuda_profiler_stop(const device_config& config) {
#ifdef DUST_USING_CUDA_PROFILER
  if (config.enabled_) {
      CUDA_CALL(cudaProfilerStop());
  }
#endif
}


class launch_control_dust {
public:
  launch_control_dust(const device_config& config,
                      size_t n_particles, size_t n_particles_each,
                      size_t n_state, size_t n_state_full,
                      size_t n_shared_int, size_t n_shared_real,
                      size_t real_size, size_t data_size);
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
                                           size_t shared_size,
                                           size_t block_size) {
  const size_t int_size = sizeof(int);

  const size_t n_pars_effective = n_particles / n_particles_each;
  const int warp_size = dust::cuda::warp_size;
  const size_t warp_block_size =
    warp_size * (n_particles_each + warp_size - 1) / warp_size;
  const size_t n_shared_int_effective = n_shared_int +
    dust::cuda::utils::align_padding(n_shared_int * int_size,
                                     real_size) / int_size;
  const size_t shared_size_int_bytes = n_shared_int_effective * int_size;

  const size_t real_align = data_size == 0 ? real_size : 16;
  const size_t n_shared_real_effective = n_shared_real +
    dust::cuda::utils::align_padding(shared_size_int_bytes +
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
    // This is the only bit of block size calculation that is
    // different to set_block_size, reasoning from John (see #246):
    //
    // > Each block runs a single set of parameters in multi_pars (as
    // > they have pars loaded into shared memory, which is shared
    // > amongst the block). We may have n_threads % n_pars = 0, in
    // > which case no problem, just run enough blocks to handle all
    // > the threads (as normal). If there is a remainder, some of
    // > last block to will have some idle threads (though these are
    // > used for the shared mem copy, but don't run a particle).
    // >
    // > Here, I found good performance with a block size up to 128,
    // > dropping off above that. But if you have a relatively large
    // > number of multi_pars and n_particles_each < 128 it's wasteful
    // > to leave some many threads idle, so use the nearest multiple
    // > of 32 up to a max of 128
    ret.block_size = std::min(static_cast<size_t>(block_size), warp_block_size);
    ret.block_count =
      n_pars_effective * (n_particles_each + ret.block_size - 1) /
      ret.block_size;
  } else {
    // If not enough particles per pars to make a whole block use
    // shared, or if shared_type too big for L1, turn it off, and run
    // in 'classic' mode where each particle is totally independent
    set_block_size(ret, block_size, n_particles);
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


inline launch_control_dust::launch_control_dust(const device_config& config,
                                                size_t n_particles,
                                                size_t n_particles_each,
                                                size_t n_state,
                                                size_t n_state_full,
                                                size_t n_shared_int,
                                                size_t n_shared_real,
                                                size_t real_size,
                                                size_t data_size) {
  if (config.enabled_) {
    const size_t shared_size = config.shared_size_;
    const size_t run_block_size = config.run_block_size_;
    run = launch_control_model(n_particles, n_particles_each,
                               n_shared_int, n_shared_real,
                               real_size, 0, shared_size, run_block_size);
    compare = launch_control_model(n_particles, n_particles_each,
                                   n_shared_int, n_shared_real,
                                   real_size, data_size, shared_size, 128);

    reorder       = launch_control_simple(128, n_particles * n_state_full);
    scatter       = launch_control_simple( 64, n_particles * n_state_full);
    index_scatter = launch_control_simple( 64, n_particles * n_state);
    interval      = launch_control_simple(128, n_particles);
  } else {
    run = launch_control{};
    compare = launch_control{};
    reorder = launch_control{};
    scatter = launch_control{};
    index_scatter = launch_control{};
    interval = launch_control{};
  }
}

}

}

#endif
