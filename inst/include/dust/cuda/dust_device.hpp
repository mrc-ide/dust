#ifndef DUST_CUDA_DUST_DEVICE_HPP
#define DUST_CUDA_DUST_DEVICE_HPP

#include <algorithm>
#include <memory>
#include <map>
#include <stdexcept>
#include <sstream>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <dust/cuda/cuda.hpp>

#include <dust/random/prng.hpp>
#include <dust/random/density.hpp>
#include <dust/filter_state.hpp>
#include <dust/filter_tools.hpp>
#include <dust/cuda/types.hpp>
#include <dust/utils.hpp>
#include <dust/particle.hpp>

#include <dust/cuda/kernels.hpp>
#include <dust/cuda/device_resample.hpp>
#include <dust/cuda/launch_control.hpp>
#include <dust/interface/cuda.hpp>

namespace dust {
// namespace cuda { //?

// We expect that the gpu versions will be very different, so let's
// not use any inheritance.  The important thing is that we hit the
// right external interface, but that's not really that defined either.

template <typename T>
class DustDevice {
public:
  typedef dust::pars_type<T> pars_type;
  typedef typename T::real_type real_type;
  typedef typename T::data_type data_type;
  typedef typename T::rng_state_type rng_state_type;
  typedef typename rng_state_type::int_type rng_int_type;

  DustDevice(const pars_type& pars, const size_t step, const size_t n_particles,
             const size_t n_threads, const std::vector<rng_int_type>& seed,
             const cuda::device_config& device_config) :
    n_pars_(0),
    n_particles_each_(n_particles),
    n_particles_total_(n_particles),
    pars_are_shared_(true),
    n_threads_(n_threads),
    // rng_(n_particles_total_ + 1, seed, deterministic), // +1 for filter
    // errors_(n_particles_total_),
    shape_({n_particles}),
    device_config_(device_config) {
    // initialise(pars, step, n_particles, true);
    // initialise_index();
  }

  DustDevice(const std::vector<pars_type>& pars, const size_t step,
             const size_t n_particles, const size_t n_threads,
             const std::vector<rng_int_type>& seed,
             const std::vector<size_t>& shape,
             const cuda::device_config& device_config) :
    n_pars_(pars.size()),
    n_particles_each_(n_particles == 0 ? 1 : n_particles),
    n_particles_total_(n_particles_each_ * pars.size()),
    pars_are_shared_(n_particles != 0),
    n_threads_(n_threads),
    // rng_(n_particles_total_ + 1, seed, deterministic), // +1 for filter
    // errors_(n_particles_total_),
    device_config_(device_config) {
    // initialise(pars, step, n_particles, true);
    // initialise_index();
    std::runtime_error("not implemented");
  }

  const std::vector<size_t>& shape() const {
    return shape_;
  }

private:
  const size_t n_pars_; // 0 in the "single" case, >=1 otherwise
  const size_t n_particles_each_; // Particles per parameter set
  const size_t n_particles_total_; // Total number of particles
  const bool pars_are_shared_; // Does the n_particles dimension exist in shape?
  size_t n_threads_;
  // rng
  // errors
  std::vector<size_t> shape_; // shape of output
  cuda::device_config device_config_;
};

// } ?
}

#endif
