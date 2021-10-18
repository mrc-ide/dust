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
    initialise(pars, step, n_particles, true);
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

  // Device support
  dust::cuda::launch_control_dust cuda_pars_; // TODO => control_ ?
  dust::cuda::device_state<real_type, rng_state_type> device_state_;
  dust::cuda::device_array<data_type> device_data_;
  std::map<size_t, size_t> device_data_offsets_;
  dust::cuda::cuda_stream kernel_stream_;
  dust::cuda::cuda_stream resample_stream_;
  bool select_needed_;
  bool select_scatter_;
  size_t device_step_;

  void initialise(const pars_type& pars, const size_t step,
                  const size_t n_particles, bool set_state) {
    const size_t n = particles_.size() == 0 ? 0 : n_state_full();
    dust::Particle<T> p(pars, step);
    if (n > 0 && p.size() != n) {
      std::stringstream msg;
      msg << "'pars' created inconsistent state size: " <<
        "expected length " << n << " but created length " <<
        p.size();
      throw std::invalid_argument(msg.str());
    }

    if (particles_.size() == n_particles) {
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < n_particles; ++i) {
        particles_[i].set_pars(p, set_state);
      }
      shared_[0] = pars.shared;
    } else {
      particles_.clear();
      particles_.reserve(n_particles);
      for (size_t i = 0; i < n_particles; ++i) {
        particles_.push_back(p);
      }
      shared_ = {pars.shared};
      initialise_device_state();
    }
    // reset_errors();
    update_device_shared();

    const size_t n_shared_int = device_state_.n_shared_int;
    const size_t n_shared_real = device_state_.n_shared_real;
    std::vector<int> shared_int(n_shared_int * n_pars_effective());
    std::vector<real_type> shared_real(n_shared_real * n_pars_effective());
    for (size_t i = 0; i < shared_.size(); ++i) {
      int * dest_int = shared_int.data() + n_shared_int * i;
      real_type * dest_real = shared_real.data() + n_shared_real * i;
      dust::cuda::device_shared_copy<T>(shared_[i], dest_int, dest_real);
    }
    device_state_.shared_int.set_array(shared_int);
    device_state_.shared_real.set_array(shared_real);
    
    set_cuda_launch();

    device_step_ = step;
    stale_host_ = false;
    stale_device_ = true;
    select_needed_ = true;
  }

  
};

// } ?
}

#endif
