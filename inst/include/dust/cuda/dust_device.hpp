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
    initialise_index();
  }

  DustDevice(const std::vector<pars_type>& pars, const size_t step,
             const size_t n_particles, const size_t n_threads,
             const std::vector<rng_int_type>& seed,
             const std::vector<size_t>& shape,
             const cuda::device_config& device_config) :
    n_pars_(pars.size()),
    n_particles_each_(n_particles == 0 ? 1 : n_particles),
    n_particles_total_(n_particles_each_ * pars.size()),
    n_state_(0),
    pars_are_shared_(n_particles != 0),
    n_threads_(n_threads),
    // rng_(n_particles_total_ + 1, seed, deterministic), // +1 for filter
    // errors_(n_particles_total_),
    device_config_(device_config) {
    // initialise(pars, step, n_particles, true);
    initialise_index();
    std::runtime_error("not implemented");
  }

  // It's the callee's responsibility to ensure that index is in
  // range [0, n-1]
  void set_index(const std::vector<size_t>& index) {
    index_ = index;
    update_device_index();
  }

  void run(const size_t step_end) {
    // before doing this we do need to have done the equivalent of
    // refresh_device of course.
    if (step_end > step_) {
      const size_t step_start = step_;
#ifdef __NVCC__
      dust::cuda::run_particles<T><<<cuda_pars_.run.block_count,
                                     cuda_pars_.run.block_size,
                                     cuda_pars_.run.shared_size_bytes,
                                     kernel_stream_.stream()>>>(
                      step_start, step_end, n_particles(),
                      n_pars_effective(),
                      device_state_.y.data(), device_state_.y_next.data(),
                      device_state_.internal_int.data(),
                      device_state_.internal_real.data(),
                      device_state_.n_shared_int,
                      device_state_.n_shared_real,
                      device_state_.shared_int.data(),
                      device_state_.shared_real.data(),
                      device_state_.rng.data(),
                      cuda_pars_.run.shared_int,
                      cuda_pars_.run.shared_real);
      kernel_stream_.sync();
#else
      dust::cuda::run_particles<T>(step_start, step_end, n_particles(),
                      n_pars_effective(),
                      device_state_.y.data(), device_state_.y_next.data(),
                      device_state_.internal_int.data(),
                      device_state_.internal_real.data(),
                      device_state_.n_shared_int,
                      device_state_.n_shared_real,
                      device_state_.shared_int.data(),
                      device_state_.shared_real.data(),
                      device_state_.rng.data(),
                      false,
                      false);
#endif

      // In the inner loop, the swap will keep the locally scoped
      // interleaved variables updated, but the interleaved variables
      // passed in have not yet been updated.  If an even number of
      // steps have been run state will have been swapped back into the
      // original place, but an on odd number of steps the passed
      // variables need to be swapped.
      if ((step_end - step_start) % 2 == 1) {
        device_state_.swap();
      }

      select_needed_ = true;
      step_ = step_end;
    }
  }

  // Simple const methods, as for Dust
  size_t n_threads() const {
    return n_threads_;
  }

  size_t n_particles() const {
    return n_particles_total_;
  }

  size_t n_state() const {
    // return index_.size(); // TODO
    return n_state_;
  }

  size_t n_state_full() const {
    return n_state_;
  }

  size_t n_pars() const {
    return n_pars_;
  }

  size_t n_pars_effective() const {
    return n_pars_ == 0 ? 1 : n_pars_;
  }

  bool pars_are_shared() const {
    return pars_are_shared_;
  }

  // size_t n_data() const {
  //   return data_.size();
  // }
  
  // const std::map<size_t, std::vector<data_type>>& data() const {
  //   return data_;
  // }

  // const std::map<size_t, size_t>& data_offsets() const {
  //   return device_data_offsets_;
  // }

  size_t step() const {
    return step_;
  }
  
  const std::vector<size_t>& shape() const {
    return shape_;
  }

private:
  const size_t n_pars_; // 0 in the "single" case, >=1 otherwise
  const size_t n_particles_each_; // Particles per parameter set
  const size_t n_particles_total_; // Total number of particles
  size_t n_state_;
  const bool pars_are_shared_; // Does the n_particles dimension exist in shape?
  size_t n_threads_;
  // rng
  // errors
  std::vector<size_t> shape_; // shape of output

  // This will want to change because we don't ever want to keep it as
  // a shared pointer.
  std::vector<dust::shared_ptr<T>> shared_;

  // This can be made more transient too
  std::vector<size_t> index_;
  
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
  size_t step_;

  // This is largely the old 
  void initialise(const pars_type& pars, const size_t step,
                  const size_t n_particles, bool set_state) {
    dust::Particle<T> p(pars, step);
    if (n_state_ == 0) { // first initialisation
      n_state_ = p.size();
    } else if (p.size() != n_state_) {
      std::stringstream msg;
      msg << "'pars' created inconsistent state size: " <<
        "expected length " << n_state_ << " but created length " <<
        p.size();
      throw std::invalid_argument(msg.str());
    }

    std::vector<Particle<T>> particles;
    particles.reserve(n_particles);
    for (size_t i = 0; i < n_particles; ++i) {
      particles.push_back(p);
    }
    shared_ = {pars.shared};

    // very likely that these can be grouped together:
    initialise_device_state();
    // update_device_shared();
    // set_cuda_launch();

    step_ = step;
    select_needed_ = true;
  }

  void initialise_device_state() {
    if (!device_config_.enabled_) {
      return;
    }
    const auto s = shared_[0];
    const size_t n_internal_int = dust::cuda::device_internal_int_size<T>(s);
    const size_t n_internal_real = dust::cuda::device_internal_real_size<T>(s);
    const size_t n_shared_int = dust::cuda::device_shared_int_size<T>(s);
    const size_t n_shared_real = dust::cuda::device_shared_real_size<T>(s);
    device_state_.initialise(n_particles(), n_state_,
                             n_pars_effective(), shared_.size(),
                             n_internal_int, n_internal_real,
                             n_shared_int, n_shared_real);
  }

  void update_device_shared() {
    if (!device_config_.enabled_) {
      return;
    }
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
  }

  void initialise_index() {
    const size_t n = n_state_full();
    index_.clear();
    index_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      index_.push_back(i);
    }
    update_device_index();
  }

  void set_cuda_launch() {
    cuda_pars_ = dust::cuda::launch_control_dust(device_config_,
                                                 n_particles(),
                                                 n_particles_each_,
                                                 n_state(),
                                                 n_state_full(),
                                                 device_state_.n_shared_int,
                                                 device_state_.n_shared_real,
                                                 sizeof(real_type),
                                                 sizeof(data_type));
  }

  void update_device_index() {
    if (!device_config_.enabled_) {
      return;
    }
    device_state_.set_device_index(index_, n_particles_total_, n_state_full());

    select_needed_ = true;
    if (!std::is_sorted(index_.cbegin(), index_.cend())) {
      select_scatter_ = true;
    } else {
      select_scatter_ = false;
    }

    // TODO: get this 64 from the original configuration, if possible.
    cuda_pars_.index_scatter =
      dust::cuda::launch_control_simple(64, n_particles_total_ * n_state());
  }
};

// } ?
}

#endif
