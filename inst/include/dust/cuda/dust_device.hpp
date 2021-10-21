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

namespace dust {

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
       const bool deterministic, const cuda::device_config& device_config) :
    n_pars_(0),
    n_particles_each_(n_particles),
    n_particles_total_(n_particles),
    pars_are_shared_(true),
    n_threads_(n_threads),
    device_config_(device_config),
    rng_(n_particles_total_ + 1, seed, deterministic), // +1 for filter
    stale_host_(false),
    stale_device_(true),
    select_needed_(true),
    select_scatter_(false) {
    initialise(pars, step, n_particles, true);
    initialise_index();
    shape_ = {n_particles};
#ifdef DUST_USING_CUDA_PROFILER
    cuda_profiler_start(device_config_);
#endif
  }

  DustDevice(const std::vector<pars_type>& pars, const size_t step,
       const size_t n_particles, const size_t n_threads,
       const std::vector<rng_int_type>& seed,
       const bool deterministic, const cuda::device_config& device_config,
       const std::vector<size_t>& shape) :
    n_pars_(pars.size()),
    n_particles_each_(n_particles == 0 ? 1 : n_particles),
    n_particles_total_(n_particles_each_ * pars.size()),
    pars_are_shared_(n_particles != 0),
    n_threads_(n_threads),
    device_config_(device_config),
    rng_(n_particles_total_ + 1, seed, deterministic),  // +1 for filter
    select_needed_(true),
    select_scatter_(false) {
    initialise(pars, step, n_particles_each_, true);
    initialise_index();
    // constructing the shape here is harder than above.
    if (n_particles > 0) {
      shape_.push_back(n_particles);
    }
    for (auto i : shape) {
      shape_.push_back(i);
    }
#ifdef DUST_USING_CUDA_PROFILER
    cuda_profiler_start(device_config_);
#endif
  }

  // We only need a destructor when running with cuda profiling; don't
  // include ond otherwise because we don't actually follow the rule
  // of 3/5/0
#ifdef DUST_USING_CUDA_PROFILER
  ~Dust() {
    cuda_profiler_stop(device_config_);
  }
#endif

  // TODO - which of these do we need to keep for the interface?
  void set_pars(const pars_type& pars, bool set_state) {
    const size_t n_particles = particles_.size();
    initialise(pars, step(), n_particles, set_state);
  }

  void set_pars(const std::vector<pars_type>& pars, bool set_state) {
    const size_t n_particles = particles_.size();
    initialise(pars, step(), n_particles / pars.size(), set_state);
  }

  // It's the callee's responsibility to ensure this is the correct length:
  //
  // * if is_matrix is false then state must be length n_state_full()
  //   and all particles get the state
  // * if is_matrix is true, state must be length (n_state_full() *
  //   n_particles()) and every particle gets a different state.
  void set_state(const std::vector<real_type>& state) {
    const size_t n_particles = particles_.size();
    const size_t n_state = n_state_full();
    const bool individual = state.size() == n_state * n_particles;
    const size_t n = individual ? 1 : n_particles_each_;
    auto it = state.begin();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < n_particles; ++i) {
      particles_[i].set_state(it + (i / n) * n_state);
    }
    stale_device_ = true;
  }

  void set_step(const size_t step) {
    device_step_ = step;
  }

  // TODO - do we still need to support this?
  void set_step(const std::vector<size_t>& step) {
    const size_t n_particles = particles_.size();
    for (size_t i = 0; i < n_particles; ++i) {
      particles_[i].set_step(step[i]);
    }
    const auto r = std::minmax_element(step.begin(), step.end());
    if (*r.second > *r.first) {
      run(*r.second);
    }
    stale_device_ = true;
  }

  // It's the callee's responsibility to ensure that index is in
  // range [0, n-1]
  void set_index(const std::vector<size_t>& index) {
    index_ = index;
    update_device_index();
  }

  void run(const size_t step_end) {
    if (step_end > device_step_) {
      const size_t step_start = device_step_;
      dust::cuda::run_particles<T><<<cuda_pars_.run.block_count,
                                     cuda_pars_.run.block_size,
                                     cuda_pars_.run.shared_size_bytes,
                                     kernel_stream_.stream()>>>(
                      step_start, step_end, particles_.size(),
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
      device_step_ = step_end;
    }
  }

  std::vector<real_type> simulate(const std::vector<size_t>& step_end) {
    const size_t n_time = step_end.size();
    // The filter snapshot class can be used to store the indexed state
    // (implements async copy, swap space, and deinterleaving)
    // Filter trajctories not used as we don't need order here
    dust::filter::filter_snapshots_device<real_type> state_store;
    state_store.resize(n_state(), n_particles(), step_end);
    for (size_t t = 0; t < n_time; ++t) {
      run_device(step_end[t]);
      state_store.store(device_state_selected());
      state_store.advance();
    }
    std::vector<real_type> ret(n_state() * n_particles() * n_time);
    state_store.history(ret.data());
    return ret;
  }

  // TODO - work out which of these are needed. They'll need a
  // D->H transfer
  void state(std::vector<real_type>& end_state) {
    return state(end_state.begin());
  }

  void state(typename std::vector<real_type>::iterator end_state) {
    size_t np = particles_.size();
    size_t index_size = index_.size();
    if (stale_host_) {
      // Run the selection and copy items back
      run_device_select();
      std::vector<real_type> y_selected(np * index_size);
      device_state_.y_selected.get_array(y_selected);

#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < np; ++i) {
        dust::utils::destride_copy(end_state + i * index_size, y_selected, i,
                                   np);
      }
    } else {
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < np; ++i) {
        particles_[i].state(index_, end_state + i * index_size);
      }
    }
  }

  // TODO: this does not use device_select. But if index is being provided
  // may not matter much
  void state(std::vector<size_t> index,
             std::vector<real_type>& end_state) {
    refresh_host();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < particles_.size(); ++i) {
      particles_[i].state(index, end_state.begin() + i * index.size());
    }
  }

  void state_full(std::vector<real_type>& end_state) {
    state_full(end_state.begin());
  }

  void state_full(typename std::vector<real_type>::iterator end_state) {
    refresh_host();
    const size_t n = n_state_full();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < particles_.size(); ++i) {
      particles_[i].state_full(end_state + i * n);
    }
  }

  void state_full(dust::cuda::device_array<real_type>& device_state, size_t dst_offset) {
    refresh_device();
    device_state.set_array(device_state_.y.data(),
                           device_state_.y.size(), dst_offset);
  }

  void reorder(const std::vector<size_t>& index) {
    size_t n_particles = particles_.size();
    size_t n_state = n_state_full();
    device_state_.scatter_index.set_array(index);
    dust::cuda::scatter_device<real_type><<<cuda_pars_.reorder.block_count,
                                          cuda_pars_.reorder.block_size,
                                          0,
                                          kernel_stream_.stream()>>>(
      device_state_.scatter_index.data(),
      device_state_.y.data(),
      device_state_.y_next.data(),
      n_state,
      n_particles,
      false);
    kernel_stream_.sync();

    device_state_.swap();
    select_needed_ = true;
  }

  // TODO - check if any other interface to this is needed?
  // e.g. something which takes std::vector weights and creates
  // a device array from them
  // device resample
  void resample(dust::cuda::device_array<real_type>& weights,
                dust::cuda::device_scan_state<real_type>& scan) {
    dust::filter::run_device_resample(n_particles(), n_pars_effective(), n_state_full(),
                                      cuda_pars_, kernel_stream_, resample_stream_,
                                      rng_.state(n_particles()),
                                      device_state_, weights, scan);
  }

  // Functions used in the device filter
  dust::cuda::device_array<size_t>& kappa() {
    return device_state_.scatter_index;
  }

  dust::cuda::device_array<real_type>& device_state_full() {
    kernel_stream_.sync();
    return device_state_.y;
  }

  dust::cuda::device_array<real_type>& device_state_selected() {
    run_device_select();
    return device_state_.y_selected;
  }

  // Maybe these funcs do argue for inheritance?
  size_t n_threads() const {
    return n_threads_;
  }

  size_t n_particles() const {
    return particles_.size();
  }

  size_t n_state() const {
    return index_.size();
  }

  size_t n_state_full() const {
    return particles_.front().size();
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

  size_t n_data() const {
    return data_.size();
  }

  const std::map<size_t, std::vector<data_type>>& data() const {
    return data_;
  }

  const std::map<size_t, size_t>& data_offsets() const {
    return device_data_offsets_;
  }

  size_t step() const {
    return particles_.front().step();
  }

  const std::vector<size_t>& shape() const {
    return shape_;
  }

  std::vector<rng_int_type> rng_state() {
    refresh_host();
    return rng_.export_state();
  }

  // TODO - make un update RNG method and call it here
  // Will need to do D->H then H->D?
  // Or just make it all run on the device might be best!
  void set_rng_state(const std::vector<rng_int_type>& rng_state) {
    refresh_host();
    rng_.import_state(rng_state);
    stale_device_ = true;
  }

  void set_n_threads(size_t n_threads) {
    n_threads_ = n_threads;
  }

  // TODO - make un update RNG method and call it here
  // see note above
  void rng_long_jump() {
    refresh_host();
    rng_.long_jump();
    stale_device_ = true;
  }

  void set_data(std::map<size_t, std::vector<data_type>> data) {
    if (rng_.deterministic()) {
      throw std::runtime_error("Can't use data with deterministic models");
    }
    data_ = data;
    initialise_device_data();
  }

  void compare_data(std::vector<real_type>& res, const std::vector<data_type>& data) {
    const size_t np = particles_.size() / n_pars_effective();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < particles_.size(); ++i) {
      res[i] = particles_[i].compare_data(data[i / np], rng_.state(i));
    }
    stale_device_ = true; // RNG use
  }

  std::vector<real_type> compare_data() {
    std::vector<real_type> res;
    auto d = device_data_offsets_.find(step());
    if (d != device_data_offsets_.end()) {
      res.resize(particles_.size());
      compare_data_device(device_state_.compare_res, d->second);
      device_state_.compare_res.get_array(res);
    }
    return res;
  }

  void compare_data(dust::cuda::device_array<real_type>& res,
                      const size_t data_offset) {
    dust::cuda::compare_particles<T><<<cuda_pars_.compare.block_count,
                                       cuda_pars_.compare.block_size,
                                       cuda_pars_.compare.shared_size_bytes,
                                       kernel_stream_.stream()>>>(
                     particles_.size(),
                     n_pars_effective(),
                     device_state_.y.data(),
                     res.data(),
                     device_state_.internal_int.data(),
                     device_state_.internal_real.data(),
                     device_state_.n_shared_int,
                     device_state_.n_shared_real,
                     device_state_.shared_int.data(),
                     device_state_.shared_real.data(),
                     device_data_.data() + data_offset,
                     device_state_.rng.data(),
                     cuda_pars_.compare.shared_int,
                     cuda_pars_.compare.shared_real);
    kernel_stream_.sync();
  }

private:
  // delete move and copy to avoid accidentally using them
  DustDevice ( const DustDevice & ) = delete;
  DustDevice ( DustDevice && ) = delete;

  const size_t n_pars_; // 0 in the "single" case, >=1 otherwise
  const size_t n_particles_each_; // Particles per parameter set
  const size_t n_particles_total_; // Total number of particles
  const bool pars_are_shared_; // Does the n_particles dimension exist in shape?
  std::vector<size_t> shape_; // shape of output
  size_t n_threads_;
  cuda::device_config device_config_;
  dust::random::prng<rng_state_type> rng_;
  std::map<size_t, std::vector<data_type>> data_;

  std::vector<size_t> index_;
  std::vector<dust::Particle<T>> particles_;
  std::vector<dust::shared_ptr<T>> shared_;

  // Device support
  dust::cuda::launch_control_dust cuda_pars_;
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
    reset_errors();
    update_device_shared();
    set_cuda_launch();

    device_step_ = step;
    stale_host_ = false;
    stale_device_ = true;
    select_needed_ = true;
  }

  void initialise(const std::vector<pars_type>& pars, const size_t step,
                  const size_t n_particles, bool set_state) {
    size_t n = particles_.size() == 0 ? 0 : n_state_full();
    std::vector<dust::Particle<T>> p;
    for (size_t i = 0; i < n_pars_; ++i) {
      p.push_back(dust::Particle<T>(pars[i], step));
      if (n > 0 && p.back().size() != n) {
        std::stringstream msg;
        msg << "'pars' created inconsistent state size: " <<
          "expected length " << n << " but parameter set " << i + 1 <<
          " created length " << p.back().size();
        throw std::invalid_argument(msg.str());
      }
      n = p.back().size(); // ensures all particles have same size
    }
    if (particles_.size() == n_particles_total_) {
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < n_particles_total_; ++i) {
        particles_[i].set_pars(p[i / n_particles], set_state);
      }
      for (size_t i = 0; i < pars.size(); ++i) {
        shared_[i] = pars[i].shared;
      }
    } else {
      particles_.clear();
      particles_.reserve(n_particles * n_pars_);
      for (size_t i = 0; i < n_pars_; ++i) {
        for (size_t j = 0; j < n_particles; ++j) {
          particles_.push_back(p[i]);
        }
        shared_.push_back(pars[i].shared);
      }
      initialise_device_state();
    }
    reset_errors();
    update_device_shared();
    set_cuda_launch();

    device_step_ = step;
    stale_host_ = false;
    stale_device_ = true;
    select_needed_ = true;
  }

  // This only gets called on construction; the size of these never
  // changes.
  void initialise_device_state() {
    if (!device_config_.enabled_) {
      return;
    }
    const auto s = shared_[0];
    const size_t n_internal_int = dust::cuda::device_internal_int_size<T>(s);
    const size_t n_internal_real = dust::cuda::device_internal_real_size<T>(s);
    const size_t n_shared_int = dust::cuda::device_shared_int_size<T>(s);
    const size_t n_shared_real = dust::cuda::device_shared_real_size<T>(s);
    device_state_.initialise(particles_.size(), n_state_full(),
                             n_pars_effective(), shared_.size(),
                             n_internal_int, n_internal_real,
                             n_shared_int, n_shared_real);
  }

  void initialise_device_data() {
    if (!device_config_.enabled_) {
      return;
    }
    std::vector<data_type> flattened_data;
    std::vector<size_t> data_offsets(n_data());
    size_t i = 0;
    for (auto & d_step : data()) {
      device_data_offsets_[d_step.first] = i;
      for (auto & d : d_step.second ) {
        flattened_data.push_back(d);
        i++;
      }
    }
    device_data_ = dust::cuda::device_array<data_type>(flattened_data.size());
    device_data_.set_array(flattened_data);
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

  void update_device_index() {
    if (!device_config_.enabled_) {
      return;
    }
    const size_t n_particles = particles_.size();
    device_state_.set_device_index(index_, n_particles, n_state_full());

    select_needed_ = true;
    if (!std::is_sorted(index_.cbegin(), index_.cend())) {
      select_scatter_ = true;
    } else {
      select_scatter_ = false;
    }

    // TODO: get this 64 from the original configuration, if possible.
    cuda_pars_.index_scatter =
      dust::cuda::launch_control_simple(64, n_particles * n_state());
  }

  // TODO - this will move into initialisation
  void refresh_device() {
    if (!device_config_.enabled_) {
      throw std::runtime_error("Can't refresh a non-existent device");
    }
    if (rng_.deterministic()) {
      throw std::runtime_error("Can't run deterministic models on GPU");
    }
    if (stale_device_) {
      const size_t np = n_particles(), ny = n_state_full();
      constexpr size_t rng_len = rng_state_type::size();
      std::vector<real_type> y_tmp(ny); // Individual particle state
      std::vector<real_type> y(np * ny); // Interleaved state of all particles
      std::vector<rng_int_type> rng(np * rng_len); // Interleaved RNG state
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < np; ++i) {
        // Interleave state
        particles_[i].state_full(y_tmp.begin());
        dust::utils::stride_copy(y.data(), y_tmp, i, np);

        // Interleave RNG state
        rng_state_type p_rng = rng_.state(i);
        size_t rng_offset = i;
        for (size_t j = 0; j < rng_len; ++j) {
          rng_offset = dust::utils::stride_copy(rng.data(), p_rng[j],
                                                rng_offset, np);
        }
      }
      // H -> D copies
      device_state_.y.set_array(y);
      device_state_.rng.set_array(rng);
      stale_device_ = false;
      select_needed_ = true;
      device_step_ = step();
    }
  }

  // TODO - presumably some of this will be needed when extracting state
  // Also should just destride on the device
  void refresh_host() {
    if (stale_host_) {
      const size_t np = n_particles(), ny = n_state_full();
      constexpr size_t rng_len = rng_state_type::size();
      std::vector<real_type> y_tmp(ny); // Individual particle state
      std::vector<real_type> y(np * ny); // Interleaved state of all particles
      std::vector<rng_int_type> rngi(np * rng_len); // Interleaved RNG state
      std::vector<rng_int_type> rng(np * rng_len); //  Deinterleaved RNG state
      // D -> H copies
      device_state_.y.get_array(y);
      device_state_.rng.get_array(rngi);
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < np; ++i) {
        dust::utils::destride_copy(y_tmp.data(), y, i, np);
        particles_[i].set_state(y_tmp.begin());

        // Destride RNG
        for (size_t j = 0; j < rng_len; ++j) {
          rng[i * rng_len + j] = rngi[i + j * np];
        }
      }
      rng_.import_state(rng, np);
      stale_host_ = false;
      set_step(device_step_);
    }
  }

  void run_select() {
    if (!select_needed_) {
      return;
    }
    size_t size_select_tmp = device_state_.select_tmp.size();
    cub::DeviceSelect::Flagged(device_state_.select_tmp.data(),
                                size_select_tmp,
                                device_state_.y.data(),
                                device_state_.index.data(),
                                device_state_.y_selected.data(),
                                device_state_.n_selected.data(),
                                device_state_.y.size(),
                                kernel_stream_.stream());
    kernel_stream_.sync();
    if (select_scatter_) {
      dust::cuda::scatter_device<real_type><<<cuda_pars_.index_scatter.block_count,
                                           cuda_pars_.index_scatter.block_size,
                                           0,
                                           kernel_stream_.stream()>>>(
        device_state_.index_state_scatter.data(),
        device_state_.y_selected.data(),
        device_state_.y_selected_swap.data(),
        n_state(),
        n_particles(),
        true);
      kernel_stream_.sync();
      device_state_.swap_selected();
    }
    select_needed_ = false;
  }

  // Set up CUDA block sizes and shared memory preferences
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
};

}

#endif