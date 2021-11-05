#ifndef DUST_CUDA_DUST_GPU_HPP
#define DUST_CUDA_DUST_GPU_HPP

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dust/gpu/call.hpp"
#include "dust/gpu/cuda.hpp"
#include "dust/gpu/device_resample.hpp"
#include "dust/gpu/filter_state.hpp"
#include "dust/gpu/kernels.hpp"
#include "dust/gpu/launch_control.hpp"
#include "dust/gpu/types.hpp"
#include "dust/filter_tools.hpp"
#include "dust/particle.hpp"
#include "dust/random/density.hpp"
#include "dust/random/prng.hpp"
#include "dust/utils.hpp"

namespace dust {

template <typename T>
class dust_gpu {
public:
  typedef T model_type;
  typedef dust::pars_type<T> pars_type;
  typedef typename T::real_type real_type;
  typedef typename T::data_type data_type;
  typedef typename T::internal_type internal_type;
  typedef typename T::shared_type shared_type;
  typedef typename T::rng_state_type rng_state_type;
  typedef typename rng_state_type::int_type rng_int_type;

  // TODO: fix this elsewhere, perhaps (see also dust/dust_cpu.hpp)
  typedef dust::filter::filter_state_device<real_type> filter_state_type;

  dust_gpu(const pars_type& pars, const size_t step, const size_t n_particles,
           const size_t n_threads, const std::vector<rng_int_type>& seed,
           const gpu::gpu_config& gpu_config) :
    n_pars_(0),
    n_particles_each_(n_particles),
    n_particles_total_(n_particles),
    n_state_full_(0),
    n_state_(0),
    pars_are_shared_(true),
    n_threads_(n_threads),
    gpu_config_(gpu_config),
    select_needed_(true),
    select_scatter_(false),
    step_(step) {
    initialise_device_state(std::vector<pars_type>(1, pars), seed);
    shape_ = {n_particles};
  }

  dust_gpu(const std::vector<pars_type>& pars, const size_t step,
           const size_t n_particles, const size_t n_threads,
           const std::vector<rng_int_type>& seed,
           const std::vector<size_t>& shape,
           const gpu::gpu_config& gpu_config) :
    n_pars_(pars.size()),
    n_particles_each_(n_particles == 0 ? 1 : n_particles),
    n_particles_total_(n_particles_each_ * pars.size()),
    n_state_full_(0), // needed for malloc size
    n_state_(0),
    pars_are_shared_(n_particles != 0),
    n_threads_(n_threads),
    gpu_config_(gpu_config),
    select_needed_(true),
    select_scatter_(false),
    step_(step) {
    initialise_device_state(pars, seed);
    // constructing the shape here is harder than above.
    if (n_particles > 0) {
      shape_.push_back(n_particles);
    }
    for (auto i : shape) {
      shape_.push_back(i);
    }
  }

  // We only need a destructor when running with cuda profiling; don't
  // include ond otherwise because we don't actually follow the rule
  // of 3/5/0
#ifdef DUST_ENABLE_CUDA_PROFILER
  ~dust_gpu() {
    cuda_profiler_stop(gpu_config_);
  }
#endif

  size_t n_threads() const {
    return n_threads_;
  }

  size_t n_particles() const {
    return n_particles_total_;
  }

  size_t n_state() const {
    return n_state_;
  }

  size_t n_state_full() const {
    return n_state_full_;
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
    return device_data_offsets_.size();
  }

  // NOTE: this is _just_ the offsets. However, when this is used
  // within the filter code we only care about the first part of the
  // map (i.e., the step to stop at) so it's ok.
  const std::map<size_t, size_t>& data() const {
    return device_data_offsets_;
  }

  size_t step() const {
    return step_;
  }

  const std::vector<size_t>& shape() const {
    return shape_;
  }

  // These two (check_errors, reset_errors) don't really exist for
  // this model because we always (currently) run particles serially
  // on the cpu (and errors on the gpu are unrecoverable). If we ever
  // set up to do openmp running of the particles (and we probably
  // should), then these need the same implementation as for dust_cpu.
  void check_errors() {
  }

  void reset_errors() {
  }

  void set_pars(const pars_type& pars, bool set_state) {
    set_pars(std::vector<pars_type>(1, pars), set_state);
  }

  void set_pars(const std::vector<pars_type>& pars, bool set_state) {
    set_device_shared(pars);
    if (set_state) {
      set_state_from_pars(pars);
    }
  }

  // It's the callee's responsibility to ensure this is the correct length:
  //
  // * if is_matrix is false then state must be length n_state_full()
  //   and all particles get the state
  // * if is_matrix is true, state must be length (n_state_full() *
  //   n_particles()) and every particle gets a different state.
  void set_state(const std::vector<real_type>& state) {
    const size_t n_particles = n_particles_total_;
    const size_t n_state = n_state_full();
    const bool individual = state.size() == n_state * n_particles;
    const size_t n = individual ? 1 : n_particles_each_;
    auto it = state.begin();
    // Interleaved state of all particles
    std::vector<real_type> y(n_particles * n_state);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < n_particles; ++i) {
      size_t at = i;
      for (size_t j = 0; j < n_state; ++j) {
        at = dust::utils::stride_copy(y.data(),
                                      *(it + (i / n) * n_state + j),
                                      at,
                                      n_particles);
      }
    }
    device_state_.y.set_array(y);
    select_needed_ = true;
  }

  void set_step(const size_t step) {
    step_ = step;
  }

  // This is prevented in interface.hpp so that we never make it here
  // (part of requiring that state setting entirely succeeds or
  // fails). We leave the same error message here though so that it's
  // easy to locate the interface component; the method must exist or
  // compilation would fail.
  void set_step(const std::vector<size_t>& step) {              // # nocov
    throw std::runtime_error("GPU doesn't support setting vector of steps"); // # nocov
  }

  // It's the callee's responsibility to ensure that index is in
  // range [0, n-1]
  void set_index(const std::vector<size_t>& index) {
    const size_t n_particles = n_particles_total_;
    n_state_ = index.size();
    device_state_.set_device_index(index, n_particles, n_state_full());

    select_needed_ = true;
    select_scatter_ = !std::is_sorted(index.cbegin(), index.cend());

    const auto block_size = cuda_pars_.index_scatter.block_size;
    cuda_pars_.index_scatter =
      dust::gpu::launch_control_simple(block_size, n_particles * n_state());
  }

  void run(const size_t step_end) {
    if (step_end > step_) {
      const size_t step_start = step_;
#ifdef __NVCC__
      dust::gpu::run_particles<T><<<cuda_pars_.run.block_count,
                                     cuda_pars_.run.block_size,
                                     cuda_pars_.run.shared_size_bytes,
                                     kernel_stream_.stream()>>>(
                      step_start, step_end, n_particles_total_,
                      n_pars_effective(),
                      device_state_.y.data(),
                      device_state_.y_next.data(),
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
      const bool use_shared_int = false;
      const bool use_shared_real = false;
      dust::gpu::run_particles<T>(step_start, step_end, n_particles_total_,
                      n_pars_effective(),
                      device_state_.y.data(),
                      device_state_.y_next.data(),
                      device_state_.internal_int.data(),
                      device_state_.internal_real.data(),
                      device_state_.n_shared_int,
                      device_state_.n_shared_real,
                      device_state_.shared_int.data(),
                      device_state_.shared_real.data(),
                      device_state_.rng.data(),
                      use_shared_int,
                      use_shared_real);
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

  std::vector<real_type> simulate(const std::vector<size_t>& step_end) {
    const size_t n_time = step_end.size();
    // The filter snapshot class can be used to store the indexed state
    // (implements async copy, swap space, and deinterleaving)
    // Filter trajectories not used as we don't need order here
    dust::filter::filter_snapshots_device<real_type> state_store;
    state_store.resize(n_state(), n_particles(), step_end);
    for (size_t t = 0; t < n_time; ++t) {
      run(step_end[t]);
      state_store.store(device_state_selected());
      state_store.advance();
    }
    std::vector<real_type> ret(n_state() * n_particles() * n_time);
    state_store.history(ret.data());
    return ret;
  }

  void state(std::vector<real_type>& end_state) {
    return state(end_state.begin());
  }

  void state(typename std::vector<real_type>::iterator end_state) {
    size_t np = n_particles_total_;
    size_t index_size = n_state_;

    // Run the selection and copy items back
    run_select();
    std::vector<real_type> y_selected(np * index_size);
    device_state_.y_selected.get_array(y_selected);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < np; ++i) {
      dust::utils::destride_copy(end_state + i * index_size, y_selected, i,
                                  np);
    }
  }

  // TODO: we should really do this via a kernel I think? Currently we
  // grab the whole state back from the device to the host, then
  // filter through it. See https://github.com/mrc-ide/dust/pull/302
  void state(std::vector<size_t> index,
             std::vector<real_type>& end_state) {
    std::vector<real_type> full_state(n_state_full_ * n_particles_total_);
    get_device_state(full_state.begin());
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < n_particles_total_; ++i) {
      size_t particle_start = i * index.size();
      for (size_t j = 0; j < index.size(); ++j) {
        end_state[particle_start + j] =
          full_state[i * n_state_full_ + index[j]];
      }
    }
  }

  void state_full(std::vector<real_type>& end_state) {
    state_full(end_state.begin());
  }

  void state_full(typename std::vector<real_type>::iterator end_state) {
    get_device_state(end_state);
  }

  void state_full(dust::gpu::device_array<real_type>& device_state,
                  size_t dst_offset) {
    device_state.set_array(device_state_.y.data(),
                           device_state_.y.size(), dst_offset);
  }

  void reorder(const std::vector<size_t>& index) {
    size_t n_particles = n_particles_total_;
    size_t n_state = n_state_full();
    device_state_.scatter_index.set_array(index);
    bool select_kernel = false;
#ifdef __NVCC__
    dust::gpu::scatter_device<real_type><<<cuda_pars_.reorder.block_count,
                                          cuda_pars_.reorder.block_size,
                                          0,
                                          kernel_stream_.stream()>>>(
      device_state_.scatter_index.data(),
      device_state_.y.data(),
      device_state_.y_next.data(),
      n_state,
      n_particles,
      select_kernel);
    kernel_stream_.sync();
#else
    dust::gpu::scatter_device<real_type>(
      device_state_.scatter_index.data(),
      device_state_.y.data(),
      device_state_.y_next.data(),
      n_state,
      n_particles,
      select_kernel);
#endif

    device_state_.swap();
    select_needed_ = true;
  }

  // NOTE: this is only used for debugging/testing, otherwise we would
  // make device_weights and scan class members.
  std::vector<size_t> resample(const std::vector<real_type>& weights) {
    dust::gpu::device_weights<real_type>
      device_weights(n_particles(), n_pars_effective());
    device_weights.weights() = weights;

    dust::gpu::device_scan_state<real_type> scan;
    scan.initialise(n_particles_total_, device_weights.weights());
    resample(device_weights.weights(), scan);

    std::vector<size_t> index(n_particles());
    device_state_.scatter_index.get_array(index);
    return index;
  }

  // Functions used in the device filter
  void resample(dust::gpu::device_array<real_type>& weights,
                dust::gpu::device_scan_state<real_type>& scan) {
    dust::filter::run_device_resample(n_particles(),
                                      n_pars_effective(),
                                      n_state_full(),
                                      cuda_pars_,
                                      kernel_stream_,
                                      resample_stream_,
                                      resample_rng_,
                                      device_state_,
                                      weights,
                                      scan);
  }

  // For the particle filter only
  dust::gpu::device_array<size_t>& filter_kappa() {
    return device_state_.scatter_index;
  }

  dust::gpu::device_array<real_type>& device_state_full() {
    kernel_stream_.sync();
    return device_state_.y;
  }

  dust::gpu::device_array<real_type>& device_state_selected() {
    run_select();
    return device_state_.y_selected;
  }

  // This function and set_rng_state are inverses of each other; the
  // layout is purposefully similar.
  std::vector<rng_int_type> rng_state() {
    const size_t np = n_particles();
    constexpr size_t rng_len = rng_state_type::size();

    std::vector<rng_int_type> rng_interleaved(np * rng_len);
    // Pull from device
    device_state_.rng.get_array(rng_interleaved);

    // De-interleaved RNG state, copied from rng_interleaved, +1 is host rng
    std::vector<rng_int_type> rng_state((np + 1) * rng_len);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < np; ++i) {
      for (size_t j = 0; j < rng_len; ++j) {
        rng_state[j + i * rng_len] = rng_interleaved[i + j * np];
      }
    }

    // Add the (host) resample state on the end
    for (size_t j = 0; j < rng_len; ++j) {
      rng_state[np * rng_len + j] = resample_rng_[j];
    }

    return rng_state;
  }

  void set_rng_state(const std::vector<rng_int_type>& rng_state) {
    const size_t np = n_particles();
    constexpr size_t rng_len = rng_state_type::size();

    // Interleaved RNG state copied from rng_state
    std::vector<rng_int_type> rng_interleaved(np * rng_len);
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < np; ++i) {
      for (size_t j = 0; j < rng_len; ++j) {
        rng_interleaved[i + j * np] = rng_state[j + i * rng_len];
      }
    }

    // Push onto device
    device_state_.rng.set_array(rng_interleaved);

    // This also imports the resample RNG, which is on the host
    for (size_t j = 0; j < rng_len; ++j) {
      resample_rng_[j] = rng_state[np * rng_len + j];
    }
  }

  void set_n_threads(size_t n_threads) {
    n_threads_ = n_threads;
  }

  void set_data(std::map<size_t, std::vector<data_type>>& data) {
    std::vector<data_type> flattened_data;
    size_t i = 0;
    for (auto & d_step : data) {
      device_data_offsets_[d_step.first] = i;
      for (auto & d : d_step.second ) {
        flattened_data.push_back(d);
        i++;
      }
    }
    device_data_ = dust::gpu::device_array<data_type>(flattened_data.size());
    device_data_.set_array(flattened_data);
  }

  std::vector<real_type> compare_data() {
    std::vector<real_type> res;
    auto d = device_data_offsets_.find(step());
    if (d != device_data_offsets_.end()) {
      res.resize(n_particles());
      compare_data(device_state_.compare_res, d->second);
      device_state_.compare_res.get_array(res);
    }
    return res;
  }

  void compare_data(dust::gpu::device_array<real_type>& res,
                    const size_t data_offset) {
#ifdef __NVCC__
    dust::gpu::compare_particles<T><<<cuda_pars_.compare.block_count,
                                       cuda_pars_.compare.block_size,
                                       cuda_pars_.compare.shared_size_bytes,
                                       kernel_stream_.stream()>>>(
                     n_particles(),
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
#else
    const bool use_shared_int = false;
    const bool use_shared_real = false;
    dust::gpu::compare_particles<T>(
                     n_particles(),
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
                     use_shared_int,
                     use_shared_real);
#endif
  }

private:
  // delete move and copy to avoid accidentally using them
  dust_gpu(const dust_gpu &) = delete;
  dust_gpu(dust_gpu &&) = delete;

  // Host quantities
  const size_t n_pars_; // 0 in the "single" case, >=1 otherwise
  const size_t n_particles_each_; // Particles per parameter set
  const size_t n_particles_total_; // Total number of particles
  size_t n_state_full_; // State size of a particle
  size_t n_state_; // State size of a particle with an index
  const bool pars_are_shared_; // Does the n_particles dimension exist in shape?

  std::vector<size_t> shape_; // shape of output
  size_t n_threads_;
  rng_state_type resample_rng_; // for the filter
  gpu::gpu_config gpu_config_;

  // GPU support
  dust::gpu::launch_control_dust cuda_pars_;
  dust::gpu::device_state<real_type, rng_state_type> device_state_;
  dust::gpu::device_array<data_type> device_data_;
  std::map<size_t, size_t> device_data_offsets_;
  dust::gpu::cuda_stream kernel_stream_;
  dust::gpu::cuda_stream resample_stream_;

  bool select_needed_;
  bool select_scatter_;
  size_t step_;

  // Naming of functions:
  //
  // Initialise called once (device_state and device_memory), to
  // allocate memory on device. Because the size of the problem
  // (number of particles, state size, number of parameter sets) is
  // constant, this never needs doing again.
  //
  // Set can be called multiple times, sets device memory from host

  // Sets state from model + pars, called from the constructors
  void initialise_device_state(const std::vector<pars_type>& pars,
                               const std::vector<rng_int_type>& seed) {
    if (n_state_full_ == 0) {
      const dust::particle<T> p(pars[0], step_);
      n_state_full_ = p.size();
      n_state_ = n_state_full_;
    }

    initialise_device_memory(pars[0].shared);
    set_device_shared(pars);
    set_state_from_pars(pars);

    // Set GPU RNG from a seed; primary reason for this construction
    // is to expand out seed correctly (e.g., it might be a 4-element
    // vector on first creation).
    dust::random::prng<rng_state_type> rng(n_particles_total_ + 1, seed);
    set_rng_state(rng.export_state());

    set_cuda_launch();

    select_needed_ = true;

    // ifdef guards not really needed here but helps guarantee
    // symmetry with destructor.
#ifdef DUST_ENABLE_CUDA_PROFILER
    cuda_profiler_start(gpu_config_);
#endif
  }

  // This only gets called on construction; the size of these never
  // changes.  Could be merged with the above really.
  void initialise_device_memory(typename dust::shared_ptr<T> s) {
    const size_t n_pars = n_pars_effective();
    const size_t n_internal_int = dust::gpu::internal_int_size<T>(s);
    const size_t n_internal_real = dust::gpu::internal_real_size<T>(s);
    const size_t n_shared_int = dust::gpu::shared_int_size<T>(s);
    const size_t n_shared_real = dust::gpu::shared_real_size<T>(s);
    device_state_.initialise(n_particles_total_, n_state_full_, n_pars,
                             n_internal_int, n_internal_real,
                             n_shared_int, n_shared_real);
  }

  void set_device_shared(const std::vector<pars_type>& pars) {
    size_t n = n_particles() == 0 ? 0 : n_state_full();
    std::vector<dust::particle<T>> p;
    for (size_t i = 0; i < n_pars_effective(); ++i) {
      p.push_back(dust::particle<T>(pars[i], step_));
      if (n > 0 && p.back().size() != n) {
        std::stringstream msg;
        msg << "'pars' created inconsistent state size: " <<
          "expected length " << n << " but parameter set " << i + 1 <<
          " created length " << p.back().size();
        throw std::invalid_argument(msg.str());
      }
      n = p.back().size(); // ensures all particles have same size
    }

    const size_t n_shared_int = device_state_.n_shared_int;
    const size_t n_shared_real = device_state_.n_shared_real;
    std::vector<int> shared_int(n_shared_int * n_pars_effective());
    std::vector<real_type> shared_real(n_shared_real * n_pars_effective());
    for (size_t i = 0; i < pars.size(); ++i) {
      int * dest_int = shared_int.data() + n_shared_int * i;
      real_type * dest_real = shared_real.data() + n_shared_real * i;
      dust::gpu::shared_copy<T>(pars[i].shared, dest_int, dest_real);
    }
    device_state_.shared_int.set_array(shared_int);
    device_state_.shared_real.set_array(shared_real);
  }

  // TODO: This update function is wildly inefficient; we should
  // probably support things like "copy one state to all the particles
  // of that parameter index", possibly as a kernel. Detecting when
  // that is the case would be important as we're probably doing the
  // wrong thing here sometimes:
  // https://github.com/mrc-ide/dust/issues/310
  void set_state_from_pars(const std::vector<pars_type>& pars) {
    const size_t n_pars = pars.size(); // or n_pars_effective();
    std::vector<std::vector<real_type>>
      state_host(n_particles() * n_pars,
                 std::vector<real_type>(n_state_full_));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < n_pars; ++i) {
      for (size_t j = 0; j < n_particles(); ++j) {
        dust::particle<T> p(pars[i], step_);
        p.state_full(state_host[i * n_particles() + j].begin());
      }
    }

    set_device_state(state_host);
  }

  // Interleaves and copies a 2D state vector
  void set_device_state(std::vector<std::vector<real_type>>& state_full) {
    const size_t np = n_particles(), ny = n_state_full();
    std::vector<real_type> y(np * ny); // Interleaved state of all particles
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < np; ++i) {
      // Interleave state
      dust::utils::stride_copy(y.data(), state_full[i], i, np);
    }
    // H -> D copy
    device_state_.y.set_array(y);
    select_needed_ = true;
  }

  // set_device_state and get_device_state are inverses of each other
  void set_device_state(std::vector<real_type>& state_full) {
    const size_t np = n_particles(), ny = n_state_full();
    std::vector<real_type> y(np * ny); // Interleaved state of all particles
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < np; ++i) {
      // Interleave state
      dust::utils::stride_copy(y.data(), state_full[i], i, np);
    }
    // H -> D copy
    device_state_.y.set_array(y);
    select_needed_ = true;
  }

  void get_device_state(typename std::vector<real_type>::iterator state_full) {
    const size_t np = n_particles(), ny = n_state_full();
    std::vector<real_type> y(np * ny); // Interleaved state of all particles
    device_state_.y.get_array(y);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < np; ++i) {
      dust::utils::destride_copy(state_full + i * ny, y, i, np);
    }
  }

  void run_select() {
    if (!select_needed_) {
      return;
    }
    const bool select_kernel = true;
#ifdef __NVCC__
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
      dust::gpu::scatter_device<real_type><<<cuda_pars_.index_scatter.block_count,
                                           cuda_pars_.index_scatter.block_size,
                                           0,
                                           kernel_stream_.stream()>>>(
        device_state_.index_state_scatter.data(),
        device_state_.y_selected.data(),
        device_state_.y_selected_swap.data(),
        n_state(),
        n_particles(),
        select_kernel);
      kernel_stream_.sync();
      device_state_.swap_selected();
    }
#else
    size_t selected_idx = 0;
    for (size_t i = 0; i < device_state_.y.size(); i++) {
      if (device_state_.index.data()[i] == 1) {
        device_state_.y_selected.data()[selected_idx] =
          device_state_.y.data()[i];
        selected_idx++;
      }
    }
    if (select_scatter_) {
      dust::gpu::scatter_device<real_type>(
        device_state_.index_state_scatter.data(),
        device_state_.y_selected.data(),
        device_state_.y_selected_swap.data(),
        n_state(),
        n_particles(),
        select_kernel);
      device_state_.swap_selected();
    }
#endif
    select_needed_ = false;
  }

  // Set up CUDA block sizes and shared memory preferences
  void set_cuda_launch() {
    cuda_pars_ = dust::gpu::launch_control_dust(gpu_config_,
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
