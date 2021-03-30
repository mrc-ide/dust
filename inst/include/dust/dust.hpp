#ifndef DUST_DUST_HPP
#define DUST_DUST_HPP

#include <algorithm>
#include <memory>
#include <map>
#include <stdexcept>
#include <sstream>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <dust/rng.hpp>
#include <dust/densities.hpp>
#include <dust/filter_tools.hpp>
#include <dust/types.hpp>
#include <dust/utils.hpp>
#include <dust/particle.hpp>
#include <dust/kernels.hpp>

namespace dust {

template <typename T>
class Dust {
public:
  typedef dust::pars_t<T> pars_t;
  typedef typename T::real_t real_t;
  typedef typename T::data_t data_t;

  Dust(const pars_t& pars, const size_t step, const size_t n_particles,
       const size_t n_threads, const std::vector<uint64_t>& seed,
       size_t device_id) :
    n_pars_(0),
    n_particles_each_(n_particles),
    n_particles_total_(n_particles),
    pars_are_shared_(true),
    n_threads_(n_threads),
    device_id_(device_id),
    rng_(n_particles_total_, seed),
    errors_(n_particles_total_),
    stale_host_(false),
    stale_device_(true) {
#ifdef __NVCC__
    initialise_device(device_id);
#endif
    initialise(pars, step, n_particles, true);
    initialise_index();
    shape_ = {n_particles};
  }

  Dust(const std::vector<pars_t>& pars, const size_t step,
       const size_t n_particles, const size_t n_threads,
       const std::vector<uint64_t>& seed, size_t device_id,
       std::vector<size_t> shape) :
    n_pars_(pars.size()),
    n_particles_each_(n_particles == 0 ? 1 : n_particles),
    n_particles_total_(n_particles_each_ * pars.size()),
    pars_are_shared_(n_particles != 0),
    n_threads_(n_threads),
    device_id_(device_id),
    rng_(n_particles_total_, seed),
    errors_(n_particles_total_),
    stale_host_(false),
    stale_device_(true) {
#ifdef __NVCC__
    initialise_device(device_id);
#endif
    initialise(pars, step, n_particles_each_, true);
    initialise_index();
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
  ~Dust() {
    CUDA_CALL_NOTHROW(cudaProfilerStop());
  }
#endif

  void reset(const pars_t& pars, const size_t step) {
    const size_t n_particles = particles_.size();
    initialise(pars, step, n_particles, true);
  }

  void reset(const std::vector<pars_t>& pars, const size_t step) {
    const size_t n_particles = particles_.size() / pars.size();
    initialise(pars, step, n_particles, true);
  }

  void set_pars(const pars_t& pars) {
    const size_t n_particles = particles_.size();
    initialise(pars, step(), n_particles, false);
  }

  void set_pars(const std::vector<pars_t>& pars) {
    const size_t n_particles = particles_.size();
    initialise(pars, step(), n_particles / pars.size(), false);
  }

  // It's the callee's responsibility to ensure that index is in
  // range [0, n-1]
  void set_index(const std::vector<size_t>& index) {
    index_ = index;
    update_device_index();
  }

  // It's the callee's responsibility to ensure this is the correct length:
  //
  // * if is_matrix is false then state must be length n_state_full()
  //   and all particles get the state
  // * if is_matrix is true, state must be length (n_state_full() *
  //   n_particles()) and every particle gets a different state.
  void set_state(const std::vector<real_t>& state, bool individual) {
    const size_t n_particles = particles_.size();
    const size_t n_state = n_state_full();
    const size_t n = individual ? 1 : n_particles_each_;
    auto it = state.begin();
    for (size_t i = 0; i < n_particles; ++i) {
      particles_[i].set_state(it + (i / n) * n_state);
    }
    stale_device_ = true;
  }

  void set_step(const size_t step) {
    const size_t n_particles = particles_.size();
    for (size_t i = 0; i < n_particles; ++i) {
      particles_[i].set_step(step);
    }
  }

  void set_step(const std::vector<size_t>& step) {
    const size_t n_particles = particles_.size();
    for (size_t i = 0; i < n_particles; ++i) {
      particles_[i].set_step(step[i]);
    }
    const auto r = std::minmax_element(step.begin(), step.end());
    if (*r.second > *r.first) {
      run(*r.second);
    }
  }

  void run(const size_t step_end) {
    refresh_host();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < particles_.size(); ++i) {
      try {
        particles_[i].run(step_end, rng_.state(i));
      } catch (std::exception const& e) {
        errors_.capture(e, i);
      }
    }
    errors_.report();
    stale_device_ = true;
  }

  template <typename U = T>
  typename std::enable_if<!dust::has_gpu_support<U>::value, void>::type
  run_device(const size_t step_end) {
    throw std::invalid_argument("GPU support not enabled for this object");
  }

  template <typename U = T>
  typename std::enable_if<dust::has_gpu_support<U>::value, void>::type
  run_device(const size_t step_end) {
    refresh_device();
    const size_t step_start = step();
#ifdef __NVCC__
    const int warp_size = dust::cuda::warp_size;
    // Set up blocks and shared memory preferences
    size_t blockSize = 128;
    size_t blockCount;
    bool use_shared_L1 = true;
    size_t n_shared_int_effective = device_state_.n_shared_int +
      dust::utils::align_padding(device_state_.n_shared_int * sizeof(int), sizeof(real_t)) / sizeof(int);
    size_t shared_size_bytes = n_shared_int_effective * sizeof(int) +
      device_state_.n_shared_real * sizeof(real_t);
    if (n_particles_each_ < warp_size || shared_size_bytes > shared_size_) {
      // If not enough particles per pars to make a whole block use
      // shared, or if shared_t too big for L1, turn it off, and run
      // in 'classic' mode where each particle is totally independent
      use_shared_L1 = false;
      shared_size_bytes = 0;
      blockCount = n_particles() * (n_particles() + blockSize - 1) / blockSize;
    } else {
      // If it's possible to make blocks with shared_t in L1 cache,
      // each block runs a pars set. Each pars set has enough blocks
      // to run all of its particles, the final block may have some
      // threads that don't do anything (hang off the end)
      blockSize = warp_size * (n_particles_each_ + warp_size - 1) / warp_size;
      blockSize = std::min(static_cast<size_t>(128), blockSize);
      blockCount = n_pars_effective() * (n_particles_each_ + blockSize - 1) /
        blockSize;
    }
    dust::run_particles<T><<<blockCount, blockSize, shared_size_bytes>>>(
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
                     use_shared_L1);
    CUDA_CALL(cudaDeviceSynchronize());
#else
    dust::run_particles<T>(step_start, step_end, particles_.size(),
                     n_pars_effective(),
                     device_state_.y.data(), device_state_.y_next.data(),
                     device_state_.internal_int.data(),
                     device_state_.internal_real.data(),
                     device_state_.n_shared_int,
                     device_state_.n_shared_real,
                     device_state_.shared_int.data(),
                     device_state_.shared_real.data(),
                     device_state_.rng.data(),
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

    stale_host_ = true;
    set_step(step_end);
  }

  std::vector<real_t> simulate(const std::vector<size_t>& step_end) {
    const size_t n_time = step_end.size();
    std::vector<real_t> ret(n_particles() * n_state() * n_time);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < particles_.size(); ++i) {
      try {
        for (size_t t = 0; t < n_time; ++t) {
          particles_[i].run(step_end[t], rng_.state(i));
          size_t offset = t * n_state() * n_particles() + i * n_state();
          particles_[i].state(index_, ret.begin() + offset);
        }
      } catch (std::exception const& e) {
        errors_.capture(e, i);
      }
    }
    errors_.report();
    return ret;
  }

  void state(std::vector<real_t>& end_state) {
    return state(end_state.begin());
  }

  // TODO: tidy this up with some templates
  void state(typename std::vector<real_t>::iterator end_state) {
    size_t np = particles_.size();
    size_t index_size = index_.size();
    if (stale_host_) {
#ifdef __NVCC__
      // Run the selection and copy items back
      run_device_select();
      std::vector<real_t> y_selected(np * index_size);
      device_state_.y_selected.get_array(y_selected);

#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < np; ++i) {
        dust::utils::destride_copy(end_state + i * index_size, y_selected, i,
                                   np);
      }
#else
      refresh_host();
#endif
    }
    // This would be better as an else, but the ifdefs are clearer this way
    if (!stale_host_) {
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < np; ++i) {
        particles_[i].state(index_, end_state + i * index_size);
      }
    }
  }

  // Used for copy of state into another block of memory on the device (used
  // for history saving)
  void state(dust::device_array<real_t>& device_state, size_t dst_offset) {
    run_device_select();
    device_state.set_array(device_state_.y_selected.data(),
                           0, dst_offset);
  }

  // TODO: this does not use device_select. But if index is being provided
  // may not matter much
  void state(std::vector<size_t> index,
             std::vector<real_t>& end_state) {
    refresh_host();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < particles_.size(); ++i) {
      particles_[i].state(index, end_state.begin() + i * index.size());
    }
  }

  void state_full(std::vector<real_t>& end_state) {
    state_full(end_state.begin());
  }

  void state_full(typename std::vector<real_t>::iterator end_state) {
    refresh_host();
    const size_t n = n_state_full();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < particles_.size(); ++i) {
      particles_[i].state_full(end_state + i * n);
    }
  }

  void state_full(dust::device_array<real_t>& device_state, size_t dst_offset) {
    refresh_device();
    device_state.set_array(device_state_.y.data(),
                           0, dst_offset);
  }

  // There are two obvious ways of reordering; we can construct a
  // completely new set of particles, like
  //
  //   std::vector<dust::Particle<T>> next;
  //   for (auto const& i: index) {
  //     next.push_back(particles_[i]);
  //   }
  //   particles_ = next;
  //
  // but this seems like a lot of churn.  The other way is to treat it
  // like a slightly weird state update where we swap around the
  // contents of the particle state (uses the set_state() and swap()
  // methods on particles).
  void reorder(const std::vector<size_t>& index) {
    if (stale_host_) {
      size_t n_particles = particles_.size();
      size_t n_state = n_state_full();
      device_state_.scatter_index.set_array(index);
#ifdef __NVCC__
      const size_t blockSize = 128;
      const size_t blockCount =
        (n_state * n_particles + blockSize - 1) / blockSize;
      dust::scatter_device<real_t><<<blockCount, blockSize>>>(
        device_state_.scatter_index.data(),
        device_state_.y.data(),
        device_state_.y_next.data(),
        n_state,
        n_particles);
      CUDA_CALL(cudaDeviceSynchronize());
#else
      dust::scatter_device<real_t>(
        device_state_.scatter_index.data(),
        device_state_.y.data(),
        device_state_.y_next.data(),
        n_state,
        n_particles);
#endif
      device_state_.swap();
    } else {
      stale_device_ = true;
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < particles_.size(); ++i) {
        size_t j = index[i];
        particles_[i].set_state(particles_[j]);
      }
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < particles_.size(); ++i) {
        particles_[i].swap();
      }
    }
  }

  std::vector<size_t> resample(const std::vector<real_t>& weights) {
    std::vector<size_t> index(n_particles());
    resample(weights, index);
    return index;
  }

  void resample(const std::vector<real_t>& weights,
                std::vector<size_t>& index) {
    auto it_weights = weights.begin();
    auto it_index = index.begin();
    if (n_pars_ == 0) {
      // One parameter set; shuffle among all particles
      const size_t np = particles_.size();
      real_t u = dust::unif_rand(rng_.state(0));
      dust::filter::resample_weight(it_weights, np, u, 0, it_index);
    } else {
      // Multiple parameter set; shuffle within each group
      // independently (and therefore in parallel)
      const size_t np = particles_.size() / n_pars_;
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < n_pars_; ++i) {
        const size_t j = i * np;
        real_t u = dust::unif_rand(rng_.state(j));
        dust::filter::resample_weight(it_weights + j, np, u, j, it_index + j);
      }
    }

    reorder(index);
    stale_device_ = true;
  }

  void resample_device(const dust::device_array<real_t>& cum_weights) {
    refresh_device();
    std::vector<real_t> shuffle_draws(n_pars_effective());
    for (size_t i = 0; i < n_pars_effective(); ++i) {
      shuffle_draws[i] = dust::unif_rand(rng_.state(0));
    }
    // TODO: eliminate this H->D?
    device_state_.resample_u.set_array(shuffle_draws);

#ifdef __NVCC__
    // Generate the scatter indices
    const size_t blockSize = 128;
    const size_t interval_blockCount =
        (n_particles + blockSize - 1) / blockSize;
    dust::find_intervals<real_t><<<blockSize, interval_blockCount>>>(
      cum_weights.data(),
      n_particles(),
      n_pars_effective(),
      device_state_.scatter_index.data(),
      device_state_.resample_u.data()
    );
    CUDA_CALL(cudaDeviceSynchronize());

    // Shuffle the particles
    const size_t scatter_blockCount =
        (n_particles * n_state + blockSize - 1) / blockSize;
    dust::scatter_device<real_t><<<blockSize, scatter_blockCount>>>(
        device_state_.scatter_index.data(),
        device_state_.y.data(),
        device_state_.y_next.data(),
        n_state,
        n_particles);
    CUDA_CALL(cudaDeviceSynchronize());
#else
    dust::find_intervals<real_t>(
      cum_weights.data(),
      n_particles(),
      n_pars_effective(),
      device_state_.scatter_index.data(),
      device_state_.resample_u.data()
    );
    dust::scatter_device<real_t>(
        device_state_.scatter_index.data(),
        device_state_.y.data(),
        device_state_.y_next.data(),
        n_state,
        n_particles);
#endif

    stale_device_ = true;
  }

  // Used in the filter
  dust::device_array<size_t> kappa() const {
    return device_state_.scatter_index;
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

  const std::map<size_t, std::vector<data_t>>& data() const {
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

  void check_errors() {
    if (errors_.unresolved()) {
      throw std::runtime_error("Errors pending; reset required");
    }
  }

  void reset_errors() {
    errors_.reset();
  }

  std::vector<uint64_t> rng_state() {
    refresh_host();
    return rng_.export_state();
  }

  void set_rng_state(const std::vector<uint64_t>& rng_state) {
    refresh_host();
    rng_.import_state(rng_state);
    stale_device_ = true;
  }

  void set_n_threads(size_t n_threads) {
    n_threads_ = n_threads;
  }

  // NOTE: it only makes sense to expose long_jump, and not jump,
  // because each rng stream is one jump away from the next.
  void rng_long_jump() {
    refresh_host();
    rng_.long_jump();
    stale_device_ = true;
  }

  void set_data(std::map<size_t, std::vector<data_t>> data) {
    data_ = data;
    initialise_device_data();
  }

  std::vector<real_t> compare_data() {
    refresh_host();
    std::vector<real_t> res;
    auto d = data_.find(step());
    if (d != data_.end()) {
      res.resize(particles_.size());
      compare_data(res, d->second);
    }
    return res;
  }

  void compare_data(std::vector<real_t>& res, const std::vector<data_t>& data) {
    const size_t np = particles_.size() / n_pars_effective();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < particles_.size(); ++i) {
      res[i] = particles_[i].compare_data(data[i / np], rng_.state(i));
    }
    stale_device_ = true; // RNG use
  }

  void compare_data_device(dust::device_array<real_t>& res,
                           const size_t data_offset) {
    refresh_device();
#ifdef __NVCC__
    const int warp_size = dust::cuda::warp_size;
    // Set up blocks and shared memory preferences
    size_t blockSize = 32;
    size_t blockCount;
    bool use_shared_L1 = true;
    size_t n_shared_int_effective = device_state_.n_shared_int +
      dust::utils::align_padding(device_state_.n_shared_int * sizeof(int), sizeof(real_t)) / sizeof(int);
    size_t n_shared_real_effective = device_state_.n_shared_real +
      dust::utils::align_padding(device_state_.n_shared_real * sizeof(real_t), 16) / sizeof(real_t);
    size_t shared_size_bytes = n_shared_int_effective * sizeof(int) +
      n_shared_real_effective * sizeof(real_t) +
      sizeof(data_t);
    if (n_particles_each_ < warp_size || shared_size_bytes > shared_size_) {
      // If not enough particles per pars to make a whole block use
      // shared, or if shared_t too big for L1, turn it off, and run
      // in 'classic' mode where each particle is totally independent
      use_shared_L1 = false;
      shared_size_bytes = 0;
      blockCount = n_particles() * (n_particles() + blockSize - 1) / blockSize;
    } else {
      // If it's possible to make blocks with shared_t in L1 cache,
      // each block runs a pars set. Each pars set has enough blocks
      // to run all of its particles, the final block may have some
      // threads that don't do anything (hang off the end)
      blockSize = warp_size * (n_particles_each_ + warp_size - 1) / warp_size;
      blockSize = std::min(static_cast<size_t>(128), blockSize);
      blockCount = n_pars_effective() * (n_particles_each_ + blockSize - 1) /
        blockSize;
    }
    dust::compare_particles<T><<<blockCount, blockSize, shared_size_bytes>>>(
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
                     use_shared_L1);
    CUDA_CALL(cudaDeviceSynchronize());
#else
    dust::compare_particles<T>(
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
                     false);
#endif
    stale_host_ = true; // RNG use
  }

private:
  const size_t n_pars_; // 0 in the "single" case, >=1 otherwise
  const size_t n_particles_each_; // Particles per parameter set
  const size_t n_particles_total_; // Total number of particles
  const bool pars_are_shared_; // Does the n_particles dimension exist in shape?
  std::vector<size_t> shape_; // shape of output
  size_t n_threads_;
  int device_id_;
  dust::pRNG<real_t> rng_;
  std::map<size_t, std::vector<data_t>> data_;
  dust::openmp_errors errors_;

  std::vector<size_t> index_;
  std::vector<dust::Particle<T>> particles_;
  std::vector<dust::shared_ptr<T>> shared_;

  // New things for device support
  dust::device_state<real_t> device_state_;
  dust::device_array<data_t> device_data_;
  std::map<size_t, size_t> device_data_offsets_;

  bool stale_host_;
  bool stale_device_;
  size_t shared_size_;

  // Sets device
  template <typename U = T>
  typename std::enable_if<!dust::has_gpu_support<U>::value, void>::type
  initialise_device(const int device_id) {
    throw std::invalid_argument("GPU support not enabled for this object");
  }

  template <typename U = T>
  typename std::enable_if<dust::has_gpu_support<U>::value, void>::type
  initialise_device(const int device_id) {
    if (device_id < 0) {
      return;
    }
#ifdef __NVCC__
    CUDA_CALL(cudaSetDevice(device_id));
    CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    int shared_size = 0;
    CUDA_CALL(cudaDeviceGetAttribute(&shared_size,
                                     cudaDevAttrMaxSharedMemoryPerBlock,
                                     device_id));
    shared_size_ = static_cast<size_t>(shared_size);

#ifdef DUST_ENABLE_CUDA_PROFILER
    CUDA_CALL(cudaProfilerStart());
#endif
#endif
  }

  void initialise(const pars_t& pars, const size_t step,
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
    stale_host_ = false;
    stale_device_ = true;
  }

  void initialise(const std::vector<pars_t>& pars, const size_t step,
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
    stale_host_ = false;
    stale_device_ = true;
  }

  // This only gets called on construction; the size of these never
  // changes.
  void initialise_device_state() {
    if (device_id_ < 0) {
      return;
    }
    const auto s = shared_[0];
    const size_t n_internal_int = dust::device_internal_int_size<T>(s);
    const size_t n_internal_real = dust::device_internal_real_size<T>(s);
    const size_t n_shared_int = dust::device_shared_int_size<T>(s);
    const size_t n_shared_real = dust::device_shared_real_size<T>(s);
    device_state_.initialise(particles_.size(), n_state_full(), n_pars_effective(),
                            shared_.size(),
                            n_internal_int, n_internal_real,
                            n_shared_int, n_shared_real);
  }

  template <typename U = T>
  typename std::enable_if<!dust::has_gpu_support<U>::value, void>::type
  initialise_device_data() {
  }

  template <typename U = T>
  typename std::enable_if<dust::has_gpu_support<U>::value, void>::type
  initialise_device_data() {
    if (device_id_ < 0) {
      return;
    }
    std::vector<data_t> flattened_data;
    std::vector<size_t> data_offsets(n_data());
    size_t i = 0;
    for (auto & d_step : data()) {
      for (auto & d : d_step.second ) {
        flattened_data.push_back(d);
      }
      device_data_offsets_[d_step.first] = i++;
    }
    device_data_ = dust::device_array<data_t>(flattened_data.size());
    device_data_.set_array(flattened_data);
  }

  template <typename U = T>
  typename std::enable_if<!dust::has_gpu_support<U>::value, void>::type
  update_device_shared() {
  }

  template <typename U = T>
  typename std::enable_if<dust::has_gpu_support<U>::value, void>::type
  update_device_shared() {
    if (device_id_ < 0) {
      return;
    }
    const size_t n_shared_int = device_state_.n_shared_int;
    const size_t n_shared_real = device_state_.n_shared_real;
    std::vector<int> shared_int(n_shared_int * n_pars_effective());
    std::vector<real_t> shared_real(n_shared_real * n_pars_effective());
    for (size_t i = 0; i < shared_.size(); ++i) {
      int * dest_int = shared_int.data() + n_shared_int * i;
      real_t * dest_real = shared_real.data() + n_shared_real * i;
      dust::device_shared_copy<T>(shared_[i], dest_int, dest_real);
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

  template <typename U = T>
  typename std::enable_if<!dust::has_gpu_support<U>::value, void>::type
  update_device_index() {
  }

  template <typename U = T>
  typename std::enable_if<dust::has_gpu_support<U>::value, void>::type
  update_device_index() {
    if (device_id_ < 0) {
      return;
    }
    size_t n_particles = particles_.size();
    std::vector<char> bool_idx(n_state_full() * n_particles, 0);
    // e.g. 4 particles with 3 states ABC stored on device as
    // [1_A, 2_A, 3_A, 4_A, 1_B, 2_B, 3_B, 4_B, 1_C, 2_C, 3_C, 4_C]
    // e.g. index [1, 3] would be
    // [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1] bool index on interleaved state
    // i.e. initialise to zero and copy 1 np times, at each offset given in
    // index
    for (auto idx_pos = index_.cbegin(); idx_pos != index_.cend(); idx_pos++) {
      std::fill_n(bool_idx.begin() + (*idx_pos * n_particles), n_particles, 1);
    }
    device_state_.index.set_array(bool_idx);
    device_state_.set_cub_tmp();
  }

  // Default noop refresh methods
  template <typename U = T>
  typename std::enable_if<!dust::has_gpu_support<U>::value, void>::type
  refresh_device() {
    stale_device_ = false;
  }

  template <typename U = T>
  typename std::enable_if<!dust::has_gpu_support<U>::value, void>::type
  refresh_host() {
    stale_host_ = false;
  }

  // Real refresh methods where we have gpu support
  template <typename U = T>
  typename std::enable_if<dust::has_gpu_support<U>::value, void>::type
  refresh_device() {
    if (device_id_ < 0) {
      throw std::runtime_error("Can't refresh a non-existent device");
    }
    if (stale_device_) {
      const size_t np = n_particles(), ny = n_state_full();
      const size_t rng_len = dust::rng_state_t<real_t>::size();
      std::vector<real_t> y_tmp(ny); // Individual particle state
      std::vector<real_t> y(np * ny); // Interleaved state of all particles
      std::vector<uint64_t> rng(np * rng_len); // Interleaved RNG state
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < np; ++i) {
        // Interleave state
        particles_[i].state_full(y_tmp.begin());
        dust::utils::stride_copy(y.data(), y_tmp, i, np);

        // Interleave RNG state
        dust::rng_state_t<real_t> p_rng = rng_.state(i);
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
    }
  }

  template <typename U = T>
  typename std::enable_if<dust::has_gpu_support<U>::value, void>::type
  refresh_host() {
    if (stale_host_) {
      const size_t np = n_particles(), ny = n_state_full();
      const size_t rng_len = dust::rng_state_t<real_t>::size();
      std::vector<real_t> y_tmp(ny); // Individual particle state
      std::vector<real_t> y(np * ny); // Interleaved state of all particles
      std::vector<uint64_t> rngi(np * rng_len); // Interleaved RNG state
      std::vector<uint64_t> rng(np * rng_len); //  Deinterleaved RNG state
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
      rng_.import_state(rng);
      stale_host_ = false;
    }
  }

  template <typename U = T>
  typename std::enable_if<!dust::has_gpu_support<U>::value, void>::type
  run_device_select() {
  }

  template <typename U = T>
  typename std::enable_if<dust::has_gpu_support<U>::value, void>::type
  run_device_select() {
    if (stale_device_) {
      return;
    }
    // TODO: noop if this has been run and device_state_.y hasn't changed
#ifdef __NVCC__
    size_t size_select_tmp = device_state_.select_tmp.size();
    cub::DeviceSelect::Flagged(device_state_.select_tmp.data(),
                                size_select_tmp,
                                device_state_.y.data(),
                                device_state_.index.data(),
                                device_state_.y_selected.data(),
                                device_state_.n_selected.data(),
                                device_state_.y.size());
#endif
  }
};

}

#endif
