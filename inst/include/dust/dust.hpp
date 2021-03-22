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

template <typename T>
class Dust {
public:
  typedef dust::pars_t<T> pars_t;
  typedef typename T::real_t real_t;
  typedef typename T::data_t data_t;

  Dust(const pars_t& pars, const size_t step, const size_t n_particles,
       const size_t n_threads, const std::vector<uint64_t>& seed,
       size_t device_id) :
    _n_pars(0),
    _n_particles_each(n_particles),
    _n_particles_total(n_particles),
    _pars_are_shared(true),
    _n_threads(n_threads),
    _device_id(device_id),
    _rng(_n_particles_total, seed),
    _errors(n_particles),
    _stale_host(false),
    _stale_device(true) {
#ifdef __NVCC__
    initialise_device(device_id);
#endif
    initialise(pars, step, n_particles, true);
    initialise_index();
    _shape = {n_particles};
  }

  Dust(const std::vector<pars_t>& pars, const size_t step,
       const size_t n_particles, const size_t n_threads,
       const std::vector<uint64_t>& seed, size_t device_id,
       std::vector<size_t> shape) :
    _n_pars(pars.size()),
    _n_particles_each(n_particles == 0 ? 1 : n_particles),
    _n_particles_total(_n_particles_each * pars.size()),
    _pars_are_shared(n_particles != 0),
    _n_threads(n_threads),
    _device_id(device_id),
    _rng(_n_particles_total, seed),
    _errors(n_particles),
    _stale_host(false),
    _stale_device(true) {
#ifdef __NVCC__
    initialise_device(device_id);
#endif
    initialise(pars, step, _n_particles_each, true);
    initialise_index();
    // constructing the shape here is harder than above.
    if (n_particles > 0) {
      _shape.push_back(n_particles);
    }
    for (auto i : shape) {
      _shape.push_back(i);
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
    const size_t n_particles = _particles.size();
    initialise(pars, step, n_particles, true);
  }

  void reset(const std::vector<pars_t>& pars, const size_t step) {
    const size_t n_particles = _particles.size() / pars.size();
    initialise(pars, step, n_particles, true);
  }

  void set_pars(const pars_t& pars) {
    const size_t n_particles = _particles.size();
    initialise(pars, step(), n_particles, false);
  }

  void set_pars(const std::vector<pars_t>& pars) {
    const size_t n_particles = _particles.size();
    initialise(pars, step(), n_particles / pars.size(), false);
  }

  // It's the callee's responsibility to ensure that index is in
  // range [0, n-1]
  void set_index(const std::vector<size_t>& index) {
    _index = index;
    update_device_index();
  }

  // It's the callee's responsibility to ensure this is the correct length:
  //
  // * if is_matrix is false then state must be length n_state_full()
  //   and all particles get the state
  // * if is_matrix is true, state must be length (n_state_full() *
  //   n_particles()) and every particle gets a different state.
  void set_state(const std::vector<real_t>& state, bool individual) {
    const size_t n_particles = _particles.size();
    const size_t n_state = n_state_full();
    const size_t n = individual ? 1 : _n_particles_each;
    auto it = state.begin();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles[i].set_state(it + (i / n) * n_state);
    }
    _stale_device = true;
  }

  void set_step(const size_t step) {
    const size_t n_particles = _particles.size();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles[i].set_step(step);
    }
  }

  void set_step(const std::vector<size_t>& step) {
    const size_t n_particles = _particles.size();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles[i].set_step(step[i]);
    }
    const auto r = std::minmax_element(step.begin(), step.end());
    if (*r.second > *r.first) {
      run(*r.second);
    }
  }

  void run(const size_t step_end) {
    refresh_host();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      try {
        _particles[i].run(step_end, _rng.state(i));
      } catch (std::exception const& e) {
        _errors.capture(e, i);
      }
    }
    _errors.report();
    _stale_device = true;
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
    // Set up blocks and shared memory preferences
    size_t blockSize = 128;
    size_t blockCount;
    bool use_shared_L1 = true;
    size_t n_shared_int_effective = _device_data.n_shared_int +
      dust::utils::align_padding(_device_data.n_shared_int * sizeof(int), sizeof(real_t)) / sizeof(int);
    size_t shared_size_bytes = n_shared_int_effective * sizeof(int) +
      _device_data.n_shared_real * sizeof(real_t);
    if (_n_particles_each < warp_size || shared_size_bytes > _shared_size) {
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
      blockSize = warp_size * (_n_particles_each + warp_size - 1) / warp_size;
      blockSize = std::min(static_cast<size_t>(128), blockSize);
      blockCount = n_pars_effective() * (_n_particles_each + blockSize - 1) /
        blockSize;
    }
    run_particles<T><<<blockCount, blockSize, shared_size_bytes>>>(
                     step_start, step_end, _particles.size(),
                     n_pars_effective(),
                     _device_data.y.data(), _device_data.y_next.data(),
                     _device_data.internal_int.data(),
                     _device_data.internal_real.data(),
                     _device_data.n_shared_int,
                     _device_data.n_shared_real,
                     _device_data.shared_int.data(),
                     _device_data.shared_real.data(),
                     _device_data.rng.data(),
                     use_shared_L1);
    CUDA_CALL(cudaDeviceSynchronize());
#else
    run_particles<T>(step_start, step_end, _particles.size(),
                     n_pars_effective(),
                     _device_data.y.data(), _device_data.y_next.data(),
                     _device_data.internal_int.data(),
                     _device_data.internal_real.data(),
                     _device_data.n_shared_int,
                     _device_data.n_shared_real,
                     _device_data.shared_int.data(),
                     _device_data.shared_real.data(),
                     _device_data.rng.data(),
                     false);
#endif

    // In the inner loop, the swap will keep the locally scoped
    // interleaved variables updated, but the interleaved variables
    // passed in have not yet been updated.  If an even number of
    // steps have been run state will have been swapped back into the
    // original place, but an on odd number of steps the passed
    // variables need to be swapped.
    if ((step_end - step_start) % 2 == 1) {
      _device_data.swap();
   }

    _stale_host = true;
    set_step(step_end);
  }

  std::vector<real_t> simulate(const std::vector<size_t>& step_end) {
    const size_t n_time = step_end.size();
    std::vector<real_t> ret(n_particles() * n_state() * n_time);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      try {
        for (size_t t = 0; t < n_time; ++t) {
          _particles[i].run(step_end[t], _rng.state(i));
          size_t offset = t * n_state() * n_particles() + i * n_state();
          _particles[i].state(_index, ret.begin() + offset);
        }
      } catch (std::exception const& e) {
        _errors.capture(e, i);
      }
    }
    _errors.report();
    return ret;
  }

  void state(std::vector<real_t>& end_state) {
    return state(end_state.begin());
  }

  // TODO: tidy this up with some templates
  void state(typename std::vector<real_t>::iterator end_state) {
      size_t np = _particles.size();
      size_t index_size = _index.size();
    if (_stale_host) {
#ifdef __NVCC__
      size_t size_select_tmp = _device_data.select_tmp.size();
      // Run the selection and copy items back
      cub::DeviceSelect::Flagged(_device_data.select_tmp.data(),
                                 size_select_tmp,
                                 _device_data.y.data(),
                                 _device_data.index.data(),
                                 _device_data.y_selected.data(),
                                 _device_data.n_selected.data(),
                                 _device_data.y.size());
      std::vector<real_t> y_selected(np * index_size);
      _device_data.y_selected.get_array(y_selected);

#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
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
    if (!_stale_host) {
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
      for (size_t i = 0; i < np; ++i) {
        _particles[i].state(_index, end_state + i * index_size);
      }
    }
  }

  // TODO: this does not use device_select. But if index is being provided
  // may not matter much
  void state(std::vector<size_t> index,
             std::vector<real_t>& end_state) {
    refresh_host();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(index, end_state.begin() + i * index.size());
    }
  }

  void state_full(std::vector<real_t>& end_state) {
    state_full(end_state.begin());
  }

  void state_full(typename std::vector<real_t>::iterator end_state) {
    refresh_host();
    const size_t n = n_state_full();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state_full(end_state + i * n);
    }
  }

  // There are two obvious ways of reordering; we can construct a
  // completely new set of particles, like
  //
  //   std::vector<dust::Particle<T>> next;
  //   for (auto const& i: index) {
  //     next.push_back(_particles[i]);
  //   }
  //   _particles = next;
  //
  // but this seems like a lot of churn.  The other way is to treat it
  // like a slightly weird state update where we swap around the
  // contents of the particle state (uses the set_state() and swap()
  // methods on particles).
  void reorder(const std::vector<size_t>& index) {
    if (_stale_host) {
      size_t n_particles = _particles.size();
      size_t n_state = n_state_full();
      _device_data.scatter_index.set_array(index);
#ifdef __NVCC__
      const size_t blockSize = 128;
      const size_t blockCount =
        (n_state * n_particles + blockSize - 1) / blockSize;
      scatter_device<real_t><<<blockCount, blockSize>>>(
        _device_data.scatter_index.data(),
        _device_data.y.data(),
        _device_data.y_next.data(),
        n_state,
        n_particles);
      CUDA_CALL(cudaDeviceSynchronize());
#else
      scatter_device<real_t>(
        _device_data.scatter_index.data(),
        _device_data.y.data(),
        _device_data.y_next.data(),
        n_state,
        n_particles);
#endif
      _device_data.swap();
    } else {
      _stale_device = true;
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
      for (size_t i = 0; i < _particles.size(); ++i) {
        size_t j = index[i];
        _particles[i].set_state(_particles[j]);
      }
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
      for (size_t i = 0; i < _particles.size(); ++i) {
        _particles[i].swap();
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
    if (_n_pars == 0) {
      // One parameter set; shuffle among all particles
      const size_t np = _particles.size();
      real_t u = dust::unif_rand(_rng.state(0));
      dust::filter::resample_weight(it_weights, np, u, 0, it_index);
    } else {
      // Multiple parameter set; shuffle within each group
      // independently (and therefore in parallel)
      const size_t np = _particles.size() / _n_pars;
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
      for (size_t i = 0; i < _n_pars; ++i) {
        const size_t j = i * np;
        real_t u = dust::unif_rand(_rng.state(j));
        dust::filter::resample_weight(it_weights + j, np, u, j, it_index + j);
      }
    }

    reorder(index);
    _stale_device = true;
  }

  size_t n_particles() const {
    return _particles.size();
  }

  size_t n_state() const {
    return _index.size();
  }

  size_t n_state_full() const {
    return _particles.front().size();
  }

  size_t n_pars() const {
    return _n_pars;
  }

  size_t n_pars_effective() const {
    return _n_pars == 0 ? 1 : _n_pars;
  }

  bool pars_are_shared() const {
    return _pars_are_shared;
  }

  size_t n_data() const {
    return _data.size();
  }

  const std::map<size_t, std::vector<data_t>>& data() const {
    return _data;
  }

  size_t step() const {
    return _particles.front().step();
  }

  const std::vector<size_t>& shape() const {
    return _shape;
  }

  void check_errors() {
    if (_errors.unresolved()) {
      throw std::runtime_error("Errors pending; reset required");
    }
  }

  void reset_errors() {
    _errors.reset();
  }

  std::vector<uint64_t> rng_state() {
    refresh_host();
    return _rng.export_state();
  }

  void set_rng_state(const std::vector<uint64_t>& rng_state) {
    refresh_host();
    _rng.import_state(rng_state);
    _stale_device = true;
  }

  void set_n_threads(size_t n_threads) {
    _n_threads = n_threads;
  }

  // NOTE: it only makes sense to expose long_jump, and not jump,
  // because each rng stream is one jump away from the next.
  void rng_long_jump() {
    refresh_host();
    _rng.long_jump();
    _stale_device = true;
  }

  void set_data(std::map<size_t, std::vector<data_t>> data) {
    _data = data;
  }

  std::vector<real_t> compare_data() {
    refresh_host();
    std::vector<real_t> res;
    auto d = _data.find(step());
    if (d != _data.end()) {
      res.resize(_particles.size());
      compare_data(res, d->second);
    }
    return res;
  }

  void compare_data(std::vector<real_t>& res, const std::vector<data_t>& data) {
    const size_t np = _particles.size() / n_pars_effective();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      res[i] = _particles[i].compare_data(data[i / np], _rng.state(i));
    }
  }

private:
  const size_t _n_pars; // 0 in the "single" case, >=1 otherwise
  const size_t _n_particles_each; // Particles per parameter set
  const size_t _n_particles_total; // Total number of particles
  const bool _pars_are_shared; // Does the n_particles dimension exist in shape?
  std::vector<size_t> _shape; // shape of output
  size_t _n_threads;
  int _device_id;
  dust::pRNG<real_t> _rng;
  std::map<size_t, std::vector<data_t>> _data;
  dust::openmp_errors _errors;

  std::vector<size_t> _index;
  std::vector<dust::Particle<T>> _particles;
  std::vector<dust::shared_ptr<T>> _shared;

  // New things for device support
  dust::device_state<real_t> _device_data;

  bool _stale_host;
  bool _stale_device;
  size_t _shared_size;

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
    _shared_size = static_cast<size_t>(shared_size);

#ifdef DUST_ENABLE_CUDA_PROFILER
    CUDA_CALL(cudaProfilerStart());
#endif
#endif
  }

  void initialise(const pars_t& pars, const size_t step,
                  const size_t n_particles, bool set_state) {
    const size_t n = _particles.size() == 0 ? 0 : n_state_full();
    dust::Particle<T> p(pars, step);
    if (n > 0 && p.size() != n) {
      std::stringstream msg;
      msg << "'pars' created inconsistent state size: " <<
        "expected length " << n << " but created length " <<
        p.size();
      throw std::invalid_argument(msg.str());
    }
    if (_particles.size() == n_particles) {
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
      for (size_t i = 0; i < n_particles; ++i) {
        _particles[i].set_pars(p, set_state);
      }
      _shared[0] = pars.shared;
    } else {
      _particles.clear();
      _particles.reserve(n_particles);
      for (size_t i = 0; i < n_particles; ++i) {
        _particles.push_back(p);
      }
      _shared = {pars.shared};
      initialise_device_data();
    }
    reset_errors();
    update_device_shared();
    _stale_host = false;
    _stale_device = true;
  }

  void initialise(const std::vector<pars_t>& pars, const size_t step,
                  const size_t n_particles, bool set_state) {
    size_t n = _particles.size() == 0 ? 0 : n_state_full();
    std::vector<dust::Particle<T>> p;
    for (size_t i = 0; i < _n_pars; ++i) {
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
    if (_particles.size() == _n_particles_total) {
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
      for (size_t i = 0; i < _n_particles_total; ++i) {
        _particles[i].set_pars(p[i / n_particles], set_state);
      }
      for (size_t i = 0; i < pars.size(); ++i) {
        _shared[i] = pars[i].shared;
      }
    } else {
      _particles.clear();
      _particles.reserve(n_particles * _n_pars);
      for (size_t i = 0; i < _n_pars; ++i) {
        for (size_t j = 0; j < n_particles; ++j) {
          _particles.push_back(p[i]);
        }
        _shared.push_back(pars[i].shared);
      }
      initialise_device_data();
    }
    reset_errors();
    update_device_shared();
    _stale_host = false;
    _stale_device = true;
  }

  // This only gets called on construction; the size of these never
  // changes.
  void initialise_device_data() {
    if (_device_id < 0) {
      return;
    }
    const auto s = _shared[0];
    const size_t n_internal_int = dust::device_internal_int_size<T>(s);
    const size_t n_internal_real = dust::device_internal_real_size<T>(s);
    const size_t n_shared_int = dust::device_shared_int_size<T>(s);
    const size_t n_shared_real = dust::device_shared_real_size<T>(s);
    _device_data.initialise(_particles.size(), n_state_full(), _shared.size(),
                            n_internal_int, n_internal_real,
                            n_shared_int, n_shared_real);
  }

  template <typename U = T>
  typename std::enable_if<!dust::has_gpu_support<U>::value, void>::type
  update_device_shared() {
  }

  template <typename U = T>
  typename std::enable_if<dust::has_gpu_support<U>::value, void>::type
  update_device_shared() {
    if (_device_id < 0) {
      return;
    }
    const size_t n_shared_int = _device_data.n_shared_int;
    const size_t n_shared_real = _device_data.n_shared_real;
    std::vector<int> shared_int(n_shared_int * n_pars_effective());
    std::vector<real_t> shared_real(n_shared_real * n_pars_effective());
    for (size_t i = 0; i < _shared.size(); ++i) {
      int * dest_int = shared_int.data() + n_shared_int * i;
      real_t * dest_real = shared_real.data() + n_shared_real * i;
      dust::device_shared_copy<T>(_shared[i], dest_int, dest_real);
    }
    _device_data.shared_int.set_array(shared_int);
    _device_data.shared_real.set_array(shared_real);
  }

  void initialise_index() {
    const size_t n = n_state_full();
    _index.clear();
    _index.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      _index.push_back(i);
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
    if (_device_id < 0) {
      return;
    }
    size_t n_particles = _particles.size();
    std::vector<char> bool_idx(n_state_full() * n_particles, 0);
    // e.g. 4 particles with 3 states ABC stored on device as
    // [1_A, 2_A, 3_A, 4_A, 1_B, 2_B, 3_B, 4_B, 1_C, 2_C, 3_C, 4_C]
    // e.g. index [1, 3] would be
    // [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1] bool index on interleaved state
    // i.e. initialise to zero and copy 1 np times, at each offset given in
    // index
    for (auto idx_pos = _index.cbegin(); idx_pos != _index.cend(); idx_pos++) {
      std::fill_n(bool_idx.begin() + (*idx_pos * n_particles), n_particles, 1);
    }
    _device_data.index.set_array(bool_idx);
    _device_data.set_cub_tmp();
  }

  // Default noop refresh methods
  template <typename U = T>
  typename std::enable_if<!dust::has_gpu_support<U>::value, void>::type
  refresh_device() {
    _stale_device = false;
  }

  template <typename U = T>
  typename std::enable_if<!dust::has_gpu_support<U>::value, void>::type
  refresh_host() {
    _stale_host = false;
  }

  // Real refresh methods where we have gpu support
  template <typename U = T>
  typename std::enable_if<dust::has_gpu_support<U>::value, void>::type
  refresh_device() {
    if (_device_id < 0) {
      throw std::runtime_error("Can't refresh a non-existent device");
    }
    if (_stale_device) {
      const size_t np = n_particles(), ny = n_state_full();
      const size_t rng_len = dust::rng_state_t<real_t>::size();
      std::vector<real_t> y_tmp(ny); // Individual particle state
      std::vector<real_t> y(np * ny); // Interleaved state of all particles
      std::vector<uint64_t> rng(np * rng_len); // Interleaved RNG state
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
      for (size_t i = 0; i < np; ++i) {
        // Interleave state
        _particles[i].state_full(y_tmp.begin());
        dust::utils::stride_copy(y.data(), y_tmp, i, np);

        // Interleave RNG state
        dust::rng_state_t<real_t> p_rng = _rng.state(i);
        size_t rng_offset = i;
        for (size_t j = 0; j < rng_len; ++j) {
          rng_offset = dust::utils::stride_copy(rng.data(), p_rng[j],
                                                rng_offset, np);
        }
      }
      // H -> D copies
      _device_data.y.set_array(y);
      _device_data.rng.set_array(rng);
      _stale_device = false;
    }
  }

  template <typename U = T>
  typename std::enable_if<dust::has_gpu_support<U>::value, void>::type
  refresh_host() {
    if (_stale_host) {
      const size_t np = n_particles(), ny = n_state_full();
      const size_t rng_len = dust::rng_state_t<real_t>::size();
      std::vector<real_t> y_tmp(ny); // Individual particle state
      std::vector<real_t> y(np * ny); // Interleaved state of all particles
      std::vector<uint64_t> rngi(np * rng_len); // Interleaved RNG state
      std::vector<uint64_t> rng(np * rng_len); //  Deinterleaved RNG state
      // D -> H copies
      _device_data.y.get_array(y);
      _device_data.rng.get_array(rngi);
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
      for (size_t i = 0; i < np; ++i) {
        dust::utils::destride_copy(y_tmp.data(), y, i, np);
        _particles[i].set_state(y_tmp.begin());

        // Destride RNG
        for (size_t j = 0; j < rng_len; ++j) {
          rng[i * rng_len + j] = rngi[i + j * np];
        }
      }
      _rng.import_state(rng);
      _stale_host = false;
    }
  }
};

#endif
