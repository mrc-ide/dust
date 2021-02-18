#ifndef DUST_DUST_HPP
#define DUST_DUST_HPP

#include <dust/rng.hpp>
#include <dust/densities.hpp>
#include <dust/tools.hpp>

#include <algorithm>
#include <memory>
#include <map>
#include <stdexcept>
#include <sstream>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif


// TODO: move these into a utilities file
template <typename T, typename U, typename Enable = void>
size_t destride_copy(T dest, U& src, size_t at, size_t stride) {
  static_assert(!std::is_reference<T>::value,
                "destride_copy should only be used with reference types");
  size_t i;
  for (i = 0; at < src.size(); ++i, at += stride) {
    dest[i] = src[at];
  }
  return i;
}

template <typename T, typename U>
size_t stride_copy(T dest, U src, size_t at, size_t stride) {
  static_assert(!std::is_reference<T>::value,
                "stride_copy should only be used with reference types");
  dest[at] = src;
  return at + stride;
}

template <typename T, typename U>
size_t stride_copy(T dest, const std::vector<U>& src, size_t at, size_t stride) {
  static_assert(!std::is_reference<T>::value,
                "stride_copy should only be used with reference types");
  for (size_t i = 0; i < src.size(); ++i, at += stride) {
    dest[at] = src[i];
  }
  return at;
}


namespace dust {
struct nothing {};
typedef nothing no_data;
typedef nothing no_internal;
typedef nothing no_shared;

// By default we do not support anything on the gpu. This name might
// change, but it does reflect our intent and it's likely that to work
// on a GPU the model will have to provide a number of things. If of
// those becomes a type (as with data, internal and shared) we could
// use the same approach as above.
template <typename T>
struct has_gpu_support : std::false_type {};

template <typename T>
using shared_ptr = std::shared_ptr<const typename T::shared_t>;

template <typename T>
struct pars_t {
  std::shared_ptr<const typename T::shared_t> shared;
  typename T::internal_t internal;

  pars_t(std::shared_ptr<const typename T::shared_t> shared_,
         typename T::internal_t internal_) :
    shared(shared_), internal(internal_) {
  }
  pars_t(typename T::shared_t shared_,
         typename T::internal_t internal_) :
    shared(std::make_shared<const typename T::shared_t>(shared_)),
    internal(internal_) {
  }
  pars_t(typename T::shared_t shared_) :
    pars_t(shared_, dust::nothing()) {
  }
  pars_t(typename T::internal_t internal_) :
    pars_t(dust::nothing(), internal_) {
  }
};

template <typename real_t>
class filter_state {
public:
  filter_state(size_t n_state, size_t n_particles, size_t n_data) :
    n_state_(n_state), n_particles_(n_particles), n_data_(n_data), offset_(0) {
    resize(n_state, n_particles, n_data);
  }

  // default constructable
  filter_state() : filter_state(0, 0, 0) {
  }

  void resize(size_t n_state, size_t n_particles, size_t n_data) {
    n_state_ = n_state;
    n_particles_ = n_particles;
    n_data_ = n_data;
    offset_ = 0;
    history_value.resize(n_state_ * n_particles_ * (n_data_ + 1));
    history_order.resize(n_particles_ * (n_data_ + 1));
    for (size_t i = 0; i < n_particles_; ++i) {
      history_order[i] = i;
    }
  }

  typename std::vector<real_t>::iterator history_value_iterator() {
    return history_value.begin() + offset_ * n_state_ * n_particles_;
  }

  typename std::vector<size_t>::iterator history_order_iterator() {
    return history_order.begin() + offset_ * n_particles_;
  }

  std::vector<real_t> history() const {
    std::vector<real_t> ret(size());
    history(ret.begin());
    return ret;
  }

  // This is a particularly unpleasant bit of bookkeeping and is
  // adapted from mcstate (see the helper files in tests for a
  // translation of the the code). As we proceed we store the values
  // of particles *before* resampling and then we store the index used
  // in resampling. We do not resample all the history at each
  // resample as that is prohibitively expensive.
  //
  // So to output sensible history we start with a particle and we
  // look to see where it "came from" in the previous step
  // (history_index) and propagate this backward in time to
  // reconstruct what is in effect a multifurcating tree.
  // This is analogous to the particle ancestor concept in the
  // particle filter literature.
  //
  // It's possible we could do this more efficiently for some subset
  // of particles too (give me the history of just one particle) by
  // breaking the function before the loop over 'k'.
  //
  // Note that we treat history_order and history_value as read-only
  // though this process so one could safely call this multiple times.
  template <typename Iterator>
  void history(Iterator ret) const {
    std::vector<size_t> index_particle(n_particles_);
    for (size_t i = 0; i < n_particles_; ++i) {
      index_particle[i] = i;
    }
    for (size_t k = 0; k < n_data_ + 1; ++k) {
      size_t i = n_data_ - k;
      auto const it_order = history_order.begin() + i * n_particles_;
      auto const it_value = history_value.begin() + i * n_state_ * n_particles_;
      auto it_ret = ret + i * n_state_ * n_particles_;
      for (size_t j = 0; j < n_particles_; ++j) {
        const size_t idx = *(it_order + index_particle[j]);
        index_particle[j] = idx;
        std::copy_n(it_value + idx * n_state_, n_state_,
                    it_ret + j * n_state_);
      }
    }
  }

  size_t size() const {
    return history_value.size();
  }

  void advance() {
    offset_++;
  }

private:
  size_t n_state_;
  size_t n_particles_;
  size_t n_data_;
  size_t offset_;
  size_t len_;
  std::vector<real_t> history_value;
  std::vector<size_t> history_order;
};

template <typename real_t>
struct device_state {
  void initialise(size_t n_particles, size_t n_state, size_t n_int,
                  size_t n_real) {
    const size_t n_rng = dust::rng_state_t<real_t>::size();
    // NOTE: not setting up yi_selected here, which was used in dustgpu
    y = dust::device_array<real_t>(n_state * n_particles);
    y_next = dust::device_array<real_t>(n_state * n_particles);
    internal_int = dust::device_array<int>(n_int * n_particles);
    internal_real = dust::device_array<real_t>(n_real * n_particles);
    rng = dust::device_array<uint64_t>(n_rng * n_particles);
  }
  void swap() {
    std::swap(y, y_next);
  }
  dust::device_array<real_t> y;
  dust::device_array<real_t> y_next;
  dust::device_array<int> internal_int;
  dust::device_array<real_t> internal_real;
  dust::device_array<uint64_t> rng;
};

// We need to compute the size of space required for integers and
// reals on the device, per particle. Because we work on the
// requirement that every particle has the same dimension we pass an
// arbitrary set of shared parameters (really the first) to
// device_work_size. The underlying model can overload this template
// for either real or int types and return the length of data
// required.
template <typename T>
size_t device_work_size_int(typename dust::shared_ptr<T> shared) {
  return 0;
}

template <typename T>
size_t device_work_size_real(typename dust::shared_ptr<T> shared) {
  return 0;
}

template <typename T>
size_t device_shared_size_int(typename dust::shared_ptr<T> shared) {
  return 0;
}

template <typename T>
size_t device_shared_size_real(typename dust::shared_ptr<T> shared) {
  return 0;
}

template <typename T>
void device_shared_copy(typename dust::shared_ptr<T> shared,
                        int * shared_int,
                        typename T::real_t * shared_real) {
}

template <typename T>
T* shared_copy(T* dest, const std::vector<T>& src) {
  memcpy(dest, src.data(), src.size() * sizeof(T));
  return dest + src.size();
}

template <typename T>
T* shared_copy(T* dest, const T src) {
  *dest = src;
  return dest + 1;
}

}

// We'll need to expand this soon to cope with shared memory, but that
// will be coming via another change to dust. Or we can do it here
// with a shared object that contains just:
// template <typename T>
// struct shared_t {
//   const real_t const * real_data;
//   const int * const * int_data;
//   // plus lengths?
// };
//
// which can just point at the data in the vector because it will live
// longer than the object.
template <typename T>
void update_device(size_t step,
                   const dust::interleaved<typename T::real_t> state,
                   dust::interleaved<int> internal_int,
                   dust::interleaved<typename T::real_t> internal_real,
                   dust::shared_ptr<T> shared,
                   dust::rng_state_t<typename T::real_t>& rng_state,
                   dust::interleaved<typename T::real_t> state_next);

template <typename T>
void run_particles(size_t step_start, size_t step_end, size_t n_particles,
                   size_t n_pars,
                   typename T::real_t * state, typename T::real_t * state_next,
                   int * internal_int, typename T::real_t * internal_real,
                   std::vector<dust::shared_ptr<T>> shared,
                   uint64_t * rng_state);

template <typename T>
class Particle {
public:
  typedef dust::pars_t<T> pars_t;
  typedef typename T::real_t real_t;
  typedef typename T::data_t data_t;

  Particle(pars_t pars, size_t step) :
    _model(pars),
    _step(step),
    _y(_model.initial(_step)),
    _y_swap(_model.size()) {
  }

  void run(const size_t step_end, dust::rng_state_t<real_t>& rng_state) {
    while (_step < step_end) {
      _model.update(_step, _y.data(), rng_state, _y_swap.data());
      _step++;
      std::swap(_y, _y_swap);
    }
  }

  void state(const std::vector<size_t>& index,
             typename std::vector<real_t>::iterator end_state) const {
    for (size_t i = 0; i < index.size(); ++i) {
      *(end_state + i) = _y[index[i]];
    }
  }

  void state_full(typename std::vector<real_t>::iterator end_state) const {
    for (size_t i = 0; i < _y.size(); ++i) {
      *(end_state + i) = _y[i];
    }
  }

  size_t size() const {
    return _y.size();
  }

  size_t step() const {
    return _step;
  }

  void swap() {
    std::swap(_y, _y_swap);
  }

  void set_step(const size_t step) {
    _step = step;
  }

  void set_state(const Particle<T>& other) {
    _y_swap = other._y;
  }

  void set_pars(const Particle<T>& other, bool set_state) {
    _model = other._model;
    _step = other._step;
    if (set_state) {
      _y = _model.initial(_step);
    }
  }

  void set_state(typename std::vector<real_t>::const_iterator state) {
    for (size_t i = 0; i < _y.size(); ++i, ++state) {
      _y[i] = *state;
    }
  }

  real_t compare_data(const data_t& data,
                      dust::rng_state_t<real_t>& rng_state) {
    return _model.compare_data(_y.data(), data, rng_state);
  }

private:
  T _model;
  size_t _step;

  std::vector<real_t> _y;
  std::vector<real_t> _y_swap;
};

template <typename T>
class Dust {
public:
  typedef dust::pars_t<T> pars_t;
  typedef typename T::real_t real_t;
  typedef typename T::data_t data_t;

  Dust(const pars_t& pars, const size_t step, const size_t n_particles,
       const size_t n_threads, const std::vector<uint64_t>& seed) :
    _n_pars(0),
    _n_particles_total(n_particles),
    _n_threads(n_threads),
    _rng(_n_particles_total, seed),
    _stale_host(false),
    _stale_device(true) {
    initialise(pars, step, n_particles, true);
    initialise_index();
  }

  Dust(const std::vector<pars_t>& pars, const size_t step,
       const size_t n_particles, const size_t n_threads,
       const std::vector<uint64_t>& seed) :
    _n_pars(pars.size()),
    _n_particles_total(n_particles * pars.size()),
    _n_threads(n_threads),
    _rng(_n_particles_total, seed),
    _stale_host(false),
    _stale_device(true) {
    initialise(pars, step, n_particles, true);
    initialise_index();
  }

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
  }

  // It's the callee's responsibility to ensure this is the correct length:
  //
  // * if is_matrix is false then state must be length n_state_full()
  //   and all particles get the state
  // * if is_matrix is true, state must be length (n_state_full() *
  //   n_particles()) and every particle gets a different state.
  void set_state(const std::vector<real_t>& state, bool is_matrix) {
    const size_t n_particles = _particles.size();
    const size_t n_state = n_state_full();
    auto it = state.begin();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles[i].set_state(it);
      if (is_matrix) {
        it += n_state;
      }
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
      _particles[i].run(step_end, _rng.state(i));
    }
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

    run_particles<T>(step_start, step_end, _particles.size(),
                     n_pars_effective(),
                     _device_data.y.data(), _device_data.y_next.data(),
                     _device_data.internal_int.data(),
                     _device_data.internal_real.data(),
                     _shared, _device_data.rng.data());

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
      for (size_t t = 0; t < n_time; ++t) {
        _particles[i].run(step_end[t], _rng.state(i));
        size_t offset = t * n_state() * n_particles() + i * n_state();
        _particles[i].state(_index, ret.begin() + offset);
      }
    }
    return ret;
  }

  void state(std::vector<real_t>& end_state) {
    return state(end_state.begin());
  }

  void state(typename std::vector<real_t>::iterator end_state) {
    refresh_host();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(_index, end_state + i * _index.size());
    }
  }

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
    refresh_host();
    const size_t n = n_state_full();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state_full(end_state.begin() + i * n);
    }
  }

  // There are two obvious ways of reordering; we can construct a
  // completely new set of particles, like
  //
  //   std::vector<Particle<T>> next;
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
    refresh_host();
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
      resample_weight(it_weights, np, u, 0, it_index);
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
        resample_weight(it_weights + j, np, u, j, it_index + j);
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

  size_t step() const {
    return _particles.front().step();
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

  std::vector<real_t> filter(bool save_history) {
    if (_data.size() == 0) {
      throw std::invalid_argument("Data has not been set for this object");
    }

    const size_t n_particles = _particles.size();
    const size_t n_particles_each = n_particles / n_pars_effective();
    std::vector<real_t> log_likelihood(n_pars_effective());
    std::vector<real_t> log_likelihood_step(n_pars_effective());
    std::vector<real_t> weights(n_particles);
    std::vector<size_t> kappa(n_particles);

    if (save_history) {
      filter_state_.resize(_index.size(), _particles.size(), _data.size());
      state(filter_state_.history_value_iterator());
      filter_state_.advance();
    }

    for (auto & d : _data) {
      run(d.first);
      compare_data(weights, d.second);

      // TODO: we should cope better with the case where all weights
      // are 0; I think that is the behaviour in the model (or rather
      // the case where there is no data and so we do not resample)
      //
      // TODO: we should cope better with the case where one filter
      // has become impossible but others continue, but that's hard!
      auto wi = weights.begin();
      for (size_t i = 0; i < n_pars_effective(); ++i) {
        log_likelihood_step[i] =
          scale_log_weights<real_t>(wi, n_particles_each);
        log_likelihood[i] += log_likelihood_step[i];
        wi += n_particles_each;
      }

      // We could move this below if wanted but we'd have to rewrite
      // the re-sort algorithm.
      if (save_history) {
        state(filter_state_.history_value_iterator());
      }

      resample(weights, kappa);

      if (save_history) {
        std::copy(kappa.begin(), kappa.end(),
                  filter_state_.history_order_iterator());
        filter_state_.advance();
      }
    }

    return log_likelihood;
  }

  const dust::filter_state<real_t>& filter_history() const {
    return filter_state_;
  }

private:
  const size_t _n_pars; // 0 in the "single" case, >=1 otherwise
  const size_t _n_particles_total; // Total number of particles
  size_t _n_threads;
  dust::pRNG<real_t> _rng;
  std::map<size_t, std::vector<data_t>> _data;

  std::vector<size_t> _index;
  std::vector<Particle<T>> _particles;
  // TODO: this is complicated where we have more than one parameter
  // set; there we need to keep track of a vector of these
  // things. This will do for now but we'll need to consider this
  // carefully in the actual GPU implementation.
  std::vector<dust::shared_ptr<T>> _shared;

  // Only used if we have data; this is going to change around a bit.
  dust::filter_state<real_t> filter_state_;

  // New things for device support
  dust::device_state<real_t> _device_data;

  bool _stale_host;
  bool _stale_device;

  void initialise(const pars_t& pars, const size_t step,
                  const size_t n_particles, bool set_state) {
    const size_t n = _particles.size() == 0 ? 0 : n_state_full();
    Particle<T> p(pars, step);
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
    _stale_host = false;
    _stale_device = true;
  }

  void initialise(const std::vector<pars_t>& pars, const size_t step,
                  const size_t n_particles, bool set_state) {
    size_t n = _particles.size() == 0 ? 0 : n_state_full();
    std::vector<Particle<T>> p;
    for (size_t i = 0; i < _n_pars; ++i) {
      p.push_back(Particle<T>(pars[i], step));
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
    _stale_host = false;
    _stale_device = true;
  }

  // This only gets called on construction; the size of these never
  // changes.
  void initialise_device_data() {
    const size_t n_int = dust::device_work_size_int<T>(_shared[0]);
    const size_t n_real = dust::device_work_size_real<T>(_shared[0]);
    _device_data.initialise(_particles.size(), n_state_full(), n_int, n_real);
  }

  void initialise_index() {
    const size_t n = n_state_full();
    _index.clear();
    _index.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      _index.push_back(i);
    }
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
        stride_copy(y.data(), y_tmp, i, np);

        // Interleave RNG state
        dust::rng_state_t<real_t> p_rng = _rng.state(i);
        size_t rng_offset = i;
        for (size_t j = 0; j < rng_len; ++j) {
          rng_offset = stride_copy(rng.data(), p_rng[j], rng_offset, np);
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
        destride_copy(y_tmp.data(), y, i, np);
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

// TODO: The exact type here for the shared memory will likely change;
// we'll need to do some translation in dust into native types in
// order to make the device copy possible. There's nothing really
// complicated in these in practice.
template <typename T>
void run_particles(size_t step_start, size_t step_end, size_t n_particles,
                   size_t n_pars,
                   typename T::real_t * state, typename T::real_t * state_next,
                   int * internal_int, typename T::real_t * internal_real,
                   std::vector<dust::shared_ptr<T>> shared,
                   uint64_t * rng_state) {
  typedef typename T::real_t real_t;
  const size_t n_particles_each = n_particles / n_pars;

  // omp here
  for (size_t i = 0; i < n_particles; ++i) {
    dust::interleaved<real_t> p_state(state, i, n_particles);
    dust::interleaved<real_t> p_state_next(state_next, i, n_particles);
    dust::interleaved<int> p_internal_int(internal_int, i, n_particles);
    dust::interleaved<real_t> p_internal_real(internal_real, i, n_particles);
    dust::interleaved<uint64_t> p_rng(rng_state, i, n_particles);
    // TODO: this needs work before moving to the device, but it might
    // not be that bad in practice. We'll need some extra code to deal
    // with the blocks (before the loop) too.
    dust::shared_ptr<T> p_shared = shared[i / n_particles_each];

    dust::rng_state_t<real_t> rng_block = dust::get_rng_state<real_t>(p_rng);
    for (size_t step = step_start; step < step_end; ++step) {
      update_device<T>(step,
                       p_state,
                       p_internal_int,
                       p_internal_real,
                       p_shared,
                       rng_block,
                       p_state_next);
      std::swap(p_state, p_state_next);
      // dust::interleaved<real_t> tmp = p_state;
      // p_state = p_state_next;
      // p_state_next = tmp;
    }
    dust::put_rng_state(rng_block, p_rng);
  }
}

template <typename T>
std::vector<typename T::real_t>
dust_simulate(const std::vector<size_t>& steps,
              const std::vector<dust::pars_t<T>>& pars,
              std::vector<typename T::real_t>& state,
              const std::vector<size_t>& index,
              const size_t n_threads,
              std::vector<uint64_t>& seed,
              bool save_state) {
  typedef typename T::real_t real_t;
  const size_t n_state_return = index.size();
  const size_t n_particles = pars.size();
  std::vector<Particle<T>> particles;
  particles.reserve(n_particles);
  for (size_t i = 0; i < n_particles; ++i) {
    particles.push_back(Particle<T>(pars[i], steps[0]));
    if (i > 0 && particles.back().size() != particles.front().size()) {
      std::stringstream msg;
      msg << "Particles have different state sizes: particle " << i + 1 <<
        " had length " << particles.back().size() << " but expected " <<
        particles.front().size();
      throw std::invalid_argument(msg.str());
    }
  }
  const size_t n_state_full = particles.front().size();

  dust::pRNG<real_t> rng(n_particles, seed);
  std::vector<real_t> ret(n_particles * n_state_return * steps.size());
  size_t n_steps = steps.size();

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (size_t i = 0; i < particles.size(); ++i) {
    particles[i].set_state(state.begin() + n_state_full * i);
    for (size_t t = 0; t < n_steps; ++t) {
      particles[i].run(steps[t], rng.state(i));
      size_t offset = t * n_state_return * n_particles + i * n_state_return;
      particles[i].state(index, ret.begin() + offset);
    }
    if (save_state) {
      particles[i].state_full(state.begin() + n_state_full * i);
    }
  }

  // To continue we'd also need the rng state:
  if (save_state) {
    rng.export_state(seed);
  }

  return ret;
}

#endif
