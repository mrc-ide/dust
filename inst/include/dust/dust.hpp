#ifndef DUST_DUST_HPP
#define DUST_DUST_HPP

#include <dust/rng.hpp>

#include <algorithm>
#include <utility>
#ifdef _OPENMP
#if _OPENMP >= 201511
#define OPENMP_HAS_MONOTONIC 1
#endif
#include <omp.h>
#endif

template <typename T>
class Particle {
public:
  typedef typename T::init_t init_t;
  typedef typename T::int_t int_t;
  typedef typename T::real_t real_t;
  typedef typename dust::RNG<real_t, int_t> rng_t;

  Particle(init_t data, size_t step) :
    _model(data),
    _step(step),
    _y(_model.initial(_step)),
    _y_swap(_model.size()) {
  }

  void run(const size_t step_end, rng_t& rng) {
    while (_step < step_end) {
      _model.update(_step, _y, rng, _y_swap);
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

  void set_state(const Particle<T> other) {
    _y_swap = other._y;
  }

  void set_state(typename std::vector<real_t>::const_iterator state) {
    for (size_t i = 0; i < _y.size(); ++i, ++state) {
      _y[i] = *state;
    }
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
  typedef typename T::init_t init_t;
  typedef typename T::int_t int_t;
  typedef typename T::real_t real_t;
  typedef typename dust::RNG<real_t, int_t> rng_t;

  Dust(const init_t data, const size_t step, const size_t n_particles,
       const size_t n_threads, const size_t seed) :
    _n_threads(n_threads),
    _rng(n_particles, seed) {
    initialise(data, step, n_particles);
  }

  void reset(const init_t data, const size_t step) {
    const size_t n_particles = _particles.size();
    initialise(data, step, n_particles);
  }

  // It's the callee's responsibility to ensure that index is in
  // range [0, n-1]
  void set_index(const std::vector<size_t>& index) {
    _index = index;
  }

  // It's the callee's responsibility to ensure this is the correct length
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
    #pragma omp parallel num_threads(_n_threads)
    {
      // Making this monotonic:static gives us a reliable sequence
      // through the data, forcing each thread to move through with a
      // stride of n_threads. However, this requires relatively recent
      // openmp (>= 4.5, released in 2015) so we will fall back on
      // ordered which will work over more versions at the risk of
      // being slower if there is any variation in how long each
      // iteration takes.
#ifdef OPENMP_HAS_MONOTONIC
      #pragma omp for schedule(monotonic:static, 1)
#else
      #pragma omp for schedule(static, 1) ordered
#endif
      for (size_t i = 0; i < _particles.size(); ++i) {
#ifndef OPENMP_HAS_MONOTONIC
        #pragma omp ordered
#endif
        _particles[i].run(step_end, pick_generator(i));
      }
    }
  }

  void state(std::vector<real_t>& end_state) const {
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(_index, end_state.begin() + i * _index.size());
    }
  }

  void state(std::vector<size_t> index,
             std::vector<real_t>& end_state) const {
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(index, end_state.begin() + i * index.size());
    }
  }

  void state_full(std::vector<real_t>& end_state) const {
    const size_t n = n_state_full();
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
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
    for (size_t i = 0; i < _particles.size(); ++i) {
      size_t j = index[i];
      _particles[i].set_state(_particles[j]);
    }
    for (auto& p : _particles) {
      p.swap();
    }
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

  size_t step() const {
    return _particles.front().step();
  }

private:
  std::vector<size_t> _index;
  const size_t _n_threads;
  dust::pRNG<real_t, int_t> _rng;
  std::vector<Particle<T>> _particles;

  // For 10 particles, 4 generators and 1, 2, 4 threads we want this:
  //
  // i:  0 1 2 3 4 5 6 7 8 9
  // g:  0 1 2 3 0 1 2 3 0 1 - rng used for the iteration
  // t1: 0 0 0 0 0 0 0 0 0 0 - thread index that executes each with 1 thread
  // t2: 0 1 0 1 0 1 0 1 0 1 - ...with 2
  // t4: 0 1 2 3 0 1 2 3 0 1 - ...with 4
  //
  // So with
  // - 1 thread: 0: (0 1 2 3)
  // - 2 threads 0: (0 2), 1: (1 3)
  // - 4 threads 0: (0), 1: (1), 2: (2), 3: (3)
  //
  // So the rng number can be picked up directly by doing
  //
  //   i % _rng.size()
  //
  // though this relies on the openmp scheduler, which technically I
  // think we should not be doing. We could derive it from the thread
  // index to provide a set of allowable rngs but this will be harder
  // to get deterministic.
  //
  // I'm not convinced that this will always do the Right Thing with
  // loop leftovers either.
  rng_t& pick_generator(const size_t i) {
    return _rng(i % _rng.size());
  }

  void initialise(const init_t data, const size_t step,
                  const size_t n_particles) {
    _particles.clear();
    _particles.reserve(n_particles);
    for (size_t i = 0; i < n_particles; ++i) {
      _particles.push_back(Particle<T>(data, step));
    }

    const size_t n = n_state_full();
    _index.clear();
    _index.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      _index.push_back(i);
    }
  }
};

#endif
