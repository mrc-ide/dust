#ifndef DUST_DUST_HPP
#define DUST_DUST_HPP

#include <dust/rng.hpp>

#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
class Particle {
public:
  typedef typename T::init_t init_t;
  typedef typename T::int_t int_t;
  typedef typename T::float_t float_t;

  Particle(init_t data, size_t step) :
    _model(data),
    _step(step),
    _y(_model.initial(_step)),
    _y_swap(_model.size()) {
  }

  void run(const size_t step_end, dust::RNG<float_t, int_t>& rng) {
    while (_step < step_end) {
      _model.update(_step, _y, rng, _y_swap);
      _step++;
      std::swap(_y, _y_swap);
    }
  }

  void state(const std::vector<size_t>& index_y,
             typename std::vector<float_t>::iterator end_state) const {
    for (size_t i = 0; i < index_y.size(); ++i) {
      *(end_state + i) = _y[index_y[i]];
    }
  }

  void state(typename std::vector<float_t>::iterator end_state) const {
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

  void update(const Particle<T> other) {
    _y_swap = other._y;
  }

private:
  T _model;
  size_t _step;

  std::vector<float_t> _y;
  std::vector<float_t> _y_swap;
};

template <typename T>
class Dust {
public:
  typedef typename T::init_t init_t;
  typedef typename T::int_t int_t;
  typedef typename T::float_t float_t;

  Dust(const init_t data, const size_t step,
       const std::vector<size_t> index_y,
       const size_t n_particles, const size_t n_threads,
       const size_t n_generators, const size_t seed) :
    _index_y(index_y),
    _n_threads(n_threads),
    _rng(n_generators, seed) {
    for (size_t i = 0; i < n_particles; ++i) {
      _particles.push_back(Particle<T>(data, step));
    }
  }

  void reset(const init_t data, const size_t step) {
    size_t n_particles = _particles.size();
    _particles.clear();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles.push_back(Particle<T>(data, step));
    }
  }

  void run(const size_t step_end) {
    #pragma omp parallel num_threads(_n_threads)
    {
      size_t thread_idx = 0;
#ifdef _OPENMP
      thread_idx = omp_get_thread_num();
#endif

      #pragma omp for schedule(static) ordered
      for (size_t i = 0; i < _particles.size(); ++i) {
        #pragma omp ordered
        _particles[i].run(step_end, pick_generator(i, thread_idx));
      }
    }
  }

  void state(std::vector<float_t>& end_state) const {
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(_index_y, end_state.begin() + i * _index_y.size());
    }
  }

  void state_full(std::vector<float_t>& end_state) const {
    const size_t n = n_state_full();
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(end_state.begin() + i * n);
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
  // contents of the particle state (uses the update() and swap()
  // methods on particles).
  void reorder(const std::vector<size_t>& index) {
    for (size_t i = 0; i < _particles.size(); ++i) {
      size_t j = index[i];
      _particles[i].update(_particles[j]);
    }
    for (auto& p : _particles) {
      p.swap();
    }
  }

  size_t n_particles() const {
    return _particles.size();
  }
  size_t n_state() const {
    return _index_y.size();
  }
  size_t n_state_full() const {
    return _particles.front().size();
  }
  size_t step() const {
    return _particles.front().step();
  }

private:
  const std::vector<size_t> _index_y;
  const size_t _n_threads;
  dust::pRNG<float_t, int_t> _rng;
  std::vector<Particle<T>> _particles;

  // This scheme means that if we have the same number of generators
  // and threads then work for thread i occurs on generator i.
  //
  // If we have more generators than threads then each thread uses m =
  // n_generators / n_threads (so if we had 8 generators and 2
  // threads, each thread would use 4 generators).  The first thread
  // will cycle alternate through the first m generators, the second
  // thread will cycle through the second m generators, etc.
  //
  // So for 8 generators, 2 threads:
  //
  //   thread 1: 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0
  //   thread 2: 4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7 4
  //
  // For 8 generators, 4 threads:
  //
  //   thread 1: 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
  //   thread 2: 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2
  //   thread 3: 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4
  //   thread 4: 6 7 6 7 6 7 6 7 6 7 6 7 6 7 6 7 6 7 6 7 6 7 6 7 6
  dust::RNG<float_t, int_t>& pick_generator(const size_t i,
                                            const size_t thread_idx) {
    const size_t m = _rng.size() / _n_threads;
    return _rng(i % m + thread_idx * m);
  }
};

#endif
