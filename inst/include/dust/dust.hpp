#ifndef DUST_DUST_HPP
#define DUST_DUST_HPP

#include <dust/rng.hpp>

#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace dust {

template <typename T>
class Particle {
public:
  typedef typename T::init_t init_t;
  Particle(init_t data, size_t step) :
    _model(data),
    _step(step),
    _y(_model.initial(_step)),
    _y_swap(_model.size()) {
  }

  void run(const size_t step_end, RNG& rng, const size_t thread_idx) {
    while (_step < step_end) {
      _model.update(_step, _y, rng, thread_idx, _y_swap);
      _step++;
      std::swap(_y, _y_swap);
    }
  }

  void state(const std::vector<size_t>& index_y,
             std::vector<double>::iterator end_state) const {
    for (size_t i = 0; i < index_y.size(); ++i) {
      *(end_state + i) = _y[index_y[i]];
    }
  }

  void state(std::vector<double>::iterator end_state) const {
    for (size_t i = 0; i < _y.size(); ++i) {
      *(end_state + i) = _y[i];
    }
  }

  size_t size() const {
    return _y.size();
  }

private:
  T _model;
  size_t _step;

  std::vector<double> _y;
  std::vector<double> _y_swap;
};

template <typename T>
class Dust {
public:
  typedef typename T::init_t init_t;
  Dust(init_t data, size_t step,
       const std::vector<size_t> index_y, const size_t n_threads,
       const double seed, const size_t n_particles) :
    _index_y(index_y),
    _n_threads(n_threads),
    _rng(n_threads, seed) {
    for (size_t i = 0; i < n_particles; ++i) {
      _particles.push_back(Particle<T>(data, step));
    }
  }

  void reset(init_t data, size_t step) {
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
        {
          _particles[i].run(step_end, _rng, thread_idx);
        }
      }
    }
  }

  void state(std::vector<double>& end_state) const {
#pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(_index_y, end_state.begin() + i * _index_y.size());
    }
  }

  void state_full(std::vector<double>& end_state) const {
    const size_t n = n_state_full();
#pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(end_state.begin() + i * n);
    }
  }

  void shuffle() {}

  size_t n_particles() const { return _particles.size(); }
  size_t n_state() const { return _index_y.size(); }
  size_t n_state_full() const { return _particles.front().size(); }

private:
  const std::vector<size_t> _index_y;
  const size_t _n_threads;
  RNG _rng;
  std::vector<Particle<T>> _particles;
};

}

#endif
