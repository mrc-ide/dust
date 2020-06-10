#include <R.h>
#include <Rinternals.h>

#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "rng.hpp"

template <typename T>
class Particle {
  public:
    Particle(SEXP data) : _internal_state(data),
                          _y(_internal_state.size()),
                          _y_swap(_internal_state.size())
    {}

    std::vector<double> run(const size_t step_end, std::vector<size_t>& _index_y, RNG& rng, const size_t thread_idx) {
      while (_step < step_end) {
        _internal_state.update(_step, _y, rng, thread_idx, _y_swap);
        _step++;
        std::swap(_y, _y_swap);
      }
      return this->state();
    }

    void state(const std::vector<size_t>& index_y,
               std::vector<double>::iterator& end_state) const {
      for (size_t i = 0; i < index_y.size(); i++) {
        *(end_state + i) = _y[index_y[i]];
      }
    }

    std::vector<double> state() const { return _y; }

  private:
    T _internal_state;
    size_t _step;

    std::vector<double> _y;
    std::vector<double> _y_swap;
};

template <typename T>
class Dust {
  public:
    Dust(SEXP data, const std::vector<size_t> index_y, const size_t n_threads,
         const double seed, const size_t n_particles)
    : _index_y(index_y), _n_threads(n_threads), _rng(n_threads, seed),
      _particles(n_particles, data)
    {}

    void reset(SEXP data) {
      size_t n_particles = _particles.size();
      _particles.clear();
      for (size_t i = 0; i < n_particles; i++) {
        _particles.push_back(T(data));
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
            _particles[i].run(step_end, _index_y, _rng, thread_idx);
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

    void shuffle() {}

    size_t n_particles() const { return _particles.size(); }

  private:
    const size_t _n_threads;
    const std::vector<size_t> _index_y;
    RNG _rng;
    std::vector<Particle<T>> _particles;
};
