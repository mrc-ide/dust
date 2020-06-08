#ifdef _OPENMP
#include <omp.h>
#endif

#include "dust.hpp"

Particle::Particle(const size_t n_y) :
  _step(0) {
    _data = ;
    _y.reserve(n_y);
    _y_swap.reserve(n_y);
    _index_y = ;
}

    dest[i] = obj->y[obj->index_y[i]];

void Particle::run(const size_t step_end, RNG& rng, const size_t thread_idx) {
  while (_step < step_end) {
    this->update(this->_data, this->_step, this->_y, rng, thread_idx, this->_y_swap);
    _step++;
    std::vector<double> y_tmp = _y;
    _y = _y_swap;
    _y_swap = y_tmp;
  }
}

std::vector<double> Particle::state() const {
  std::vector<double> read_state;
  read_state.reserve(_y.size());
  for (size_t i = 0; i < index_y.size(); i++) {
    read_state[i] = _y[_index_y[i]];
  }
  return read_state;
}

Dust::Dust(const size_t n_y, const size_t n_particles, const size_t n_threads, 
           const double seed) 
  : _n_particles(n_particles), _n_threads(n_threads), 
    _particles(n_particles, n_y), _rng(n_threads, seed) {}

void Dust::run(const size_t step_end) {
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

std::vector<std::vector<double>> Dust::state() const {
  std::vector<std::vector<double>> return_state(_particles.size());
  #pragma omp parallel for schedule(static) num_threads(obj->n_threads)
  for (i = 0; i < _particles.size(); ++i) {
    return_state[i] = _particles[i].state()
  }
  return(return_state);
}

void Dust::shuffle() {
  
}