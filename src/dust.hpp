#include <R.h>
#include <Rinternals.h>

#include "rng.hpp"

class Particle {
  public:
    Particle(const size_t n_y);
    
    void run(const size_t step_end, RNG& rng, const size_t thread_idx);
    std::vector<double> state() const;

  private:
    void * _data;
    void * _update;
    size_t _step;
    
    std::vector<double> _y;
    std::vector<double> _y_swap;
    std::vector<size_t> _index_y;
}

class Dust {
  public:
    Dust(const size_t n_y, const size_t n_particles, const size_t n_threads, 
         const double seed);
    
    void run(const size_t step_end);
    std::vector<std::vector<double>> state() const;
    void shuffle();

    size_t n_particles() const { return _particles.size() };

  private:
    size_t _n_threads;
    std::vector<Particle> _particles;
    RNG _rng; 
}

