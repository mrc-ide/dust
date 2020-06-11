#ifndef DUST_RNG_HPP
#define DUST_RNG_HPP

#include <random>
#include <dust/xoshiro.hpp>

namespace dust {

class RNG {
public:
  RNG(const size_t n_threads, const uint64_t seed);

  double runif(const size_t thread_idx);
  double rnorm(const size_t thread_idx, double mu, double sd);
  template <class T = int> T rbinom(const size_t thread_idx, double p, int n);
  template <class T = int> T rpois(const size_t thread_idx, double lambda);

private:
  std::vector<Xoshiro> _generators;
};

// Constructor builds vector of XOSHIRO providing non-overlapping
// streams of random numbers
inline RNG::RNG(const size_t n_threads, const uint64_t seed) {
    Xoshiro rng(seed);
    for (size_t i = 0; i < n_threads; i++) {
        _generators.push_back(rng);
        rng.jump();
    }
}

// Standard C++11 distributions

// Should just do rng()/max
inline double RNG::runif(const size_t thread_idx) {
  static std::uniform_real_distribution<double> unif_dist(0, 1);
  return(unif_dist(_generators[thread_idx]));
}

inline double RNG::rnorm(const size_t thread_idx, const double mu, const double sd) {
  std::normal_distribution<double> norm(mu, sd);
  return(norm(_generators[thread_idx]));
}

class RNG2 {
public:
  RNG2(dust::Xoshiro generator) : _generator(generator) {}

  double runif(double min, double max) {
    static std::uniform_real_distribution<double> unif_dist(min, max);
    return(unif_dist(_generator));
  }
  double rnorm(double mu, double sd) {
    std::normal_distribution<double> norm(mu, sd);
    return(norm(_generator));
  }
  template <class T = int> T rbinom(double p, int n);
  template <class T = int> T rpois(double lambda);

private:
  dust::Xoshiro _generator;
};


class pRNG {
public:
  pRNG(const size_t n, const uint64_t seed) {
    dust::Xoshiro rng(seed);
    for (size_t i = 0; i < n; ++i) {
      _rngs.push_back(RNG2(rng));
      rng.jump();
    }
  }

  RNG2& operator()(size_t index) {
    return _rngs[index];
  }

  size_t size() const {
    return _rngs.size();
  }

private:
  std::vector<RNG2> _rngs;
};

}

#endif
