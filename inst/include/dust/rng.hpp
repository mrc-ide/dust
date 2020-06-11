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

}

#endif
