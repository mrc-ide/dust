#include <dust/rng.hpp>

// Bindings for using this in C code
double tmp(int seed) {
  dust::RNG r(1, seed);
  double ret = r.rnorm(0, 0, 1);
  return ret;
}


// // https://stackoverflow.com/a/24237867
// extern "C" RNG* C_RNG_alloc(const size_t n_threads, const uint64_t seed) noexcept {
//    try {
//      return static_cast<RNG*>(new RNG(n_threads, seed));
//    }
//    catch (...) {
//      return nullptr;
//    }
//  }

// extern "C" void C_RNG_free(RNG *obj) noexcept {
//    try {
//      RNG* typed_obj = static_cast<RNG*>(obj);
//      delete typed_obj;
//    }
//    catch (...) {
//        // Ignore
//    }
// }

// extern "C" double C_runif(RNG* r, const size_t thread_idx) {
//     return r->runif(thread_idx);
// }

// extern "C" int C_rbinom(RNG* r, const size_t thread_idx, const double p, const int n) {
//     return static_cast<int>(r->rbinom(thread_idx, p, n));
// }

// extern "C" int C_rpois(RNG* r, const size_t thread_idx, const double lambda) {
//     return static_cast<int>(r->rpois(thread_idx, lambda));
// }

// extern "C" double C_rnorm(RNG* r, const size_t thread_idx, const double mu, const double sd) {
//     return r->rnorm(thread_idx, mu, sd);
// }
