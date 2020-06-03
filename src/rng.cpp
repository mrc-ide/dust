#include "rng.hpp"

#include <trng/binomial_dist.hpp>

RNG::RNG(const size_t n_threads):
 _rng_array(n_threads, trng::lcg64_shift) {}

// https://stackoverflow.com/a/24237867
extern "C" RNG* C_RNG_alloc(const size_t n_threads) noexcept {
   try {
     return static_cast<RNG*>(new RNG(n_threads));
   }
   catch (...) {
     return nullptr;
   }
 }

extern "C" void C_RNG_free(RNG *obj) noexcept {
   try {
     RNG* typed_obj = static_cast<RNG*>(obj);
     delete typed_obj;
   }
   catch (...) {
       // Ignore
   }
 }

trng::lcg64_shift RNG::get_generator(const size_t thread_idx) {
    if (thread_idx < _rng_array.size()) {
        return(_rng_array[thread_idx]);
    } else {
        throw std::runtime_error("Thread idx out of range");
    }
}

void RNG::jump(const size_t thread_idx, const size_t rand_per_it) {
    _rng_array[thread_idx].jump(thread_idx * rand_per_it);
}

extern "C" void C_jump(RNG* r, const size_t thread_idx, const size_t rand_per_it) {
    r->jump(thread_idx, rand_per_it);
}

int RNG::rbinom(const size_t thread_idx, const double p, const int n) {
    trng::binomial_dist(p, n);
    return(_rng_array[thread_idx]);
}

extern "C" int C_rbinom(RNG* r, const size_t thread_idx, const double p, const int n) { 
    return r->rbinom(thread_idx, p, n); 
}