#include <random>

#include "rng.hpp"

static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

uint64_t splitmix64(uint64_t seed) {
    uint64_t z = (seed += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}

XOSHIRO::XOSHIRO(uint64_t seed) {
    this->set_seed(seed);
}

void XOSHIRO::set_seed(uint64_t seed) {
    // normal brain: for i in 1:4
    // advanced brain: -funroll-loops
    // galaxy brain:
    _state[0] = splitmix64(seed);
    _state[1] = splitmix64(_state[0]);
    _state[2] = splitmix64(_state[1]);
    _state[3] = splitmix64(_state[2]);
}

uint64_t XOSHIRO::operator()() {
    const uint64_t result = rotl(_state[1] * 5, 7) * 9;

	const uint64_t t = _state[1] << 17;

	_state[2] ^= _state[0];
	_state[3] ^= _state[1];
	_state[1] ^= _state[2];
	_state[0] ^= _state[3];

	_state[2] ^= t;

	_state[3] = rotl(_state[3], 45);

	return result;
}

/* This is the jump function for the generator. It is equivalent
   to 2^128 calls to next(); it can be used to generate 2^128
   non-overlapping subsequences for parallel computations. */
void XOSHIRO::jump() {
	static const uint64_t JUMP[] = \
        { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

	uint64_t s0 = 0;
	uint64_t s1 = 0;
	uint64_t s2 = 0;
	uint64_t s3 = 0;
	for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for(int b = 0; b < 64; b++) {
			if (JUMP[i] & UINT64_C(1) << b) {
				s0 ^= _state[0];
				s1 ^= _state[1];
				s2 ^= _state[2];
				s3 ^= _state[3];
			}
			this->gen_rand();	
		}
		
	_state[0] = s0;
	_state[1] = s1;
	_state[2] = s2;
	_state[3] = s3;
}

/* This is the long-jump function for the generator. It is equivalent to
   2^192 calls to next(); it can be used to generate 2^64 starting points,
   from each of which jump() will generate 2^64 non-overlapping
   subsequences for parallel distributed computations. */
void XOSHIRO::long_jump() {
    static const uint64_t LONG_JUMP[] = \
        { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };

	uint64_t s0 = 0;
	uint64_t s1 = 0;
	uint64_t s2 = 0;
	uint64_t s3 = 0;
	for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
		for(int b = 0; b < 64; b++) {
			if (LONG_JUMP[i] & UINT64_C(1) << b) {
				s0 ^= _state[0];
				s1 ^= _state[1];
				s2 ^= _state[2];
				s3 ^= _state[3];
			}
			next();	
		}
		
	s[0] = s0;
	s[1] = s1;
	s[2] = s2;
	s[3] = s3;
}


RNG::RNG(const size_t n_threads, const uint64_t seed) {
    XOSHIRO rng(seed);
    for (size_t i = 0; i < n_threads; i++) {
        _generators.push_back(rng);
        rng.jump();
    }
}

// Should just do rng()/max
double RNG::runif(const size_t thread_idx) {
    static std::uniform_int_distribution<int> unif_dist(0, 1);
    return(unif_dist(_generators[thread_idx]));
}

double RNG::rnorm(const size_t thread_idx, const double mu, const double sd) {
    std::normal_distribution<double> norm(mu, sd);
    return(norm(_generators[thread_idx]));
}

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

extern "C" double C_runif(RNG* r, const size_t thread_idx) {
    return r->runif(thread_idx);
}

extern "C" int C_rbinom(RNG* r, const size_t thread_idx, const double p, const int n) {
    return static_cast<int>(r->rbinom(thread_idx, p, n));
}

extern "C" double C_rnorm(RNG* r, const size_t thread_idx, const double mu, const double sd) {
    return r->rnorm(thread_idx, mu, sd);
}
