#include <vector>
#include <climits>

#include <trng/lcg64_shift.hpp>

#define XOSHIRO_WIDTH 4

class XOSHIRO {
    public:
        // Definitions to be used as URNG in C++11
        typedef size_t result_type;
        static size_t min() { return 0; }
        static size_t max() { return ULLONG_MAX; }
        uint64_t operator()(); // generate random number

        // Constructor
        XOSHIRO(uint64_t seed);
        
        // Change internal state
        void set_seed(uint64_t seed);
        void jump();
        void long_jump();

    private:
        uint64_t _state[XOSHIRO_WIDTH];
};
class RNG {
    public:
        RNG(const size_t n_threads, const uint64_t seed);

        double runif(const size_t thread_idx);
        double rnorm(const size_t thread_idx, double mu, double sd);
        template <class T = int> T rbinom(const size_t thread_idx, double p, int n);
        template <class T = int> T rpois(const size_t thread_idx, double lambda);

    private:
        std::vector<XOSHIRO> _generators;
};
