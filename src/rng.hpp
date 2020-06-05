#include <vector>

#include <trng/lcg64_shift.hpp>

class RNG {
    public:
        RNG(const size_t n_threads);

        void jump(const size_t thread_idx, const size_t rand_per_it);
        trng::lcg64_shift get_generator(const size_t thread_idx);

        double runif(const size_t thread_idx);
        double rnorm(const size_t thread_idx, double mu, double sd);
        int rbinom(const size_t thread_idx, double p, int n);
        
        template <class T = int> T rbinom_tf(const size_t thread_idx, double p, int n);

    private:
        std::vector<trng::lcg64_shift> _rng_array;
};
