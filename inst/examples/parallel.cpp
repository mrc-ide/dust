#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

class parallel {
public:
  typedef double real_t;
  typedef dust::no_data data_t;
  typedef dust::no_internal internal_t;
  typedef dust::random::xoshiro256starstar_state rng_state_type;
  struct shared_t {
    real_t sd;
  };

  parallel(const dust::pars_type<parallel>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 2;
  }

  std::vector<real_t> initial(size_t step) {
#ifdef _OPENMP
    static bool has_openmp = true;
#else
    static bool has_openmp = false;
#endif
    std::vector<real_t> ret = {0, (real_t) has_openmp};
    return ret;
  }

  void update(size_t step, const real_t * state, rng_state_type& rng_state,
              real_t * state_next) {
    real_t mean = state[0];
    state_next[0] = dust::random::normal<real_t>(rng_state, mean, shared->sd);
#ifdef _OPENMP
    state_next[1] = (real_t) omp_get_thread_num();
#else
    state_next[1] = -1;
#endif
  }

private:
  dust::shared_ptr<parallel> shared;
};

namespace dust {
template <>
dust::pars_type<parallel> dust_pars<parallel>(cpp11::list pars) {
  parallel::real_t sd = cpp11::as_cpp<parallel::real_t>(pars["sd"]);
  parallel::shared_t shared{sd};
  return dust::pars_type<parallel>(shared);
}
}
