#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

class parallel {
public:
  using real_type = double;
  using data_type = dust::no_data;
  using internal_type = dust::no_internal;
  using rng_state_type = dust::random::generator<real_type>;
  struct shared_type {
    real_type sd;
  };

  parallel(const dust::pars_type<parallel>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 2;
  }

  std::vector<real_type> initial(size_t time, rng_state_type& rng_state) {
#ifdef _OPENMP
    static bool has_openmp = true;
#else
    static bool has_openmp = false;
#endif
    std::vector<real_type> ret = {0, (real_type) has_openmp};
    return ret;
  }

  void update(size_t time, const real_type * state, rng_state_type& rng_state,
              real_type * state_next) {
    real_type mean = state[0];
    state_next[0] = dust::random::normal<real_type>(rng_state, mean, shared->sd);
#ifdef _OPENMP
    state_next[1] = (real_type) omp_get_thread_num();
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
  parallel::real_type sd = cpp11::as_cpp<parallel::real_type>(pars["sd"]);
  parallel::shared_type shared{sd};
  return dust::pars_type<parallel>(shared);
}
}
