#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

class parallel {
public:
  typedef double real_t;
  typedef dust::no_data data_t;
  typedef dust::no_internal internal_t;
  struct shared_t {
    double sd;
  };

  parallel(const dust::pars_t<parallel>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 2;
  }

  std::vector<double> initial(size_t step) {
#ifdef _OPENMP
    static bool has_openmp = true;
#else
    static bool has_openmp = false;
#endif
    std::vector<double> ret = {0, (double) has_openmp};
    return ret;
  }

  void update(size_t step, const double * state,
              dust::rng_state_t<real_t>& rng_state,
              double * state_next) {
    double mean = state[0];
    state_next[0] = dust::distr::rnorm(rng_state, mean, shared->sd);
#ifdef _OPENMP
    state_next[1] = (double) omp_get_thread_num();
#else
    state_next[1] = -1;
#endif
  }

private:
  dust::shared_ptr<parallel> shared;
};

#include <cpp11/list.hpp>
template <>
dust::pars_t<parallel> dust_pars<parallel>(cpp11::list pars) {
  parallel::real_t sd = cpp11::as_cpp<parallel::real_t>(pars["sd"]);
  parallel::shared_t shared{sd};
  return dust::pars_t<parallel>(shared);
}
