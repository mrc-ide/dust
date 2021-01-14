#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

class parallel {
public:
  typedef double real_t;
  typedef no_data data_t;
  struct init_t {
    double sd;
  };
  parallel(const init_t& pars) : pars_(pars) {
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
    state_next[0] = dust::distr::rnorm(rng_state, mean, pars_.sd);
#ifdef _OPENMP
    state_next[1] = (double) omp_get_thread_num();
#else
    state_next[1] = -1;
#endif
  }

private:
  init_t pars_;
};

#include <cpp11/list.hpp>
template <>
parallel::init_t dust_pars<parallel>(cpp11::list pars) {
  parallel::real_t sd = cpp11::as_cpp<parallel::real_t>(pars["sd"]);
  return parallel::init_t{sd};
}
