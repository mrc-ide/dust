class walk {
public:
  typedef double real_t;
  typedef no_data data_t;
  struct init_t {
    real_t sd;
  };
  walk(const init_t& pars) : pars_(pars) {
  }
  size_t size() const {
    return 1;
  }
  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> ret = {0};
    return ret;
  }
  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state,
              real_t * state_next) {
    real_t mean = state[0];
    state_next[0] = dust::distr::rnorm(rng_state, mean, pars_.sd);
  }

private:
  init_t pars_;
};

#include <cpp11/list.hpp>
template <>
walk::init_t dust_pars<walk>(cpp11::list pars) {
  walk::real_t sd = cpp11::as_cpp<walk::real_t>(pars["sd"]);
  return walk::init_t{sd};
}
