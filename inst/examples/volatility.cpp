class volatility {
public:
  typedef double real_t;
  typedef dust::no_data data_t;
  typedef dust::no_internal internal_t;

  struct shared_t {
    real_t alpha;
    real_t sigma;
    real_t x0;
  };

  volatility(const dust::pars_t<volatility>& pars) : shared(pars.shared) {
  }

  size_t size() {
    return 1;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(1);
    state[0] = shared->x0;
    return state;
  }

  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state, real_t * state_next) {
    const real_t x = state[0];
    state_next[0] = shared->alpha * x +
      shared->sigma * dust::distr::rnorm(rng_state, 0, 1);
  }

private:
  dust::shared_ptr<volatility> shared;
};

#include <cpp11/list.hpp>
template <>
dust::pars_t<volatility> dust_pars<volatility>(cpp11::list pars) {
  volatility::real_t x0 = 0;
  volatility::real_t alpha = 0.91;
  volatility::real_t sigma = 1;

  SEXP r_alpha = pars["alpha"];
  if (r_alpha != R_NilValue) {
    alpha = cpp11::as_cpp<volatility::real_t>(r_alpha);
  }
  SEXP r_sigma = pars["sigma"];
  if (r_sigma != R_NilValue) {
    sigma = cpp11::as_cpp<volatility::real_t>(r_sigma);
  }

  volatility::shared_t shared{alpha, sigma, x0};
  return dust::pars_t<volatility>(shared);
}
