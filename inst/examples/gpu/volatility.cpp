class volatility {
public:
  typedef int int_t;
  typedef double real_t;
  struct init_t {
    real_t alpha;
    real_t sigma;
    real_t x0;
  };
  volatility(const init_t& data): data_(data) {
  }
  size_t size() {
    return 1;
  }
  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(1);
    state[0] = data_.x0;
    return state;
  }
  #ifdef __NVCC__
  __device__
  #endif
  void update(size_t step, const dust::state_t<real_t>& state,
              dust::rng_state_t<real_t>& rng_state,
              dust::state_t<real_t>& state_next) {
    const real_t x = state.state_ptr[0];
    const real_t x = state[0];
    state_next.state_ptr[0] = data_.alpha * x +
      data_.sigma * dust::distr::rnorm(rng_state, 0, 1);
  }

private:
  init_t data_;
};

#include <cpp11/list.hpp>
template <>
volatility::init_t dust_data<volatility>(cpp11::list data) {
  volatility::real_t x0 = 0;
  volatility::real_t alpha = 0.91;
  volatility::real_t sigma = 1;

  SEXP r_alpha = data["alpha"];
  if (r_alpha != R_NilValue) {
    alpha = cpp11::as_cpp<volatility::real_t>(r_alpha);
  }
  SEXP r_sigma = data["sigma"];
  if (r_sigma != R_NilValue) {
    sigma = cpp11::as_cpp<volatility::real_t>(r_sigma);
  }

  return volatility::init_t{alpha, sigma, x0};
}
