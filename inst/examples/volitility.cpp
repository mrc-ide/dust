class volitility {
public:
  typedef int int_t;
  typedef double real_t;
  struct init_t {
    real_t alpha;
    real_t sigma;
    real_t x0;
  };
  volitility(const init_t& data): data_(data) {
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
  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state, real_t * state_next) {
    const real_t x = state[0];
    state_next[0] = data_.alpha * x +
      data_.sigma * dust::distr::rnorm(rng_state, 0, 1);
  }

private:
  init_t data_;
};

#include <cpp11/list.hpp>
template <>
volitility::init_t dust_data<volitility>(cpp11::list data) {
  volitility::real_t x0 = 0;
  volitility::real_t alpha = 0.91;
  volitility::real_t sigma = 1;

  SEXP r_alpha = data["alpha"];
  if (r_alpha != R_NilValue) {
    alpha = cpp11::as_cpp<volitility::real_t>(r_alpha);
  }
  SEXP r_sigma = data["sigma"];
  if (r_sigma != R_NilValue) {
    sigma = cpp11::as_cpp<volitility::real_t>(r_sigma);
  }

  return volitility::init_t{alpha, sigma, x0};
}
