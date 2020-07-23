class sirs {
public:
  typedef double real_t;

  struct init_t {
    real_t S0;
    real_t I0;
    real_t R0;
    real_t alpha;
    real_t beta;
    real_t gamma;
    real_t dt;
  };

  sirs(const init_t& data): internal(data) {
  }

  size_t size() {
    return 3;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(3);
    state[0] = internal.S0;
    state[1] = internal.I0;
    state[2] = internal.R0;
    return state;
  }

#ifdef __NVCC__
  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state,
              real_t * state_next) {
#else
  void update(size_t step, const std::vector<real_t>& state,
              dust::rng_state_t<real_t>& rng_state,
              std::vector<real_t>& state_next) {
#endif
    real_t S = state[0];
    real_t I = state[1];
    real_t R = state[2];
    real_t N = S + I + R;

    real_t p_SI = 1 - exp(- internal.beta * I / (real_t) N);
    real_t p_IR = 1 - exp(-(internal.gamma));
    real_t p_RS = 1 - exp(- internal.alpha);

    real_t n_SI = dust::distr::rbinom(rng_state, S, p_SI * internal.dt);
    real_t n_IR = dust::distr::rbinom(rng_state, I, p_IR * internal.dt);
    real_t n_RS = dust::distr::rbinom(rng_state, R, p_RS * internal.dt);

    state_next[0] = S - n_SI + n_RS;
    state_next[1] = I + n_SI - n_IR;
    state_next[2] = R + n_IR - n_RS;
  }
private:
  init_t internal;
};

#include <cpp11/list.hpp>
template <>
sirs::init_t dust_data<sirs>(cpp11::list data) {
  // Initial state values
  sirs::real_t I0 = 10.0;
  sirs::real_t S0 = 1000.0;
  sirs::real_t R0 = 0.0;

  // Default rates
  sirs::real_t alpha = 0.1;
  sirs::real_t beta = 0.2;
  sirs::real_t gamma = 0.1;

  // Time scaling
  sirs::real_t dt = 1.0;

  // Accept beta and gamma as optional elements
  SEXP r_beta = data["beta"];
  if (r_beta != R_NilValue) {
    beta = cpp11::as_cpp<sirs::real_t>(r_beta);
  }
  SEXP r_gamma = data["gamma"];
  if (r_gamma != R_NilValue) {
    gamma = cpp11::as_cpp<sirs::real_t>(r_gamma);
  }

  return sirs::init_t{S0, I0, R0, alpha, beta, gamma, dt};
}
