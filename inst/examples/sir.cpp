class sir {
public:
  typedef float real_t;
  struct init_t {
    real_t S0;
    real_t I0;
    real_t R0;
    real_t beta;
    real_t gamma;
    real_t dt;
  };

  sir(const init_t& data) : data_(data) {
  }

  size_t size() const {
    return 3;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> ret = {data_.S0, data_.I0, data_.R0};
    return ret;
  }

#ifdef __NVCC__
  __device__
#endif
  void update(size_t step, const std::vector<real_t>& state,
              dust::rng_state_t<real_t>& rng_state,
              std::vector<real_t>& state_next) {
    real_t S = state[0];
    real_t I = state[1];
    real_t R = state[2];
    real_t N = S + I + R;

    real_t p_SI = 1 - std::exp(-(data_.beta) * I / N);
    real_t p_IR = 1 - std::exp(-(data_.gamma));
    real_t n_IR = dust::distr::rbinom(rng_state, I, p_IR * data_.dt);
    real_t n_SI = dust::distr::rbinom(rng_state, S, p_SI * data_.dt);

    state_next[0] = S - n_SI;
    state_next[1] = I + n_SI - n_IR;
    state_next[2] = R + n_IR;
  }

private:
  init_t data_;
};

#include <cpp11/list.hpp>
template <>
sir::init_t dust_data<sir>(cpp11::list data) {
  // Initial state values
  real_t I0 = 10.0;
  real_t S0 = 1000.0;
  real_t R0 = 0.0;

  // Default rates
  real_t beta = 0.2;
  real_t gamma = 0.1;

  // Time scaling
  real_t dt = 0.25;

  // Accept beta and gamma as optional elements
  SEXP r_beta = data["beta"];
  if (r_beta != R_NilValue) {
    beta = cpp11::as_cpp<real_t>(r_beta);
  }
  SEXP r_gamma = data["gamma"];
  if (r_gamma != R_NilValue) {
    gamma = cpp11::as_cpp<real_t>(r_gamma);
  }

  return sir::init_t{S0, I0, R0, beta, gamma, dt};
}

template <>
cpp11::sexp dust_info<sir>(const sir::init_t& data) {
  using namespace cpp11::literals;
  // Information about state order
  cpp11::writable::strings vars({"S", "I", "R"});
  // Information about parameter values
  cpp11::list pars = cpp11::writable::list({"beta"_nm = data.beta,
                                            "gamma"_nm = data.gamma});
  return cpp11::writable::list({"vars"_nm = vars, "pars"_nm = pars});
}
