class sir {
public:
  typedef double real_t;
  typedef dust::no_data data_t;

  struct init_t {
    double S0;
    double I0;
    double R0;
    double beta;
    double gamma;
    double dt;
    size_t freq;
  };

  sir(const init_t& pars) : pars_(pars) {
  }

  size_t size() const {
    return 5;
  }

  std::vector<double> initial(size_t step) {
    std::vector<double> ret = {pars_.S0, pars_.I0, pars_.R0, 0, 0};
    return ret;
  }

  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state,
              real_t * state_next) {
    double S = state[0];
    double I = state[1];
    double R = state[2];
    double cumulative_incidence = state[3];

    double N = S + I + R;

    double p_SI = 1 - std::exp(-(pars_.beta) * I / N);
    double p_IR = 1 - std::exp(-(pars_.gamma));
    double n_IR = dust::distr::rbinom(rng_state, I, p_IR * pars_.dt);
    double n_SI = dust::distr::rbinom(rng_state, S, p_SI * pars_.dt);

    state_next[0] = S - n_SI;
    state_next[1] = I + n_SI - n_IR;
    state_next[2] = R + n_IR;
    state_next[3] = cumulative_incidence + n_SI;
    // Little trick here to compute daily incidence by accumulating
    // incidence from the first day.
    state_next[4] = (step % pars_.freq == 0) ? n_SI : state[4] + n_SI;
  }

private:
  init_t pars_;
};

#include <cpp11/list.hpp>

// Helper function for accepting values with defaults
inline double with_default(double default_value, cpp11::sexp value) {
  return value == R_NilValue ? default_value : cpp11::as_cpp<double>(value);
}

template <>
sir::init_t dust_pars<sir>(cpp11::list pars) {
  // Initial state values
  double I0 = 10.0;
  double S0 = 1000.0;
  double R0 = 0.0;

  // Rates, which can be set based on the provided pars
  // [[dust::param(beta, required = FALSE)]]
  double beta = with_default(0.2, pars["beta"]);
  // [[dust::param(gamma, required = FALSE)]]
  double gamma = with_default(0.1, pars["gamma"]);

  // Time scaling
  size_t freq = 4;
  double dt = 1.0 / static_cast<double>(freq);

  return sir::init_t{S0, I0, R0, beta, gamma, dt, freq};
}

template <>
cpp11::sexp dust_info<sir>(const sir::init_t& pars) {
  using namespace cpp11::literals;
  // Information about state order
  cpp11::writable::strings vars({"S", "I", "R", "inc"});
  // Information about parameter values
  cpp11::list p = cpp11::writable::list({"beta"_nm = pars.beta,
                                         "gamma"_nm = pars.gamma});
  return cpp11::writable::list({"vars"_nm = vars, "pars"_nm = p});
}
