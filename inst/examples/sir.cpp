class sir {
public:
  typedef double real_t;
  struct data_t {
    real_t incidence;
  };
  typedef dust::no_internal internal_t;

  struct shared_t {
    real_t S0;
    real_t I0;
    real_t R0;
    real_t beta;
    real_t gamma;
    real_t dt;
    size_t freq;
    // Observation parameters
    real_t exp_noise;
  };

  sir(const dust::pars_t<sir>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 5;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> ret = {shared->S0, shared->I0, shared->R0, 0, 0};
    return ret;
  }

  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state,
              real_t * state_next) {
    real_t S = state[0];
    real_t I = state[1];
    real_t R = state[2];
    real_t cumulative_incidence = state[3];

    real_t N = S + I + R;

    real_t p_SI = 1 - std::exp(-(shared->beta) * I / N);
    real_t p_IR = 1 - std::exp(-(shared->gamma));
    real_t n_IR = dust::distr::rbinom(rng_state, I, p_IR * shared->dt);
    real_t n_SI = dust::distr::rbinom(rng_state, S, p_SI * shared->dt);

    state_next[0] = S - n_SI;
    state_next[1] = I + n_SI - n_IR;
    state_next[2] = R + n_IR;
    state_next[3] = cumulative_incidence + n_SI;
    // Little trick here to compute daily incidence by accumulating
    // incidence from the first day.
    state_next[4] = (step % shared->freq == 0) ? n_SI : state[4] + n_SI;
  }

  real_t compare_data(const real_t * state, const data_t& data,
                      dust::rng_state_t<real_t>& rng_state) {
    const real_t incidence_modelled = state[4];
    const real_t incidence_observed = data.incidence;
    const real_t lambda = incidence_modelled +
      dust::distr::rexp(rng_state, shared->exp_noise);
    return dust::dpois(incidence_observed, lambda, true);
  }

private:
  dust::shared_ptr<sir> shared;
};

// This is required to activate gpu support. We can possibly do this
// automatically based on the existance of the template below though.
template <>
struct dust::has_gpu_support<sir> : std::true_type {};

template <>
void update_device<sir>(size_t step,
                        const dust::interleaved<sir::real_t> state,
                        dust::interleaved<int> internal_int,
                        dust::interleaved<sir::real_t> internal_real,
                        dust::shared_ptr<sir> shared,
                        dust::rng_state_t<sir::real_t>& rng_state,
                        dust::interleaved<sir::real_t> state_next) {
  typedef sir::real_t real_t;
  real_t S = state[0];
  real_t I = state[1];
  real_t R = state[2];
  real_t cumulative_incidence = state[3];

  real_t N = S + I + R;

  real_t p_SI = 1 - std::exp(-(shared->beta) * I / N);
  real_t p_IR = 1 - std::exp(-(shared->gamma));
  real_t n_IR = dust::distr::rbinom(rng_state, I, p_IR * shared->dt);
  real_t n_SI = dust::distr::rbinom(rng_state, S, p_SI * shared->dt);

  state_next[0] = S - n_SI;
  state_next[1] = I + n_SI - n_IR;
  state_next[2] = R + n_IR;
  state_next[3] = cumulative_incidence + n_SI;
  // Little trick here to compute daily incidence by accumulating
  // incidence from the first day.
  state_next[4] = (step % shared->freq == 0) ? n_SI : state[4] + n_SI;
}

#include <cpp11/list.hpp>

// Helper function for accepting values with defaults
inline double with_default(double default_value, cpp11::sexp value) {
  return value == R_NilValue ? default_value : cpp11::as_cpp<double>(value);
}

template <>
dust::pars_t<sir> dust_pars<sir>(cpp11::list pars) {
  typedef sir::real_t real_t;
  // Initial state values
  real_t I0 = 10.0;
  real_t S0 = 1000.0;
  real_t R0 = 0.0;

  // Rates, which can be set based on the provided pars
  // [[dust::param(beta, required = FALSE)]]
  real_t beta = with_default(0.2, pars["beta"]);
  // [[dust::param(gamma, required = FALSE)]]
  real_t gamma = with_default(0.1, pars["gamma"]);

  // Time scaling
  size_t freq = 4;
  real_t dt = 1.0 / static_cast<real_t>(freq);

  // Compare function
  // [[dust::param(exp_noise, required = FALSE)]]
  real_t exp_noise = with_default(1e6, pars["exp_noise"]);

  sir::shared_t shared{S0, I0, R0, beta, gamma, dt, freq, exp_noise};
  return dust::pars_t<sir>(shared);
}

template <>
cpp11::sexp dust_info<sir>(const dust::pars_t<sir>& pars) {
  using namespace cpp11::literals;
  // Information about state order
  cpp11::writable::strings vars({"S", "I", "R", "cases_cumul", "cases_inc"});
  // Information about parameter values
  cpp11::list p = cpp11::writable::list({"beta"_nm = pars.shared->beta,
                                         "gamma"_nm = pars.shared->gamma});
  return cpp11::writable::list({"vars"_nm = vars, "pars"_nm = p});
}

// The way that this is going to work is we will process a list
// *outside* of the C that will take (say) a df and convert it
// row-wise into a list with elements `step` and `data`, we will pass
// that in here. Then this function will be called once per data
// element to create the struct that will be used for future work.
template <>
sir::data_t dust_data<sir>(cpp11::list data) {
  return sir::data_t{cpp11::as_cpp<double>(data["incidence"])};
}
