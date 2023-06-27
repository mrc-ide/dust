class sir {
public:
  using real_type = double;
  using internal_type = dust::no_internal;
  using rng_state_type = dust::random::generator<real_type>;
  
  struct data_type {
    real_type incidence;
  };

  struct shared_type {
    real_type S0;
    real_type I0;
    real_type R0;
    real_type beta;
    real_type gamma;
    real_type dt;
    size_t freq;
    // Observation parameters
    real_type exp_noise;
  };

  sir(const dust::pars_type<sir>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 5;
  }

  std::vector<real_type> initial(size_t time, rng_state_type& rng_state) {
    std::vector<real_type> ret = {shared->S0, shared->I0, shared->R0, 0, 0};
    return ret;
  }

  void update(size_t time, const real_type * state, rng_state_type& rng_state,
              real_type * state_next) {
    real_type S = state[0];
    real_type I = state[1];
    real_type R = state[2];
    real_type cumulative_incidence = state[3];

    real_type N = S + I + R;

    real_type p_SI = 1 - std::exp(-(shared->beta) * I / N);
    real_type p_IR = 1 - std::exp(-(shared->gamma));
    real_type n_IR = dust::random::binomial<real_type>(rng_state, I,
                                                       p_IR * shared->dt);
    real_type n_SI = dust::random::binomial<real_type>(rng_state, S,
                                                       p_SI * shared->dt);

    state_next[0] = S - n_SI;
    state_next[1] = I + n_SI - n_IR;
    state_next[2] = R + n_IR;
    state_next[3] = cumulative_incidence + n_SI;
    // Little trick here to compute daily incidence by accumulating
    // incidence from the first day.
    state_next[4] = (time % shared->freq == 0) ? n_SI : state[4] + n_SI;
  }

  real_type compare_data(const real_type * state, const data_type& data,
                         rng_state_type& rng_state) {
    const real_type incidence_observed = data.incidence;
    if (std::isnan(data.incidence)) {
      return 0;
    }
    const real_type incidence_modelled = state[4];
    const real_type lambda = incidence_modelled +
      dust::random::exponential(rng_state, shared->exp_noise);
    return dust::density::poisson(incidence_observed, lambda, true);
  }

private:
  dust::shared_ptr<sir> shared;
};

// Helper function for accepting values with defaults
inline double with_default(double default_value, cpp11::sexp value) {
  return value == R_NilValue ? default_value : cpp11::as_cpp<double>(value);
}

namespace dust {

template <>
dust::pars_type<sir> dust_pars<sir>(cpp11::list pars) {
  using real_type = sir::real_type;
  // Initial state values
  // [[dust::param(I0, required = FALSE)]]
  real_type I0 = with_default(10, pars["I0"]);
  real_type S0 = 1000.0;
  real_type R0 = 0.0;

  // Rates, which can be set based on the provided pars
  // [[dust::param(beta, required = FALSE)]]
  real_type beta = with_default(0.2, pars["beta"]);
  // [[dust::param(gamma, required = FALSE)]]
  real_type gamma = with_default(0.1, pars["gamma"]);

  // Time scaling
  size_t freq = 4;
  real_type dt = 1.0 / static_cast<real_type>(freq);

  // Compare function
  // [[dust::param(exp_noise, required = FALSE)]]
  real_type exp_noise = with_default(1e6, pars["exp_noise"]);

  sir::shared_type shared{S0, I0, R0, beta, gamma, dt, freq, exp_noise};
  return dust::pars_type<sir>(shared);
}

template <>
cpp11::sexp dust_info<sir>(const dust::pars_type<sir>& pars) {
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
// row-wise into a list with elements `time` and `data`, we will pass
// that in here. Then this function will be called once per data
// element to create the struct that will be used for future work.
template <>
sir::data_type dust_data<sir>(cpp11::list data) {
  return sir::data_type{cpp11::as_cpp<sir::real_type>(data["incidence"])};
}

}
