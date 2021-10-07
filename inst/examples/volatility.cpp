class volatility {
public:
  typedef double real_t;
  struct data_t {
    real_t observed;
  };
  typedef dust::no_internal internal_t;
  typedef dust::random::xoshiro256starstar_state rng_state_type;

  struct shared_t {
    real_t alpha;
    real_t sigma;
    real_t gamma;
    real_t tau;
    real_t x0;
  };

  volatility(const dust::pars_type<volatility>& pars) : shared(pars.shared) {
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
              rng_state_type& rng_state, real_t * state_next) {
    const real_t x = state[0];
    state_next[0] = shared->alpha * x +
      shared->sigma * dust::random::normal<real_t>(rng_state, 0, 1);
  }

  real_t compare_data(const real_t * state, const data_t& data,
                      rng_state_type& rng_state) {
    return dust::density::normal(data.observed, shared->gamma * state[0],
                                 shared->tau, true);
  }

private:
  dust::shared_ptr<volatility> shared;
};

// Helper function for accepting values with defaults
inline double with_default(double default_value, cpp11::sexp value) {
  return value == R_NilValue ? default_value : cpp11::as_cpp<double>(value);
}

namespace dust {

template <>
dust::pars_type<volatility> dust_pars<volatility>(cpp11::list pars) {
  typedef volatility::real_t real_t;
  real_t x0 = 0;
  real_t alpha = with_default(0.91, pars["alpha"]);
  real_t sigma = with_default(1, pars["sigma"]);
  real_t gamma = with_default(1, pars["gamma"]);
  real_t tau = with_default(1, pars["tau"]);

  volatility::shared_t shared{alpha, sigma, gamma, tau, x0};
  return dust::pars_type<volatility>(shared);
}

template <>
volatility::data_t dust_data<volatility>(cpp11::list data) {
  return volatility::data_t{cpp11::as_cpp<double>(data["observed"])};
}

}
