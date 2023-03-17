class volatility {
public:
  using real_type = double;
  using internal_type = dust::no_internal;
  using rng_state_type = dust::random::generator<real_type>;

  struct data_type {
    real_type observed;
  };

  struct shared_type {
    real_type alpha;
    real_type sigma;
    real_type gamma;
    real_type tau;
    real_type x0;
  };

  volatility(const dust::pars_type<volatility>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 1;
  }

  std::vector<real_type> initial(size_t time, rng_state_type& rng_state) {
    std::vector<real_type> state(1);
    state[0] = shared->x0;
    return state;
  }

  void update(size_t time, const real_type * state,
              rng_state_type& rng_state, real_type * state_next) {
    const real_type x = state[0];
    state_next[0] = shared->alpha * x +
      shared->sigma * dust::random::normal<real_type>(rng_state, 0, 1);
  }

  real_type compare_data(const real_type * state, const data_type& data,
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
  using real_type = volatility::real_type;
  real_type x0 = 0;
  real_type alpha = with_default(0.91, pars["alpha"]);
  real_type sigma = with_default(1, pars["sigma"]);
  real_type gamma = with_default(1, pars["gamma"]);
  real_type tau = with_default(1, pars["tau"]);

  volatility::shared_type shared{alpha, sigma, gamma, tau, x0};
  return dust::pars_type<volatility>(shared);
}

template <>
volatility::data_type dust_data<volatility>(cpp11::list data) {
  return volatility::data_type{cpp11::as_cpp<double>(data["observed"])};
}

}
