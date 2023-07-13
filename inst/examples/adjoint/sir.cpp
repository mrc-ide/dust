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
  };

  sir(const dust::pars_type<sir>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 5;
  }

  std::vector<real_type> initial(size_t time, rng_state_type& rng_state) {
    return std::vector<real_type> {shared->S0, shared->I0, shared->R0, 0, 0};
  }

  void update(size_t time, const real_type * state, rng_state_type& rng_state,
              real_type * state_next) {
    const real_type S = state[0];
    const real_type I = state[1];
    const real_type R = state[2];
    const real_type cases_cumul = state[3];
    const real_type cases_inc = state[4];
    const real_type N = S + I + R;
    const real_type p_inf = shared->beta * I / N * shared->dt;
    const real_type p_IR = 1 - dust::math::exp(-shared->gamma * shared->dt);
    const real_type p_SI = 1 - dust::math::exp(-p_inf);
    const real_type n_SI = S * p_SI;
    const real_type n_IR = I * p_IR;
    state_next[0] = S - n_SI;
    state_next[1] = I + n_SI - n_IR;
    state_next[2] = R + n_IR;
    state_next[3] = cases_cumul + n_SI;
    state_next[4] = (time % shared->freq == 0) ? n_SI : (cases_inc + n_SI);
  }

  size_t adjoint_size() const {
    return size() + 3;
  }
  // The interpretation here is slightly odd and very different to the
  // above, with adjoint and adjoint_next not quite being correct, and
  // with this function only writing the *additional* contribution of
  // the initial conditions into a vector adjoint_next which is
  // already largely full.
  void adjoint_initial(size_t time, const real_type * state,
                       const real_type * adjoint,
                       real_type * adjoint_next) {
    const real_type adj_I = adjoint[1];
    adjoint_next[0] = adjoint[0];
    adjoint_next[1] = adjoint[1];
    adjoint_next[2] = adjoint[2];
    adjoint_next[3] = adjoint[3];
    adjoint_next[4] = adjoint[4];
    adjoint_next[5] = adjoint[5];
    adjoint_next[6] = adjoint[6];
    adjoint_next[7] = adjoint[7] + adj_I;
  }

  void adjoint_update(size_t time, const real_type * state,
                      const real_type * adjoint,
                      real_type * adjoint_next) {
    // Same unpack as above.
    const real_type S = state[0];
    const real_type I = state[1];
    const real_type R = state[2];
    // const real_type cases_cumul = state[3];
    // const real_type cases_inc = state[4];

    // Same intermediates as above; saving these is probably more work
    // than wanted.
    const real_type N = S + I + R;
    const real_type p_inf = shared->beta * I / N * shared->dt;
    const real_type p_IR = 1 - dust::math::exp(-shared->gamma * shared->dt);
    const real_type p_SI = 1 - dust::math::exp(-p_inf);
    // const real_type n_SI = S * p_SI;
    // const real_type n_IR = I * p_IR;

    // Also unpack adjoint - same order as before followed by
    // derivative with respect to parameters
    const real_type adj_S = adjoint[0];
    const real_type adj_I = adjoint[1];
    const real_type adj_R = adjoint[2];
    const real_type adj_cases_cumul = adjoint[3];
    const real_type adj_cases_inc = adjoint[4];
    const real_type adj_beta = adjoint[5];
    const real_type adj_gamma = adjoint[6];
    // const real_type adj_I0 = adjoint[7];

    const real_type adj_n_IR = -adj_I + adj_R;
    const real_type adj_n_SI = adj_cases_cumul + adj_cases_inc + adj_I - adj_S;
    const real_type adj_p_SI = S * adj_n_SI;
    const real_type adj_p_inf = dust::math::exp(-p_inf) * adj_p_SI;
    const real_type adj_N = -(shared->beta * I / (N * N) * shared->dt) * adj_p_inf;
    const real_type adj_p_IR = I * adj_n_IR;

    adjoint_next[0] = adj_N + p_SI * adj_n_SI + adj_S;
    adjoint_next[1] = adj_N + p_IR * adj_n_IR + shared->beta / N * shared->dt * adj_p_inf + adj_I;
    adjoint_next[2] = adj_N + adj_R;
    adjoint_next[3] = adj_cases_cumul;
    adjoint_next[4] = adj_cases_inc * (time % shared->freq == 0 ? 0 : 1);

    // adjoint_parameters: accumulates feedback
    adjoint_next[5] = adj_beta + I / N * shared->dt * adj_p_inf;
    adjoint_next[6] = adj_gamma + dust::math::exp(-shared->gamma * shared->dt) * shared->dt * adj_p_IR;
    adjoint_next[7] = 0;
  }

  void adjoint_compare_data(size_t time,
                            const real_type * state, const data_type& data,
                            const real_type * adjoint,
                            real_type * adjoint_next) {
    const real_type cases_inc = state[4];
    adjoint_next[0] = adjoint[0];
    adjoint_next[1] = adjoint[1];
    adjoint_next[2] = adjoint[2];
    adjoint_next[3] = adjoint[3];
    adjoint_next[4] = adjoint[4] + data.incidence / cases_inc - 1;
    adjoint_next[5] = adjoint[5];
    adjoint_next[6] = adjoint[6];
    adjoint_next[7] = adjoint[7];
  }

  real_type compare_data(const real_type * state, const data_type& data,
                         rng_state_type& rng_state) {
    const real_type incidence_modelled = state[4];
    const real_type incidence_observed = data.incidence;
    const real_type lambda = incidence_modelled;
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

  sir::shared_type shared{S0, I0, R0, beta, gamma, dt, freq};
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
  cpp11::writable::strings adjoint({"beta", "gamma", "I0"});
  return cpp11::writable::list({"vars"_nm = vars,
                                "pars"_nm = p,
                                "adjoint"_nm = adjoint});
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
