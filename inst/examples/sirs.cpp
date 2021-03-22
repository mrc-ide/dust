class sirs {
public:
  typedef double real_t;
  typedef dust::no_data data_t;
  typedef dust::no_internal internal_t;

  struct shared_t {
    real_t S0;
    real_t I0;
    real_t R0;
    real_t alpha;
    real_t beta;
    real_t gamma;
    real_t dt;
  };

  sirs(const dust::pars_t<sirs>& pars): shared(pars.shared) {
  }

  size_t size() {
    return 3;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(3);
    state[0] = shared->S0;
    state[1] = shared->I0;
    state[2] = shared->R0;
    return state;
  }

  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state,
              real_t * state_next) {
    real_t S = state[0];
    real_t I = state[1];
    real_t R = state[2];
    real_t N = S + I + R;

    real_t p_SI = 1 - exp(- shared->beta * I / N);
    real_t p_IR = 1 - exp(-(shared->gamma));
    real_t p_RS = 1 - exp(- shared->alpha);

    real_t n_SI = dust::distr::rbinom(rng_state, S, p_SI * shared->dt);
    real_t n_IR = dust::distr::rbinom(rng_state, I, p_IR * shared->dt);
    real_t n_RS = dust::distr::rbinom(rng_state, R, p_RS * shared->dt);

    state_next[0] = S - n_SI + n_RS;
    state_next[1] = I + n_SI - n_IR;
    state_next[2] = R + n_IR - n_RS;
  }

private:
  dust::shared_ptr<sirs> shared;
};

namespace dust {
template <>
dust::pars_t<sirs> dust_pars<sirs>(cpp11::list pars) {
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
  // [[dust::param(beta, required = FALSE, default = 0.2)]]
  SEXP r_beta = pars["beta"];
  if (r_beta != R_NilValue) {
    beta = cpp11::as_cpp<sirs::real_t>(r_beta);
  }
  // [[dust::param(gamma, required = FALSE, default = 0.1)]]
  SEXP r_gamma = pars["gamma"];
  if (r_gamma != R_NilValue) {
    gamma = cpp11::as_cpp<sirs::real_t>(r_gamma);
  }

  sirs::shared_t shared{S0, I0, R0, alpha, beta, gamma, dt};
  return dust::pars_t<sirs>(shared);
}

template <>
struct has_gpu_support<sirs> : std::true_type {};

template <>
size_t device_shared_int_size<sirs>(dust::shared_ptr<sirs> shared) {
  return 0;
}

template <>
size_t device_shared_real_size<sirs>(dust::shared_ptr<sirs> shared) {
  return 4;
}

template <>
void device_shared_copy<sirs>(dust::shared_ptr<sirs> shared,
                              int * shared_int,
                              sirs::real_t * shared_real) {
  typedef sirs::real_t real_t;
  shared_real = dust::shared_copy<real_t>(shared_real, shared->alpha);
  shared_real = dust::shared_copy<real_t>(shared_real, shared->beta);
  shared_real = dust::shared_copy<real_t>(shared_real, shared->gamma);
  shared_real = dust::shared_copy<real_t>(shared_real, shared->dt);
}

template <>
DEVICE void update_device<sirs>(size_t step,
                                const dust::interleaved<sirs::real_t> state,
                                dust::interleaved<int> internal_int,
                                dust::interleaved<sirs::real_t> internal_real,
                                const int * shared_int,
                                const sirs::real_t * shared_real,
                                dust::rng_state_t<sirs::real_t>& rng_state,
                                dust::interleaved<sirs::real_t> state_next) {
  typedef sirs::real_t real_t;
  const real_t alpha = shared_real[0];
  const real_t beta = shared_real[1];
  const real_t gamma = shared_real[2];
  const real_t dt = shared_real[3];
  const real_t S = state[0];
  const real_t I = state[1];
  const real_t R = state[2];
  const real_t N = S + I + R;
  const real_t p_SI = 1 - exp(- beta * I / N);
  const real_t p_IR = 1 - exp(- gamma);
  const real_t p_RS = 1 - exp(- alpha);
  const real_t n_SI = dust::distr::rbinom(rng_state, S, p_SI * dt);
  const real_t n_IR = dust::distr::rbinom(rng_state, I, p_IR * dt);
  const real_t n_RS = dust::distr::rbinom(rng_state, R, p_RS * dt);
  state_next[0] = S - n_SI + n_RS;
  state_next[1] = I + n_SI - n_IR;
  state_next[2] = R + n_IR - n_RS;
}

}
