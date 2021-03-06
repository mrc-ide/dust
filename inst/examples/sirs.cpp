class sirs {
public:
  typedef double real_t;
  typedef dust::no_internal internal_t;

  // ALIGN(16) is required before the data_t definition when using NVCC
  // This is so when loaded into shared memory it is aligned correctly
  struct ALIGN(16) data_t {
    double incidence;
  };

  struct shared_t {
    real_t S0;
    real_t I0;
    real_t R0;
    real_t alpha;
    real_t beta;
    real_t gamma;
    real_t dt;
    size_t freq;
    real_t exp_noise;
  };

  sirs(const dust::pars_t<sirs>& pars): shared(pars.shared) {
  }

  size_t size() {
    return 4;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(4);
    state[0] = shared->S0;
    state[1] = shared->I0;
    state[2] = shared->R0;
    state[3] = 0;
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
    state_next[3] = (step % shared->freq == 0) ? n_SI : state[3] + n_SI;
  }

  real_t compare_data(const real_t * state, const data_t& data,
                      dust::rng_state_t<real_t>& rng_state) {
    const real_t incidence_modelled = state[3];
    const real_t incidence_observed = data.incidence;
    const real_t lambda = incidence_modelled +
      dust::distr::rexp(rng_state, shared->exp_noise);
    return dust::dpois(incidence_observed, lambda, true);
  }

private:
  dust::shared_ptr<sirs> shared;
};

// Helper function for accepting values with defaults
inline double with_default(double default_value, cpp11::sexp value) {
  return value == R_NilValue ? default_value : cpp11::as_cpp<double>(value);
}

namespace dust {
template <>
dust::pars_t<sirs> dust_pars<sirs>(cpp11::list pars) {
  // Initial state values
  sirs::real_t I0 = 10.0;
  sirs::real_t S0 = 1000.0;
  sirs::real_t R0 = 0.0;

  // Time scaling
  // [[dust::param(freq, required = FALSE, default = 1)]]
  size_t freq = std::max(1.0, with_default(1.0, pars["freq"]));
  sirs::real_t dt = 1 / static_cast<sirs::real_t>(freq);

  sirs::real_t exp_noise = 1e6;

  // [[dust::param(alpha, required = FALSE, default = 0.1)]]
  sirs::real_t alpha = with_default(0.1, pars["alpha"]);

  // [[dust::param(beta, required = FALSE, default = 0.2)]]
  sirs::real_t beta = with_default(0.2, pars["beta"]);

  // [[dust::param(gamma, required = FALSE, default = 0.1)]]
  sirs::real_t gamma = with_default(0.1, pars["gamma"]);

  sirs::shared_t shared{S0, I0, R0, alpha, beta, gamma, dt, freq, exp_noise};
  return dust::pars_t<sirs>(shared);
}

template <>
sirs::data_t dust_data<sirs>(cpp11::list data) {
  return sirs::data_t{cpp11::as_cpp<double>(data["incidence"])};
}

template <>
struct has_gpu_support<sirs> : std::true_type {};

template <>
size_t device_shared_int_size<sirs>(dust::shared_ptr<sirs> shared) {
  return 1;
}

template <>
size_t device_shared_real_size<sirs>(dust::shared_ptr<sirs> shared) {
  return 5;
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
  shared_real = dust::shared_copy<real_t>(shared_real, shared->exp_noise);
  shared_int = dust::shared_copy<int>(shared_int, shared->freq);
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
  const int freq = shared_int[0];
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
  state_next[3] = (step % freq == 0) ? n_SI : state[3] + n_SI;
}

template <>
DEVICE sirs::real_t compare_device<sirs>(const dust::interleaved<sirs::real_t> state,
                           const sirs::data_t& data,
                           dust::interleaved<int> internal_int,
                           dust::interleaved<sirs::real_t> internal_real,
                           const int * shared_int,
                           const sirs::real_t * shared_real,
                           dust::rng_state_t<sirs::real_t>& rng_state) {
  typedef sirs::real_t real_t;
  const real_t exp_noise = shared_real[4];
  const real_t incidence_modelled = state[3];
  const real_t incidence_observed = data.incidence;
  const real_t lambda = incidence_modelled +
    dust::distr::rexp(rng_state, exp_noise);
  return dust::dpois(incidence_observed, lambda, true);
}

}
