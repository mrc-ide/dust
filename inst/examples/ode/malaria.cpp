// [[dust::time_type(continuous)]]
class malaria {
public:
  using real_type = double;
  using internal_type = dust::no_internal;
  using rng_state_type = dust::random::generator<real_type>;

  struct data_type {
    int tested;
    int positive;
  };

  struct shared_type {
    real_type a; // Biting rate (bites per human per mosquito)

    real_type bh;  // Pr(transmission vector to human)
    real_type bv;  // Pr(transmission human to vector)
    int n_rates;   // number of exposure compartments
    real_type mu;  // -log(Pr(vector survival))
    real_type r;   // Rate of recovery
    real_type tau; // Length in mosquito latency period

    real_type initial_Ih; // Initial infected humans
    real_type initial_Iv; // Initial infected vectors
    real_type initial_Sv; // Initial susceptible vector

    real_type beta_volatility; // Volatility of random walk
  };

  malaria(const dust::pars_type<malaria>& pars) :
    shared(pars.shared) {
  }

  // State is modelled as [Sh, Ih, Sv, Iv, beta, Ev...]
  size_t n_variables() const {
    return shared->n_rates + 5;
  }

  // Output is [host prevelance, vector prevelance, total exposed vectors, total vectors]
  size_t n_output() const {
    return 4;
  }

  std::vector<real_type> initial(size_t t, rng_state_type& rng) {
    std::vector<real_type> state(shared->n_rates + 5);
    state[0] = 1 - shared->initial_Ih; // Sh
    state[1] = shared->initial_Ih;     // Ih
    state[2] = shared->initial_Sv;     // Sv
    state[3] = shared->initial_Iv;     // Iv
    state[4] = shared->mu;             // beta
    // Ev in the remaining compartments, zeroed at start:
    for (size_t i = 5; i < state.size(); ++i) {
      state[i] = 0;
    }
    return state;
  }

  void rhs(double t, const std::vector<double>& state, std::vector<double>& dstatedt) {
    const real_type Sh = state[0];
    const real_type Ih = state[1];
    const real_type Sv = state[2];
    const real_type Iv = state[3];
    const real_type beta = state[4];
    const real_type * Ev = state.data() + 5;
    const real_type foi_h = shared->a * shared->bh * Iv;
    const real_type foi_v = shared->a * shared->bv * Ih;
    dstatedt[5] = foi_v * Sv - (shared->n_rates / shared->tau) * Ev[0] - shared->mu * Ev[0];
    for (int i = 1; i < shared->n_rates; ++i) {
      dstatedt[5 + i] = (shared->n_rates / shared->tau) * Ev[i - 1] - (shared->n_rates / shared->tau) * Ev[i] - shared->mu * Ev[i];
    }
    dstatedt[0] = - foi_h * Sh + shared->r * Ih;
    dstatedt[1] = foi_h * Sh - shared->r * Ih;
    dstatedt[2] = beta * shared->initial_Sv - foi_v * Sv - shared->mu * Sv;
    dstatedt[3] = (shared->n_rates / shared->tau) * Ev[shared->n_rates - 1] - shared->mu * Iv;
  }

  void output(double t, const std::vector<double>& state, std::vector<double>& output) {
    const real_type Sh = state[0];
    const real_type Ih = state[1];
    const real_type Sv = state[2];
    const real_type Iv = state[3];
    const real_type * Ev = state.data() + 5;
    const real_type Ev_tot = std::accumulate(Ev, Ev + shared->n_rates, (real_type)0); // sum of Ev
    const real_type N = Sh + Ih;
    const real_type V = Sv + Ev_tot + Iv;
    output[0] = Ih / N;
    output[1] = Iv / V;
    output[2] = Ev_tot;
    output[3] = V;
  }

  void update_stochastic(double t, const std::vector<double>& state, rng_state_type& rng_state, std::vector<double>& state_next) {
    const real_type beta = state[4];
    state_next[4] = beta * dust::math::exp(dust::random::normal<real_type>(rng_state, 0, shared->beta_volatility));
  }

  real_type compare_data(const real_type * state, const data_type& data,
                         rng_state_type& rng_state) {
    const real_type Ih = state[1]; // Ih
    return dust::density::binomial(data.positive, data.tested, Ih, true);
  }

private:
  std::shared_ptr<const shared_type> shared;
};

template <typename T>
T with_default(T default_value, const char * name, cpp11::list user) {
  const cpp11::sexp value = user[name];
  return value == R_NilValue ? default_value : cpp11::as_cpp<T>(value);
}

namespace dust {
template<>
dust::pars_type<malaria> dust_pars<malaria>(cpp11::list user) {
  using real_type = typename malaria::real_type;
  malaria::shared_type shared;

  // [[dust::param(a, required = FALSE)]]
  shared.a = with_default<real_type>(1.0 / 3.0, "a", user);
  // [[dust::param(n_rates, required = FALSE)]]
  shared.n_rates = with_default<int>(15, "n_rates", user);
  // [[dust::param(r, required = FALSE)]]
  shared.r = with_default<real_type>(0.01, "r", user);
  // [[dust::param(tau, required = FALSE)]]
  shared.tau = with_default<real_type>(12, "tau", user);

  // [[dust::param(initial_Ih, required = FALSE)]]
  shared.initial_Ih = with_default<real_type>(0.8, "initial_Ih", user);
  // [[dust::param(initial_Iv, required = FALSE)]]
  shared.initial_Iv = with_default<real_type>(1, "initial_Iv", user);
  // [[dust::param(initial_Sv, required = FALSE)]]
  shared.initial_Sv = with_default<real_type>(100, "initial_Sv", user);

  shared.beta_volatility = 0.5;
  shared.bh = 0.05;
  shared.bv = 0.05;
  const real_type p = 0.9; // Daily probability of vector survival
  shared.mu = - dust::math::log(p);

  return dust::pars_type<malaria>(shared);
}

template <>
cpp11::sexp dust_info<malaria>(const dust::pars_type<malaria>& pars) {
  cpp11::writable::strings vars({"Sh", "Ih", "Sv", "Iv", "beta", "Ev"});
  return vars;
}

template <>
malaria::data_type dust_data<malaria>(cpp11::list data) {
  const int tested = cpp11::as_cpp<int>(data["tested"]);
  const int positive = cpp11::as_cpp<int>(data["positive"]);
  return malaria::data_type{tested, positive};
}
}
