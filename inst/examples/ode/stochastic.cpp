// [[dust::time_type(continuous)]]
class logistic {
public:
  using real_type = double;
  using data_type = dust::no_data;
  using internal_type = dust::no_internal;
  using rng_state_type = dust::random::generator<real_type>;

  struct shared_type {
    real_type r1;
    real_type K1;
    real_type r2;
    real_type K2;
    real_type v;
  };

  logistic(const dust::pars_type<logistic>& pars): shared(pars.shared) {
  }

  void rhs(real_type t,
           const std::vector<real_type> &y,
           std::vector<real_type> &dydt) const {
    const real_type N1 = y[0];
    const real_type N2 = y[1];
    const real_type w  = y[2];
    dydt[0] = shared->r1 * N1 * (1 - N1 / (shared->K1 * w));
    dydt[1] = shared->r2 * N2 * (1 - N2 / (shared->K2 * w));

    // For now we just keep the stochastic component within the same
    // general vector, but it might be better to have a separate
    // argument here that holds the stochastic part (that does
    // complicate initialisation slightly, but not impossibly).
    dydt[2] = 0;
  }

  void output(real_type t,
              const std::vector<real_type>& y,
              std::vector<real_type>& output) {
  }

  void update_stochastic(real_type t, const std::vector<real_type>& y,
                         rng_state_type& rng_state,
                         std::vector<real_type>& y_next) {
    const real_type r = dust::random::normal<real_type>(rng_state, 0, shared->v);
    y_next[2] = y[2] * std::exp(r);
  }

  std::vector<real_type> initial(real_type time, rng_state_type& rng_state) {
    std::vector<real_type> ret = {1, 1, 1};
    return ret;
  }

  size_t n_variables() const {
    return 3;
  }

  size_t n_output() const {
    return 0;
  }

private:
  dust::shared_ptr<logistic> shared;
};

namespace dust {

template <>
dust::pars_type<logistic> dust_pars<logistic>(cpp11::list pars) {
  using real_type = logistic::real_type;
  // [[dust::param(r1, required = TRUE)]]
  real_type r1 = cpp11::as_cpp<real_type>(pars["r1"]);
  // [[dust::param(K1, required = TRUE)]]
  real_type K1 = cpp11::as_cpp<real_type>(pars["K1"]);
  // [[dust::param(r2, required = TRUE)]]
  real_type r2 = cpp11::as_cpp<real_type>(pars["r2"]);
  // [[dust::param(K2, required = TRUE)]]
  real_type K2 = cpp11::as_cpp<real_type>(pars["K2"]);
  // [[dust::param(v, required = TRUE)]]
  real_type v = cpp11::as_cpp<real_type>(pars["v"]);

  logistic::shared_type shared{r1, K1, r2, K2, v};
  return dust::pars_type<logistic>(shared);
}

}
