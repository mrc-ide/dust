// [[dust::time_type(continuous)]]
class logistic {
public:
  using data_type = mode::no_data;
  using internal_type = mode::no_internal;
  using rng_state_type = dust::random::generator<double>;

  struct shared_type {
    double r1;
    double K1;
    double r2;
    double K2;
  };

  logistic(const mode::pars_type<logistic>& pars): shared(pars.shared) {
  }

  void rhs(double t,
           const std::vector<double>& y,
           std::vector<double>& dydt) const {
    const double N1 = y[0];
    const double N2 = y[1];
    dydt[0] = shared->r1 * N1 * (1 - N1 / shared->K1);
    dydt[1] = shared->r2 * N2 * (1 - N2 / shared->K2);
  }

  void output(double t,
         const std::vector<double>& y,
         std::vector<double>& output) {
    const double N1 = y[0];
    const double N2 = y[1];
    output[0] = N1 + N2;
  }

  std::vector<double> initial(double time) {
    std::vector<double> ret = {1, 1};
    return ret;
  }

  void update_stochastic(double t, const std::vector<double>& y,
                         rng_state_type& rng_state,
                         std::vector<double>& y_next) {
  }

  size_t n_variables() const {
    return 2;
  }

  size_t n_output() const {
    return 1;
  }

private:
  mode::shared_ptr<logistic> shared;
};

namespace mode {

template <>
mode::pars_type<logistic> mode_pars<logistic>(cpp11::list pars) {
  // [[dust::param(r1, required = TRUE)]]
  double r1 = cpp11::as_cpp<double>(pars["r1"]);
  // [[dust::param(K1, required = TRUE)]]
  double K1 = cpp11::as_cpp<double>(pars["K1"]);
  // [[dust::param(r2, required = TRUE)]]
  double r2 = cpp11::as_cpp<double>(pars["r2"]);
  // [[dust::param(K2, required = TRUE)]]
  double K2 = cpp11::as_cpp<double>(pars["K2"]);

  logistic::shared_type shared{r1, K1, r2, K2};
  return mode::pars_type<logistic>(shared);
}

template <>
cpp11::sexp mode_info<logistic>(const mode::pars_type<logistic>& pars) {
  return cpp11::writable::strings({"N1", "N2"});
}

}
