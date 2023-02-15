// [[dust::time_type(continuous)]]
class logistic {
public:
  using real_type = double;
  using data_type = dust::no_data;
  using internal_type = dust::no_internal;
  using rng_state_type = dust::random::generator<real_type>;

  struct shared_type {
    size_t n;
    std::vector<real_type> r;
    std::vector<real_type> K;
  };

  logistic(const dust::pars_type<logistic>& pars): shared(pars.shared) {
  }

  void rhs(real_type t,
           const std::vector<real_type>& y,
           std::vector<real_type>& dydt) const {
    for (size_t i = 0; i < shared->n; ++i) {
      dydt[i] = shared->r[i] * y[i] * (1 - y[i] / shared->K[i]);
    }
  }

  void output(real_type t,
              const std::vector<real_type>& y,
              std::vector<real_type>& output) {
    real_type tot = 0;
    for (size_t i = 0; i < shared->n; ++i) {
      tot += y[i];
    }
    output[0] = tot;
  }

  std::vector<real_type> initial(real_type time) {
    return std::vector<real_type>(shared->n, 1);
  }

  void update_stochastic(real_type t, const std::vector<real_type>& y,
                         rng_state_type& rng_state,
                         std::vector<real_type>& y_next) {
  }

  size_t n_variables() const {
    return shared->n;
  }

  size_t n_output() const {
    return 1;
  }

private:
  dust::shared_ptr<logistic> shared;
};

namespace dust {

template <>
dust::pars_type<logistic> dust_pars<logistic>(cpp11::list pars) {
  using real_type = logistic::real_type;
  // [[dust::param(r, required = TRUE)]]
  std::vector<real_type> r = cpp11::as_cpp<std::vector<real_type>>(pars["r"]);
  // [[dust::param(K, required = TRUE)]]
  std::vector<real_type> K = cpp11::as_cpp<std::vector<real_type>>(pars["K"]);
  if (r.size() != K.size()) {
    cpp11::stop("Expected 'r' and 'K' to have the same size");
  }
  const size_t n = r.size();
  if (n == 0) {
    cpp11::stop("'r' and 'K' must have length of at least 1");
  }
  logistic::shared_type shared{n, r, K};
  return dust::pars_type<logistic>(shared);
}

template <>
cpp11::sexp dust_info<logistic>(const dust::pars_type<logistic>& pars) {
  using namespace cpp11::literals;
  return cpp11::writable::list({"n"_nm = pars.shared->n});
}

}
