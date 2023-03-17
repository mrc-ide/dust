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
    real_type v;
    bool random_initial;
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

  std::vector<real_type> initial(real_type time, rng_state_type& rng_state) {
    std::vector<real_type> y(shared->n, 1);
    if (shared->random_initial) {
      for (size_t i = 0; i < shared->n; ++i) {
        y[i] *= std::exp(dust::random::random_normal<real_type>(rng_state));
      }
    }
    return y;
  }

  void update_stochastic(real_type t, const std::vector<real_type>& y,
                         rng_state_type& rng_state,
                         std::vector<real_type>& y_next) {
    for (size_t i = 0; i < shared->n; ++i) {
      const auto r = dust::random::normal<real_type>(rng_state, 0, shared->v);
      y_next[i] = y[i] * std::exp(r);
    }
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

template <typename real_type>
std::vector<real_type> user_vector(const char * name, cpp11::list pars) {
  auto value = cpp11::as_cpp<cpp11::doubles>(pars[name]);
  return std::vector<real_type>(value.begin(), value.end());
}

namespace dust {

template <>
dust::pars_type<logistic> dust_pars<logistic>(cpp11::list pars) {
  using real_type = logistic::real_type;
  // [[dust::param(r, required = TRUE)]]
  const auto r = user_vector<real_type>("r", pars);
  // [[dust::param(K, required = TRUE)]]
  const auto K = user_vector<real_type>("K", pars);
  const size_t n = r.size();
  if (n == 0) {
    cpp11::stop("'r' and 'K' must have length of at least 1");
  }
  // [[dust::param(v, required = FALSE)]]
  cpp11::sexp r_v = pars["v"];
  // [[dust::param(random_initial, required = FALSE)]]
  const bool random_initial = pars["random_initial"] == R_NilValue ? false :
    cpp11::as_cpp<bool>(pars["random_initial"]);
  const real_type v = r_v == R_NilValue ? 0.1 : cpp11::as_cpp<real_type>(r_v);
  logistic::shared_type shared{n, r, K, v, random_initial};
  return dust::pars_type<logistic>(shared);
}

template <>
cpp11::sexp dust_info<logistic>(const dust::pars_type<logistic>& pars) {
  using namespace cpp11::literals;
  return cpp11::writable::list({"n"_nm = pars.shared->n});
}

}
