class variable {
public:
  typedef double real_t;
  typedef no_data data_t;
  struct init_t {
    size_t len;
    double mean;
    double sd;
  };

  variable(const init_t& pars) : pars_(pars) {
  }

  size_t size() const {
    return pars_.len;
  }

  std::vector<double> initial(size_t step) {
    std::vector<double> ret;
    for (size_t i = 0; i < pars_.len; ++i) {
      ret.push_back(i + 1);
    }
    return ret;
  }

  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state,
              real_t * state_next) {
    for (size_t i = 0; i < pars_.len; ++i) {
      state_next[i] =
        dust::distr::rnorm(rng_state, state[i] + pars_.mean, pars_.sd);
    }
  }

private:
  init_t pars_;
};

#include <cpp11/list.hpp>
template <>
variable::init_t dust_pars<variable>(cpp11::list pars) {
  const size_t len = cpp11::as_cpp<int>(pars["len"]);
  double mean = 0, sd = 1;

  SEXP r_mean = pars["mean"];
  if (r_mean != R_NilValue) {
    mean = cpp11::as_cpp<double>(r_mean);
  }

  SEXP r_sd = pars["sd"];
  if (r_sd != R_NilValue) {
    sd = cpp11::as_cpp<double>(r_sd);
  }

  return variable::init_t{len, mean, sd};
}
