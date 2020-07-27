class variable {
public:
  typedef double real_t;
  struct init_t {
    size_t len;
    double mean;
    double sd;
  };

  variable(const init_t& data) : data_(data) {
  }

  size_t size() const {
    return data_.len;
  }

  std::vector<double> initial(size_t step) {
    std::vector<double> ret;
    for (size_t i = 0; i < data_.len; ++i) {
      ret.push_back(i + 1);
    }
    return ret;
  }

  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state,
              real_t * state_next) {
    for (size_t i = 0; i < data_.len; ++i) {
      state_next[i] =
        dust::distr::rnorm(rng_state, state[i] + data_.mean, data_.sd);
    }
  }

private:
  init_t data_;
};

#include <cpp11/list.hpp>
template <>
variable::init_t dust_data<variable>(cpp11::list data) {
  const size_t len = cpp11::as_cpp<int>(data["len"]);
  double mean = 0, sd = 1;

  SEXP r_mean = data["mean"];
  if (r_mean != R_NilValue) {
    mean = cpp11::as_cpp<double>(r_mean);
  }

  SEXP r_sd = data["sd"];
  if (r_sd != R_NilValue) {
    sd = cpp11::as_cpp<double>(r_sd);
  }

  return variable::init_t{len, mean, sd};
}
