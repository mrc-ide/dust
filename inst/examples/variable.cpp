class variable {
public:
  typedef int int_t;
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

  void update(size_t step, const std::vector<double> state,
              dust::RNG<double, int>& rng, std::vector<double>& state_next) {
    for (size_t i = 0; i < data_.len; ++i) {
      state_next[i] = rng.rnorm(state[i] + data_.mean, data_.sd);
    }
  }

private:
  init_t data_;
};

#include <Rcpp.h>
template <>
variable::init_t dust_data<variable>(Rcpp::List data) {
  const size_t len = Rcpp::as<int>(data["len"]);
  double mean = 0, sd = 1;
  if (data.containsElementNamed("mean")) {
    mean = Rcpp::as<double>(data["mean"]);
  }
  if (data.containsElementNamed("sd")) {
    sd = Rcpp::as<double>(data["sd"]);
  }
  return variable::init_t{len, mean, sd};
}
