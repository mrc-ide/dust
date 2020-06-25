class walk {
public:
  struct init_t {
    double sd;
  };
  walk(const init_t& data) : data_(data) {
  }
  size_t size() const {
    return 1;
  }
  std::vector<double> initial(size_t step) {
    std::vector<double> ret = {0};
    return ret;
  }
  void update(size_t step, const std::vector<double>& state,
              dust::RNG<double, int>& rng, std::vector<double>& state_next) {
    double mean = state[0];
    state_next[0] = rng.rnorm(mean, data_.sd);
  }
private:
  init_t data_;
};

#include <Rcpp.h>
template <>
walk::init_t dust_data<walk>(Rcpp::List data) {
  double sd = Rcpp::as<double>(data["sd"]);
  return walk::init_t{sd};
}
