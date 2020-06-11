class walk {
public:
  typedef SEXP init_t;
  walk(SEXP data) : sd(REAL(data)[0]) {
  }
  size_t size() const {
    return 1;
  }
  std::vector<double> initial(size_t step) {
    std::vector<double> ret = {0};
    return ret;
  }
  void update(size_t step, const std::vector<double> state, dust::RNG& rng,
              std::vector<double>& state_next) {
    double mean = state[0];
    state_next[0] = rng.rnorm(mean, sd);
  }
private:
  double sd;
};
