class walk {
public:
  typedef int int_t;
  typedef double real_t;
  struct init_t {
    real_t sd;
  };
  walk(const init_t& data) : data_(data) {
  }
  size_t size() const {
    return 1;
  }
  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> ret = {0};
    return ret;
  }
  void update(size_t step, const std::vector<real_t>& state,
              dust::RNG<real_t, int>& rng, std::vector<real_t>& state_next) {
    real_t mean = state[0];
    state_next[0] = rng.rnorm(mean, data_.sd);
  }
private:
  init_t data_;
};

#include <cpp11/list.hpp>
template <>
walk::init_t dust_data<walk>(cpp11::list data) {
  walk::real_t sd = cpp11::as_cpp<walk::real_t>(data["sd"]);
  return walk::init_t{sd};
}
