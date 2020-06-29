#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

class parallel {
public:
  typedef int int_t;
  typedef double real_t;
  struct init_t {
    double sd;
  };
  parallel(const init_t& data) : data_(data) {
  }
  size_t size() const {
    return 2;
  }
  std::vector<double> initial(size_t step) {
#ifdef _OPENMP
    static bool has_openmp = true;
#else
    static bool has_openmp = false;
#endif
    std::vector<double> ret = {0, (double) has_openmp};
    return ret;
  }
  void update(size_t step, const std::vector<double>& state,
              dust::RNG<double, int>& rng, std::vector<double>& state_next) {
    double mean = state[0];
    state_next[0] = rng.rnorm(mean, data_.sd);
#ifdef _OPENMP
    state_next[1] = (double) omp_get_thread_num();
#else
    state_next[1] = -1;
#endif
  }

private:
  init_t data_;
};

#include <Rcpp.h>
template <>
parallel::init_t dust_data<parallel>(Rcpp::List data) {
  double sd = Rcpp::as<double>(data["sd"]);
  return parallel::init_t{sd};
}
