class sir {
public:
  typedef int int_t;
  typedef double float_t;
  struct init_t {
    double beta;
    double dt;
    double gamma;
    double I0;
    double initial_I;
    double initial_R;
    double initial_S;
    double p_IR;
    double S0;
    double steps_per_day;
  };

  sir(const init_t& data) : data_(data) {
  }

  size_t size() const {
    return 3;
  }

  std::vector<double> initial(size_t step) {
    std::vector<double> ret =
      {data_.initial_S, data_.initial_I, data_.initial_R};
    return ret;
  }

  void update(size_t step, const std::vector<double> state,
              dust::RNG<double, int>& rng, std::vector<double>& state_next) {
    double S = state[0];
    double I = state[1];
    double R = state[2];
    double N = S + I + R;
    double n_IR = rng.rbinom(round(I), data_.p_IR * data_.dt);
    double p_SI = 1 - std::exp(-(data_.beta) * I / (double) N);
    double n_SI = rng.rbinom(round(S), p_SI * data_.dt);
    state_next[2] = R + n_IR;
    state_next[1] = I + n_SI - n_IR;
    state_next[0] = S - n_SI;
  }

private:
  init_t data_;
};

#include <Rcpp.h>
template <>
sir::init_t dust_data<sir>(Rcpp::List data) {
  double initial_R = 0.0;
  double beta = 0.2;
  double gamma = 0.1;
  double I0 = 10.0;
  double S0 = 1000.0;
  double steps_per_day = 4;
  // Some boilerplate needed here in order to set user parameters
  // given a default that exists, though we'll never be using it like
  // that?
  double dt = 1 / (double) steps_per_day;
  double initial_I = I0;
  double initial_S = S0;
  double p_IR = 1 - std::exp(-(gamma));
  return sir::init_t{beta, dt, gamma, I0, initial_I, initial_R, initial_S,
      p_IR, S0, steps_per_day};
}
