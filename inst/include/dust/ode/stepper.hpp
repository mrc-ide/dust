#ifndef DUST_ODE_STEPPER_HPP
#define DUST_ODE_STEPPER_HPP

#include <algorithm>
#include <cmath>
#include <vector>

#include "dust/ode/initial_step_size.hpp"
#include "dust/utils.hpp"

namespace dust {
namespace ode {

template <typename Model>
class stepper {
public:
  using real_type = typename Model::real_type;
  using rng_state_type = typename Model::rng_state_type;

  stepper(Model m, real_type t) :
    m(m), n_var(m.n_variables()), n_out(m.n_output()),
    y(n_var), y_next(n_var), y_stiff(n_var), k1(n_var),
    k2(n_var), k3(n_var), k4(n_var),
    k5(n_var), k6(n_var), output(n_out),
    needs_initialise(true) {
    const auto y = m.initial(t);
    set_state(y.begin());
  }

  void step(real_type t, real_type h) {
    if (needs_initialise) {
      std::fill(k1.begin(), k1.end(), 0);
      m.rhs(t, y, k1);
      needs_initialise = false;
    }
    for (size_t i = 0; i < n_var; ++i) { // 22
      y_next[i] = y[i] + h * A21 * k1[i];
    }
    m.rhs(t + C2 * h, y_next, k2);
    for (size_t i = 0; i < n_var; ++i) { // 23
      y_next[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
    }
    m.rhs(t + C3 * h, y_next, k3);
    for (size_t i = 0; i < n_var; ++i) { // 24
      y_next[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
    }
    m.rhs(t + C4 * h, y_next, k4);
    for (size_t i = 0; i < n_var; ++i) { // 25
      y_next[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] +
                              A54 * k4[i]);
    }
    m.rhs(t + C5 * h, y_next, k5);
    for (size_t i = 0; i < n_var; ++i) { // 26
      y_stiff[i] = y[i] + h * (A61 * k1[i] + A62 * k2[i] +
                               A63 * k3[i] + A64 * k4[i] +
                               A65 * k5[i]);
    }
    const real_type t_next = t + h;
    m.rhs(t_next, y_stiff, k6);
    for (size_t i = 0; i < n_var; ++i) { // 27
      y_next[i] = y[i] + h * (A71 * k1[i] + A73 * k3[i] + A74 * k4[i] +
                              A75 * k5[i] + A76 * k6[i]);
    }
    m.rhs(t_next, y_next, k2);

    for (size_t i = 0; i < n_var; ++i) {
      k4[i] = h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] +
                   E5 * k5[i] + E6 * k6[i] + E7 * k2[i]);
    }
  }

  real_type error(real_type atol, real_type rtol) {
    real_type err = 0.0;
    for (size_t i = 0; i < n_var; ++i) {
      const real_type sk = atol + rtol *
                               std::max(std::abs(y[i]), std::abs(y_next[i]));
      err += utils::square(k4[i] / sk);
    }
    return std::sqrt(err / n_var);
  }

  void set_state(typename std::vector<real_type>::const_iterator state) {
    for (size_t i = 0; i < n_var; ++i, ++state) {
      y[i] = *state;
    }
    needs_initialise = true;
  }

  void set_state(typename std::vector<real_type>::const_iterator state,
                 const std::vector<size_t>& index) {
    for (size_t i = 0; i < index.size(); ++i, ++state) {
      y[index[i]] = *state;
    }
    needs_initialise = true;
  }

  // store future y values in y_next, future dydt in k2 these will
  // then be swapped into place (see "swap" below). It's important to
  // move the y values into y_next and not one of the k* vectors that
  // hold derivatives in case the model is stochastic and does not
  // explicitly set a derivative for an equation (in which case they
  // should be zero).
  void set_state(const stepper<Model>& other) {
    std::copy(other.k1.begin(), other.k1.end(), k2.begin());
    std::copy(other.y.begin(), other.y.end(), y_next.begin());
  }

  // to be called after "set_state(other)", see above
  // to populate desired y, k1 values
  void swap() {
    std::swap(k1, k2);
    std::swap(y, y_next);
  }

  void set_model(Model new_model) {
    m = new_model;
    needs_initialise = true;
  }

  void set_model(Model new_model, real_type t) {
    m = new_model;
    const auto y = m.initial(t);
    set_state(y.begin());
    needs_initialise = true;
  }

  void initialise(real_type t) {
    std::fill(k1.begin(), k1.end(), 0);
    m.rhs(t, y, k1);
  }

  void update_stochastic(real_type t, rng_state_type& rng_state) {
    // Slightly odd construction here - we copy y into y_next so that
    // they both hold the same values, then do the step to update from
    // y_next to y so that at the end of this step 'y' holds the
    // current values (and then the derivative calculation in
    // initialise works as expected).
    std::copy_n(y.begin(), n_var, y_next.begin()); // from y to y_next
    m.update_stochastic(t, y_next, rng_state, y);  // from y_next to y
    needs_initialise = true;
  }

  void step_complete(real_type t, real_type h) {
    std::copy_n(k2.begin(), n_var, k1.begin()); // k1 = k2
    std::copy_n(y_next.begin(), n_var, y.begin()); // y = y_next
  }

  const std::vector<real_type>& state() const {
    return y;
  }

  void state(real_type t,
             const std::vector<size_t>& index,
             typename std::vector<real_type>::iterator end_state) {
    auto n = index.size();
    bool have_run_output = false;
    for (size_t i = 0; i < n; ++i, ++end_state) {
      auto j = index[i];
      if (j < n_var) {
        *end_state = y[j];
      } else {
        if (!have_run_output) {
          m.output(t, y, output);
          have_run_output = true;
        }
        *end_state = output[j - n_var];
      }
    }
  }

  void state(real_type t,
             typename std::vector<real_type>::iterator end_state) {
    auto n = n_var + n_out;
    bool have_run_output = false;
    for (size_t i = 0; i < n; ++i, ++end_state) {
      if (i < n_var) {
        *end_state = y[i];
      } else {
        if (!have_run_output) {
          m.output(t, y, output);
          have_run_output = true;
        }
        *end_state = output[i - n_var];
      }
    }
  }

  real_type init_step_size(real_type t, control<real_type> ctl) {
    return initial_step_size(m, t, y, ctl);
  }

  real_type compare_data(const typename Model::data_type& data,
                      rng_state_type& rng_state) {
    return m.compare_data(y.data(), data, rng_state);
  }

private:
  Model m;
  size_t n_var;
  size_t n_out;
  std::vector<real_type> y;
  std::vector<real_type> y_next;
  std::vector<real_type> y_stiff;
  std::vector<real_type> k1;
  std::vector<real_type> k2;
  std::vector<real_type> k3;
  std::vector<real_type> k4;
  std::vector<real_type> k5;
  std::vector<real_type> k6;
  std::vector<real_type> output;
  bool needs_initialise;

  static constexpr real_type C2 = 0.2;
  static constexpr real_type C3 = 0.3;
  static constexpr real_type C4 = 0.8;
  static constexpr real_type C5 = 8.0 / 9.0;
  static constexpr real_type A21 = 0.2;
  static constexpr real_type A31 = 3.0 / 40.0;
  static constexpr real_type A32 = 9.0 / 40.0;
  static constexpr real_type A41 = 44.0 / 45.0;
  static constexpr real_type A42 = -56.0 / 15.0;
  static constexpr real_type A43 = 32.0 / 9.0;
  static constexpr real_type A51 = 19372.0 / 6561.0;
  static constexpr real_type A52 = -25360.0 / 2187.0;
  static constexpr real_type A53 = 64448.0 / 6561.0;
  static constexpr real_type A54 = -212.0 / 729.0;
  static constexpr real_type A61 = 9017.0 / 3168.0;
  static constexpr real_type A62 = -355.0 / 33.0;
  static constexpr real_type A63 = 46732.0 / 5247.0;
  static constexpr real_type A64 = 49.0 / 176.0;
  static constexpr real_type A65 = -5103.0 / 18656.0;
  static constexpr real_type A71 = 35.0 / 384.0;
  static constexpr real_type A73 = 500.0 / 1113.0;
  static constexpr real_type A74 = 125.0 / 192.0;
  static constexpr real_type A75 = -2187.0 / 6784.0;
  static constexpr real_type A76 = 11.0 / 84.0;
  static constexpr real_type E1 = 71.0 / 57600.0;
  static constexpr real_type E3 = -71.0 / 16695.0;
  static constexpr real_type E4 = 71.0 / 1920.0;
  static constexpr real_type E5 = -17253.0 / 339200.0;
  static constexpr real_type E6 = 22.0 / 525.0;
  static constexpr real_type E7 = -1.0 / 40.0;
  // ---- DENSE OUTPUT OF SHAMPINE (1986)
  static constexpr real_type D1 = -12715105075.0 / 11282082432.0;
  static constexpr real_type D3 = 87487479700.0 / 32700410799.0;
  static constexpr real_type D4 = -10690763975.0 / 1880347072.0;
  static constexpr real_type D5 = 701980252875.0 / 199316789632.0;
  static constexpr real_type D6 = -1453857185.0 / 822651844.0;
  static constexpr real_type D7 = 69997945.0 / 29380423.0;
};

}
}

#endif
