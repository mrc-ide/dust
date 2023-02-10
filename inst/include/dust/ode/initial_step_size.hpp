#ifndef DUST_ODE_INITIAL_STEP_SIZE_HPP
#define DUST_ODE_INITIAL_STEP_SIZE_HPP

#include <cmath>
#include <cstddef>
#include <vector>

#include "dust/ode/control.hpp"
#include "dust/utils.hpp"

namespace dust {
namespace ode {

template <typename Model>
double
initial_step_size(Model m, double t, std::vector<double> y, control ctl) {
  const double order = 5;
  const size_t n = m.n_variables();
  std::vector<double> f0(n);
  std::vector<double> f1(n);
  std::vector<double> y1(n);

  // Compute a first guess for explicit Euler as
  //   h = 0.01 * norm (y0) / norm (f0)
  // the increment for explicit euler is small compared to the solution
  m.rhs(t, y, f0);

  double norm_f = 0.0;
  double norm_y = 0.0;

  for (size_t i = 0; i < n; ++i) {
    const double sk = ctl.atol + ctl.rtol * std::abs(y[i]);
    norm_f += utils::square(f0[i] / sk);
    norm_y += utils::square(y[i] / sk);
  }
  // TODO what are these magic numbers
  double h = (norm_f <= 1e-10 || norm_y <= 1e-10) ?
             1e-6 : std::sqrt(norm_y / norm_f) * 0.01;
  h = std::min(h, ctl.step_size_max);

  // Perform an explicit Euler step
  for (size_t i = 0; i < n; ++i) {
    y1[i] = y[i] + h * f0[i];
  }
  m.rhs(t + h, y1, f1);

  // Estimate the second derivative of the solution:
  double der2 = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double sk = ctl.atol + ctl.rtol * std::abs(y[i]);
    der2 += utils::square((f1[i] - f0[i]) / sk);
  }
  der2 = std::sqrt(der2) / h;

  // Step size is computed such that
  //   h^order * max(norm(f0), norm(der2)) = 0.01
  const double der12 = std::max(std::abs(der2), std::sqrt(norm_f));
  const double h1 = (der12 <= 1e-15) ?
                    std::max(1e-6, std::abs(h) * 1e-3) :
                    std::pow(0.01 / der12, 1.0 / order);
  h = std::min(std::min(100 * std::abs(h), h1), ctl.step_size_max);
  return h;
}

}
}

#endif
