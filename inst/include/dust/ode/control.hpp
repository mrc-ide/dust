#ifndef DUST_ODE_CONTROL_HPP
#define DUST_ODE_CONTROL_HPP

#include <cstddef>
#include <limits>

namespace dust {
namespace ode {

template <typename real_type>
struct control {
  // TODO: I've had to un-const these for a bit
  size_t max_steps = 10000;
  real_type atol = 1e-6;
  real_type rtol = 1e-6;
  real_type step_size_min = 1e-8;
  real_type step_size_max = std::numeric_limits<real_type>::infinity();
  real_type factor_safe = 0.9;
  real_type factor_min = 0.2;  // from dopri5.f:276, retard.f:328
  real_type factor_max = 10.0; // from dopri5.f:281, retard.f:333
  real_type beta = 0.04;
  real_type constant = 0.2 - 0.04 * 0.75; // 0.04 is beta
  bool debug_record_step_times = false;

  control(size_t max_steps, real_type atol, real_type rtol,
          real_type step_size_min, real_type step_size_max,
          bool debug_record_step_times) :
      max_steps(max_steps), atol(atol), rtol(rtol),
      step_size_min(step_size_min),
      step_size_max(step_size_max),
      debug_record_step_times(debug_record_step_times) {}

  control() {}
};

}
}

#endif
