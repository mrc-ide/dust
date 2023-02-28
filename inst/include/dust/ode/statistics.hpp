#ifndef DUST_ODE_STATISTICS_HPP
#define DUST_ODE_STATISTICS_HPP

#include <cstddef>
#include <vector>

namespace dust {
namespace ode {

template <typename real_type>
struct statistics {
  size_t n_steps;
  size_t n_steps_accepted;
  size_t n_steps_rejected;
  std::vector<real_type> step_times;

  void reset() {
    n_steps = 0;
    n_steps_accepted = 0;
    n_steps_rejected = 0;
    step_times.resize(0);
  }
};

}
}

#endif
