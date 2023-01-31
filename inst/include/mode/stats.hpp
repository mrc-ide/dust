#pragma once
namespace mode {

struct stats {
  size_t n_steps;
  size_t n_steps_accepted;
  size_t n_steps_rejected;
  std::vector<double> step_times;

  void reset() {
    n_steps = 0;
    n_steps_accepted = 0;
    n_steps_rejected = 0;
    step_times.resize(0);
  }
};

}
