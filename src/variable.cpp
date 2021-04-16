// Generated by dust (version 0.9.0) - do not edit
#include <cpp11.hpp>
[[cpp11::register]]
SEXP dust_variable_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, cpp11::sexp device_id);

[[cpp11::register]]
SEXP dust_variable_run(SEXP ptr, size_t step_end, bool device);

[[cpp11::register]]
SEXP dust_variable_simulate(SEXP ptr, cpp11::sexp step_end);

[[cpp11::register]]
SEXP dust_variable_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_variable_set_state(SEXP ptr, SEXP r_state, SEXP r_step);

[[cpp11::register]]
SEXP dust_variable_reset(SEXP ptr, cpp11::list r_pars, size_t step);

[[cpp11::register]]
SEXP dust_variable_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
size_t dust_variable_step(SEXP ptr);

[[cpp11::register]]
void dust_variable_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_variable_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_variable_set_pars(SEXP ptr, cpp11::list r_pars);

[[cpp11::register]]
SEXP dust_variable_rng_state(SEXP ptr, bool last_only);

[[cpp11::register]]
SEXP dust_variable_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_variable_set_data(SEXP ptr, cpp11::list data);

[[cpp11::register]]
SEXP dust_variable_compare_data(SEXP ptr, bool device);

[[cpp11::register]]
SEXP dust_variable_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot,
                          bool device);

[[cpp11::register]]
cpp11::sexp dust_variable_capabilities();

[[cpp11::register]]
void dust_variable_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_variable_n_state(SEXP ptr);

[[cpp11::register]]
cpp11::sexp dust_variable_device_info();

#include <dust/dust.hpp>
#include <dust/interface.hpp>

class variable {
public:
  typedef double real_t;
  typedef dust::no_data data_t;
  typedef dust::no_internal internal_t;

  struct shared_t {
    size_t len;
    real_t mean;
    real_t sd;
  };

  variable(const dust::pars_t<variable>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return shared->len;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> ret;
    for (size_t i = 0; i < shared->len; ++i) {
      ret.push_back(i + 1);
    }
    return ret;
  }

  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state,
              real_t * state_next) {
    for (size_t i = 0; i < shared->len; ++i) {
      state_next[i] =
        dust::distr::rnorm(rng_state, state[i] + shared->mean, shared->sd);
    }
  }

private:
  dust::shared_ptr<variable> shared;
};

namespace dust {
template <>
struct has_gpu_support<variable> : std::true_type {};

template <>
size_t device_shared_int_size<variable>(dust::shared_ptr<variable> shared) {
  return 1;
}

template <>
size_t device_shared_real_size<variable>(dust::shared_ptr<variable> shared) {
  return 2;
}

template <>
void device_shared_copy<variable>(dust::shared_ptr<variable> shared,
                                  int * shared_int,
                                  variable::real_t * shared_real) {
  typedef variable::real_t real_t;
  shared_int = dust::shared_copy<int>(shared_int, shared->len);
  shared_real = dust::shared_copy<real_t>(shared_real, shared->mean);
  shared_real = dust::shared_copy<real_t>(shared_real, shared->sd);
}

template <>
DEVICE
void update_device<variable>(size_t step,
                             const dust::interleaved<variable::real_t> state,
                             dust::interleaved<int> internal_int,
                             dust::interleaved<variable::real_t> internal_real,
                             const int * shared_int,
                             const variable::real_t * shared_real,
                             dust::rng_state_t<variable::real_t>& rng_state,
                             dust::interleaved<variable::real_t> state_next) {
  typedef variable::real_t real_t;
  const size_t len = shared_int[0];
  const real_t mean = shared_real[0];
  const real_t sd = shared_real[1];
  for (size_t i = 0; i < len; ++i) {
    state_next[i] = dust::distr::rnorm(rng_state, state[i] + mean, sd);
  }
}

template <>
dust::pars_t<variable> dust_pars<variable>(cpp11::list pars) {
  typedef variable::real_t real_t;
  const size_t len = cpp11::as_cpp<int>(pars["len"]);
  real_t mean = 0, sd = 1;

  SEXP r_mean = pars["mean"];
  if (r_mean != R_NilValue) {
    mean = cpp11::as_cpp<real_t>(r_mean);
  }

  SEXP r_sd = pars["sd"];
  if (r_sd != R_NilValue) {
    sd = cpp11::as_cpp<real_t>(r_sd);
  }

  variable::shared_t shared{len, mean, sd};
  return dust::pars_t<variable>(shared);
}

}

SEXP dust_variable_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, cpp11::sexp device_id) {
  return dust::r::dust_alloc<variable>(r_pars, pars_multi, step, r_n_particles,
                                        n_threads, r_seed, device_id);
}

SEXP dust_variable_run(SEXP ptr, size_t step_end, bool device) {
  return dust::r::dust_run<variable>(ptr, step_end, device);
}

SEXP dust_variable_simulate(SEXP ptr, cpp11::sexp step_end) {
  return dust::r::dust_simulate<variable>(ptr, step_end);
}

SEXP dust_variable_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<variable>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_variable_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  dust::r::dust_set_state<variable>(ptr, r_state, r_step);
  return R_NilValue;
}

SEXP dust_variable_reset(SEXP ptr, cpp11::list r_pars, size_t step) {
  return dust::r::dust_reset<variable>(ptr, r_pars, step);
}

SEXP dust_variable_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<variable>(ptr, r_index);
}

size_t dust_variable_step(SEXP ptr) {
  return dust::r::dust_step<variable>(ptr);
}

void dust_variable_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<variable>(ptr, r_index);
}

SEXP dust_variable_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<variable>(ptr, r_weights);
}

SEXP dust_variable_set_pars(SEXP ptr, cpp11::list r_pars) {
  return dust::r::dust_set_pars<variable>(ptr, r_pars);
}

SEXP dust_variable_rng_state(SEXP ptr, bool last_only) {
  return dust::r::dust_rng_state<variable>(ptr, last_only);
}

SEXP dust_variable_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<variable>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_variable_set_data(SEXP ptr, cpp11::list data) {
  dust::r::dust_set_data<variable>(ptr, data);
  return R_NilValue;
}

SEXP dust_variable_compare_data(SEXP ptr, bool device) {
  return dust::r::dust_compare_data<variable>(ptr, device);
}

SEXP dust_variable_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot,
                          bool device) {
  return dust::r::dust_filter<variable>(ptr, save_trajectories, step_snapshot, device);
}

cpp11::sexp dust_variable_capabilities() {
  return dust::r::dust_capabilities<variable>();
}

void dust_variable_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<variable>(ptr, n_threads);
}

int dust_variable_n_state(SEXP ptr) {
  return dust::r::dust_n_state<variable>(ptr);
}

cpp11::sexp dust_variable_device_info() {
  return dust::cuda::device_info<variable>();
}
