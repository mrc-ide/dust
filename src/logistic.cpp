// Generated by dust (version 0.14.8) - do not edit
#include <cpp11.hpp>

[[cpp11::register]]
cpp11::sexp dust_logistic_gpu_info();
[[cpp11::register]]
SEXP dust_ode_logistic_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time,
                         cpp11::sexp r_n_particles, int n_threads,
                         cpp11::sexp r_seed, bool deterministic,
                         cpp11::sexp gpu_config, cpp11::sexp ode_control);

[[cpp11::register]]
cpp11::sexp dust_ode_logistic_capabilities();

[[cpp11::register]]
SEXP dust_ode_logistic_run(SEXP ptr, cpp11::sexp r_time_end);

[[cpp11::register]]
SEXP dust_ode_logistic_simulate(SEXP ptr, cpp11::sexp time_end);

[[cpp11::register]]
SEXP dust_ode_logistic_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_ode_logistic_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                           SEXP r_time, SEXP r_set_initial_state,
                                           SEXP index, SEXP reset_step_size);

[[cpp11::register]]
SEXP dust_ode_logistic_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
SEXP dust_ode_logistic_time(SEXP ptr);

[[cpp11::register]]
void dust_ode_logistic_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_ode_logistic_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_ode_logistic_rng_state(SEXP ptr, bool first_only, bool last_only);

[[cpp11::register]]
SEXP dust_ode_logistic_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_ode_logistic_set_data(SEXP ptr, cpp11::list data, bool shared);

[[cpp11::register]]
SEXP dust_ode_logistic_compare_data(SEXP ptr);

[[cpp11::register]]
SEXP dust_ode_logistic_filter(SEXP ptr, SEXP time_end,
                                     bool save_trajectories,
                                     cpp11::sexp time_snapshot,
                                     cpp11::sexp min_log_likelihood);

[[cpp11::register]]
void dust_ode_logistic_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_ode_logistic_n_state(SEXP ptr);

[[cpp11::register]]
void dust_ode_logistic_set_stochastic_schedule(SEXP ptr, SEXP time);

[[cpp11::register]]
SEXP dust_ode_logistic_ode_statistics(SEXP ptr);
#include <dust/r/dust.hpp>

// [[dust::time_type(continuous)]]
class logistic {
public:
  using real_type = double;
  using data_type = dust::no_data;
  using internal_type = dust::no_internal;
  using rng_state_type = dust::random::generator<real_type>;

  struct shared_type {
    size_t n;
    std::vector<real_type> r;
    std::vector<real_type> K;
    real_type v;
    bool random_initial;
  };

  logistic(const dust::pars_type<logistic>& pars): shared(pars.shared) {
  }

  void rhs(real_type t,
           const std::vector<real_type>& y,
           std::vector<real_type>& dydt) const {
    for (size_t i = 0; i < shared->n; ++i) {
      dydt[i] = shared->r[i] * y[i] * (1 - y[i] / shared->K[i]);
    }
  }

  void output(real_type t,
              const std::vector<real_type>& y,
              std::vector<real_type>& output) {
    real_type tot = 0;
    for (size_t i = 0; i < shared->n; ++i) {
      tot += y[i];
    }
    output[0] = tot;
  }

  std::vector<real_type> initial(real_type time, rng_state_type& rng_state) {
    std::vector<real_type> y(shared->n, 1);
    if (shared->random_initial) {
      for (size_t i = 0; i < shared->n; ++i) {
        y[i] *= std::exp(dust::random::random_normal<real_type>(rng_state));
      }
    }
    return y;
  }

  void update_stochastic(real_type t, const std::vector<real_type>& y,
                         rng_state_type& rng_state,
                         std::vector<real_type>& y_next) {
    for (size_t i = 0; i < shared->n; ++i) {
      const auto r = dust::random::normal<real_type>(rng_state, 0, shared->v);
      y_next[i] = y[i] * std::exp(r);
    }
  }

  size_t n_variables() const {
    return shared->n;
  }

  size_t n_output() const {
    return 1;
  }

private:
  dust::shared_ptr<logistic> shared;
};

template <typename real_type>
std::vector<real_type> user_vector(const char * name, cpp11::list pars) {
  auto value = cpp11::as_cpp<cpp11::doubles>(pars[name]);
  return std::vector<real_type>(value.begin(), value.end());
}

namespace dust {

template <>
dust::pars_type<logistic> dust_pars<logistic>(cpp11::list pars) {
  using real_type = logistic::real_type;
  // [[dust::param(r, required = TRUE)]]
  const auto r = user_vector<real_type>("r", pars);
  // [[dust::param(K, required = TRUE)]]
  const auto K = user_vector<real_type>("K", pars);
  const size_t n = r.size();
  if (n == 0) {
    cpp11::stop("'r' and 'K' must have length of at least 1");
  }
  // [[dust::param(v, required = FALSE)]]
  cpp11::sexp r_v = pars["v"];
  // [[dust::param(random_initial, required = FALSE)]]
  const bool random_initial = pars["random_initial"] == R_NilValue ? false :
    cpp11::as_cpp<bool>(pars["random_initial"]);
  const real_type v = r_v == R_NilValue ? 0.1 : cpp11::as_cpp<real_type>(r_v);
  logistic::shared_type shared{n, r, K, v, random_initial};
  return dust::pars_type<logistic>(shared);
}

template <>
cpp11::sexp dust_info<logistic>(const dust::pars_type<logistic>& pars) {
  using namespace cpp11::literals;
  return cpp11::writable::list({"n"_nm = pars.shared->n});
}

}

cpp11::sexp dust_logistic_gpu_info() {
  return dust::gpu::r::gpu_info();
}
using model_ode = dust::dust_ode<logistic>;

cpp11::sexp dust_ode_logistic_capabilities() {
  return dust::r::dust_capabilities<model_ode>();
}

SEXP dust_ode_logistic_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time,
                             cpp11::sexp r_n_particles, int n_threads,
                             cpp11::sexp r_seed, bool deterministic,
                             cpp11::sexp gpu_config, cpp11::sexp ode_control) {
  return dust::r::dust_ode_alloc<logistic>(r_pars, pars_multi, r_time, r_n_particles,
                                        n_threads, r_seed, deterministic,
                                        gpu_config, ode_control);
}

SEXP dust_ode_logistic_run(SEXP ptr, cpp11::sexp r_time_end) {
  return dust::r::dust_run<model_ode>(ptr, r_time_end);
}

SEXP dust_ode_logistic_simulate(SEXP ptr, cpp11::sexp r_time_end) {
  return dust::r::dust_simulate<model_ode>(ptr, r_time_end);
}

SEXP dust_ode_logistic_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<model_ode>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_ode_logistic_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                           SEXP r_time, SEXP r_set_initial_state, SEXP index, SEXP reset_step_size) {
  return dust::r::dust_update_state<model_ode>(ptr, r_pars, r_state, r_time,
                                                      r_set_initial_state, index, reset_step_size);
}

SEXP dust_ode_logistic_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<model_ode>(ptr, r_index);
}

SEXP dust_ode_logistic_time(SEXP ptr) {
  return dust::r::dust_time<model_ode>(ptr);
}

void dust_ode_logistic_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<model_ode>(ptr, r_index);
}

SEXP dust_ode_logistic_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<model_ode>(ptr, r_weights);
}

SEXP dust_ode_logistic_rng_state(SEXP ptr, bool first_only, bool last_only) {
  return dust::r::dust_rng_state<model_ode>(ptr, first_only, last_only);
}

SEXP dust_ode_logistic_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<model_ode>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_ode_logistic_set_data(SEXP ptr, cpp11::list data,
                                       bool shared) {
  dust::r::dust_set_data<model_ode>(ptr, data, shared);
  return R_NilValue;
}

SEXP dust_ode_logistic_compare_data(SEXP ptr) {
  return dust::r::dust_compare_data<model_ode>(ptr);
}

SEXP dust_ode_logistic_filter(SEXP ptr, SEXP time_end,
                                     bool save_trajectories,
                                     cpp11::sexp time_snapshot,
                                     cpp11::sexp min_log_likelihood) {
  return dust::r::dust_filter<model_ode>(ptr, time_end,
                                                save_trajectories,
                                                time_snapshot,
                                                min_log_likelihood);
}

void dust_ode_logistic_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<model_ode>(ptr, n_threads);
}

int dust_ode_logistic_n_state(SEXP ptr) {
  return dust::r::dust_n_state<model_ode>(ptr);
}

void dust_ode_logistic_set_stochastic_schedule(SEXP ptr, SEXP time) {
  dust::r::dust_set_stochastic_schedule<model_ode>(ptr, time);
}

SEXP dust_ode_logistic_ode_statistics(SEXP ptr) {
  return dust::r::dust_ode_statistics<model_ode>(ptr);
}
