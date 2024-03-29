# Generated by cpp11: do not edit by hand

density_binomial <- function(x, size, prob, log) {
  .Call(`_dust_density_binomial`, x, size, prob, log)
}

density_normal <- function(x, mu, sd, log) {
  .Call(`_dust_density_normal`, x, mu, sd, log)
}

density_negative_binomial_mu <- function(x, size, mu, log, is_float) {
  .Call(`_dust_density_negative_binomial_mu`, x, size, mu, log, is_float)
}

density_negative_binomial_prob <- function(x, size, prob, log) {
  .Call(`_dust_density_negative_binomial_prob`, x, size, prob, log)
}

density_beta_binomial <- function(x, size, prob, rho, log) {
  .Call(`_dust_density_beta_binomial`, x, size, prob, rho, log)
}

density_poisson <- function(x, lambda, log) {
  .Call(`_dust_density_poisson`, x, lambda, log)
}

dust_rng_alloc <- function(r_seed, n_streams, deterministic, is_float) {
  .Call(`_dust_dust_rng_alloc`, r_seed, n_streams, deterministic, is_float)
}

dust_rng_jump <- function(ptr, is_float) {
  invisible(.Call(`_dust_dust_rng_jump`, ptr, is_float))
}

dust_rng_long_jump <- function(ptr, is_float) {
  invisible(.Call(`_dust_dust_rng_long_jump`, ptr, is_float))
}

dust_rng_random_real <- function(ptr, n, n_threads, is_float) {
  .Call(`_dust_dust_rng_random_real`, ptr, n, n_threads, is_float)
}

dust_rng_random_normal <- function(ptr, n, n_threads, algorithm, is_float) {
  .Call(`_dust_dust_rng_random_normal`, ptr, n, n_threads, algorithm, is_float)
}

dust_rng_uniform <- function(ptr, n, r_min, r_max, n_threads, is_float) {
  .Call(`_dust_dust_rng_uniform`, ptr, n, r_min, r_max, n_threads, is_float)
}

dust_rng_exponential <- function(ptr, n, r_rate, n_threads, is_float) {
  .Call(`_dust_dust_rng_exponential`, ptr, n, r_rate, n_threads, is_float)
}

dust_rng_normal <- function(ptr, n, r_mean, r_sd, n_threads, algorithm, is_float) {
  .Call(`_dust_dust_rng_normal`, ptr, n, r_mean, r_sd, n_threads, algorithm, is_float)
}

dust_rng_binomial <- function(ptr, n, r_size, r_prob, n_threads, is_float) {
  .Call(`_dust_dust_rng_binomial`, ptr, n, r_size, r_prob, n_threads, is_float)
}

dust_rng_nbinomial <- function(ptr, n, r_size, r_prob, n_threads, is_float) {
  .Call(`_dust_dust_rng_nbinomial`, ptr, n, r_size, r_prob, n_threads, is_float)
}

dust_rng_hypergeometric <- function(ptr, n, r_n1, r_n2, r_k, n_threads, is_float) {
  .Call(`_dust_dust_rng_hypergeometric`, ptr, n, r_n1, r_n2, r_k, n_threads, is_float)
}

dust_rng_gamma <- function(ptr, n, r_a, r_b, n_threads, is_float) {
  .Call(`_dust_dust_rng_gamma`, ptr, n, r_a, r_b, n_threads, is_float)
}

dust_rng_poisson <- function(ptr, n, r_lambda, n_threads, is_float) {
  .Call(`_dust_dust_rng_poisson`, ptr, n, r_lambda, n_threads, is_float)
}

dust_rng_cauchy <- function(ptr, n, r_location, r_scale, n_threads, is_float) {
  .Call(`_dust_dust_rng_cauchy`, ptr, n, r_location, r_scale, n_threads, is_float)
}

dust_rng_multinomial <- function(ptr, n, r_size, r_prob, n_threads, is_float) {
  .Call(`_dust_dust_rng_multinomial`, ptr, n, r_size, r_prob, n_threads, is_float)
}

dust_rng_state <- function(ptr, is_float) {
  .Call(`_dust_dust_rng_state`, ptr, is_float)
}

dust_rng_pointer_init <- function(n_streams, seed, long_jump, algorithm) {
  .Call(`_dust_dust_rng_pointer_init`, n_streams, seed, long_jump, algorithm)
}

dust_rng_pointer_sync <- function(obj, algorithm) {
  invisible(.Call(`_dust_dust_rng_pointer_sync`, obj, algorithm))
}

test_rng_pointer_get <- function(obj, n_streams) {
  .Call(`_dust_test_rng_pointer_get`, obj, n_streams)
}

dust_logistic_gpu_info <- function() {
  .Call(`_dust_dust_logistic_gpu_info`)
}

dust_ode_logistic_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
  .Call(`_dust_dust_ode_logistic_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
}

dust_ode_logistic_capabilities <- function() {
  .Call(`_dust_dust_ode_logistic_capabilities`)
}

dust_ode_logistic_run <- function(ptr, r_time_end) {
  .Call(`_dust_dust_ode_logistic_run`, ptr, r_time_end)
}

dust_ode_logistic_simulate <- function(ptr, time_end) {
  .Call(`_dust_dust_ode_logistic_simulate`, ptr, time_end)
}

dust_ode_logistic_run_adjoint <- function(ptr) {
  .Call(`_dust_dust_ode_logistic_run_adjoint`, ptr)
}

dust_ode_logistic_set_index <- function(ptr, r_index) {
  .Call(`_dust_dust_ode_logistic_set_index`, ptr, r_index)
}

dust_ode_logistic_update_state <- function(ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size) {
  .Call(`_dust_dust_ode_logistic_update_state`, ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size)
}

dust_ode_logistic_state <- function(ptr, r_index) {
  .Call(`_dust_dust_ode_logistic_state`, ptr, r_index)
}

dust_ode_logistic_time <- function(ptr) {
  .Call(`_dust_dust_ode_logistic_time`, ptr)
}

dust_ode_logistic_reorder <- function(ptr, r_index) {
  invisible(.Call(`_dust_dust_ode_logistic_reorder`, ptr, r_index))
}

dust_ode_logistic_resample <- function(ptr, r_weights) {
  .Call(`_dust_dust_ode_logistic_resample`, ptr, r_weights)
}

dust_ode_logistic_rng_state <- function(ptr, first_only, last_only) {
  .Call(`_dust_dust_ode_logistic_rng_state`, ptr, first_only, last_only)
}

dust_ode_logistic_set_rng_state <- function(ptr, rng_state) {
  .Call(`_dust_dust_ode_logistic_set_rng_state`, ptr, rng_state)
}

dust_ode_logistic_set_data <- function(ptr, data, shared) {
  .Call(`_dust_dust_ode_logistic_set_data`, ptr, data, shared)
}

dust_ode_logistic_compare_data <- function(ptr) {
  .Call(`_dust_dust_ode_logistic_compare_data`, ptr)
}

dust_ode_logistic_filter <- function(ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood) {
  .Call(`_dust_dust_ode_logistic_filter`, ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood)
}

dust_ode_logistic_set_n_threads <- function(ptr, n_threads) {
  invisible(.Call(`_dust_dust_ode_logistic_set_n_threads`, ptr, n_threads))
}

dust_ode_logistic_n_state <- function(ptr) {
  .Call(`_dust_dust_ode_logistic_n_state`, ptr)
}

dust_ode_logistic_set_stochastic_schedule <- function(ptr, time) {
  invisible(.Call(`_dust_dust_ode_logistic_set_stochastic_schedule`, ptr, time))
}

dust_ode_logistic_ode_statistics <- function(ptr) {
  .Call(`_dust_dust_ode_logistic_ode_statistics`, ptr)
}

cpp_openmp_info <- function() {
  .Call(`_dust_cpp_openmp_info`)
}

dust_sir_gpu_info <- function() {
  .Call(`_dust_dust_sir_gpu_info`)
}

dust_cpu_sir_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
  .Call(`_dust_dust_cpu_sir_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
}

dust_cpu_sir_capabilities <- function() {
  .Call(`_dust_dust_cpu_sir_capabilities`)
}

dust_cpu_sir_run <- function(ptr, r_time_end) {
  .Call(`_dust_dust_cpu_sir_run`, ptr, r_time_end)
}

dust_cpu_sir_simulate <- function(ptr, time_end) {
  .Call(`_dust_dust_cpu_sir_simulate`, ptr, time_end)
}

dust_cpu_sir_run_adjoint <- function(ptr) {
  .Call(`_dust_dust_cpu_sir_run_adjoint`, ptr)
}

dust_cpu_sir_set_index <- function(ptr, r_index) {
  .Call(`_dust_dust_cpu_sir_set_index`, ptr, r_index)
}

dust_cpu_sir_update_state <- function(ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size) {
  .Call(`_dust_dust_cpu_sir_update_state`, ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size)
}

dust_cpu_sir_state <- function(ptr, r_index) {
  .Call(`_dust_dust_cpu_sir_state`, ptr, r_index)
}

dust_cpu_sir_time <- function(ptr) {
  .Call(`_dust_dust_cpu_sir_time`, ptr)
}

dust_cpu_sir_reorder <- function(ptr, r_index) {
  invisible(.Call(`_dust_dust_cpu_sir_reorder`, ptr, r_index))
}

dust_cpu_sir_resample <- function(ptr, r_weights) {
  .Call(`_dust_dust_cpu_sir_resample`, ptr, r_weights)
}

dust_cpu_sir_rng_state <- function(ptr, first_only, last_only) {
  .Call(`_dust_dust_cpu_sir_rng_state`, ptr, first_only, last_only)
}

dust_cpu_sir_set_rng_state <- function(ptr, rng_state) {
  .Call(`_dust_dust_cpu_sir_set_rng_state`, ptr, rng_state)
}

dust_cpu_sir_set_data <- function(ptr, data, shared) {
  .Call(`_dust_dust_cpu_sir_set_data`, ptr, data, shared)
}

dust_cpu_sir_compare_data <- function(ptr) {
  .Call(`_dust_dust_cpu_sir_compare_data`, ptr)
}

dust_cpu_sir_filter <- function(ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood) {
  .Call(`_dust_dust_cpu_sir_filter`, ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood)
}

dust_cpu_sir_set_n_threads <- function(ptr, n_threads) {
  invisible(.Call(`_dust_dust_cpu_sir_set_n_threads`, ptr, n_threads))
}

dust_cpu_sir_n_state <- function(ptr) {
  .Call(`_dust_dust_cpu_sir_n_state`, ptr)
}

dust_cpu_sir_set_stochastic_schedule <- function(ptr, time) {
  invisible(.Call(`_dust_dust_cpu_sir_set_stochastic_schedule`, ptr, time))
}

dust_cpu_sir_ode_statistics <- function(ptr) {
  .Call(`_dust_dust_cpu_sir_ode_statistics`, ptr)
}

dust_sirs_gpu_info <- function() {
  .Call(`_dust_dust_sirs_gpu_info`)
}

dust_cpu_sirs_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
  .Call(`_dust_dust_cpu_sirs_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
}

dust_cpu_sirs_capabilities <- function() {
  .Call(`_dust_dust_cpu_sirs_capabilities`)
}

dust_cpu_sirs_run <- function(ptr, r_time_end) {
  .Call(`_dust_dust_cpu_sirs_run`, ptr, r_time_end)
}

dust_cpu_sirs_simulate <- function(ptr, time_end) {
  .Call(`_dust_dust_cpu_sirs_simulate`, ptr, time_end)
}

dust_cpu_sirs_run_adjoint <- function(ptr) {
  .Call(`_dust_dust_cpu_sirs_run_adjoint`, ptr)
}

dust_cpu_sirs_set_index <- function(ptr, r_index) {
  .Call(`_dust_dust_cpu_sirs_set_index`, ptr, r_index)
}

dust_cpu_sirs_update_state <- function(ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size) {
  .Call(`_dust_dust_cpu_sirs_update_state`, ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size)
}

dust_cpu_sirs_state <- function(ptr, r_index) {
  .Call(`_dust_dust_cpu_sirs_state`, ptr, r_index)
}

dust_cpu_sirs_time <- function(ptr) {
  .Call(`_dust_dust_cpu_sirs_time`, ptr)
}

dust_cpu_sirs_reorder <- function(ptr, r_index) {
  invisible(.Call(`_dust_dust_cpu_sirs_reorder`, ptr, r_index))
}

dust_cpu_sirs_resample <- function(ptr, r_weights) {
  .Call(`_dust_dust_cpu_sirs_resample`, ptr, r_weights)
}

dust_cpu_sirs_rng_state <- function(ptr, first_only, last_only) {
  .Call(`_dust_dust_cpu_sirs_rng_state`, ptr, first_only, last_only)
}

dust_cpu_sirs_set_rng_state <- function(ptr, rng_state) {
  .Call(`_dust_dust_cpu_sirs_set_rng_state`, ptr, rng_state)
}

dust_cpu_sirs_set_data <- function(ptr, data, shared) {
  .Call(`_dust_dust_cpu_sirs_set_data`, ptr, data, shared)
}

dust_cpu_sirs_compare_data <- function(ptr) {
  .Call(`_dust_dust_cpu_sirs_compare_data`, ptr)
}

dust_cpu_sirs_filter <- function(ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood) {
  .Call(`_dust_dust_cpu_sirs_filter`, ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood)
}

dust_cpu_sirs_set_n_threads <- function(ptr, n_threads) {
  invisible(.Call(`_dust_dust_cpu_sirs_set_n_threads`, ptr, n_threads))
}

dust_cpu_sirs_n_state <- function(ptr) {
  .Call(`_dust_dust_cpu_sirs_n_state`, ptr)
}

dust_cpu_sirs_set_stochastic_schedule <- function(ptr, time) {
  invisible(.Call(`_dust_dust_cpu_sirs_set_stochastic_schedule`, ptr, time))
}

dust_cpu_sirs_ode_statistics <- function(ptr) {
  .Call(`_dust_dust_cpu_sirs_ode_statistics`, ptr)
}

dust_gpu_sirs_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
  .Call(`_dust_dust_gpu_sirs_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
}

dust_gpu_sirs_capabilities <- function() {
  .Call(`_dust_dust_gpu_sirs_capabilities`)
}

dust_gpu_sirs_run <- function(ptr, r_time_end) {
  .Call(`_dust_dust_gpu_sirs_run`, ptr, r_time_end)
}

dust_gpu_sirs_simulate <- function(ptr, time_end) {
  .Call(`_dust_dust_gpu_sirs_simulate`, ptr, time_end)
}

dust_gpu_sirs_run_adjoint <- function(ptr) {
  .Call(`_dust_dust_gpu_sirs_run_adjoint`, ptr)
}

dust_gpu_sirs_set_index <- function(ptr, r_index) {
  .Call(`_dust_dust_gpu_sirs_set_index`, ptr, r_index)
}

dust_gpu_sirs_update_state <- function(ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size) {
  .Call(`_dust_dust_gpu_sirs_update_state`, ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size)
}

dust_gpu_sirs_state <- function(ptr, r_index) {
  .Call(`_dust_dust_gpu_sirs_state`, ptr, r_index)
}

dust_gpu_sirs_time <- function(ptr) {
  .Call(`_dust_dust_gpu_sirs_time`, ptr)
}

dust_gpu_sirs_reorder <- function(ptr, r_index) {
  invisible(.Call(`_dust_dust_gpu_sirs_reorder`, ptr, r_index))
}

dust_gpu_sirs_resample <- function(ptr, r_weights) {
  .Call(`_dust_dust_gpu_sirs_resample`, ptr, r_weights)
}

dust_gpu_sirs_rng_state <- function(ptr, first_only, last_only) {
  .Call(`_dust_dust_gpu_sirs_rng_state`, ptr, first_only, last_only)
}

dust_gpu_sirs_set_rng_state <- function(ptr, rng_state) {
  .Call(`_dust_dust_gpu_sirs_set_rng_state`, ptr, rng_state)
}

dust_gpu_sirs_set_data <- function(ptr, data, shared) {
  .Call(`_dust_dust_gpu_sirs_set_data`, ptr, data, shared)
}

dust_gpu_sirs_compare_data <- function(ptr) {
  .Call(`_dust_dust_gpu_sirs_compare_data`, ptr)
}

dust_gpu_sirs_filter <- function(ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood) {
  .Call(`_dust_dust_gpu_sirs_filter`, ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood)
}

dust_gpu_sirs_set_n_threads <- function(ptr, n_threads) {
  invisible(.Call(`_dust_dust_gpu_sirs_set_n_threads`, ptr, n_threads))
}

dust_gpu_sirs_n_state <- function(ptr) {
  .Call(`_dust_dust_gpu_sirs_n_state`, ptr)
}

dust_gpu_sirs_set_stochastic_schedule <- function(ptr, time) {
  invisible(.Call(`_dust_dust_gpu_sirs_set_stochastic_schedule`, ptr, time))
}

dust_gpu_sirs_ode_statistics <- function(ptr) {
  .Call(`_dust_dust_gpu_sirs_ode_statistics`, ptr)
}

test_cuda_pars <- function(r_gpu_config, n_particles, n_particles_each, n_state, n_state_full, n_shared_int, n_shared_real, data_size, shared_size) {
  .Call(`_dust_test_cuda_pars`, r_gpu_config, n_particles, n_particles_each, n_state, n_state_full, n_shared_int, n_shared_real, data_size, shared_size)
}

test_interpolate_search <- function(target, x) {
  .Call(`_dust_test_interpolate_search`, target, x)
}

test_interpolate_constant1 <- function(t, y, z) {
  .Call(`_dust_test_interpolate_constant1`, t, y, z)
}

test_interpolate_linear1 <- function(t, y, z) {
  .Call(`_dust_test_interpolate_linear1`, t, y, z)
}

test_interpolate_spline1 <- function(t, y, z) {
  .Call(`_dust_test_interpolate_spline1`, t, y, z)
}

test_xoshiro_run <- function(obj) {
  .Call(`_dust_test_xoshiro_run`, obj)
}

cpp_scale_log_weights <- function(w) {
  .Call(`_dust_cpp_scale_log_weights`, w)
}

dust_variable_gpu_info <- function() {
  .Call(`_dust_dust_variable_gpu_info`)
}

dust_cpu_variable_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
  .Call(`_dust_dust_cpu_variable_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
}

dust_cpu_variable_capabilities <- function() {
  .Call(`_dust_dust_cpu_variable_capabilities`)
}

dust_cpu_variable_run <- function(ptr, r_time_end) {
  .Call(`_dust_dust_cpu_variable_run`, ptr, r_time_end)
}

dust_cpu_variable_simulate <- function(ptr, time_end) {
  .Call(`_dust_dust_cpu_variable_simulate`, ptr, time_end)
}

dust_cpu_variable_run_adjoint <- function(ptr) {
  .Call(`_dust_dust_cpu_variable_run_adjoint`, ptr)
}

dust_cpu_variable_set_index <- function(ptr, r_index) {
  .Call(`_dust_dust_cpu_variable_set_index`, ptr, r_index)
}

dust_cpu_variable_update_state <- function(ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size) {
  .Call(`_dust_dust_cpu_variable_update_state`, ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size)
}

dust_cpu_variable_state <- function(ptr, r_index) {
  .Call(`_dust_dust_cpu_variable_state`, ptr, r_index)
}

dust_cpu_variable_time <- function(ptr) {
  .Call(`_dust_dust_cpu_variable_time`, ptr)
}

dust_cpu_variable_reorder <- function(ptr, r_index) {
  invisible(.Call(`_dust_dust_cpu_variable_reorder`, ptr, r_index))
}

dust_cpu_variable_resample <- function(ptr, r_weights) {
  .Call(`_dust_dust_cpu_variable_resample`, ptr, r_weights)
}

dust_cpu_variable_rng_state <- function(ptr, first_only, last_only) {
  .Call(`_dust_dust_cpu_variable_rng_state`, ptr, first_only, last_only)
}

dust_cpu_variable_set_rng_state <- function(ptr, rng_state) {
  .Call(`_dust_dust_cpu_variable_set_rng_state`, ptr, rng_state)
}

dust_cpu_variable_set_data <- function(ptr, data, shared) {
  .Call(`_dust_dust_cpu_variable_set_data`, ptr, data, shared)
}

dust_cpu_variable_compare_data <- function(ptr) {
  .Call(`_dust_dust_cpu_variable_compare_data`, ptr)
}

dust_cpu_variable_filter <- function(ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood) {
  .Call(`_dust_dust_cpu_variable_filter`, ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood)
}

dust_cpu_variable_set_n_threads <- function(ptr, n_threads) {
  invisible(.Call(`_dust_dust_cpu_variable_set_n_threads`, ptr, n_threads))
}

dust_cpu_variable_n_state <- function(ptr) {
  .Call(`_dust_dust_cpu_variable_n_state`, ptr)
}

dust_cpu_variable_set_stochastic_schedule <- function(ptr, time) {
  invisible(.Call(`_dust_dust_cpu_variable_set_stochastic_schedule`, ptr, time))
}

dust_cpu_variable_ode_statistics <- function(ptr) {
  .Call(`_dust_dust_cpu_variable_ode_statistics`, ptr)
}

dust_gpu_variable_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
  .Call(`_dust_dust_gpu_variable_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
}

dust_gpu_variable_capabilities <- function() {
  .Call(`_dust_dust_gpu_variable_capabilities`)
}

dust_gpu_variable_run <- function(ptr, r_time_end) {
  .Call(`_dust_dust_gpu_variable_run`, ptr, r_time_end)
}

dust_gpu_variable_simulate <- function(ptr, time_end) {
  .Call(`_dust_dust_gpu_variable_simulate`, ptr, time_end)
}

dust_gpu_variable_run_adjoint <- function(ptr) {
  .Call(`_dust_dust_gpu_variable_run_adjoint`, ptr)
}

dust_gpu_variable_set_index <- function(ptr, r_index) {
  .Call(`_dust_dust_gpu_variable_set_index`, ptr, r_index)
}

dust_gpu_variable_update_state <- function(ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size) {
  .Call(`_dust_dust_gpu_variable_update_state`, ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size)
}

dust_gpu_variable_state <- function(ptr, r_index) {
  .Call(`_dust_dust_gpu_variable_state`, ptr, r_index)
}

dust_gpu_variable_time <- function(ptr) {
  .Call(`_dust_dust_gpu_variable_time`, ptr)
}

dust_gpu_variable_reorder <- function(ptr, r_index) {
  invisible(.Call(`_dust_dust_gpu_variable_reorder`, ptr, r_index))
}

dust_gpu_variable_resample <- function(ptr, r_weights) {
  .Call(`_dust_dust_gpu_variable_resample`, ptr, r_weights)
}

dust_gpu_variable_rng_state <- function(ptr, first_only, last_only) {
  .Call(`_dust_dust_gpu_variable_rng_state`, ptr, first_only, last_only)
}

dust_gpu_variable_set_rng_state <- function(ptr, rng_state) {
  .Call(`_dust_dust_gpu_variable_set_rng_state`, ptr, rng_state)
}

dust_gpu_variable_set_data <- function(ptr, data, shared) {
  .Call(`_dust_dust_gpu_variable_set_data`, ptr, data, shared)
}

dust_gpu_variable_compare_data <- function(ptr) {
  .Call(`_dust_dust_gpu_variable_compare_data`, ptr)
}

dust_gpu_variable_filter <- function(ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood) {
  .Call(`_dust_dust_gpu_variable_filter`, ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood)
}

dust_gpu_variable_set_n_threads <- function(ptr, n_threads) {
  invisible(.Call(`_dust_dust_gpu_variable_set_n_threads`, ptr, n_threads))
}

dust_gpu_variable_n_state <- function(ptr) {
  .Call(`_dust_dust_gpu_variable_n_state`, ptr)
}

dust_gpu_variable_set_stochastic_schedule <- function(ptr, time) {
  invisible(.Call(`_dust_dust_gpu_variable_set_stochastic_schedule`, ptr, time))
}

dust_gpu_variable_ode_statistics <- function(ptr) {
  .Call(`_dust_dust_gpu_variable_ode_statistics`, ptr)
}

dust_volatility_gpu_info <- function() {
  .Call(`_dust_dust_volatility_gpu_info`)
}

dust_cpu_volatility_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
  .Call(`_dust_dust_cpu_volatility_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
}

dust_cpu_volatility_capabilities <- function() {
  .Call(`_dust_dust_cpu_volatility_capabilities`)
}

dust_cpu_volatility_run <- function(ptr, r_time_end) {
  .Call(`_dust_dust_cpu_volatility_run`, ptr, r_time_end)
}

dust_cpu_volatility_simulate <- function(ptr, time_end) {
  .Call(`_dust_dust_cpu_volatility_simulate`, ptr, time_end)
}

dust_cpu_volatility_run_adjoint <- function(ptr) {
  .Call(`_dust_dust_cpu_volatility_run_adjoint`, ptr)
}

dust_cpu_volatility_set_index <- function(ptr, r_index) {
  .Call(`_dust_dust_cpu_volatility_set_index`, ptr, r_index)
}

dust_cpu_volatility_update_state <- function(ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size) {
  .Call(`_dust_dust_cpu_volatility_update_state`, ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size)
}

dust_cpu_volatility_state <- function(ptr, r_index) {
  .Call(`_dust_dust_cpu_volatility_state`, ptr, r_index)
}

dust_cpu_volatility_time <- function(ptr) {
  .Call(`_dust_dust_cpu_volatility_time`, ptr)
}

dust_cpu_volatility_reorder <- function(ptr, r_index) {
  invisible(.Call(`_dust_dust_cpu_volatility_reorder`, ptr, r_index))
}

dust_cpu_volatility_resample <- function(ptr, r_weights) {
  .Call(`_dust_dust_cpu_volatility_resample`, ptr, r_weights)
}

dust_cpu_volatility_rng_state <- function(ptr, first_only, last_only) {
  .Call(`_dust_dust_cpu_volatility_rng_state`, ptr, first_only, last_only)
}

dust_cpu_volatility_set_rng_state <- function(ptr, rng_state) {
  .Call(`_dust_dust_cpu_volatility_set_rng_state`, ptr, rng_state)
}

dust_cpu_volatility_set_data <- function(ptr, data, shared) {
  .Call(`_dust_dust_cpu_volatility_set_data`, ptr, data, shared)
}

dust_cpu_volatility_compare_data <- function(ptr) {
  .Call(`_dust_dust_cpu_volatility_compare_data`, ptr)
}

dust_cpu_volatility_filter <- function(ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood) {
  .Call(`_dust_dust_cpu_volatility_filter`, ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood)
}

dust_cpu_volatility_set_n_threads <- function(ptr, n_threads) {
  invisible(.Call(`_dust_dust_cpu_volatility_set_n_threads`, ptr, n_threads))
}

dust_cpu_volatility_n_state <- function(ptr) {
  .Call(`_dust_dust_cpu_volatility_n_state`, ptr)
}

dust_cpu_volatility_set_stochastic_schedule <- function(ptr, time) {
  invisible(.Call(`_dust_dust_cpu_volatility_set_stochastic_schedule`, ptr, time))
}

dust_cpu_volatility_ode_statistics <- function(ptr) {
  .Call(`_dust_dust_cpu_volatility_ode_statistics`, ptr)
}

dust_walk_gpu_info <- function() {
  .Call(`_dust_dust_walk_gpu_info`)
}

dust_cpu_walk_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
  .Call(`_dust_dust_cpu_walk_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
}

dust_cpu_walk_capabilities <- function() {
  .Call(`_dust_dust_cpu_walk_capabilities`)
}

dust_cpu_walk_run <- function(ptr, r_time_end) {
  .Call(`_dust_dust_cpu_walk_run`, ptr, r_time_end)
}

dust_cpu_walk_simulate <- function(ptr, time_end) {
  .Call(`_dust_dust_cpu_walk_simulate`, ptr, time_end)
}

dust_cpu_walk_run_adjoint <- function(ptr) {
  .Call(`_dust_dust_cpu_walk_run_adjoint`, ptr)
}

dust_cpu_walk_set_index <- function(ptr, r_index) {
  .Call(`_dust_dust_cpu_walk_set_index`, ptr, r_index)
}

dust_cpu_walk_update_state <- function(ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size) {
  .Call(`_dust_dust_cpu_walk_update_state`, ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size)
}

dust_cpu_walk_state <- function(ptr, r_index) {
  .Call(`_dust_dust_cpu_walk_state`, ptr, r_index)
}

dust_cpu_walk_time <- function(ptr) {
  .Call(`_dust_dust_cpu_walk_time`, ptr)
}

dust_cpu_walk_reorder <- function(ptr, r_index) {
  invisible(.Call(`_dust_dust_cpu_walk_reorder`, ptr, r_index))
}

dust_cpu_walk_resample <- function(ptr, r_weights) {
  .Call(`_dust_dust_cpu_walk_resample`, ptr, r_weights)
}

dust_cpu_walk_rng_state <- function(ptr, first_only, last_only) {
  .Call(`_dust_dust_cpu_walk_rng_state`, ptr, first_only, last_only)
}

dust_cpu_walk_set_rng_state <- function(ptr, rng_state) {
  .Call(`_dust_dust_cpu_walk_set_rng_state`, ptr, rng_state)
}

dust_cpu_walk_set_data <- function(ptr, data, shared) {
  .Call(`_dust_dust_cpu_walk_set_data`, ptr, data, shared)
}

dust_cpu_walk_compare_data <- function(ptr) {
  .Call(`_dust_dust_cpu_walk_compare_data`, ptr)
}

dust_cpu_walk_filter <- function(ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood) {
  .Call(`_dust_dust_cpu_walk_filter`, ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood)
}

dust_cpu_walk_set_n_threads <- function(ptr, n_threads) {
  invisible(.Call(`_dust_dust_cpu_walk_set_n_threads`, ptr, n_threads))
}

dust_cpu_walk_n_state <- function(ptr) {
  .Call(`_dust_dust_cpu_walk_n_state`, ptr)
}

dust_cpu_walk_set_stochastic_schedule <- function(ptr, time) {
  invisible(.Call(`_dust_dust_cpu_walk_set_stochastic_schedule`, ptr, time))
}

dust_cpu_walk_ode_statistics <- function(ptr) {
  .Call(`_dust_dust_cpu_walk_ode_statistics`, ptr)
}
