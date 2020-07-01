#include <dust/dust.hpp>
#include <dust/interface.hpp>

{{model}}

// [[Rcpp::export(rng = false)]]
SEXP dust_{{name}}_alloc(Rcpp::List r_data, size_t step, size_t n_particles,
                size_t n_threads, size_t n_generators, size_t seed) {
  return dust_alloc<{{type}}>(r_data, step, n_particles, n_threads,
                              n_generators, seed);
}

// [[Rcpp::export(rng = false)]]
SEXP dust_{{name}}_run(SEXP ptr, size_t step_end) {
  return dust_run<{{type}}>(ptr, step_end);
}

// [[Rcpp::export(rng = false)]]
SEXP dust_{{name}}_set_index_y(SEXP ptr, Rcpp::IntegerVector r_index_y) {
  dust_set_index_y<{{type}}>(ptr, r_index_y);
  return R_NilValue;
}

// [[Rcpp::export(rng = false)]]
SEXP dust_{{name}}_reset(SEXP ptr, Rcpp::List r_data, size_t step) {
  return dust_reset<{{type}}>(ptr, r_data, step);
}

// [[Rcpp::export(rng = false)]]
SEXP dust_{{name}}_state(SEXP ptr, SEXP r_index) {
  return dust_state<{{type}}>(ptr, r_index);
}

// [[Rcpp::export(rng = false)]]
size_t dust_{{name}}_step(SEXP ptr) {
  return dust_step<{{type}}>(ptr);
}

// [[Rcpp::export(rng = false)]]
void dust_{{name}}_reorder(SEXP ptr, Rcpp::IntegerVector r_index) {
  return dust_reorder<{{type}}>(ptr, r_index);
}
