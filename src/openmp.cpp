#ifdef _OPENMP
#include <omp.h>
#endif

#include <Rcpp.h>

// [[Rcpp::export(rng = false, name = "cpp_openmp_info")]]
Rcpp::List openmp_info() {
#ifdef _OPENMP
  int num_procs = omp_get_num_procs();
  int max_threads = omp_get_max_threads();
  int thread_limit = omp_get_thread_limit();
#else
  int num_procs = NA_INTEGER;
  int max_threads = NA_INTEGER;
  int thread_limit = NA_INTEGER;
#endif
  using Rcpp::_;
  return Rcpp::List::create(_["num_procs"] = num_procs,
                            _["max_threads"] = max_threads,
                            _["thread_limit"] = thread_limit);
}
