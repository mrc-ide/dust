#ifdef _OPENMP
#include <omp.h>
#endif

#include <Rcpp.h>

// [[Rcpp::export(rng = false, name = "cpp_openmp_info")]]
Rcpp::List openmp_info() {
#ifdef _OPENMP
  const int num_procs = omp_get_num_procs();
  const int max_threads = omp_get_max_threads();
  const int thread_limit = omp_get_thread_limit();
  static int openmp_version = _OPENMP;
  static bool has_openmp = true;
  static bool has_monotonic = _OPENMP >= 201511;
#else
  static int num_procs = NA_INTEGER;
  static int max_threads = NA_INTEGER;
  static int thread_limit = NA_INTEGER;
  static int openmp_version = NA_INTEGER;
  static bool has_openmp = false;
  static bool has_monotonic = false;
#endif
  using Rcpp::_;
  return Rcpp::List::create(_["num_procs"] = num_procs,
                            _["max_threads"] = max_threads,
                            _["thread_limit"] = thread_limit,
                            _["openmp_version"] = openmp_version,
                            _["has_openmp"] = has_openmp,
                            _["has_monotonic"] = has_monotonic);
}
