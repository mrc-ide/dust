#ifdef _OPENMP
#include <omp.h>
#endif

#include <cpp11/list.hpp>
#include <cpp11/list_of.hpp>

[[cpp11::register]]
cpp11::writable::list cpp_openmp_info() {
#ifdef _OPENMP
  const int num_procs = omp_get_num_procs();
  const int max_threads = omp_get_max_threads();
  const int thread_limit = omp_get_thread_limit();
  static int openmp_version = _OPENMP;
  static bool has_openmp = true;
#else
  static int num_procs = NA_INTEGER;
  static int max_threads = NA_INTEGER;
  static int thread_limit = NA_INTEGER;
  static int openmp_version = NA_INTEGER;
  static bool has_openmp = false;
#endif
  using namespace cpp11::literals;
  return cpp11::writable::list({"num_procs"_nm = num_procs,
                                "max_threads"_nm = max_threads,
                                "thread_limit"_nm = thread_limit,
                                "openmp_version"_nm = openmp_version,
                                "has_openmp"_nm = has_openmp});
}
