#include <vector>
#include <cpp11.hpp>
#include <dust/filter_tools.hpp>

[[cpp11::register]]
cpp11::list cpp_scale_log_weights(std::vector<double> w) {
  double average = dust::filter::scale_log_weights<double>(w.begin(), w.size());
  using namespace cpp11::literals;
  return cpp11::writable::list({"average"_nm = average, "weights"_nm = w});
}
