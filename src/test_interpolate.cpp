#include <cpp11.hpp>
#include <dust/interpolate/interpolate.hpp>

[[cpp11::register]]
int test_interpolate_search(double target, std::vector<double> x) {
  return dust::interpolate::internal::interpolate_search(target, x, false);
}

[[cpp11::register]]
double test_interpolate_constant1(std::vector<double> x, std::vector<double> y,
                                  double z) {
  auto obj = dust::interpolate::InterpolateConstant<double>(x, y);
  return obj.eval(z);
}

[[cpp11::register]]
double test_interpolate_linear1(std::vector<double> x, std::vector<double> y,
                                  double z) {
  auto obj = dust::interpolate::InterpolateLinear<double>(x, y);
  return obj.eval(z);
}
