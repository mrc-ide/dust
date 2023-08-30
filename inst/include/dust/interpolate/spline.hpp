#ifndef DUST_INTERPOLATE_SPLINE_HPP
#define DUST_INTERPOLATE_SPLINE_HPP

#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace dust {
namespace interpolate {
namespace spline {

template <typename T>
using tridiagonal = std::tuple<std::vector<T>, std::vector<T>, std::vector<T>>;

template <typename T>
T square(T x) {
  return x * x;
}

template <typename T>
tridiagonal<T> calculate_a(std::vector<T> x) {
  const auto n = x.size();
  std::vector<T> a0(n);
  std::vector<T> a1(n);
  std::vector<T> a2(n);

  // Left boundary
  a0[0] = 0; // will be ignored
  a1[0] = 2 / (x[1] - x[0]);
  a2[0] = 1 / (x[1] - x[0]);
  // Middle elements
  for (size_t i = 1; i < n - 1; ++i) {
    const T x0 = x[i - 1];
    const T x1 = x[i];
    const T x2 = x[i + 1];
    a0[i] = 1 / (x1 - x0);
    a1[i] = 2 * (1 / (x1 - x0) + 1 / (x2 - x1));
    a2[i] = 1 / (x2 - x1);
  }
  // Right boundary
  a0[n - 1] = 1 / (x[n - 1] - x[n - 2]);
  a1[n - 1] = 2 / (x[n - 1] - x[n - 2]);
  a2[n - 1] = 0; // will be ignored

  return tridiagonal<T>(a0, a1, a2);
}

template <typename T>
std::vector<T> calculate_b(std::vector<T> x, std::vector<T> y) {
  const auto n = x.size();
  const auto nm1 = n - 1;
  std::vector<T> b(n);
  // Left boundary
  b[0] = 3 * (y[1] - y[0]) / square(x[1] - x[0]);
  // Middle elements
  for (size_t i = 1; i < nm1; ++i) {
    const auto x0 = x[i - 1];
    const auto x1 = x[i];
    const auto x2 = x[i + 1];
    const auto y0 = y[i - 1];
    const auto y1 = y[i];
    const auto y2 = y[i + 1];
    b[i] = 3 * ((y1 - y0) / square(x1 - x0) +
                (y2 - y1) / square(x2 - x1));
  }
  // Right boundary
  b[nm1] = 3 * (y[nm1] - y[nm1 - 1]) / square(x[nm1] - x[nm1 - 1]);
  return b;
}

template <typename T>
void solve_tridiagonal(const tridiagonal<T>& m,
                       std::vector<T>& x) {
  const std::vector<T>& a = std::get<0>(m);
  const std::vector<T>& c = std::get<2>(m);
  std::vector<T> b = std::get<1>(m);
  const auto n = a.size();

  for (size_t i = 1; i < n; ++i) {
    const auto fac = a[i] / b[i - 1];
    b[i] -= fac * c[i - 1];
    x[i] -= fac * x[i - 1];
  }

  x[n - 1] /= b[n - 1];
  for (int i = n - 2; i >= 0; i--) {
    x[i] = (x[i] - c[i] * x[i + 1]) / b[i];
  }
}

}
}
}

#endif
