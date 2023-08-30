#ifndef DUST_INTERPOLATE_INTERPOLATE_HPP
#define DUST_INTERPOLATE_INTERPOLATE_HPP

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace dust {
namespace interpolate {

namespace internal {

template <typename T>
size_t interpolate_search(T target, std::vector<T> container,
                          bool allow_extrapolate_rhs) {
  if (target < container[0]) {
    throw std::runtime_error("Trying to interpolate off lhs");
  }
  const auto n = container.size();
  auto lower = std::lower_bound(container.begin(), container.end(), target);

  if (lower == container.end()) {
    if (allow_extrapolate_rhs || container[n - 1] == target) {
      return n - 1;
    } else {
      throw std::runtime_error("Trying to interpolate off rhs");
    }
  }

  const auto i = std::distance(container.begin(), lower);
  return container[i] != target ? i - 1 : i;
}

}

template <typename T>
class InterpolateConstant {
private:
  std::vector<T> x_;
  std::vector<T> y_;
public:
  InterpolateConstant(const std::vector<T>& x, const std::vector<T>& y) :
    x_(x), y_(y) {
  }

  T eval(T z) {
    size_t i = internal::interpolate_search(z, x_, true);
    return y_[i];
  }
};

template <typename T>
class InterpolateLinear {
private:
  std::vector<T> x_;
  std::vector<T> y_;
public:
  InterpolateLinear(const std::vector<T>& x, const std::vector<T>& y) :
    x_(x), y_(y) {
  }

  T eval(T z) {
    size_t i = internal::interpolate_search(z, x_, false);
    const size_t n = x_.size() - 1;
    if (i == n) {
      return y_[n];
    }
    T x0 = x_[i], x1 = x_[i + 1], y0 = y_[i], y1 = y_[i + 1];
    return y0 + (y1 - y0) * (z - x0) / (x1 - x0);
  }
};

}
}

#endif
