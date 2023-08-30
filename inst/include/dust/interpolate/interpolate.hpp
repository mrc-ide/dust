#ifndef DUST_INTERPOLATE_INTERPOLATE_HPP
#define DUST_INTERPOLATE_INTERPOLATE_HPP

#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <dust/interpolate/spline.hpp>

namespace dust {
namespace interpolate {

namespace internal {

template <typename T>
size_t interpolate_search(T target, std::vector<T> container,
                          bool allow_extrapolate_rhs) {
  if (target < container[0]) {
    const auto delta = container[0] - target;
    std::stringstream msg;
    msg << "Tried to interpolate at time = " << target <<
      ", which is " << delta << " before the first time (" <<
      container[0] << ")";
    throw std::runtime_error(msg.str());
  }
  const auto n = container.size();
  auto lower = std::lower_bound(container.begin(), container.end(), target);

  if (lower == container.end()) {
    if (allow_extrapolate_rhs || container[n - 1] == target) {
      return n - 1;
    } else {
      std::stringstream msg;
      const auto delta = target - container[n - 1];
      msg << "Tried to interpolate at time = " << target <<
        ", which is " << delta << " after the last time (" <<
        container[n - 1] << ")";
      throw std::runtime_error(msg.str());
    }
  }

  const auto i = std::distance(container.begin(), lower);
  return container[i] != target ? i - 1 : i;
}

}

template <typename T>
class InterpolateConstant {
private:
  std::vector<T> t_;
  std::vector<T> y_;
public:
  InterpolateConstant(const std::vector<T>& t, const std::vector<T>& y) :
    t_(t), y_(y) {
    // TODO: check sizes
  }

  InterpolateConstant() {}

  T eval(T z) const {
    size_t i = internal::interpolate_search(z, t_, true);
    return y_[i];
  }
};

template <typename T>
class InterpolateLinear {
private:
  std::vector<T> t_;
  std::vector<T> y_;
public:
  InterpolateLinear(const std::vector<T>& t, const std::vector<T>& y) :
    t_(t), y_(y) {
    // TODO: check sizes
  }

  InterpolateLinear() {}

  T eval(T z) const {
    size_t i = internal::interpolate_search(z, t_, false);
    const size_t n = t_.size() - 1;
    if (i == n) {
      return y_[n];
    }
    T t0 = t_[i], t1 = t_[i + 1], y0 = y_[i], y1 = y_[i + 1];
    return y0 + (y1 - y0) * (z - t0) / (t1 - t0);
  }
};

template <typename T>
class InterpolateSpline {
private:
  std::vector<T> t_;
  std::vector<T> y_;
  std::vector<T> k_;

public:
  InterpolateSpline(const std::vector<T> t, const std::vector<T>& y) :
    t_(t), y_(y) {
    const auto a = spline::calculate_a<T>(t);
    auto b = spline::calculate_b<T>(t, y);
    spline::solve_tridiagonal(a, b);
    k_ = b;
  }

  InterpolateSpline() {}

  T eval(T z) const {
    size_t i = internal::interpolate_search(z, t_, false);
    const size_t n = t_.size() - 1;
    if (i == n) {
      return y_[n];
    }
    const T t0 = t_[i], t1 = t_[i + 1], y0 = y_[i], y1 = y_[i + 1];
    const T k0 = k_[i], k1 = k_[i + 1];
    const T t = (z - t0) / (t1 - t0);
    const T a =  k0 * (t1 - t0) - (y1 - y0);
    const T b = -k1 * (t1 - t0) + (y1 - y0);
    return (1 - t) * y0 + t * y1 + t * (1 - t) * (a * (1 - t) + b * t);
  }
};

}
}

#endif
