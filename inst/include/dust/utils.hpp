#ifndef DUST_UTILS_HPP
#define DUST_UTILS_HPP

#include <algorithm>
#include <numeric>
#include <sstream>
#include <vector>

namespace dust {
namespace utils {

template <typename T, typename U, typename Enable = void>
size_t destride_copy(T dest, U& src, size_t at, size_t stride) {
  static_assert(!std::is_reference<T>::value,
                "destride_copy should only be used with reference types");
  size_t i;
  for (i = 0; at < src.size(); ++i, at += stride) {
    dest[i] = src[at];
  }
  return i;
}

template <typename T, typename U>
size_t stride_copy(T dest, U src, size_t at, size_t stride) {
  static_assert(!std::is_reference<T>::value,
                "stride_copy should only be used with reference types");
  dest[at] = src;
  return at + stride;
}

template <typename T, typename U>
size_t stride_copy(T dest, const std::vector<U>& src, size_t at,
                   size_t stride) {
  static_assert(!std::is_reference<T>::value,
                "stride_copy should only be used with reference types");
  for (size_t i = 0; i < src.size(); ++i, at += stride) {
    dest[at] = src[i];
  }
  return at;
}

class openmp_errors {
public:
  openmp_errors() : openmp_errors(0) {
  }
  openmp_errors(size_t len) :
    count(0), err(len), seen(len) {
  }

  void reset() {
    count = 0;
    std::fill(seen.begin(), seen.end(), false);
    std::fill(err.begin(), err.end(), "");
  }

  bool unresolved() const {
    return count > 0;
  }

  template <typename T>
  void capture(const T& e, size_t i) {
    err[i] = e.what();
    seen[i] = true;
  }

  void report(bool clear = false, const char *title = "particles",
              size_t n_max = 4) {
    count = std::accumulate(std::begin(seen), std::end(seen), 0);
    if (count == 0) {
      return;
    }

    std::stringstream msg;
    msg << count << " " << title << " reported errors.";

    const size_t n_report = std::min(n_max, count);
    for (size_t i = 0, j = 0; i < seen.size() && j < n_report; ++i) {
      if (seen[i]) {
        msg << std::endl << "  - " << i + 1 << ": " << err[i];
        ++j;
      }
    }
    if (n_report < count) {
      msg << std::endl << "  - (and " << (count - n_report) << " more)";
    }

    if (clear) {
      reset();
    }

    throw std::runtime_error(msg.str());
  }

private:
  size_t count;
  std::vector<std::string> err;
  std::vector<bool> seen;
};

template<typename T>
T square(T x) {
  return x * x;
}

template<typename T>
T clamp(T x, T min, T max) {
  return std::max(std::min(x, max), min);
}

template <typename real_type, typename It>
bool all_zero(It begin, It end) {
  return std::all_of(begin, end, [](real_type x) { return x == 0; });
}

}
}

#endif
