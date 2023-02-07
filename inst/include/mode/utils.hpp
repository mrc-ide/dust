#ifndef MODE_UTILS_HPP
#define MODE_UTILS_HPP

#include <algorithm>

namespace mode {

template<typename T>
T square(T x) {
  return x * x;
}

template<typename T>
T clamp(T x, T min, T max) {
  return std::max(std::min(x, max), min);
}

}

#endif
