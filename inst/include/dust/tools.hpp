#ifndef DUST_TOOLS_HPP
#define DUST_TOOLS_HPP

#include <algorithm>
#include <numeric>

template <typename real_t>
void resample_weight(typename std::vector<real_t>::const_iterator w,
                     size_t n, real_t u, size_t offset,
                     typename std::vector<size_t>::iterator idx) {
  const real_t tot = std::accumulate(w, w + n, static_cast<real_t>(0));
  real_t ww = 0.0, uu = tot * u / n, du = tot / n;

  size_t j = offset;
  for (size_t i = 0; i < n; ++i) {
    while (ww < uu) {
      ww += *w;
      ++w;
      ++j;
    }
    uu += du;
    *idx = j == 0 ? 0 : j - 1;
    ++idx;
  }
}

#endif
