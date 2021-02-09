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

template <typename real_t>
real_t scale_log_weights(typename std::vector<real_t>::iterator w, size_t n) {
  real_t max_w = -std::numeric_limits<real_t>::infinity();
  auto wi = w;
  for (size_t i = 0; i < n; ++i, ++wi) {
    if (std::isnan(*wi)) {
      *wi = -std::numeric_limits<real_t>::infinity();
    } else {
      max_w = std::max(max_w, *wi);
    }
  }
  real_t tot = 0.0;
  wi = w;
  for (size_t i = 0; i < n; ++i, ++wi) {
    *wi = std::exp(*wi - max_w);
    tot += *wi;
  }
  return std::log(tot / n) + max_w;
}

#endif
