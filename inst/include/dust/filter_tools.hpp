#ifndef DUST_FILTER_TOOLS_HPP
#define DUST_FILTER_TOOLS_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace dust {
namespace filter {

template <typename real_type>
void resample_weight(typename std::vector<real_type>::const_iterator w,
                     size_t n, real_type u, size_t offset,
                     typename std::vector<size_t>::iterator idx) {
  const real_type tot = std::accumulate(w, w + n, static_cast<real_type>(0));
  real_type ww = 0.0, uu0 = tot * u / n, du = tot / n;
  size_t j = offset;
  const size_t end = n + offset;
  for (size_t i = 0; i < n; ++i) {
    // We could accumulate uu by adding du at each iteration but that
    // suffers roundoff error here with floats.
    const real_type uu = uu0 + i * du;
    // The second clause (i.e., j - offset < n) should never be hit
    // but prevents any invalid read if we have pathalogical 'u' that
    // is within floating point eps of 1
    while (ww < uu && j < end) {
      ww += *w;
      ++w;
      ++j;
    }
    *idx = j == 0 ? 0 : j - 1;
    ++idx;
  }
}

// Given some vector of log probabilities 'w' we want to compute a
// vector of numbers such that their ratio equals the exponential of
// their difference, along with the log average value of the numbers.
//
// We can't just do exp(w) because most of the numbers are impossibly
// small. Instead we scale them so that the largest value of exp(w)
// will be 1 and this preserves the relative probabilities because all
// numbers are multiplied by the same value.
//
// Returns scaled weights by modifying 'w' and returns the single
// value of the average log weight.
template <typename real_type>
real_type scale_log_weights(typename std::vector<real_type>::iterator w, size_t n) {
  if (n == 1) {
    return *w;
  }
  real_type max_w = -std::numeric_limits<real_type>::infinity();
  auto wi = w;
  for (size_t i = 0; i < n; ++i, ++wi) {
    if (std::isnan(*wi)) {
      *wi = -std::numeric_limits<real_type>::infinity();
    } else {
      max_w = std::max(max_w, *wi);
    }
  }
  real_type tot = 0.0;
  wi = w;
  for (size_t i = 0; i < n; ++i, ++wi) {
    *wi = std::exp(*wi - max_w);
    tot += *wi;
  }
  return std::log(tot / n) + max_w;
}


template <typename real_type>
bool early_exit(const std::vector<real_type>& log_likelihood,
                const std::vector<real_type>& min) {
  for (auto x : log_likelihood) {
    if (x == -std::numeric_limits<real_type>::infinity()) {
      return true;
    }
  }

  if (min.size() == 0) {
    return false;
  }

  if (log_likelihood.size() == 1) {
    return log_likelihood[0] < min[0];
  }

  if (min.size() == 1) {
    const auto log_likelihood_tot = std::accumulate(log_likelihood.begin(),
                                                    log_likelihood.end(),
                                                    static_cast<real_type>(0));
    return log_likelihood_tot < min[0];
  }

  for (size_t i = 0; i < log_likelihood.size(); ++i) {
    if (log_likelihood[i] >= min[i]) {
      return false;
    }
  }

  return true;
}


}
}

#endif
