#ifndef DUST_FILTER_TOOLS_HPP
#define DUST_FILTER_TOOLS_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "dust/random/random.hpp"
#include "dust/utils.hpp"

namespace dust {
namespace filter {

template <typename real_type>
void resample_weight(typename std::vector<real_type>::const_iterator w,
                     size_t n, real_type u, size_t offset,
                     typename std::vector<size_t>::iterator idx) {
  const real_type tot = std::accumulate(w, w + n, static_cast<real_type>(0));
  if (tot == 0 && dust::utils::all_zero<real_type>(w, w + n)) {
    for (size_t i = 0; i < n; ++i) {
      *idx = offset + i;
      ++idx;
    }
  } else {
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
}

template <typename real_type, typename rng_state_type>
void resample_index(const std::vector<real_type>& weights,
                    const size_t n_pars, const size_t n_particles,
                    const size_t n_threads,
                    std::vector<size_t>& index, rng_state_type& rng_state) {
  auto it_weights = weights.begin();
  auto it_index = index.begin();
  if (n_pars == 0) {
    // One parameter set; shuffle among all particles
    real_type u = dust::random::random_real<real_type>(rng_state);
    dust::filter::resample_weight(it_weights, n_particles, u, 0, it_index);
  } else {
    // Multiple parameter set; shuffle within each group
    // independently (and therefore in parallel)
    std::vector<real_type> u;
    for (size_t i = 0; i < n_pars; ++i) {
      u.push_back(dust::random::random_real<real_type>(rng_state));
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (size_t i = 0; i < n_pars; ++i) {
      const size_t j = i * n_particles;
      dust::filter::resample_weight(it_weights + j, n_particles, u[i], j,
                                    it_index + j);
    }
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

  if (max_w == -std::numeric_limits<real_type>::infinity()) {
    return max_w;
  }

  real_type tot = 0.0;
  wi = w;
  for (size_t i = 0; i < n; ++i, ++wi) {
    *wi = std::exp(*wi - max_w);
    tot += *wi;
  }

  return std::log(tot / n) + max_w;
}


// log_likelihood is a vector over parameter sets (this will be length
// 1, 2, ...)
//
// min is a vector of length 0, 1, or log_likelihood.size()
//
// - if min is length 0 we're not doing any early exit and the only
//   exit condition is that the likelihood has become impossible for
//   any parameter set.
//
// - if log_likelihood is length 1 (single parameter set) then min
//   must be the same length and we exit simply when the ll drops
//   below min
//
// - if min is a scalar but log_likelihood is a vector, we exit once
//   the total log likelihood (sum over all parameters) drops below
//   min
//
// - if min is a vector the same length as log_likelihood then we exit
//   if *all* log likelihoods drop below their min (note this is
//   slightly different to the -Inf condition, which is *any* below
//   -Inf)
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
