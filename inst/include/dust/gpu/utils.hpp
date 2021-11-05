#ifndef DUST_GPU_UTILS_HPP
#define DUST_GPU_UTILS_HPP

#include <algorithm>
#include <numeric>
#include <vector>

#include "dust/gpu/cuda.hpp"

namespace dust {
namespace gpu {
namespace utils {

// Translates index in y (full state) to index in y_selected
// Maps: 3 4 5 10 9 1 -> 1 2 3 5 4 0
template <typename T>
std::vector<size_t> sort_indexes(const T &v) {
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::vector<size_t> idx_offset = idx;

  // Find the idx order so that v[idx] is sort(v)
  // in the example this gives 5 0 1 2 4 3
  std::stable_sort(
    idx.begin(), idx.end(),
    [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  // Now sort these idx, as we want to these indices in the original order
  // in the example this gives 1 2 3 5 4 0
  std::stable_sort(
    idx_offset.begin(), idx_offset.end(),
    [&idx](size_t i1, size_t i2) { return idx[i1] < idx[i2]; });
  return idx_offset;
}


template <typename T, typename U = T>
inline __host__ __device__ T align_padding(const T offset, const U align) {
  T remainder = offset % align;
  return remainder ? align - remainder : 0;
}

}
}
}

#endif
