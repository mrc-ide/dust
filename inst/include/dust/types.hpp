#ifndef DUST_TYPES_HPP
#define DUST_TYPES_HPP

#include <numeric>
#include <sstream>
#include <vector>

namespace dust {

struct nothing {};
typedef nothing no_data;
typedef nothing no_internal;
typedef nothing no_shared;

// By default we do not support anything on the gpu. This name might
// change, but it does reflect our intent and it's likely that to work
// on a GPU the model will have to provide a number of things. If of
// those becomes a type (as with data, internal and shared) we could
// use the same approach as above.
template <typename T>
struct has_gpu_support : std::false_type {};

template <typename T>
using shared_ptr = std::shared_ptr<const typename T::shared_t>;

template <typename T>
struct pars_t {
  std::shared_ptr<const typename T::shared_t> shared;
  typename T::internal_t internal;

  pars_t(std::shared_ptr<const typename T::shared_t> shared_,
         typename T::internal_t internal_) :
    shared(shared_), internal(internal_) {
  }
  pars_t(typename T::shared_t shared_,
         typename T::internal_t internal_) :
    shared(std::make_shared<const typename T::shared_t>(shared_)),
    internal(internal_) {
  }
  pars_t(typename T::shared_t shared_) :
    pars_t(shared_, dust::nothing()) {
  }
  pars_t(typename T::internal_t internal_) :
    pars_t(dust::nothing(), internal_) {
  }
};

template <typename real_t>
struct device_state {
  void initialise(size_t n_particles, size_t n_state, size_t n_shared_len_,
                  size_t n_internal_int, size_t n_internal_real,
                  size_t n_shared_int_, size_t n_shared_real_) {
    n_shared_len = n_shared_len_;
    n_shared_int = n_shared_int_;
    n_shared_real = n_shared_real_;
    const size_t n_rng = dust::rng_state_t<real_t>::size();
    y = dust::device_array<real_t>(n_state * n_particles);
    y_next = dust::device_array<real_t>(n_state * n_particles);
    y_selected = dust::device_array<real_t>(n_state * n_particles);
    internal_int = dust::device_array<int>(n_internal_int * n_particles);
    internal_real = dust::device_array<real_t>(n_internal_real * n_particles);
    shared_int = dust::device_array<int>(n_shared_int * n_shared_len);
    shared_real = dust::device_array<real_t>(n_shared_real * n_shared_len);
    rng = dust::device_array<uint64_t>(n_rng * n_particles);
    index = dust::device_array<char>(n_state * n_particles);
    n_selected = dust::device_array<int>(1);
    scatter_index = dust::device_array<size_t>(n_particles);
    set_cub_tmp();
  }
  void swap() {
    std::swap(y, y_next);
  }
  // TODO - use GPU templates
  void set_cub_tmp() {
#ifdef __NVCC__
    // Free the array before running cub function below
    size_t tmp_bytes = 0;
    select_tmp.set_size(tmp_bytes);
    cub::DeviceSelect::Flagged(select_tmp.data(),
                               tmp_bytes,
                               y.data(),
                               index.data(),
                               y_selected.data(),
                               n_selected.data(),
                               y.size());
    select_tmp.set_size(tmp_bytes);
#endif
  }

  size_t n_shared_len;
  size_t n_shared_int;
  size_t n_shared_real;
  dust::device_array<real_t> y;
  dust::device_array<real_t> y_next;
  dust::device_array<real_t> y_selected;
  dust::device_array<int> internal_int;
  dust::device_array<real_t> internal_real;
  dust::device_array<int> shared_int;
  dust::device_array<real_t> shared_real;
  dust::device_array<uint64_t> rng;
  dust::device_array<char> index;
  dust::device_array<int> n_selected;
  dust::device_array<void> select_tmp;
  dust::device_array<size_t> scatter_index;
};

// We need to compute the size of space required for integers and
// reals on the device, per particle. Because we work on the
// requirement that every particle has the same dimension we pass an
// arbitrary set of shared parameters (really the first) to
// device_internal_size. The underlying model can overload this template
// for either real or int types and return the length of data
// required.
template <typename T>
size_t device_internal_int_size(typename dust::shared_ptr<T> shared) {
  return 0;
}

template <typename T>
size_t device_internal_real_size(typename dust::shared_ptr<T> shared) {
  return 0;
}

template <typename T>
size_t device_shared_int_size(typename dust::shared_ptr<T> shared) {
  return 0;
}

template <typename T>
size_t device_shared_real_size(typename dust::shared_ptr<T> shared) {
  return 0;
}

template <typename T>
void device_shared_copy(typename dust::shared_ptr<T> shared,
                        int * shared_int,
                        typename T::real_t * shared_real) {
}

template <typename T>
T* shared_copy(T* dest, const std::vector<T>& src) {
  memcpy(dest, src.data(), src.size() * sizeof(T));
  return dest + src.size();
}

template <typename T>
T* shared_copy(T* dest, const T src) {
  *dest = src;
  return dest + 1;
}

template <typename T>
struct device_ptrs {
  int * shared_int;
  typename T::real_t * shared_real;
  typename T::data_t * data;
};

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

  void report(size_t n_max = 4) {
    count = std::accumulate(std::begin(seen), std::end(seen), 0);
    if (count == 0) {
      return;
    }

    std::stringstream msg;
    msg << count << " particles reported errors.";

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

    throw std::runtime_error(msg.str());
  }

private:
  size_t count;
  std::vector<std::string> err;
  std::vector<bool> seen;
};

}

#endif
