#ifndef DUST_TYPES_HPP
#define DUST_TYPES_HPP

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
class filter_state {
public:
  filter_state(size_t n_state, size_t n_particles, size_t n_data) :
    n_state_(n_state), n_particles_(n_particles), n_data_(n_data), offset_(0) {
    resize(n_state, n_particles, n_data);
  }

  // default constructable
  filter_state() : filter_state(0, 0, 0) {
  }

  void resize(size_t n_state, size_t n_particles, size_t n_data) {
    n_state_ = n_state;
    n_particles_ = n_particles;
    n_data_ = n_data;
    offset_ = 0;
    history_value.resize(n_state_ * n_particles_ * (n_data_ + 1));
    history_order.resize(n_particles_ * (n_data_ + 1));
    for (size_t i = 0; i < n_particles_; ++i) {
      history_order[i] = i;
    }
  }

  typename std::vector<real_t>::iterator history_value_iterator() {
    return history_value.begin() + offset_ * n_state_ * n_particles_;
  }

  typename std::vector<size_t>::iterator history_order_iterator() {
    return history_order.begin() + offset_ * n_particles_;
  }

  std::vector<real_t> history() const {
    std::vector<real_t> ret(size());
    history(ret.begin());
    return ret;
  }

  // This is a particularly unpleasant bit of bookkeeping and is
  // adapted from mcstate (see the helper files in tests for a
  // translation of the the code). As we proceed we store the values
  // of particles *before* resampling and then we store the index used
  // in resampling. We do not resample all the history at each
  // resample as that is prohibitively expensive.
  //
  // So to output sensible history we start with a particle and we
  // look to see where it "came from" in the previous step
  // (history_index) and propagate this backward in time to
  // reconstruct what is in effect a multifurcating tree.
  // This is analogous to the particle ancestor concept in the
  // particle filter literature.
  //
  // It's possible we could do this more efficiently for some subset
  // of particles too (give me the history of just one particle) by
  // breaking the function before the loop over 'k'.
  //
  // Note that we treat history_order and history_value as read-only
  // though this process so one could safely call this multiple times.
  template <typename Iterator>
  void history(Iterator ret) const {
    std::vector<size_t> index_particle(n_particles_);
    for (size_t i = 0; i < n_particles_; ++i) {
      index_particle[i] = i;
    }
    for (size_t k = 0; k < n_data_ + 1; ++k) {
      size_t i = n_data_ - k;
      auto const it_order = history_order.begin() + i * n_particles_;
      auto const it_value = history_value.begin() + i * n_state_ * n_particles_;
      auto it_ret = ret + i * n_state_ * n_particles_;
      for (size_t j = 0; j < n_particles_; ++j) {
        const size_t idx = *(it_order + index_particle[j]);
        index_particle[j] = idx;
        std::copy_n(it_value + idx * n_state_, n_state_,
                    it_ret + j * n_state_);
      }
    }
  }

  size_t size() const {
    return history_value.size();
  }

  void advance() {
    offset_++;
  }

private:
  size_t n_state_;
  size_t n_particles_;
  size_t n_data_;
  size_t offset_;
  size_t len_;
  std::vector<real_t> history_value;
  std::vector<size_t> history_order;
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
    scatter_index = dust::device_array<int>(n_state * n_particles);
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
  dust::device_array<int> scatter_index;
};

// We need to compute the size of space required for integers and
// reals on the device, per particle. Because we work on the
// requirement that every particle has the same dimension we pass an
// arbitrary set of shared parameters (really the first) to
// device_internal_size. The underlying model can overload this template
// for either real or int types and return the length of data
// required.
template <typename T>
size_t device_internal_size_int(typename dust::shared_ptr<T> shared) {
  return 0;
}

template <typename T>
size_t device_internal_size_real(typename dust::shared_ptr<T> shared) {
  return 0;
}

template <typename T>
size_t device_shared_size_int(typename dust::shared_ptr<T> shared) {
  return 0;
}

template <typename T>
size_t device_shared_size_real(typename dust::shared_ptr<T> shared) {
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

}

#endif
