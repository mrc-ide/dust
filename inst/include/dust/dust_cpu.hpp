#ifndef DUST_DUST_CPU_HPP
#define DUST_DUST_CPU_HPP

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dust/gpu/cuda.hpp"
#include "dust/filter_state.hpp"
#include "dust/filter_tools.hpp"
#include "dust/particle.hpp"
#include "dust/random/density.hpp"
#include "dust/random/random.hpp"
#include "dust/utils.hpp"

namespace dust {

template <typename T>
class dust_cpu {
public:
  using model_type = T;
  using pars_type = dust::pars_type<T>;
  using real_type = typename T::real_type;
  using data_type = typename T::data_type;
  using internal_type = typename T::internal_type;
  using shared_type = typename T::shared_type;
  using rng_state_type = typename T::rng_state_type;
  using rng_int_type = typename rng_state_type::int_type;

  // TODO: fix this elsewhere, perhaps (see also cuda/dust_gpu.hpp)
  using filter_state_type = dust::filter::filter_state_host<real_type>;

  dust_cpu(const pars_type& pars, const size_t time, const size_t n_particles,
           const size_t n_threads, const std::vector<rng_int_type>& seed,
           const bool deterministic) :
    n_pars_(0),
    n_particles_each_(n_particles),
    n_particles_total_(n_particles),
    pars_are_shared_(true),
    n_threads_(n_threads),
    rng_(n_particles_total_ + 1, seed, deterministic), // +1 for filter
    errors_(n_particles_total_) {
    initialise(pars, time, n_particles, true);
    initialise_index();
    shape_ = {n_particles};
  }

  dust_cpu(const std::vector<pars_type>& pars, const size_t time,
           const size_t n_particles, const size_t n_threads,
           const std::vector<rng_int_type>& seed,
           const bool deterministic,
           const std::vector<size_t>& shape) :
    n_pars_(pars.size()),
    n_particles_each_(n_particles == 0 ? 1 : n_particles),
    n_particles_total_(n_particles_each_ * pars.size()),
    pars_are_shared_(n_particles != 0),
    n_threads_(n_threads),
    rng_(n_particles_total_ + 1, seed, deterministic),  // +1 for filter
    errors_(n_particles_total_) {
    initialise(pars, time, n_particles_each_, true);
    initialise_index();
    // constructing the shape here is harder than above.
    if (n_particles > 0) {
      shape_.push_back(n_particles);
    }
    for (auto i : shape) {
      shape_.push_back(i);
    }
  }

  void set_pars(const pars_type& pars, bool set_state) {
    const size_t n_particles = particles_.size();
    initialise(pars, time(), n_particles, set_state);
  }

  void set_pars(const std::vector<pars_type>& pars, bool set_state) {
    const size_t n_particles = particles_.size();
    initialise(pars, time(), n_particles / pars.size(), set_state);
  }

  // It's the callee's responsibility to ensure this is the correct length:
  //
  // * if is_matrix is false then state must be length n_state_full()
  //   and all particles get the state
  // * if is_matrix is true, state must be length (n_state_full() *
  //   n_particles()) and every particle gets a different state.
  void set_state(const std::vector<real_type>& state,
                 const std::vector<size_t>& index) {
    const size_t n_particles = particles_.size();
    const bool use_index = index.size() > 0;
    const size_t n_state = use_index ? index.size() : n_state_full();
    const bool individual = state.size() == n_state * n_particles;
    const size_t n = individual ? 1 : n_particles_each_;
    auto it = state.begin();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < n_particles; ++i) {
      const auto it_i = it + (i / n) * n_state;
      if (use_index) {
        particles_[i].set_state(it_i, index);
      } else {
        particles_[i].set_state(it_i);
      }
    }
  }

  void set_time(const size_t time) {
    const size_t n_particles = particles_.size();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < n_particles; ++i) {
      particles_[i].set_time(time);
    }
  }

  // It's the callee's responsibility to ensure that index is in
  // range [0, n-1]
  void set_index(const std::vector<size_t>& index) {
    index_ = index;
  }

  void run(const size_t time_end) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < particles_.size(); ++i) {
      try {
        particles_[i].run(time_end, rng_.state(i));
      } catch (std::exception const& e) {
        errors_.capture(e, i);
      }
    }
    errors_.report();
  }

  std::vector<real_type> simulate(const std::vector<size_t>& time_end) {
    const size_t n_time = time_end.size();
    std::vector<real_type> ret(n_particles() * n_state() * n_time);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < particles_.size(); ++i) {
      try {
        for (size_t t = 0; t < n_time; ++t) {
          particles_[i].run(time_end[t], rng_.state(i));
          size_t offset = t * n_state() * n_particles() + i * n_state();
          particles_[i].state(index_, ret.begin() + offset);
        }
      } catch (std::exception const& e) {
        errors_.capture(e, i);
      }
    }
    errors_.report();
    return ret;
  }

  void state(std::vector<real_type>& end_state) {
    return state(end_state.begin());
  }

  void state(typename std::vector<real_type>::iterator end_state) {
    size_t np = particles_.size();
    size_t index_size = index_.size();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < np; ++i) {
      particles_[i].state(index_, end_state + i * index_size);
    }
  }

  void state(std::vector<size_t> index,
             std::vector<real_type>& end_state) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < particles_.size(); ++i) {
      particles_[i].state(index, end_state.begin() + i * index.size());
    }
  }

  void state_full(std::vector<real_type>& end_state) {
    state_full(end_state.begin());
  }

  void state_full(typename std::vector<real_type>::iterator end_state) {
    const size_t n = n_state_full();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < particles_.size(); ++i) {
      particles_[i].state_full(end_state + i * n);
    }
  }

  // There are two obvious ways of reordering; we can construct a
  // completely new set of particles, like
  //
  //   std::vector<dust::particle<T>> next;
  //   for (auto const& i: index) {
  //     next.push_back(particles_[i]);
  //   }
  //   particles_ = next;
  //
  // but this seems like a lot of churn.  The other way is to treat it
  // like a slightly weird state update where we swap around the
  // contents of the particle state (uses the set_state() and swap()
  // methods on particles).
  void reorder(const std::vector<size_t>& index) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < particles_.size(); ++i) {
      size_t j = index[i];
      particles_[i].set_state(particles_[j]);
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < particles_.size(); ++i) {
      particles_[i].swap();
    }
  }

  std::vector<size_t> resample(const std::vector<real_type>& weights) {
    std::vector<size_t> index(n_particles());
    resample(weights, index);
    return index;
  }

  void resample(const std::vector<real_type>& weights,
                std::vector<size_t>& index) {
    auto it_weights = weights.begin();
    auto it_index = index.begin();
    if (n_pars_ == 0) {
      // One parameter set; shuffle among all particles
      const size_t np = particles_.size();
      real_type u = dust::random::uniform<real_type>(rng_.state(n_particles_total_),
                                               0, 1);
      dust::filter::resample_weight(it_weights, np, u, 0, it_index);
    } else {
      // Multiple parameter set; shuffle within each group
      // independently (and therefore in parallel)
      const size_t np = particles_.size() / n_pars_;
      std::vector<real_type> u;
      for (size_t i = 0; i < n_pars_; ++i) {
        u.push_back(dust::random::uniform<real_type>(rng_.state(n_particles_total_),
                                                  0, 1));
      }
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < n_pars_; ++i) {
        const size_t j = i * np;
        dust::filter::resample_weight(it_weights + j, np, u[i], j, it_index + j);
      }
    }

    reorder(index);
  }

  size_t n_threads() const {
    return n_threads_;
  }

  size_t n_particles() const {
    return particles_.size();
  }

  size_t n_state() const {
    return index_.size();
  }

  size_t n_state_full() const {
    return particles_.front().size();
  }

  size_t n_pars() const {
    return n_pars_;
  }

  size_t n_pars_effective() const {
    return n_pars_ == 0 ? 1 : n_pars_;
  }

  bool pars_are_shared() const {
    return pars_are_shared_;
  }

  size_t n_data() const {
    return data_.size();
  }

  const std::map<size_t, std::vector<data_type>>& data() const {
    return data_;
  }

  size_t time() const {
    return particles_.front().time();
  }

  const std::vector<size_t>& shape() const {
    return shape_;
  }

  void check_errors() {
    if (errors_.unresolved()) {
      throw std::runtime_error("Errors pending; reset required");
    }
  }

  void reset_errors() {
    errors_.reset();
  }

  std::vector<rng_int_type> rng_state() {
    return rng_.export_state();
  }

  void set_rng_state(const std::vector<rng_int_type>& rng_state) {
    rng_.import_state(rng_state);
  }

  void set_n_threads(size_t n_threads) {
    n_threads_ = n_threads;
  }

  // NOTE: it only makes sense to expose long_jump, and not jump,
  // because each rng stream is one jump away from the next.
  void rng_long_jump() {
    rng_.long_jump();
  }

  bool deterministic() const {
    return rng_.deterministic();
  }

  void set_data(std::map<size_t, std::vector<data_type>> data,
                bool data_is_shared) {
    data_ = data;
    data_is_shared_ = data_is_shared;
  }

  std::vector<real_type> compare_data() {
    std::vector<real_type> res;
    auto d = data_.find(time());
    if (d != data_.end()) {
      res.resize(particles_.size());
      compare_data(res, d->second);
    }
    return res;
  }

  void compare_data(std::vector<real_type>& res, const std::vector<data_type>& data) {
    const size_t np = particles_.size() / n_pars_effective();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < particles_.size(); ++i) {
      const size_t j = data_is_shared_ ? 0 : i / np;
      res[i] = particles_[i].compare_data(data[j], rng_.state(i));
    }
  }

private:
  // delete move and copy to avoid accidentally using them
  dust_cpu(const dust_cpu &) = delete;
  dust_cpu(dust_cpu &&) = delete;

  const size_t n_pars_; // 0 in the "single" case, >=1 otherwise
  const size_t n_particles_each_; // Particles per parameter set
  const size_t n_particles_total_; // Total number of particles
  const bool pars_are_shared_; // Does the n_particles dimension exist in shape?
  std::vector<size_t> shape_; // shape of output
  size_t n_threads_;
  dust::random::prng<rng_state_type> rng_;
  std::map<size_t, std::vector<data_type>> data_;
  bool data_is_shared_;
  dust::utils::openmp_errors errors_;

  std::vector<size_t> index_;
  std::vector<dust::particle<T>> particles_;
  std::vector<dust::shared_ptr<T>> shared_;

  void initialise(const pars_type& pars, const size_t time,
                  const size_t n_particles, bool set_state) {
    const size_t n = particles_.size() == 0 ? 0 : n_state_full();
    dust::particle<T> p(pars, time);
    if (n > 0 && p.size() != n) {
      std::stringstream msg;
      msg << "'pars' created inconsistent state size: " <<
        "expected length " << n << " but created length " <<
        p.size();
      throw std::invalid_argument(msg.str());
    }

    if (particles_.empty()) {
      particles_.reserve(n_particles);
      for (size_t i = 0; i < n_particles; ++i) {
        particles_.push_back(p);
      }
      shared_ = {pars.shared};
    } else {
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < n_particles; ++i) {
        particles_[i].set_pars(p, set_state);
      }
      shared_[0] = pars.shared;
    }
    reset_errors();
  }

  void initialise(const std::vector<pars_type>& pars, const size_t time,
                  const size_t n_particles, bool set_state) {
    size_t n = particles_.size() == 0 ? 0 : n_state_full();
    std::vector<dust::particle<T>> p;
    for (size_t i = 0; i < n_pars_; ++i) {
      p.push_back(dust::particle<T>(pars[i], time));
      if (n > 0 && p.back().size() != n) {
        std::stringstream msg;
        msg << "'pars' created inconsistent state size: " <<
          "expected length " << n << " but parameter set " << i + 1 <<
          " created length " << p.back().size();
        throw std::invalid_argument(msg.str());
      }
      n = p.back().size(); // ensures all particles have same size
    }

    if (particles_.empty()) {
      particles_.reserve(n_particles * n_pars_);
      for (size_t i = 0; i < n_pars_; ++i) {
        for (size_t j = 0; j < n_particles; ++j) {
          particles_.push_back(p[i]);
        }
        shared_.push_back(pars[i].shared);
      }
    } else {
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
      for (size_t i = 0; i < n_particles_total_; ++i) {
        particles_[i].set_pars(p[i / n_particles], set_state);
      }
      for (size_t i = 0; i < pars.size(); ++i) {
        shared_[i] = pars[i].shared;
      }
    }
    reset_errors();
  }

  void initialise_index() {
    const size_t n = n_state_full();
    index_.clear();
    index_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      index_.push_back(i);
    }
  }
};

}

#endif
