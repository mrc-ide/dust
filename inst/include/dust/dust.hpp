#ifndef DUST_DUST_HPP
#define DUST_DUST_HPP

#include <dust/rng.hpp>
#include <dust/densities.hpp>
#include <dust/tools.hpp>

#include <algorithm>
#include <memory>
#include <map>
#include <stdexcept>
#include <sstream>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace dust {
struct nothing {};
typedef nothing no_data;
typedef nothing no_internal;
typedef nothing no_shared;

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

// If we store history, we need to save:
//
// history order (n_particles * n_data)
// history value (n_history_state * n_particles * n_data)
//
// restart (not yet thinking about this)
//
// implies that we should save an index

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
    history_index.resize(n_particles_ * (n_data_ + 1));
    for (size_t i = 0; i < n_particles_; ++i) {
      history_index[i] = i;
    }
  }

  typename std::vector<real_t>::iterator history_value_iterator() {
    return history_value.begin() + offset_ * n_state_ * n_particles_;
  }

  typename std::vector<size_t>::iterator history_index_iterator() {
    return history_index.begin() + offset_ * n_particles_;
  }

  void advance() {
    offset_++;
  }

  std::vector<real_t> history_value;
  std::vector<size_t> history_index;

private:
  size_t n_state_;
  size_t n_particles_;
  size_t n_data_;
  size_t offset_;
  size_t len_;
};
}

template <typename T>
class Particle {
public:
  typedef dust::pars_t<T> pars_t;
  typedef typename T::real_t real_t;
  typedef typename T::data_t data_t;

  Particle(pars_t pars, size_t step) :
    _model(pars),
    _step(step),
    _y(_model.initial(_step)),
    _y_swap(_model.size()) {
  }

  void run(const size_t step_end, dust::rng_state_t<real_t>& rng_state) {
    while (_step < step_end) {
      _model.update(_step, _y.data(), rng_state, _y_swap.data());
      _step++;
      std::swap(_y, _y_swap);
    }
  }

  void state(const std::vector<size_t>& index,
             typename std::vector<real_t>::iterator end_state) const {
    for (size_t i = 0; i < index.size(); ++i) {
      *(end_state + i) = _y[index[i]];
    }
  }

  void state_full(typename std::vector<real_t>::iterator end_state) const {
    for (size_t i = 0; i < _y.size(); ++i) {
      *(end_state + i) = _y[i];
    }
  }

  size_t size() const {
    return _y.size();
  }

  size_t step() const {
    return _step;
  }

  void swap() {
    std::swap(_y, _y_swap);
  }

  void set_step(const size_t step) {
    _step = step;
  }

  void set_state(const Particle<T>& other) {
    _y_swap = other._y;
  }

  size_t set_pars(const pars_t& pars) {
    auto m = T(pars);
    bool ret = m.size();
    if (m.size() == _model.size()) {
      _model = m;
      ret = 0;
    }
    return ret;
  }

  void set_state(typename std::vector<real_t>::const_iterator state) {
    for (size_t i = 0; i < _y.size(); ++i, ++state) {
      _y[i] = *state;
    }
  }

  real_t compare_data(const data_t& data,
                      dust::rng_state_t<real_t>& rng_state) {
    return _model.compare_data(_y.data(), data, rng_state);
  }

private:
  T _model;
  size_t _step;

  std::vector<real_t> _y;
  std::vector<real_t> _y_swap;
};

template <typename T>
class Dust {
public:
  typedef dust::pars_t<T> pars_t;
  typedef typename T::real_t real_t;
  typedef typename T::data_t data_t;

  Dust(const pars_t& pars, const size_t step, const size_t n_particles,
       const size_t n_threads, const std::vector<uint64_t>& seed) :
    _n_pars(0),
    _n_particles_total(n_particles),
    _n_threads(n_threads),
    _rng(_n_particles_total, seed) {
    initialise(pars, step, n_particles);
  }

  Dust(const std::vector<pars_t>& pars, const size_t step,
       const size_t n_particles, const size_t n_threads,
       const std::vector<uint64_t>& seed) :
    _n_pars(pars.size()),
    _n_particles_total(n_particles * pars.size()),
    _n_threads(n_threads),
    _rng(_n_particles_total, seed) {
    initialise(pars, step, n_particles);
  }

  void reset(const pars_t& pars, const size_t step) {
    const size_t n_particles = _particles.size();
    initialise(pars, step, n_particles);
  }

  void reset(const std::vector<pars_t>& pars, const size_t step) {
    const size_t n_particles = _particles.size() / pars.size();
    initialise(pars, step, n_particles);
  }

  void set_pars(const pars_t pars) {
    const size_t n_particles = _particles.size();
    std::vector<size_t> err(n_particles);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < n_particles; ++i) {
      err[i] = _particles[i].set_pars(pars);
    }
    for (size_t i = 0; i < n_particles; ++i) {
      if (err[i] > 0) {
        std::stringstream msg;
        msg << "Tried to initialise a particle with a different state size:" <<
          " particle " << i + 1 << " had state size " <<
          _particles[i].size() << " but new pars implies state size " <<
          err[i];
        throw std::invalid_argument(msg.str());
      }
    }
  }

  // It's the callee's responsibility to ensure that index is in
  // range [0, n-1]
  void set_index(const std::vector<size_t>& index) {
    _index = index;
  }

  // It's the callee's responsibility to ensure this is the correct length:
  //
  // * if is_matrix is false then state must be length n_state_full()
  //   and all particles get the state
  // * if is_matrix is true, state must be length (n_state_full() *
  //   n_particles()) and every particle gets a different state.
  void set_state(const std::vector<real_t>& state, bool is_matrix) {
    const size_t n_particles = _particles.size();
    const size_t n_state = n_state_full();
    auto it = state.begin();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles[i].set_state(it);
      if (is_matrix) {
        it += n_state;
      }
    }
  }

  void set_step(const size_t step) {
    const size_t n_particles = _particles.size();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles[i].set_step(step);
    }
  }

  void set_step(const std::vector<size_t>& step) {
    const size_t n_particles = _particles.size();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles[i].set_step(step[i]);
    }
    const auto r = std::minmax_element(step.begin(), step.end());
    if (*r.second > *r.first) {
      run(*r.second);
    }
  }

  void run(const size_t step_end) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].run(step_end, _rng.state(i));
    }
  }

  void state(std::vector<real_t>& end_state) const {
    return state(end_state.begin());
  }

  void state(typename std::vector<real_t>::iterator end_state) const {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(_index, end_state + i * _index.size());
    }
  }

  void state(std::vector<size_t> index,
             std::vector<real_t>& end_state) const {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(index, end_state.begin() + i * index.size());
    }
  }

  void state_full(std::vector<real_t>& end_state) const {
    const size_t n = n_state_full();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state_full(end_state.begin() + i * n);
    }
  }

  // There are two obvious ways of reordering; we can construct a
  // completely new set of particles, like
  //
  //   std::vector<Particle<T>> next;
  //   for (auto const& i: index) {
  //     next.push_back(_particles[i]);
  //   }
  //   _particles = next;
  //
  // but this seems like a lot of churn.  The other way is to treat it
  // like a slightly weird state update where we swap around the
  // contents of the particle state (uses the set_state() and swap()
  // methods on particles).
  void reorder(const std::vector<size_t>& index) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      size_t j = index[i];
      _particles[i].set_state(_particles[j]);
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].swap();
    }
  }

  std::vector<size_t> resample(const std::vector<real_t>& weights) {
    std::vector<size_t> index(n_particles());
    resample(weights, index);
    return index;
  }

  void resample(const std::vector<real_t>& weights,
                std::vector<size_t>& index) {
    auto it_weights = weights.begin();
    auto it_index = index.begin();
    if (_n_pars == 0) {
      // One parameter set; shuffle among all particles
      const size_t np = _particles.size();
      real_t u = dust::unif_rand(_rng.state(0));
      resample_weight(it_weights, np, u, 0, it_index);
    } else {
      // Multiple parameter set; shuffle within each group
      // independently (and therefore in parallel)
      const size_t np = _particles.size() / _n_pars;
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
      for (size_t i = 0; i < _n_pars; ++i) {
        const size_t j = i * np;
        real_t u = dust::unif_rand(_rng.state(j));
        resample_weight(it_weights + j, np, u, j, it_index + j);
      }
    }

    reorder(index);
  }

  size_t n_particles() const {
    return _particles.size();
  }

  size_t n_state() const {
    return _index.size();
  }

  size_t n_state_full() const {
    return _particles.front().size();
  }

  size_t n_pars() const {
    return _n_pars;
  }

  size_t n_pars_effective() const {
    return _n_pars == 0 ? 1 : _n_pars;
  }

  size_t step() const {
    return _particles.front().step();
  }

  std::vector<uint64_t> rng_state() {
    return _rng.export_state();
  }

  void set_rng_state(const std::vector<uint64_t>& rng_state) {
    _rng.import_state(rng_state);
  }

  void set_n_threads(size_t n_threads) {
    _n_threads = n_threads;
  }

  // NOTE: it only makes sense to expose long_jump, and not jump,
  // because each rng stream is one jump away from the next.
  void rng_long_jump() {
    _rng.long_jump();
  }

  void set_data(std::map<size_t, std::vector<data_t>> data) {
    _data = data;
  }

  std::vector<real_t> compare_data() {
    std::vector<real_t> res;
    auto d = _data.find(step());
    if (d != _data.end()) {
      res.resize(_particles.size());
      compare_data(res, d->second);
    }
    return res;
  }

  void compare_data(std::vector<real_t>& res, const std::vector<data_t>& data) {
    const size_t np = _particles.size() / n_pars_effective();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      res[i] = _particles[i].compare_data(data[i / np], _rng.state(i));
    }
  }

  std::vector<real_t> filter(bool save_history) {
    if (_data.size() == 0) {
      throw std::invalid_argument("Data has not been set for this object");
    }

    const size_t n_particles = _particles.size();
    const size_t n_particles_each = n_particles / n_pars_effective();
    std::vector<real_t> log_likelihood(n_pars_effective());
    std::vector<real_t> log_likelihood_step(n_pars_effective());
    std::vector<real_t> weights(n_particles);
    std::vector<size_t> kappa(n_particles);

    if (save_history) {
      filter_state_.resize(_index.size(), _particles.size(), _data.size());
      state(filter_state_.history_value_iterator());
      filter_state_.advance();
    }

    for (auto & d : _data) {
      run(d.first);
      compare_data(weights, d.second);

      // TODO: we should cope better with the case where all weights
      // are 0; I think that is the behaviour in the model (or rather
      // the case where there is no data and so we do not resample)
      //
      // TODO: we should cope better with the case where one filter
      // has become impossible but others continue, but that's hard!
      auto wi = weights.begin();
      for (size_t i = 0; i < n_pars_effective(); ++i) {
        log_likelihood_step[i] =
          scale_log_weights<real_t>(wi, n_particles_each);
        log_likelihood[i] += log_likelihood_step[i];
        wi += n_particles_each;
      }

      // We could move this below if wanted but we'd have to rewrite
      // the re-sort algorithm.
      if (save_history) {
        state(filter_state_.history_value_iterator());
      }

      resample(weights, kappa);

      if (save_history) {
        std::copy(kappa.begin(), kappa.end(),
                  filter_state_.history_index_iterator());
        filter_state_.advance();
      }
    }

    return log_likelihood;
  }

  const dust::filter_state<real_t>& filter_history() const {
    return filter_state_;
  }

private:
  const size_t _n_pars; // 0 in the "single" case, >=1 otherwise
  const size_t _n_particles_total; // Total number of particles
  size_t _n_threads;
  dust::pRNG<real_t> _rng;
  std::map<size_t, std::vector<data_t>> _data;

  std::vector<size_t> _index;
  std::vector<Particle<T>> _particles;

  // Only used if we have data; this is going to change around a bit.
  dust::filter_state<real_t> filter_state_;

  void initialise(const pars_t& pars, const size_t step,
                  const size_t n_particles) {
    _particles.clear();
    _particles.reserve(n_particles);
    for (size_t i = 0; i < n_particles; ++i) {
      _particles.push_back(Particle<T>(pars, step));
    }
    initialise_index();
  }

  void initialise(const std::vector<pars_t>& pars, const size_t step,
                  const size_t n_particles) {
    // NOTE: we select the initialise function at runtime, but should
    // always get it right. We might throw otherwise?
    //
    // We can throw here so need to make a new copy of particles.
    std::vector<Particle<T>> particles;
    particles.reserve(n_particles * _n_pars);
    for (size_t i = 0; i < _n_pars; ++i) {
      for (size_t j = 0; j < n_particles; ++j) {
        particles.push_back(Particle<T>(pars[i], step));
      }
      if (i > 0) {
        const size_t n_old = particles.front().size();
        const size_t n_new = particles.back().size();
        if (n_old != n_new) {
          std::stringstream msg;
          msg << "Pars created different state sizes: pars " << i + 1 <<
            " (of " << _n_pars << ") had length " << n_new <<
            " but expected " << n_old;
          throw std::invalid_argument(msg.str());
        }
      }
    }
    _particles = particles;
    initialise_index();
  }

  void initialise_index() {
    const size_t n = n_state_full();
    _index.clear();
    _index.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      _index.push_back(i);
    }
  }
};


template <typename T>
std::vector<typename T::real_t>
dust_simulate(const std::vector<size_t>& steps,
              const std::vector<dust::pars_t<T>>& pars,
              std::vector<typename T::real_t>& state,
              const std::vector<size_t>& index,
              const size_t n_threads,
              std::vector<uint64_t>& seed,
              bool save_state) {
  typedef typename T::real_t real_t;
  const size_t n_state_return = index.size();
  const size_t n_particles = pars.size();
  std::vector<Particle<T>> particles;
  particles.reserve(n_particles);
  for (size_t i = 0; i < n_particles; ++i) {
    particles.push_back(Particle<T>(pars[i], steps[0]));
    if (i > 0 && particles.back().size() != particles.front().size()) {
      std::stringstream msg;
      msg << "Particles have different state sizes: particle " << i + 1 <<
        " had length " << particles.back().size() << " but expected " <<
        particles.front().size();
      throw std::invalid_argument(msg.str());
    }
  }
  const size_t n_state_full = particles.front().size();

  dust::pRNG<real_t> rng(n_particles, seed);
  std::vector<real_t> ret(n_particles * n_state_return * steps.size());
  size_t n_steps = steps.size();

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (size_t i = 0; i < particles.size(); ++i) {
    particles[i].set_state(state.begin() + n_state_full * i);
    for (size_t t = 0; t < n_steps; ++t) {
      particles[i].run(steps[t], rng.state(i));
      size_t offset = t * n_state_return * n_particles + i * n_state_return;
      particles[i].state(index, ret.begin() + offset);
    }
    if (save_state) {
      particles[i].state_full(state.begin() + n_state_full * i);
    }
  }

  // To continue we'd also need the rng state:
  if (save_state) {
    rng.export_state(seed);
  }

  return ret;
}

#endif
