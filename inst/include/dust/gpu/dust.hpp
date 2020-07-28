#ifndef DUST_GPU_DUST_HPP
#define DUST_GPU_DUST_HPP

#include <dust/gpu/cuda.cuh>
#include <dust/gpu/rng.hpp>

#include <algorithm>
#include <numeric>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cuda_profiler_api.h>
#include <thrust/device_vector.h>
#include <thrust/swap.h>
#include <cub/device/device_select.cuh>

namespace dust {

template <typename real_t>
struct state_t {
  real_t* state_ptr;
  size_t state_stride;
};

}

template <typename T, typename real_t>
__global__
void run_particles(T* models,
                   real_t* particle_y,
                   real_t* particle_y_swap,
                   dust::RNGptr rng_state,
                   size_t y_len,
                   size_t n_particles,
                   size_t step,
                   size_t step_end) {
  int p_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (p_idx < n_particles) {
    dust::rng_state_t<real_t> rng = dust::loadRNG<real_t>(rng_state, p_idx);

    // Read state into shared memory
    extern __shared__ real_t y_shared[];
    for (int state_idx = 0; state_idx < y_len, state_idx++) {
      y_shared[threadIdx.x + state_idx * blockDim.x] =
        particle_y[p_idx + state_idx * n_particles];
      y_shared[threadIdx.x + state_idx * blockDim.x + blockDim.x * y_len] =
        particle_y_swap[p_idx + state_idx * n_particles];
    }
    dust::state_t<real_t> particle_y_p = {y_shared + threadIdx.x, blockDim.x};
    dust::state_t<real_t> particle_y_p_swap =
      {y_shared + threadIdx.x + blockDim.x * y_len, blockDim.x};

    for (int curr_step = step; curr_step < step_end; ++curr_step) {
      // Run the model forward a step
      models[p_idx].update(curr_step,
                           particle_y_p,
                           rng,
                           particle_y_p_swap);
      __syncwarp();

      // Update state
      real_t* tmp = particle_y_p.state_ptr;
      particle_y_p.state_ptr = particle_y_p_swap.state_ptr;
      particle_y_p_swap.state_ptr = tmp;
    }
    dust::putRNG(rng, rng_state, p_idx);
    // Write back state from shared memory
    for (int state_idx = 0; state_idx < y_len, state_idx++) {
      particle_y[p_idx + state_idx * n_particles] =
        y_shared[threadIdx.x + state_idx * blockDim.x];
      particle_y_swap[p_idx + state_idx * n_particles] =
        y_shared[threadIdx.x + state_idx * blockDim.x + blockDim.x * y_len];
    }
  }
}

template <typename T>
class Dust {
public:
  typedef typename T::init_t init_t;
  typedef typename T::real_t real_t;

  Dust(const init_t data, const size_t step, const size_t n_particles,
       const size_t n_threads, const size_t seed) :
    _n_threads(n_threads),
    _n_particles(n_particles),
    _steps(n_particles),
    _rng(n_particles, seed),
    _y_flat(0),
    _y_swap_flat(0),
    _models(nullptr),
    _y_device(nullptr),
    _y_swap_device(nullptr),
    _d_index(nullptr),
    _d_y_out(nullptr),
    _d_tmp(nullptr),
    _d_num_selected_out(nullptr),
    _temp_storage_bytes(0) {
    cudaProfilerStart();
    initialise(data, step, n_particles);
  }

  // NB - if you call cudaDeviceReset() this destructor will error due to
  // double free
  ~Dust() {
    CUDA_CALL(cudaFree(_y_device));
    CUDA_CALL(cudaFree(_y_swap_device));
    CUDA_CALL(cudaFree(_models));
    CUDA_CALL(cudaFree(_d_index));
    CUDA_CALL(cudaFree(_d_tmp));
    CUDA_CALL(cudaFree(_d_y_out));
    CUDA_CALL(cudaFree(_d_num_selected_out));
    cudaProfilerStop();
  }

  void reset(const init_t data, const size_t step) {
    initialise(data, step, _n_particles);
  }

  // It's the callee's responsibility to ensure that index is in
  // range [0, n-1]
  void set_index(const std::vector<size_t>& index) {
    _index = index;
    index_to_device();
  }

  // It's the callee's responsibility to ensure this is the correct length:
  //
  // * if is_matrix is false then state must be length n_state_full()
  //   and all particles get the state
  // * if is_matrix is true, state must be length (n_state_full() *
  //   n_particles()) and every particle gets a different state.
  void set_state(const std::vector<real_t>& state, bool is_matrix) {
    auto it = state.begin();
    for (size_t i = 0; i < _n_particles; ++i) {
      for (size_t j = 0; j < _state_size; ++j, ++it) {
        _y_flat[i + j * _n_particles] = *it;
      }
      if (!is_matrix) {
        it = state.begin();
      }
    }
    y_to_device();
  }

  void set_step(const size_t step) {
    std::fill(_steps.begin(), _steps.end(), step);
  }

  void set_step(const std::vector<size_t>& step) {
    for (size_t i = 0; i < _n_particles; ++i) {
      _steps[i] = step[i];
    }
    const auto r = std::minmax_element(step.begin(), step.end());
    if (*r.second > *r.first) {
      run(*r.second);
    }
  }

  void run(const size_t step_end) {
    const size_t blockSize = 32; // Check later
    const size_t blockCount = (_n_particles + blockSize - 1) / blockSize;
    run_particles<<<blockCount, blockSize, 2 * _state_size * blockSize * sizeof(real_t)>>>
    (
      _models,
      _y_device,
      _y_swap_device,
      _rng.state_ptr(),
      _state_size,
      _n_particles,
      this->step(),
      step_end
    );
    // write step end back to particles
    for (size_t i = 0; i < _n_particles; ++i) {
      _steps[i] = step_end;
    }
    cudaDeviceSynchronize();
  }

  void state(std::vector<real_t>& end_state) {
    cub::DeviceSelect::Flagged(_d_tmp, _temp_storage_bytes,
                               _y_device, _d_index,
                               _d_y_out, _d_num_selected_out,
                               _n_particles * _state_size);
    std::vector<real_t> y_flat_selected(_n_particles * _index.size());
    CUDA_CALL(cudaMemcpy(y_flat_selected.data(), _d_y_out, y_flat_selected.size() * sizeof(real_t),
                         cudaMemcpyDefault));

    #pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (size_t i = 0; i < _n_particles; ++i) {
      for (size_t j = 0; j < _index.size(); j++) {
        end_state[j + i * _index.size()] = y_flat_selected[i + j * _n_particles];
      }
    }
  }

  void state(std::vector<size_t> index,
             std::vector<real_t>& end_state) {
    y_to_host();
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (size_t i = 0; i < _n_particles; ++i) {
      for (size_t j = 0; j < index.size(); j++) {
        end_state[j + i * index.size()] = _y_flat[i + index[j] * _n_particles];
      }
    }
  }

  void state_full(std::vector<real_t>& end_state) {
    y_to_host();
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (size_t i = 0; i < _n_particles; ++i) {
      for (size_t j = 0; j < _state_size; j++) {
        end_state[j + i * _state_size] = _y_flat[i + j * _n_particles];
      }
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

  // TODO - write a kernel to do this copy in device memory
  // NB see scatter
  /*
  void reorder(const std::vector<size_t>& index) {
    for (size_t i = 0; i < _particles.size(); ++i) {
      size_t j = index[i];
      _particles[i].set_state(_particles[j]);
    }
    for (auto& p : _particles) {
      p.swap();
    }
  }
  */
  void reorder(const std::vector<size_t>& index) {
  }

  size_t n_particles() const {
    return _n_particles;
  }

  size_t n_state() const {
    return _index.size();
  }

  size_t n_state_full() const {
    return _state_size;
  }

  size_t step() const {
    return _steps.front();
  }

  std::vector<uint64_t> rng_state() {
    return _rng.export_state();
  }

private:
  // delete move and copy to avoid accidentally using them
  Dust ( const Dust & ) = delete;
  Dust ( Dust && ) = delete;

  size_t _n_particles;
  size_t _state_size;
  std::vector<size_t> _index;
  std::vector<size_t> _steps;
  const size_t _n_threads;
  dust::pRNG<real_t> _rng;
  std::vector<real_t> _y_flat;
  std::vector<real_t> _y_swap_flat;

  // Device memory
  T* _models;
  real_t* _y_device;
  real_t* _y_swap_device;
  bool* _d_index;
  real_t* _d_y_out;
  void* _d_tmp;
  size_t* _d_num_selected_out;
  size_t _temp_storage_bytes;

  void initialise(const init_t data, const size_t step,
                  const size_t n_particles) {
    std::fill(_steps.begin(), _steps.end(), step);

    CUDA_CALL(cudaFree(_y_device));
    CUDA_CALL(cudaFree(_y_swap_device));
    CUDA_CALL(cudaFree(_models));
    CUDA_CALL(cudaFree(_d_index));

    T model(data);
    std::vector<T> models(n_particles, model);
    CUDA_CALL(cudaMalloc((void** )&_models, models.size() * sizeof(T)));
    CUDA_CALL(cudaMemcpy(_models, models.data(), models.size() * sizeof(T),
                         cudaMemcpyDefault));
    _state_size = model.size();

    _y_flat.clear();
    _y_flat.resize(n_particles * model.size());
    _y_swap_flat.clear();
    _y_swap_flat.resize(n_particles * model.size());
    std::vector<real_t> y(model.initial(step));
    std::vector<real_t> y_swap(model.size());
    auto y_flat_it = _y_flat.begin();
    auto y_flat_swap_it = _y_swap_flat.begin();
    for (auto i = 0; i < y.size(); i++) {
      std::fill_n(y_flat_it, _n_particles, y[i]);
      std::fill_n(y_flat_swap_it, _n_particles, y_swap[i]);
      y_flat_it += _n_particles;
      y_flat_swap_it += _n_particles;
    }

    CUDA_CALL(cudaMalloc((void** )&_y_device, _y_flat.size() * sizeof(real_t)));
    CUDA_CALL(cudaMemcpy(_y_device, _y_flat.data(), _y_flat.size() * sizeof(real_t),
                         cudaMemcpyDefault));
    CUDA_CALL(cudaMalloc((void** )&_y_swap_device, _y_swap_flat.size() * sizeof(real_t)));
    CUDA_CALL(cudaMemcpy(_y_swap_device, _y_swap_flat.data(), _y_swap_flat.size() * sizeof(real_t),
                         cudaMemcpyDefault));

    // Set the index
    const size_t n = n_state_full();
    _index.clear();
    _index.resize(n);
    std::iota(_index.begin(), _index.end(), 0);
    CUDA_CALL(cudaMalloc((void** )&_d_index, n_state_full() * n_particles * sizeof(bool)));
    index_to_device();
  }

  void index_to_device() {
    CUDA_CALL(cudaFree(_d_tmp));
    CUDA_CALL(cudaFree(_d_y_out));
    CUDA_CALL(cudaFree(_d_num_selected_out));

    std::vector<char> bool_idx(n_state_full() * _n_particles, 0); // NB: vector<bool> is specialised and can't be used here
    for (auto idx_pos = _index.cbegin(); idx_pos != _index.cend(); idx_pos++) {
      std::fill_n(bool_idx.begin() + (*idx_pos * _n_particles), _n_particles, 1);
    }
    CUDA_CALL(cudaMemcpy(_d_index, bool_idx.data(), bool_idx.size() * sizeof(char),
                         cudaMemcpyHostToDevice));

    // Allocate temporary and output storage
    CUDA_CALL(cudaMalloc((void**)&_d_y_out, n_state() * _n_particles * sizeof(real_t)));
    CUDA_CALL(cudaMalloc((void**)&_d_num_selected_out, 1 * sizeof(size_t)));
    // Determine temporary device storage requirements
    cub::DeviceSelect::Flagged(_d_tmp, _temp_storage_bytes,
                               _y_device, _d_index, _d_y_out,
                               _d_num_selected_out, _state_size * _n_particles);
    CUDA_CALL(cudaMalloc((void**)&_d_tmp, _temp_storage_bytes));
  }

  void y_to_host() {
    CUDA_CALL(cudaMemcpy(_y_flat.data(), _y_device, _y_flat.size() * sizeof(real_t),
                         cudaMemcpyDefault));
  }
  void y_swap_to_device() {
    CUDA_CALL(cudaMemcpy(_y_swap_device, _y_swap_flat.data(), _y_swap_flat.size() * sizeof(real_t),
                         cudaMemcpyDefault));
  }
  void y_to_device() {
    CUDA_CALL(cudaMemcpy(_y_device, _y_flat.data(), _y_flat.size() * sizeof(real_t),
                         cudaMemcpyDefault));
  }
};

#endif
