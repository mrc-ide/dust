#ifndef DUST_GPU_RNG_HPP
#define DUST_GPU_RNG_HPP

#include <dust/gpu/xoshiro.hpp>
#include <dust/gpu/distr/binomial.hpp>
#include <dust/gpu/distr/normal.hpp>
#include <dust/gpu/distr/poisson.hpp>
#include <dust/gpu/distr/uniform.hpp>

namespace dust {

struct RNGptr {
  uint64_t* state_ptr;
  unsigned int state_stride;
  unsigned int particle_stride;
};

// Read state from global memory
template <typename T>
__device__
rng_state_t<T> loadRNG(RNGptr& d_rng_state, int p_idx) {
  rng_state_t<T> rng_state;
  for (int i = 0; i < XOSHIRO_WIDTH; i++) {
    int j = p_idx * d_rng_state.particle_stride + i * d_rng_state.state_stride;
    state.s[i] = d_rng_state.state_ptr[j];
  }
  return rng_state;
}

// Write state into global memory
template <typename T>
__device__
void putRNG(rng_state_t<T>& rng, RNGptr& d_rng_state, int p_idx) {
  for (int i = 0; i < XOSHIRO_WIDTH; i++) {
    int j = p_idx * d_rng_state.particle_stride + i * d_rng_state.state_stride;
    d_rng_state.state_ptr[j] = rng_state.s[i];
  }
}

template <typename real_t>
class pRNG { // # nocov
public:
  pRNG(const size_t n, const uint64_t seed) {
    dust::Xoshiro rng(seed);
    for (int i = 0; i < n; i++) {
      _rngs.push_back(rng);
      rng.jump();
    }

    CUDA_CALL(cudaMalloc((void** )&_d_rng_state.state_ptr,
                          n * XOSHIRO_WIDTH * sizeof(uint64_t)));
    // Set to be interleaved
    _d_rng_state.state_stride = n;
    _d_rng_state.particle_stride = 1;
    // This might be replaced by runtime error if it becomes more
    // tuneable than above, or a static_assert if it stays n and 1
    // always.
    assert(_d_rng_state.state_stride * _d_rng_state.particle_stride == n)
    put_state_device();
  }

  ~pRNG() {
    CUDA_CALL(cudaFree(_d_rng_state.state_ptr));
  }

  RNGptr state_ptr() { return _d_rng_state; }

  size_t size() const {
    return _rngs.size();
  }

  void jump() {
    get_state_device();
    for (size_t i = 0; i < _rngs.size(); ++i) {
      _rngs[i].jump();
    }
    put_state_device();
  }

  void long_jump() {
    get_state_device();
    for (size_t i = 0; i < _rngs.size(); ++i) {
      _rngs[i].long_jump();
    }
    put_state_device();
  }

  std::vector<uint64_t> export_state() {
    std::vector<uint64_t> state;
    const size_t n = XOSHIRO_WIDTH; // TODO: nicer in cpu version
    state.reserve(size() * n);
    for (size_t i = 0; i < size(); ++i) {
      uint64_t * s = _rngs[i].get_rng_state();
      for (size_t j = 0; j < n; ++j) {
        state.push_back(s[j]);
      }
    }
    return state;
  }

private:
  // delete move and copy to avoid accidentally using them
  pRNG ( const pRNG & ) = delete;
  pRNG ( pRNG && ) = delete;

  void put_state_device() {
    std::vector<uint64_t> flattened_state(size() * XOSHIRO_WIDTH);
    for (int i = 0; i < size(); i++) {
      uint64_t* current_state = _rngs[i].get_rng_state();
      for (int state_idx = 0; state_idx < XOSHIRO_WIDTH; state_idx++) {
        flattened_state[i * _d_rng_state.particle_stride +
                        state_idx * _d_rng_state.state_stride] =
          current_state[state_idx];
      }
    }
    CUDA_CALL(cudaMemcpy(_d_rng_state.state_ptr, flattened_state.data(),
                         flattened_state.size() * sizeof(uint64_t),
                         cudaMemcpyDefault));
    cudaDeviceSynchronize();
  }

  void get_state_device() {
    std::vector<uint64_t> flattened_state(size() * XOSHIRO_WIDTH);
    CUDA_CALL(cudaMemcpy(flattened_state.data(), _d_rng_state.state_ptr,
                         flattened_state.size() * sizeof(uint64_t),
                         cudaMemcpyDefault));
    cudaDeviceSynchronize();

    for (int i = 0; i < size(); i++) {
      std::vector<uint64_t> state(XOSHIRO_WIDTH);
      for (int state_idx = 0; state_idx < XOSHIRO_WIDTH; state_idx++) {
        state[i] = flattened_state[i * _d_rng_state.particle_stride +
                                   state_idx * _d_rng_state.state_stride];
      }
      _rngs[i].set_state(state);
    }
  }

  // Host memory
  std::vector<dust::Xoshiro> _rngs;

  // Device memory
  RNGptr _d_rng_state;
};

}

#endif
