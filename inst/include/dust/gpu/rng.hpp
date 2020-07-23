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
__device__
RNGState loadRNG(RNGptr& rng_state, int p_idx) {
  RNGState state;
  for (int i = 0; i < XOSHIRO_WIDTH; i++) {
    state.s[i] = rng_state.state_ptr[p_idx * rng_state.particle_stride +
                                      i * rng_state.state_stride];
  }
  return state;
}

// Write state into global memory
__device__
void putRNG(RNGState& rng, RNGptr& rng_state, int p_idx) {
  for (int i = 0; i < XOSHIRO_WIDTH; i++) {
    rng_state.state_ptr[p_idx * rng_state.particle_stride +
                        i * rng_state.state_stride] = rng.s[i];
  }
}

template <typename real_t, typename int_t>
class pRNG { // # nocov
public:
  pRNG(const size_t n, const uint64_t seed) {
    dust::Xoshiro rng(seed);
    //std::vector<dust::distr::rnorm<real_t>> rnorm_buffers;
    for (int i = 0; i < n; i++) {
      //rnorm_buffers.push_back(dust::distr::rnorm<real_t>());
      _rngs.push_back(rng);
      rng.jump();
    }

    //CUDA_CALL(cudaMalloc((void** )&_rnorm_buffers,
    //                     n * sizeof(dust::distr::rnorm<real_t>)));
    //CUDA_CALL(cudaMemcpy(_rnorm_buffers, rnorm_buffers.data(),
    //                     n * sizeof(dust::distr::rnorm<real_t>),
    //                     cudaMemcpyDefault));

    CUDA_CALL(cudaMalloc((void** )&_d_rng_state.state_ptr,
                          n * XOSHIRO_WIDTH * sizeof(uint64_t)));
    // Set to be interleaved
    _d_rng_state.state_stride = n;
    _d_rng_state.particle_stride = 1;
    put_state_device();
  }

  ~pRNG() {
    CUDA_CALL(cudaFree(_d_rng_state.state_ptr));
    //CUDA_CALL(cudaFree(_rnorm_buffers));
  }

  RNGptr state_ptr() { return _d_rng_state; }
  //dust::distr::rnorm<real_t>* rnorm_ptr() { return _rnorm_buffers; }

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

private:
  // delete move and copy to avoid accidentally using them
  pRNG ( const pRNG & ) = delete;
  pRNG ( pRNG && ) = delete;

  void put_state_device() {
    std::vector<uint64_t> interleaved_state(size() * XOSHIRO_WIDTH);
    for (int i = 0; i < size(); i++) {
      uint64_t* current_state = _rngs[i].get_rng_state();
      for (int state_idx = 0; state_idx < XOSHIRO_WIDTH; state_idx++) {
        interleaved_state[i * _d_rng_state.particle_stride +
                          state_idx * _d_rng_state.state_stride] =
          current_state[state_idx];
      }
    }
    CUDA_CALL(cudaMemcpy(_d_rng_state.state_ptr, interleaved_state.data(),
                         interleaved_state.size() * sizeof(uint64_t),
                         cudaMemcpyDefault));
    cudaDeviceSynchronize();
  }

  void get_state_device() {
    std::vector<uint64_t> interleaved_state(size() * XOSHIRO_WIDTH);
    CUDA_CALL(cudaMemcpy(interleaved_state.data(), _d_rng_state.state_ptr,
                         interleaved_state.size() * sizeof(uint64_t),
                         cudaMemcpyDefault));
    cudaDeviceSynchronize();

    for (int i = 0; i < size(); i++) {
      std::vector<uint64_t> state(XOSHIRO_WIDTH);
      for (int state_idx = 0; state_idx < XOSHIRO_WIDTH; state_idx++) {
        state[i] = interleaved_state[i * _d_rng_state.particle_stride +
                                     state_idx * _d_rng_state.state_stride];
      }
      _rngs[i].set_state(state);
    }
  }

  // Host memory
  std::vector<dust::Xoshiro> _rngs;

  // Device memory
  RNGptr _d_rng_state;
  //dust::distr::rnorm<real_t>* _rnorm_buffers;
};

}

#endif
