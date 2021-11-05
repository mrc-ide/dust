// -*- c++ -*-
#include <cpp11.hpp>
#include <dust/gpu/gpu_info.hpp>
#include <dust/interface/gpu_info.hpp>

cpp11::sexp dust_gpu_info() {
  return dust::gpu::interface::gpu_info<float>();
}
