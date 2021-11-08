// -*- c++ -*-
#include <cpp11.hpp>
#include <dust/gpu/gpu_info.hpp>
#include <dust/r/gpu_info.hpp>

cpp11::sexp dust_gpu_info() {
  return dust::gpu::r::gpu_info();
}
