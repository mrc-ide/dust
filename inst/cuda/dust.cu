// -*- c++ -*-
#include <cpp11.hpp>
#include <dust/cuda/device_info.hpp>
#include <dust/interface/cuda_device_info.hpp>

cpp11::sexp dust_device_info() {
  return dust::cuda::device_info<float>();
}
