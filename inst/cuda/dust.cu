// -*- c++ -*-
#include <cpp11.hpp>
#include <dust/device_info.hpp>

cpp11::sexp dust_device_info() {
  return dust::cuda::device_info<float>();
}
