// -*- c++ -*-
#include <cpp11.hpp>
#include <dust/device_info.hpp>

cpp11::sexp dust_device_info() {
  return dust_device_info<void>();
}