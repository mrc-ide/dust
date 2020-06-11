#ifndef DUST_INTERFACE_HPP
#define DUST_INTERFACE_HPP

#include <dust/dust.hpp>
#include <dust/util.hpp>

template <typename T>
extern "C" void test_walk_finalise(SEXP ptr) {
  dust_walk *obj = dust::util::read_r_pointer<T>(ptr, false);
  if (obj) {
    delete obj;
  }
  if (ptr) {
    R_ClearExternalPtr(ptr);
  }
}

#endif
