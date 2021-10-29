/// IMPORTANT; changes here must be reflected into inst/template/dust.hpp
// #include <dust/cuda/dust_device.hpp> // should this only be used when device is needed?
#include <dust/dust.hpp>
#include <dust/interface/dust.hpp>

/// Can we move this later? in that case we can simplify a little.
{{model}}

cpp11::sexp dust_{{name}}_capabilities() {
  return dust::r::dust_capabilities<{{class}}>();
}

cpp11::sexp dust_{{name}}_device_info() {
  return dust::cuda::device_info<{{class}}::real_type>();
}
