/// IMPORTANT; changes here must be reflected into inst/template/dust.hpp
#include <dust/r/dust.hpp>

/// Can we move this later? in that case we can simplify a little.
{{model}}

cpp11::sexp dust_{{name}}_gpu_info() {
  return dust::gpu::r::gpu_info();
}
