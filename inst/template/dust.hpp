/// IMPORTANT; changes here must be reflected into inst/template/dust.cpp
#include <cpp11.hpp>

[[cpp11::register]]
cpp11::sexp dust_{{name}}_capabilities();

[[cpp11::register]]
cpp11::sexp dust_{{name}}_device_info();
