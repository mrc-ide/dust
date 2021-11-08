{{header}}
#ifndef DUST_RANDOM_BINOMIAL_GAMMA_TABLES_HPP
#define DUST_RANDOM_BINOMIAL_GAMMA_TABLES_HPP

#include "dust/random/cuda_compatibility.hpp"

namespace dust {
namespace random {

CONSTANT int k_tail_values_max_f = {{k_max_f}};
CONSTANT int k_tail_values_max_d = {{k_max_d}};

CONSTANT
float k_tail_values_f[] = {
{{values_float}}
};

CONSTANT
double k_tail_values_d[] = {
{{values_double}}
};

}
}

#endif
