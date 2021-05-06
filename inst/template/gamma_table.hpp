{{header}}
#ifndef DUST_DISTR_GAMMA_TABLE_HPP
#define DUST_DISTR_GAMMA_TABLE_HPP


namespace dust {
namespace distr {

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
