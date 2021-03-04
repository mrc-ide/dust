{{header}}
#ifndef DUST_DISTR_GAMMA_TABLE_HPP
#define DUST_DISTR_GAMMA_TABLE_HPP

#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#else
#define CONSTANT const
#endif

CONSTANT int k_tail_values_max = {{max_k}};

CONSTANT
float k_tail_values_f[] = {
{{values_float}}
};

CONSTANT
double k_tail_values_d[] = {
{{values_double}}
};

#endif
