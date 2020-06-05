#include <math.h>
#include <Rmath.h>
#include "dust.h"

SEXP r_binom_test(SEXP r_type) {
    int type = (int) INTEGER(r_type)[0];
  
    RNG* rng = C_RNG_alloc(1);

    long long sum = 0;
    for (int rep = 0; rep < 100; rep++) {
        for (int i = 0; i < 20; i++) {
                for (int j = 0; j < 20; j++) {
                    double p = 1/pow(2,(double)i);
                    int N = 1 << j;
                    if (type == 0) {
                        sum += C_rbinom(rng, 0, p, N); 
                    } else {
                        sum += Rf_rbinom((double)N, p);
                    }
            }
        }
    }
    printf("sum: %lld\n", sum);
    C_RNG_free(rng);
    return ScalarInteger(0);
}