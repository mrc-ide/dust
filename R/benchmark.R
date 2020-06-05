library(bench)

dyn.load("src/dust.so")

run_benchmark <- function() {
  bench::mark(
    "trng_rbinom" = {.Call(Cbinom_test, as.integer(0))},
    "tf_rbinom" = {.Call(Cbinom_test, as.integer(1))},
    "R_rbinom" = {.Call(Cbinom_test, as.integer(2))},
    check = FALSE)
}