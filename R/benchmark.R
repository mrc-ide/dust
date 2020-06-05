library(bench)

dyn.load("src/dust.so")

run_benchmark <- function() {
  bench::mark(
    "tf_rbinom" = {.Call(Cbinom_test, as.integer(0))},
    "R_rbinom" = {.Call(Cbinom_test, as.integer(1))},
    check = FALSE)
}