library(bench)

dyn.load("src/dust.so")

run_benchmark <- function() {
  bench::mark(
    .Call(Cbinom_test, as.integer(0)),
    .Call(Cbinom_test, as.integer(1)),
    check = FALSE)
}