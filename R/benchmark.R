library(bench)

dyn.load("src/dust.so")

run_benchmark <- function() {
  bench::mark(
    .Call("Cbinom_test", as.integer(0), PACKAGE = "dust"),
    .Call("Cbinom_test", as.integer(1), PACKAGE = "dust"),
    check = FALSE)
}