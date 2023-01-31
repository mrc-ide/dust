logistic_analytic <- function(r, K, times, y0) {
  sapply(times, function(t) K / (1 + (K / y0 - 1) * exp(-r * t)))
}

logistic_rhs <- function(t, y0, parms) {
  K <- parms$K
  r <- parms$r
  r * y0 * (1 - y0 / K)
}

logistic_dde <- function(r, K, times, y0) {
  dde::dopri(y0, times, logistic_rhs,
             list(r = r, K = K),
             tcrit = times,
             return_time = FALSE,
             return_by_column = FALSE)
}

example_logistic <- function() {
  path <- dust_file("examples/ode/logistic.cpp")
  list(generator = dust(path, quiet = TRUE),
       pars = list(r1 = 0.1, r2 = 0.2, K1 = 100, K2 = 100))
}
