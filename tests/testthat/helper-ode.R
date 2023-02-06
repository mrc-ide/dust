logistic_analytic <- function(r, k, times, y0) {
  sapply(times, function(t) k / (1 + (k / y0 - 1) * exp(-r * t)))
}

logistic_rhs <- function(t, y0, parms) {
  k <- parms$k
  r <- parms$r
  r * y0 * (1 - y0 / k)
}

logistic_dde <- function(r, k, times, y0) {
  dde::dopri(y0, times, logistic_rhs,
             list(r = r, k = k),
             tcrit = times,
             return_time = FALSE,
             return_by_column = FALSE)
}

example_logistic <- function() {
  path <- dust_file("examples/ode/logistic.cpp")
  list(generator = dust(path, quiet = TRUE),
       pars = list(r1 = 0.1, r2 = 0.2, K1 = 100, K2 = 100))
}
