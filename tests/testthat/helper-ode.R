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
  list(generator = dust_example("logistic"),
       pars = list(r = c(0.1, 0.2), K = c(100, 100)))
}
