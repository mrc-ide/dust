context("densities")

## This one is the easiest:
test_that("dpois agrees", {
  lambda <- rexp(50)
  x <- as.integer(runif(length(lambda), 0, 50))
  expect_equal(dpois(x, lambda, TRUE),
               dust_dpois(x, lambda, TRUE))
  expect_equal(dpois(x, lambda, FALSE),
               dust_dpois(x, lambda, FALSE))
})


test_that("dbinom agrees", {
  size <- as.integer(0:50)
  prob <- runif(length(size))
  x <- as.integer(runif(length(size), 0, size))
  expect_equal(dust_dbinom(x, size, prob, TRUE),
               dbinom(x, size, prob, TRUE))
  expect_equal(dust_dbinom(x, size, prob, FALSE),
               dbinom(x, size, prob, FALSE))
})


test_that("dnbinom agrees", {
  size <- as.integer(0:50)
  prob <- runif(length(size))
  mu <- size * (1 - prob) / prob
  x <- as.integer(runif(length(size), 0, size))
  expect_equal(dust_dnbinom(x, size, mu, TRUE),
               dnbinom(x, size, mu = mu, log = TRUE))
  expect_equal(dust_dnbinom(x, size, mu, FALSE),
               dnbinom(x, size, mu = mu, log = FALSE))
})


test_that("dbetabinom agrees", {
  ## Directly from sircovid v0.9.5
  dbetabinom <- function(x, size, prob, rho, log = FALSE) {
    a <- prob * (1 / rho - 1)
    b <- (1 - prob) * (1 / rho - 1)
    out <- lchoose(size, x) + lbeta(x + a, size - x + b) - lbeta(a, b)
    if (!log) {
      out <- exp(out)
    }
    out
  }

  size <- as.integer(0:50)
  prob <- runif(length(size))
  rho <- runif(length(size))
  x <- as.integer(runif(length(size), 0, size))
  expect_equal(dust_dbetabinom(x, size, prob, rho, TRUE),
               dbetabinom(x, size, prob, rho, TRUE))
  expect_equal(dust_dbetabinom(x, size, prob, rho, FALSE),
               dbetabinom(x, size, prob, rho, FALSE))
})
