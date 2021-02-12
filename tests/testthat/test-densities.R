context("densities")

## This one is the easiest:
test_that("dpois agrees", {
  lambda <- rexp(50)
  x <- as.integer(runif(length(lambda), 0, 50))
  expect_equal(dpois(x, lambda, TRUE),
               dust_dpois(x, lambda, TRUE))
  expect_equal(dpois(x, lambda, FALSE),
               dust_dpois(x, lambda, FALSE))

  ## Corner cases
  expect_equal(dust_dpois(0L, 0, TRUE), dpois(0, 0, TRUE))
  expect_equal(dust_dpois(0L, 0, FALSE), dpois(0, 0, FALSE))
  expect_equal(dpois(1L, 0, TRUE), dust_dpois(1L, 0, TRUE))
})


test_that("dnorm agrees", {
  mu <- runif(50, -100, 100)
  x <- rnorm(length(mu), mu, sd = runif(length(mu), max = 100))
  sd <- runif(length(x), max = 100)
  expect_equal(dnorm(x, mu, sd, TRUE),
               dust_dnorm(x, mu, sd, TRUE))
  expect_equal(dnorm(x, mu, sd, FALSE),
               dust_dnorm(x, mu, sd, FALSE))

  ## Corner cases
  expect_equal(dust_dnorm(1, 1, 0, TRUE), dnorm(1, 1, 0, TRUE))
  expect_equal(dust_dnorm(1, 1, 0, FALSE), dnorm(1, 1, 0, FALSE))
  expect_equal(dust_dnorm(0, 1, 0, TRUE), dnorm(0, 1, 0, TRUE))
  expect_equal(dust_dnorm(0, 1, 0, FALSE), dnorm(0, 1, 0, FALSE))
})


test_that("dbinom agrees", {
  size <- as.integer(0:50)
  prob <- runif(length(size))
  x <- as.integer(runif(length(size), 0, size))
  expect_equal(dust_dbinom(x, size, prob, TRUE),
               dbinom(x, size, prob, TRUE))
  expect_equal(dust_dbinom(x, size, prob, FALSE),
               dbinom(x, size, prob, FALSE))

  ## Corner cases
  expect_equal(dust_dbinom(0L, 0L, 0, TRUE), dbinom(0, 0, 0, TRUE))
  expect_equal(dust_dbinom(0L, 0L, 0, FALSE), dbinom(0, 0, 0, FALSE))
  expect_equal(dust_dbinom(0L, 0L, 0.5, TRUE), dbinom(0, 0, 0.5, TRUE))
  expect_equal(dust_dbinom(10L, 0L, 0, TRUE), dbinom(10L, 0L, 0, TRUE))
  expect_equal(dust_dbinom(10L, 4L, 0.5, TRUE), dbinom(10L, 4L, 0.5, TRUE))
})


test_that("dnbinom agrees", {
  size <- as.integer(1:50)
  prob <- runif(length(size))
  mu <- size * (1 - prob) / prob
  x <- sample(size, replace = TRUE)
  expect_equal(dust_dnbinom(x, size, mu, TRUE),
               dnbinom(x, size, mu = mu, log = TRUE))
  expect_equal(dust_dnbinom(x, size, mu, FALSE),
               dnbinom(x, size, mu = mu, log = FALSE))

  ## size > x case which was implemented incorrectly in <= v0.6.5
  expect_equal(
    dnbinom(511, 2, mu = 6.65, log = TRUE),
    dust_dnbinom(511L, 2L, 6.65, TRUE))

  ## Corner cases
  expect_equal(dust_dnbinom(0L, 0L, 0, TRUE),
               dnbinom(0, 0, mu = 0, log = TRUE))
  expect_equal(dust_dnbinom(0L, 0L, 0, FALSE),
               dnbinom(0, 0, mu = 0, log = FALSE))
  expect_equal(dust_dnbinom(0L, 0L, 0.5, FALSE),
               dnbinom(0, 0, mu = 0.5, log = FALSE))
  expect_equal(dust_dnbinom(10L, 0L, 1, TRUE),
               suppressWarnings(dnbinom(10L, 0L, mu = 1, log = TRUE)))
  expect_equal(dust_dnbinom(10L, 1L, 1, TRUE),
               dnbinom(10L, 1L, mu = 1, log = TRUE))
  expect_equal(dust_dnbinom(10L, 0L, 1, FALSE),
               suppressWarnings(dnbinom(10L, 0L, mu = 1, log = FALSE)))
  expect_equal(dust_dnbinom(0L, 10L, 1, TRUE),
               suppressWarnings(dnbinom(0L, 10L, mu = 1, log = TRUE)))
  ## We disagree with R here; we *could* return NaN but -Inf seems
  ## more sensible, and is what R returns if mu = eps
  expect_equal(dust_dnbinom(10L, 0L, 0, TRUE), -Inf)
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

  expect_equal(dust_dbetabinom(0L, 0L, 0, 0, TRUE), 0)
  expect_equal(dust_dbetabinom(0L, 0L, 0, 0, FALSE), 1)
  expect_equal(dust_dbetabinom(0L, 0L, 0.5, 0, FALSE), 1)

  expect_identical(dust_dbetabinom(10L, 0L, 0.5, 0.1, FALSE), 0)
  expect_identical(dust_dbetabinom(10L, 2L, 0.5, 0.4, FALSE), 0)
  expect_identical(dust_dbetabinom(10L, 0L, 0.5, 0.1, TRUE), -Inf)
  expect_identical(dust_dbetabinom(10L, 2L, 0.5, 0.4, TRUE), -Inf)
})
