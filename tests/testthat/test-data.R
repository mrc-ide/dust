context("data")

test_that("Can construct dust_data", {
  d <- data.frame(step = seq(0, 50, by = 10), a = runif(6), b = runif(6))
  res <- dust_data(d)
  expect_is(res, "list")
  expect_null(names(res))
  expect_length(res, 6)
  expect_equal(lengths(res), rep(2, 6))
  expect_identical(lapply(res, "[[", 1), list(0L, 10L, 20L, 30L, 40L, 50L))
  expect_identical(res[[4]][[2]], as.list(d[4, ]))
})


test_that("Can validate step", {
  d <- data.frame(step = seq(0, 50, by = 10), a = runif(6), b = runif(6))
  expect_error(
    dust_data(d, "col"),
    "'col' is not a column in d")

  d$step[[4]] <- -40
  expect_error(
    dust_data(d),
    "All elements in column 'step' must be nonnegative")

  d$step[[4]] <- 40.1
  expect_error(
    dust_data(d),
    "All elements in column 'step' must be integer-like")

  d$step[[4]] <- 50
  expect_error(
    dust_data(d),
    "All elements in column 'step' must be unique")
})


test_that("rounding errors are converted to integers", {
  d <- data.frame(t = seq(0, 50, by = 10), a = runif(6), b = runif(6))
  d$t <- d$t + sample(c(-1e-14, 1e-14), 6, replace = TRUE)
  res <- dust_data(d, "t")
  expect_identical(lapply(res, "[[", 1), list(0L, 10L, 20L, 30L, 40L, 50L))
})
