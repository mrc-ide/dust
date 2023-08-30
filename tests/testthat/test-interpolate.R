test_that("binary search", {
  t <- 0:9 + 0.5
  eps <- 1e-7

  expect_equal(test_interpolate_search(0.5, t), 0)
  expect_equal(test_interpolate_search(0.5 + eps, t), 0)
  expect_equal(test_interpolate_search(1.5 - eps, t), 0)

  expect_equal(test_interpolate_search(1.5, t), 1)
  expect_equal(test_interpolate_search(1.5 + eps, t), 1)
  expect_equal(test_interpolate_search(2.5 - eps, t), 1)

  expect_equal(test_interpolate_search(5.5, t), 5)
  expect_equal(test_interpolate_search(5.5 + eps, t), 5)
  expect_equal(test_interpolate_search(6.5 - eps, t), 5)

  expect_equal(test_interpolate_search(9.5 - eps, t), 8)
  expect_equal(test_interpolate_search(9.5, t), 9)

  expect_error(
    test_interpolate_search(0, t),
    paste("Tried to interpolate at time = 0, which is 0.5",
          "before the first time (0.5)"),
    fixed = TRUE)
  expect_error(
    test_interpolate_search(9.5 + eps, t),
    paste("Tried to interpolate at time = .+",
          "which is .+ after the last time \\(9.5\\)"))
})


test_that("can work with simple constant interpolation", {
  set.seed(1)
  t <- as.numeric(0:10)
  y <- runif(length(t))
  expect_error(
    test_interpolate_constant1(t, y, 0 - 1e-8),
    "Tried to interpolate.+before the first time")
  expect_equal(test_interpolate_constant1(t, y, 0), y[[1]])
  expect_equal(test_interpolate_constant1(t, y, 1 - 1e-8), y[[1]])
  expect_equal(test_interpolate_constant1(t, y, 1), y[[2]])
  expect_equal(test_interpolate_constant1(t, y, 2), y[[3]])

  z <- vapply(t, function(z) test_interpolate_constant1(t, y, z), numeric(1))
  expect_equal(z, y)

  expect_equal(test_interpolate_constant1(t, y, 10 - 1e-8), y[[10]])
  expect_equal(test_interpolate_constant1(t, y, 10), y[[11]])
  expect_equal(test_interpolate_constant1(t, y, 100), y[[11]])
})


test_that("can work with simple linear interpolation", {
  set.seed(1)
  t <- as.numeric(0:10)
  y <- runif(length(t))
  expect_error(
    test_interpolate_linear1(t, y, 0 - 1e-8),
    "Tried to interpolate.+before the first time")
  expect_error(
    test_interpolate_linear1(t, y, 10 + 1e-8),
    "Tried to interpolate.+after the last time")
  cmp <- approxfun(t, y)

  expect_equal(test_interpolate_linear1(t, y, 0), y[[1]])
  expect_equal(test_interpolate_linear1(t, y, 1 - 1e-8), cmp(1 - 1e-8))
  expect_equal(test_interpolate_linear1(t, y, 0.5), cmp(0.5))
  expect_equal(test_interpolate_linear1(t, y, 1), y[[2]])
  expect_equal(test_interpolate_linear1(t, y, 2), y[[3]])

  z <- vapply(t, function(z) test_interpolate_linear1(t, y, z), numeric(1))
  expect_equal(z, y)
})


test_that("can work with simple spline interpolation", {
  set.seed(1)
  t <- as.numeric(0:10)
  y <- runif(length(t))

  expect_error(
    test_interpolate_spline1(t, y, 0 - 1e-8),
    "Tried to interpolate.+before the first time")
  expect_error(
    test_interpolate_spline1(t, y, 10 + 1e-8),
    "Tried to interpolate.+after the last time")
  cmp <- splinefun(t, y, method = "natural")

  z <- vapply(t, function(z) test_interpolate_spline1(t, y, z), numeric(1))
  expect_equal(z, y)

  expect_equal(test_interpolate_spline1(t, y, 0), y[[1]])
  expect_equal(test_interpolate_spline1(t, y, 1 - 1e-8), cmp(1 - 1e-8))
  expect_equal(test_interpolate_spline1(t, y, 0.5), cmp(0.5))
  expect_equal(test_interpolate_spline1(t, y, 1), y[[2]])
  expect_equal(test_interpolate_spline1(t, y, 2), y[[3]])
})


test_that("Check that time values are sensible", {
  t <- c(0, 1, 1, 2)
  y <- c(0, 1, 2, 3)
  err <- expect_error(
    test_interpolate_spline1(t, y, 1),
    "Times for spline must be strictly increasing but were not around index 2:")
  expect_match(err$message, "[0, 1, 1]", fixed = TRUE)
})
