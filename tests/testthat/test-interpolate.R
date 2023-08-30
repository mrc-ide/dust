test_that("binary search", {
  x <- 0:9 + 0.5
  eps <- 1e-7

  expect_error(test_interpolate_search(0, x),
               "Trying to interpolate off lhs")

  expect_equal(test_interpolate_search(0.5, x), 0)
  expect_equal(test_interpolate_search(0.5 + eps, x), 0)
  expect_equal(test_interpolate_search(1.5 - eps, x), 0)

  expect_equal(test_interpolate_search(1.5, x), 1)
  expect_equal(test_interpolate_search(1.5 + eps, x), 1)
  expect_equal(test_interpolate_search(2.5 - eps, x), 1)

  expect_equal(test_interpolate_search(5.5, x), 5)
  expect_equal(test_interpolate_search(5.5 + eps, x), 5)
  expect_equal(test_interpolate_search(6.5 - eps, x), 5)

  expect_equal(test_interpolate_search(9.5 - eps, x), 8)
  expect_equal(test_interpolate_search(9.5, x), 9)

  expect_error(test_interpolate_search(9.5, + eps),
               "Trying to interpolate off rhs")
})


test_that("can work with simple constant interpolation", {
  set.seed(1)
  x <- as.numeric(0:10)
  y <- runif(length(x))
  expect_error(
    test_interpolate_constant1(x, y, 0 - 1e-8),
    "Trying to interpolate off lhs")
  expect_equal(test_interpolate_constant1(x, y, 0), y[[1]])
  expect_equal(test_interpolate_constant1(x, y, 1 - 1e-8), y[[1]])
  expect_equal(test_interpolate_constant1(x, y, 1), y[[2]])
  expect_equal(test_interpolate_constant1(x, y, 2), y[[3]])

  z <- vapply(x, function(z) test_interpolate_constant1(x, y, z), numeric(1))
  expect_equal(z, y)

  expect_equal(test_interpolate_constant1(x, y, 10 - 1e-8), y[[10]])
  expect_equal(test_interpolate_constant1(x, y, 10), y[[11]])
  expect_equal(test_interpolate_constant1(x, y, 100), y[[11]])
})


test_that("can work with simple linear interpolation", {
  set.seed(1)
  x <- as.numeric(0:10)
  y <- runif(length(x))
  expect_error(
    test_interpolate_linear1(x, y, 0 - 1e-8),
    "Trying to interpolate off lhs")
  expect_error(
    test_interpolate_linear1(x, y, 10 + 1e-8),
    "Trying to interpolate off rhs")
  cmp <- approxfun(x, y)

  expect_equal(test_interpolate_linear1(x, y, 0), y[[1]])
  expect_equal(test_interpolate_linear1(x, y, 1 - 1e-8), cmp(1 - 1e-8))
  expect_equal(test_interpolate_linear1(x, y, 0.5), cmp(0.5))
  expect_equal(test_interpolate_linear1(x, y, 1), y[[2]])
  expect_equal(test_interpolate_linear1(x, y, 2), y[[3]])

  z <- vapply(x, function(z) test_interpolate_linear1(x, y, z), numeric(1))
  expect_equal(z, y)
})
