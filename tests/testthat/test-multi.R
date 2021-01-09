context("multi")

test_that("create trivial multi dust object", {
  res <- dust_example("walk")
  ## res <- dust(dust_file("examples/walk.cpp"), workdir = "tmp")
  obj1 <- res$new(list(sd = 1), 0, 10, seed = 1L, data_multi = FALSE)
  obj2 <- res$new(list(list(sd = 1)), 0, 10, seed = 1L, data_multi = TRUE)

  expect_identical(obj2$name(), obj1$name())
  expect_identical(obj2$param(), obj1$param())
  expect_identical(obj2$n_threads(), obj1$n_threads())
  expect_identical(obj2$has_openmp(), obj1$has_openmp())
  expect_identical(obj2$step(), obj1$step())

  expect_equal(obj2$data(), list(obj1$data()))
  expect_equal(obj2$info(), list(obj1$info()))

  expect_identical(obj2$rng_state(), obj1$rng_state())

  expect_identical(obj1$n_data(), 0L)
  expect_identical(obj2$n_data(), 1L)

  expect_identical(obj2$state(), array(obj1$state(), c(1, 10, 1)))
  expect_identical(obj2$state(1L), array(obj1$state(1L), c(1, 10, 1)))

  y1 <- obj1$run(1)
  y2 <- obj2$run(1)
})


test_that("create trivial 2 element mulitdust object", {
  res <- dust_example("walk")
  ## res <- dust(dust_file("examples/walk.cpp"), workdir = "tmp")
  obj1 <- res$new(list(sd = 1), 0, 10, seed = 1L, data_multi = FALSE)
  obj2 <- res$new(list(list(sd = 1), list(sd = 1)), 0, 5, seed = 1L,
                  data_multi = TRUE)

  expect_identical(obj2$n_data(), 2L)
  expect_equal(obj2$state(), array(obj1$state(), c(1, 5, 2)))

  y1 <- obj1$run(1)
  y2 <- obj2$run(1)
  expect_equal(y2, array(y1, c(1, 5, 2)))
})
