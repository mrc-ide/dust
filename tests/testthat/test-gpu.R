context("gpu")

test_that("Can run device version of model on cpu", {
  np <- 100
  len <- 20
  gen <- dust_example("variable")
  mod1 <- gen$new(list(len = len), 0, np, seed = 1L)
  mod2 <- gen$new(list(len = len), 0, np, seed = 1L)

  expect_identical(
    mod1$run(10),
    mod2$run(10, TRUE))
  expect_identical(
    mod1$run(13),
    mod2$run(13, TRUE))
})


test_that("Can use both device and cpu run functions", {
  np <- 100
  len <- 20
  gen <- dust_example("variable")
  mod1 <- gen$new(list(len = len), 0, np, seed = 1L)
  mod2 <- gen$new(list(len = len), 0, np, seed = 1L)
  mod3 <- gen$new(list(len = len), 0, np, seed = 1L)

  expect_identical(
    mod1$run(10),
    mod2$run(10))
  expect_identical(
    mod1$run(13),
    mod2$run(13, TRUE))
  expect_identical(
    mod1$run(19, TRUE),
    mod2$run(19))
  expect_identical(
    mod1$state(),
    mod3$run(19))
})


test_that("Raise suitable error if model does not support GPU", {
  gen <- dust_example("walk")
  mod <- gen$new(list(sd = 1), 0, 100, seed = 1L)
  expect_error(
    mod$run(10, TRUE),
    "GPU support not enabled for this object")
})
