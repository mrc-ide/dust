context("gpu")

test_that("Can run device version of model on cpu", {
  gen <- dust_example("variable")
  mod1 <- gen$new(list(len = 10), 0, 100, seed = 1L)
  mod2 <- gen$new(list(len = 10), 0, 100, seed = 1L)

  expect_identical(
    mod1$run(10),
    mod2$run(10, TRUE))
})


test_that("Raise suitable error if model does not support GPU", {
  gen <- dust_example("walk")
  mod <- gen$new(list(sd = 1), 0, 100, seed = 1L)
  expect_error(
    mod$run(10, TRUE),
    "GPU support not enabled for this object")
})
