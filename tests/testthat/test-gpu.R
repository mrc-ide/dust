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


test_that("Can run multiple parameter sets", {
  res <- dust_example("variable")
  p <- list(list(len = 10, sd = 1), list(len = 10, sd = 10))
  mod1 <- res$new(p, 0, 10, seed = 1L, pars_multi = TRUE)
  mod2 <- res$new(p, 0, 10, seed = 1L, pars_multi = TRUE)
  expect_identical(
    mod1$run(10),
    mod2$run(10, TRUE))
  expect_identical(
    mod1$run(13),
    mod2$run(13, TRUE))
})


test_that("Can generate cuda compatible code", {
  info <- list(
    has_cuda = TRUE,
    cuda_version = numeric_version("10.1.0"),
    devices = data.frame(id = 0, version = 75L),
    path_cuda_lib = "/path/to/cuda",
    path_cub_include = "/path/to/cub")
  cuda <- cuda_options(info, FALSE, FALSE, FALSE)

  workdir <- tempfile()
  cache <- new.env(parent = emptyenv())
  res <- generate_dust(dust_file("examples/sirs.cpp"), TRUE, workdir, cuda,
                       cache)

  expect_setequal(
    dir(file.path(res$path, "src")),
    c("dust.hpp", "dust.cu", "cpp11.cpp", "Makevars"))

  txt <- readLines(file.path(res$path, "src", "Makevars"))
  expect_match(txt, "-L/path/to/cuda", all = FALSE, fixed = TRUE)
  expect_match(txt, "-I/path/to/cub", all = FALSE, fixed = TRUE)
  expect_match(txt, "-gencode=arch=compute_75,code=sm_75", all = FALSE)
})
