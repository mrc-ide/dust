context("gpu")

test_that("Can run device version of model on cpu", {
  np <- 100
  len <- 20
  gen <- dust_example("variable")
  mod1 <- gen$new(list(len = len), 0, np, seed = 1L)
  mod2 <- gen$new(list(len = len), 0, np, seed = 1L, device_id = 0L)

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
  mod1 <- gen$new(list(len = len), 0, np, seed = 1L, device_id = 0L)
  mod2 <- gen$new(list(len = len), 0, np, seed = 1L, device_id = 0L)
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
  mod2 <- res$new(p, 0, 10, seed = 1L, pars_multi = TRUE, device_id = 0L)
  expect_identical(
    mod1$run(10),
    mod2$run(10, TRUE))
  expect_identical(
    mod1$run(13),
    mod2$run(13, TRUE))
})


test_that("Can reorder on the device", {
  res <- dust_example("variable")
  p <- list(list(len = 10, sd = 1), list(len = 10, sd = 10))

  np <- 13
  mod1 <- res$new(p, 0, np, seed = 1L, pars_multi = TRUE)
  mod2 <- res$new(p, 0, np, seed = 1L, pars_multi = TRUE, device_id = 0L)
  mod1$set_index(integer(0))
  mod2$set_index(integer(0))

  idx <- cbind(sample(np, np, TRUE), sample(np, np, TRUE))

  expect_identical(
    mod1$run(10),
    mod2$run(10, TRUE))

  mod1$reorder(idx)
  mod2$reorder(idx)

  expect_identical(
    mod1$run(13),
    mod2$run(13, TRUE))

  expect_identical(mod1$state(), mod2$state())
})


test_that("Can generate cuda compatible code", {
  info <- list(
    has_cuda = TRUE,
    cuda_version = numeric_version("10.1.0"),
    devices = data.frame(id = 0, version = 75L),
    path_cuda_lib = "/path/to/cuda",
    path_cub_include = "/path/to/cub")
  cuda <- cuda_options(info, FALSE, FALSE, FALSE, NULL)

  workdir <- tempfile()
  cache <- new.env(parent = emptyenv())
  res <- generate_dust(dust_file("examples/sirs.cpp"), TRUE, workdir, cuda,
                       cache)

  expect_setequal(
    dir(file.path(res$path, "src")),
    c("dust.hpp", "dust.cu", "Makevars"))

  txt <- readLines(file.path(res$path, "src", "Makevars"))
  expect_match(txt, "-L/path/to/cuda", all = FALSE, fixed = TRUE)
  expect_match(txt, "-I/path/to/cub", all = FALSE, fixed = TRUE)
  expect_match(txt, "-gencode=arch=compute_75,code=sm_75", all = FALSE)
})


test_that("Generate default cuda configuration", {
  skip_if_not_installed("mockery")
  mock_create <- mockery::mock(stop("no nvcc found!"))
  res <- with_mock(
    "dust:::cuda_create_test_package" = mock_create,
    cuda_configuration())
  expect_equal(res,
               list(has_cuda = FALSE,
                    cuda_version = NULL,
                    devices = NULL,
                    path_cuda_lib = NULL,
                    path_cub_include = NULL))
})


test_that("Retrieve configuration", {
  skip_if_not_installed("mockery")
  mock_create <- mockery::mock(stop("no nvcc found!"))
  mock_ctp <- mockery::mock(mock_create_test_package())
  res <- with_mock(
    "dust:::cuda_create_test_package" = mock_ctp,
    cuda_configuration(quiet = TRUE))
  mockery::expect_called(mock_ctp, 1)
  expect_equal(res, c(example_cuda_config(),
                      list(path_cuda_lib = NULL, path_cub_include = NULL)))
})


test_that("Report error if requested", {
  skip_if_not_installed("mockery")
  mock_create <- mockery::mock(stop("no nvcc found!"))
  res <- with_mock(
    "dust:::cuda_create_test_package" = mock_create,
    expect_message(cuda_configuration(),
                   "nvcc detection reported failure:.*no nvcc found!"))
})


test_that("locate cub", {
  skip_if_not_installed("mockery")
  path_good <- tempfile()
  path_bad <- tempfile()
  dir.create(file.path(path_good, "cub"), FALSE, TRUE)

  version_10 <- numeric_version("10.1.243")
  version_11 <- numeric_version("11.1.243")
  expect_error(
    cuda_path_cub_include(version_10, path_bad),
    "Did not find directory 'cub' within '.+' \\(via provided argument\\)")
  expect_equal(cuda_path_cub_include(version_10, path_good), path_good)

  expect_error(
    withr::with_envvar(
      c(DUST_PATH_CUB_INCLUDE = path_bad),
      cuda_path_cub_include(version_10, NULL)),
    "Did not find directory 'cub' within '.+' \\(via environment variable")
  withr::with_envvar(
    c(DUST_PATH_CUB_INCLUDE = path_good),
    expect_equal(cuda_path_cub_include(version_10, NULL), path_good))
  withr::with_envvar(
    c(DUST_PATH_CUB_INCLUDE = path_good),
    expect_null(cuda_path_cub_include(version_11, NULL)))

  res <- with_mock(
    "dust:::cuda_cub_path_default" = mockery::mock(path_bad, path_good, NULL),
    expect_error(
      cuda_path_cub_include(version_10, NULL),
      "Did not find directory 'cub' within '.+' \\(via default path"),
    expect_equal(cuda_path_cub_include(version_10, NULL), path_good),
    expect_error(
      cuda_path_cub_include(version_10, NULL),
      "Did not find cub headers"))
})


test_that("locate cuda libs", {
  skip_if_not_installed("mockery")
  path_good <- tempfile()
  path_bad <- tempfile()
  dir.create(path_good, FALSE, TRUE)
  file.create(file.path(path_good, "libcudart.so"))

  expect_error(
    cuda_path_cuda_lib(path_bad),
    "Did not find 'libcudart' within '.+' \\(via provided argument\\)")
  expect_equal(cuda_path_cuda_lib(path_good), path_good)

  expect_error(
    withr::with_envvar(
      c(DUST_PATH_CUDA_LIB = path_bad),
      cuda_path_cuda_lib(NULL)),
    "Did not find 'libcudart' within '.+' \\(via environment variable")
  withr::with_envvar(
    c(DUST_PATH_CUDA_LIB = path_good),
    expect_equal(cuda_path_cuda_lib(NULL), path_good))

  withr::with_envvar(
    c(DUST_PATH_CUDA_LIB = NA_character_),
    expect_null(cuda_path_cuda_lib(NULL)))
})


test_that("cuda_cub_path_default returns sensible values by R version", {
  expect_null(cuda_cub_path_default("3.6.3"))
  testthat::skip_if(getRversion() < numeric_version("4.0.0"), "old R")
  expect_equal(cuda_cub_path_default("4.0.0"),
               file.path(tools::R_user_dir("dust", "data"), "cub"))
})


test_that("install cub", {
  skip_on_cran()
  skip_if_offline()

  path <- tempfile()
  expect_equal(cuda_install_cub(path, quiet = TRUE), path)
  expect_setequal(dir(path), c("cub", "LICENSE.TXT"))
  expect_true(file.exists(file.path(path, "cub", "cub.cuh")))

  expect_error(cuda_install_cub(path, quiet = TRUE),
               "Path already exists")
})


test_that("Set the cuda options", {
  info <- example_cuda_config()
  expect_mapequal(
    cuda_options(info, FALSE, FALSE, FALSE, NULL)$flags,
    list(nvcc_flags = "-O2",
         gencode = "-gencode=arch=compute_75,code=sm_75",
         cub_include = "",
         lib_flags = ""))

  expect_mapequal(
    cuda_options(info, TRUE, TRUE, FALSE, NULL)$flags,
    list(nvcc_flags = paste("-g -G -O0 -pg --generate-line-info",
                            "-DDUST_ENABLE_CUDA_PROFILER"),
         gencode = "-gencode=arch=compute_75,code=sm_75",
         cub_include = "",
         lib_flags = ""))

  info$path_cub_include <- "/path/to/cub"
  info$path_cuda_lib <- "/path/to/cuda"

  expect_mapequal(
    cuda_options(info, FALSE, FALSE, TRUE, NULL)$flags,
    list(nvcc_flags = "-O2 --use_fast_math",
         gencode = "-gencode=arch=compute_75,code=sm_75",
         cub_include = "-I/path/to/cub",
         lib_flags = "-L/path/to/cuda"))

  expect_mapequal(
    cuda_options(info, FALSE, FALSE, FALSE, "--maxregcount=100")$flags,
    list(nvcc_flags = "-O2 --maxregcount=100",
         gencode = "-gencode=arch=compute_75,code=sm_75",
         cub_include = "-I/path/to/cub",
         lib_flags = "-L/path/to/cuda"))
  expect_mapequal(
    cuda_options(info, FALSE, FALSE, FALSE,
                 c("--maxregcount=100", "--use_fast_math"))$flags,
    list(nvcc_flags = "-O2 --maxregcount=100 --use_fast_math",
         gencode = "-gencode=arch=compute_75,code=sm_75",
         cub_include = "-I/path/to/cub",
         lib_flags = "-L/path/to/cuda"))
})


test_that("can create sensible cuda options", {
  skip_if_not_installed("mockery")
  opts <- cuda_options(example_cuda_config(), FALSE, FALSE, FALSE, NULL)
  mock_dust_cuda_options <- mockery::mock(opts, cycle = TRUE)

  mockery::stub(cuda_check, "dust_cuda_options", mock_dust_cuda_options)
  expect_null(cuda_check(NULL))
  expect_null(cuda_check(FALSE))
  expect_equal(cuda_check(TRUE), opts)
  expect_equal(cuda_check(opts), opts)
  expect_error(cuda_check("something"),
               "'x' must be a cuda_options")
})


test_that("Can generate test package code", {
  res <- cuda_create_test_package("/path/to/cuda")
  expect_true(file.exists(res$path))
  expect_setequal(
    dir(res$path),
    c("DESCRIPTION", "NAMESPACE", "src"))
  expect_setequal(
    dir(file.path(res$path, "src")),
    c("dust.cu", "dust.hpp", "Makevars"))
  txt <- readLines(file.path(res$path, "src", "Makevars"))
  expect_match(txt, "-L/path/to/cuda", all = FALSE, fixed = TRUE)
  expect_false(any(grepl("gencode", txt)))
})


test_that("High-level interface caches", {
  skip_if_not_installed("mockery")
  prev <- cache$cuda
  on.exit(cache$cuda <- NULL)

  cfg1 <- list(has_cuda = FALSE)
  cfg2 <- example_cuda_config()
  cache$cuda <- NULL
  mock_cuda_configuration <- mockery::mock(cfg1, cfg2)
  path_lib <- "/path/to/lib"
  path_include <- "/path/to/include"

  mockery::stub(dust_cuda_configuration, "cuda_configuration",
                mock_cuda_configuration)

  ## Cache miss, call:
  expect_identical(
    dust_cuda_configuration(path_lib, path_include, FALSE, TRUE),
    cfg1)
  expect_identical(cache$cuda, cfg1)
  mockery::expect_called(mock_cuda_configuration, 1)
  expect_equal(
    mockery::mock_args(mock_cuda_configuration)[[1]],
    list(path_lib, path_include, FALSE))

  ## Cache hit, no call:
  expect_identical(
    dust_cuda_configuration(path_lib, path_include, FALSE),
    cfg1)
  mockery::expect_called(mock_cuda_configuration, 1)

  ## Cache invalidation, call:
  expect_identical(
    dust_cuda_configuration(path_lib, path_include, FALSE, TRUE),
    cfg2)
  expect_identical(cache$cuda, cfg2)
  mockery::expect_called(mock_cuda_configuration, 2)
  expect_equal(
    mockery::mock_args(mock_cuda_configuration)[[1]],
    list(path_lib, path_include, FALSE))

  ## Cache hit, no call:
  expect_identical(
    dust_cuda_configuration(path_lib, path_include, FALSE),
    cfg2)
  mockery::expect_called(mock_cuda_configuration, 2)
})


test_that("high level interface to cuda options", {
  skip_if_not_installed("mockery")

  cfg1 <- example_cuda_config()
  cfg2 <- list(has_cuda = FALSE)
  mock_cuda_configuration <- mockery::mock(cfg1, cfg2)

  mockery::stub(dust_cuda_options, "dust_cuda_configuration",
                mock_cuda_configuration)

  path_lib <- "/path/cuda/lib"
  res <- dust_cuda_options(path_cuda_lib = path_lib)
  expect_identical(res, cuda_options(cfg1, FALSE, FALSE, FALSE, NULL))
  mockery::expect_called(mock_cuda_configuration, 1)
  expect_equal(
    mockery::mock_args(mock_cuda_configuration)[[1]],
    list(path_cuda_lib = path_lib))

  expect_error(
    dust_cuda_options(path_cuda_lib = path_lib),
    "cuda not supported on this machine")

  mockery::expect_called(mock_cuda_configuration, 2)
  expect_equal(
    mockery::mock_args(mock_cuda_configuration)[[2]],
    list(path_cuda_lib = path_lib))
})


test_that("Can provide device id", {
  np <- 100
  len <- 20
  gen <- dust_example("variable")
  expect_error(
    gen$new(list(len = len), 0, np, device_id = 2),
    "Invalid 'device_id' 2, must be at most 0")
  mod <- gen$new(list(len = len), 0, np, device_id = -10)
  expect_equal(r6_private(mod)$device_id_, -10)
  mod <- gen$new(list(len = len), 0, np, device_id = NULL)
  expect_equal(r6_private(mod)$device_id_, -1)
  mod <- gen$new(list(len = len), 0, np, device_id = 0L)
  expect_equal(r6_private(mod)$device_id_, 0)
})


test_that("Error if using gpu features without device", {
  np <- 100
  len <- 20
  gen <- dust_example("variable")
  mod <- gen$new(list(len = len), 0, np, seed = 1L, device_id = -1L)
  expect_error(
    mod$run(10, TRUE),
    "Can't refresh a non-existent device")
})


test_that("Can provide device id to non-gpu model with no effect", {
  np <- 10
  gen <- dust_example("sir")
  expect_error(
    gen$new(list(), 0, np, device_id = 2),
    "Invalid 'device_id' 2, must be at most 0")
  mod <- gen$new(list(), 0, np, device_id = -10)
  expect_equal(r6_private(mod)$device_id_, -10)
  mod <- gen$new(list(), 0, np, device_id = NULL)
  expect_equal(r6_private(mod)$device_id_, -1)
  mod <- gen$new(list(), 0, np, device_id = 0L)
  expect_equal(r6_private(mod)$device_id_, 0)
})


test_that("Can use sirs gpu model", {
  gen <- dust(dust_file("examples/sirs.cpp"), quiet = TRUE)
  np <- 100
  len <- 20

  mod1 <- gen$new(list(), 0, np, seed = 1L)
  mod2 <- gen$new(list(), 0, np, seed = 1L, device_id = 0L)

  expect_identical(
    mod1$run(10),
    mod2$run(10, TRUE))
  expect_identical(
    mod1$run(13),
    mod2$run(13, TRUE))

  # Test that device_select run is cached
  mod1$set_index(1L)
  mod2$set_index(1L)
  expect_identical(
    mod1$run(15),
    mod2$run(15, TRUE))
  expect_identical(
    mod1$run(15),
    mod2$run(15, TRUE))
})

test_that("Missing GPU comparison function errors", {
  dat <- example_volatility()
  gen <- dust_example("volatility")
  mod <- gen$new(list(sd = 1), 0, 100, seed = 1L)
  mod$set_data(dust_data(dat$data))
  mod$run(10)
  expect_null(mod$compare_data(TRUE))
})

test_that("Comparison function can be run on the GPU", {
  dat <- example_sirs()

  np <- 10

  mod_h <- dat$model$new(list(), 0, np, seed = 10L)
  mod_h$set_data(dat$dat_dust)
  mod_h$run(10)
  weights_h <- mod_h$compare_data()

  mod_d <- dat$model$new(list(), 0, np, seed = 10L, device_id = 0L)
  mod_d$set_data(dat$dat_dust)
  mod_d$run(10)
  weights_d <- mod_d$compare_data(TRUE)

  expect_identical(weights_h, weights_d)
})

test_that("Can run a single particle filter on the GPU", {
  dat <- example_sirs()

  np <- 10

  mod_h <- dat$model$new(list(), 0, np, seed = 10L)
  mod_h$set_data(dat$dat_dust)
  ans_h <- mod_h$filter(save_trajectories = TRUE,
                        step_snapshot = c(4, 16))

  mod_d <- dat$model$new(list(), 0, np, seed = 10L, device_id = 0L)
  mod_d$set_data(dat$dat_dust)
  ans_d <- mod_d$filter(device = TRUE,
                        save_trajectories = TRUE,
                        step_snapshot = c(4, 16))

  expect_equal(ans_h$log_likelihood, ans_d$log_likelihood)
  expect_identical(ans_h$trajectories, ans_d$trajectories)
  expect_identical(ans_h$snapshots, ans_d$snapshots)
})

test_that("Can run multiple particle filters on the GPU", {
  dat <- example_sirs()

  np <- 10
  pars <- list(list(beta = 0.2), list(beta = 0.1))

  mod_h <- dat$model$new(pars, 0, np, seed = 10L, pars_multi = TRUE)
  mod_h$set_data(dust_data(dat$dat, multi = 2))
  ans_h <- mod_h$filter(save_trajectories = TRUE,
                        step_snapshot = c(4, 16))

  mod_d <- dat$model$new(pars, 0, np, seed = 10L, pars_multi = TRUE,
                         device_id = 0L)
  mod_d$set_data(dust_data(dat$dat, multi = 2))
  ans_d <- mod_d$filter(device = TRUE,
                        save_trajectories = TRUE,
                        step_snapshot = c(4, 16))

  expect_equal(ans_h$log_likelihood, ans_d$log_likelihood)
  expect_identical(ans_h$trajectories, ans_d$trajectories)
  expect_identical(ans_h$snapshots, ans_d$snapshots)
})