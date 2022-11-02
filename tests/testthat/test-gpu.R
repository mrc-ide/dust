test_that("Can run gpu version of model on cpu", {
  np <- 100
  len <- 20
  gen <- dust_example("variable")
  mod1 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = NULL)
  mod2 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)

  expect_false(mod1$uses_gpu())
  expect_false(mod2$uses_gpu())
  expect_true(mod2$uses_gpu(TRUE))

  expect_identical(
    mod1$run(10),
    mod2$run(10))
  expect_identical(
    mod1$run(13),
    mod2$run(13))
})


test_that("Raise suitable errors if models do not support GPU", {
  gen <- dust_example("walk")
  mod <- gen$new(list(sd = 1), 0, 100, seed = 1L)
  expect_error(
    gen$new(list(sd = 1), 0, 100, seed = 1L, gpu_config = 0L),
    "GPU support not enabled for this object")
})


test_that("Can run multiple parameter sets", {
  res <- dust_example("variable")
  p <- list(list(len = 3, sd = 1), list(len = 3, sd = 1))
  mod1 <- res$new(p, 0, 5, seed = 1L, pars_multi = TRUE)
  mod2 <- res$new(p, 0, 5, seed = 1L, pars_multi = TRUE, gpu_config = 0L)

  y1 <- mod1$run(10)
  y2 <- mod2$run(10)

  expect_identical(
    mod1$run(10),
    mod2$run(10))
  expect_identical(
    mod1$run(13),
    mod2$run(13))
})


test_that("Can reorder on the gpu", {
  res <- dust_example("variable")
  p <- list(list(len = 10, sd = 1), list(len = 10, sd = 10))

  np <- 13
  mod1 <- res$new(p, 0, np, seed = 1L, pars_multi = TRUE)
  mod2 <- res$new(p, 0, np, seed = 1L, pars_multi = TRUE, gpu_config = 0L)
  mod1$set_index(2L)
  mod2$set_index(2L)

  idx <- cbind(sample(np, np, TRUE), sample(np, np, TRUE))

  expect_identical(
    mod1$run(10),
    mod2$run(10))

  mod1$reorder(idx)
  mod2$reorder(idx)

  expect_identical(
    mod1$run(13),
    mod2$run(13))

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
  res <- generate_dust(dust_file("examples/sirs.cpp"), TRUE, workdir, cuda,
                       TRUE, TRUE)

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
  mockery::stub(cuda_configuration, "cuda_create_test_package",
                mock_create)
  res <- cuda_configuration(quiet = TRUE)
  mockery::expect_called(mock_create, 1)
  expect_equal(res,
               list(has_cuda = FALSE,
                    cuda_version = NULL,
                    devices = NULL,
                    path_cuda_lib = NULL,
                    path_cub_include = NULL))
})


test_that("Retrieve configuration", {
  skip_if_not_installed("mockery")
  mock_create <- mockery::mock(mock_create_test_package())
  mockery::stub(cuda_configuration, "cuda_create_test_package",
                mock_create)
  res <- cuda_configuration(quiet = TRUE)
  mockery::expect_called(mock_create, 1)
  expect_equal(res, c(example_cuda_config(),
                      list(path_cuda_lib = NULL, path_cub_include = NULL)))
})


test_that("Report error if requested", {
  skip_if_not_installed("mockery")
  mock_create <- mockery::mock(stop("no nvcc found!"))
  mockery::stub(cuda_configuration, "cuda_create_test_package",
                mock_create)
  expect_message(cuda_configuration(),
                 "nvcc detection reported failure:.*no nvcc found!")
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

  mock_cuda_path <- mockery::mock(path_bad, path_good, NULL)
  mockery::stub(cuda_path_cub_include,
                "cuda_cub_path_default",
                mock_cuda_path)
  expect_error(
    cuda_path_cub_include(version_10, NULL),
    "Did not find directory 'cub' within '.+' \\(via default path")
  expect_equal(cuda_path_cub_include(version_10, NULL), path_good)
  expect_error(
    cuda_path_cub_include(version_10, NULL),
    "Did not find cub headers")
})


test_that("locate cuda libs", {
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
  expect_message(p <- cuda_install_cub(path, quiet = TRUE),
                 "Installing cub headers")
  expect_equal(p, path)
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
    gen$new(list(len = len), 0, np, gpu_config = 2),
    "Invalid 'device_id' 2, must be at most 0")
  expect_error(
    gen$new(list(len = len), 0, np, gpu_config = -1),
    "'device_id' must be non-negative")
  mod <- gen$new(list(len = len), 0, np, gpu_config = NULL)
  expect_equal(r6_private(mod)$gpu_config_$device_id, NULL)
  mod <- gen$new(list(len = len), 0, np, gpu_config = 0L)
  expect_equal(r6_private(mod)$gpu_config_$device_id, 0)
})


test_that("Can control device run block size", {
  np <- 100
  len <- 20
  gen <- dust_example("variable")
  mod <- gen$new(list(len = len), 0, np,
                 gpu_config = list(device_id = 0, run_block_size = 512))
  expect_equal(r6_private(mod)$gpu_config_,
               list(real_gpu = FALSE,
                    device_id = 0,
                    shared_size = 0,
                    run_block_size = 512))
})


test_that("Can use sirs gpu model", {
  gen <- dust_example("sirs")
  np <- 100
  len <- 20

  mod1 <- gen$new(list(), 0, np, seed = 1L)
  mod2 <- gen$new(list(), 0, np, seed = 1L, gpu_config = 0L)

  expect_identical(
    mod1$run(10),
    mod2$run(10))
  expect_identical(
    mod1$run(13),
    mod2$run(13))

  # Test that device_select run is cached
  mod1$set_index(1L)
  mod2$set_index(1L)
  expect_identical(
    mod1$run(15),
    mod2$run(15))
  expect_identical(
    mod1$run(15),
    mod2$run(15))
})


test_that("Can simulate sirs gpu model", {
  res <- dust_example("sirs")

  times <- seq(0, 100, by = 10)
  np <- 20
  mod_d <- res$new(list(), 0, np, seed = 1L, gpu_config = 0L)
  mod_d$set_index(c(1, 3))
  y <- mod_d$simulate(times)
  expect_equal(dim(y), c(2, np, length(times)))

  mod_h <- res$new(list(), 0, np, seed = 1L)
  expect_identical(mod_h$simulate(times)[c(1, 3), , , drop = FALSE], y)
})


test_that("Comparison function can be run on the GPU", {
  dat <- example_sirs()

  np <- 10

  mod_h <- dat$model$new(list(), 0, np, seed = 10L)
  mod_h$set_data(dat$dat_dust)
  mod_h$run(4)
  weights_h <- mod_h$compare_data()

  mod_d <- dat$model$new(list(), 0, np, seed = 10L, gpu_config = 0L)
  mod_d$set_data(dat$dat_dust)
  mod_d$run(4)
  weights_d <- mod_d$compare_data()

  expect_identical(weights_h, weights_d)
})


test_that("Can run a single particle filter on the GPU", {
  dat <- example_sirs()

  np <- 10

  mod_h <- dat$model$new(list(), 0, np, seed = 10L)
  mod_h$set_data(dat$dat_dust)
  ans_h <- mod_h$filter(save_trajectories = TRUE,
                        time_snapshot = c(4, 16))

  mod_d <- dat$model$new(list(), 0, np, seed = 10L, gpu_config = 0L)
  mod_d$set_data(dat$dat_dust)
  ans_d <- mod_d$filter(save_trajectories = TRUE,
                        time_snapshot = c(4, 16))

  expect_equal(ans_h$log_likelihood, ans_d$log_likelihood)
  expect_identical(ans_h$trajectories, ans_d$trajectories)
  expect_identical(ans_h$snapshots, ans_d$snapshots)
})


test_that("Can run particle filter without collecting state on GPU", {
  dat <- example_sirs()

  np <- 10

  mod_h <- dat$model$new(list(), 0, np, seed = 10L)
  mod_h$set_data(dat$dat_dust)
  ans_h <- mod_h$filter()

  mod_d <- dat$model$new(list(), 0, np, seed = 10L, gpu_config = 0L)
  mod_d$set_data(dat$dat_dust)
  ans_d <- mod_d$filter()

  expect_equal(ans_h$log_likelihood, ans_d$log_likelihood)
})


test_that("Can run GPU kernels using shared memory", {
  dat <- example_sirs()

  # Larger particle size makes multiple blocks be used
  np <- 256

  mod_h <- dat$model$new(list(), 0, np, seed = 10L)
  mod_h$set_data(dat$dat_dust)
  ans_h <- mod_h$filter(save_trajectories = TRUE,
                        time_snapshot = c(4, 16))

  mod_d <- dat$model$new(list(), 0, np, seed = 10L, gpu_config = 0L)
  mod_d$set_data(dat$dat_dust)
  ans_d <- mod_d$filter(save_trajectories = TRUE,
                        time_snapshot = c(4, 16))

  expect_equal(ans_h$log_likelihood, ans_d$log_likelihood)
  expect_identical(ans_h$trajectories, ans_d$trajectories)
  expect_identical(ans_h$snapshots, ans_d$snapshots)
})


test_that("Can run multiple particle filters on the GPU", {
  dat <- example_sirs()

  np <- 10
  pars <- list(list(beta = 0.2, I0 = 5), list(beta = 0.1, I0 = 20))

  mod_h <- dat$model$new(pars, 0, np, seed = 10L, pars_multi = TRUE)
  mod_h$set_data(dust_data(dat$dat, multi = 2))
  ans_h <- mod_h$filter(save_trajectories = TRUE,
                        time_snapshot = c(4, 16))

  mod_d <- dat$model$new(pars, 0, np, seed = 10L, pars_multi = TRUE,
                         gpu_config = 0L)
  mod_d$set_data(dust_data(dat$dat, multi = 2))
  ans_d <- mod_d$filter(save_trajectories = TRUE,
                        time_snapshot = c(4, 16))

  expect_equal(ans_h$log_likelihood, ans_d$log_likelihood)
  expect_identical(ans_h$trajectories, ans_d$trajectories)
  expect_identical(ans_h$snapshots, ans_d$snapshots)
})


test_that("Can run and simulate with nontrivial index", {
  np <- 100
  len <- 20
  gen <- dust_example("variable")

  # Test run
  mod1 <- gen$new(list(len = len), 0, np, seed = 1L)
  mod2 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)

  index <- c(4:7, 19:16, 10:12)

  mod1$set_index(index)
  mod2$set_index(index)

  expect_identical(
    mod1$run(10),
    mod2$run(10))
  expect_identical(
    mod1$run(13),
    mod2$run(13))

  # Test simulate
  times <- seq(0, 100, by = 10)

  mod3 <- gen$new(list(len = len), 0, np, seed = 1L)
  mod4 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)
  mod3$set_index(index)
  mod4$set_index(index)

  y3 <- mod3$simulate(times)
  y4 <- mod4$simulate(times)

  expect_equal(dim(y3), c(length(index), np, length(times)))
  expect_identical(y3, y4)
})


test_that("Can fit a small model into shared", {
  n_state <- 100
  n_state_full <- 202
  res <- test_cuda_pars(0, 2000, 2000,
                        n_state, n_state_full,
                        20, 30, 0,
                        40000)
  expect_true(res$run$shared_int)
  expect_true(res$run$shared_real)
  expect_true(res$compare$shared_int)
  expect_true(res$compare$shared_real)
  ## 200 is 20 * 4 + 30 * 4
  expect_equal(res$run$shared_size_bytes, 200)
  expect_equal(res$compare$shared_size_bytes, 200)

  expect_equal(res$run$block_size, 128)
  expect_equal(res$run$block_count, 16)
  expect_equal(res$compare$block_size, 128)
  expect_equal(res$compare$block_count, 16)

  expect_equal(res$reorder, create_launch_control(128, 3157))
  expect_equal(res$scatter, create_launch_control(64, 6313))
  expect_equal(res$index_scatter, create_launch_control(64, 3125))
  expect_equal(res$interval, create_launch_control(128, 3157))
})


test_that("Can fit a small model into shared, with data", {
  n_state_full <- n_state <- 100
  res <- test_cuda_pars(0, 2000, 2000,
                        n_state, n_state_full,
                        20, 30, 32,
                        40000)
  expect_true(res$run$shared_int)
  expect_true(res$run$shared_real)
  expect_true(res$compare$shared_int)
  expect_true(res$compare$shared_real)
  ## 300 is 20 * 4 + 30 * 4
  expect_equal(res$run$shared_size_bytes, 200) #
  ## 240 is 20 * 4 + 30 * 4 + 8 + 32; the +8 here is for the alignment
  expect_equal(res$compare$shared_size_bytes, 240)
})


test_that("Will spill a large model out of shared, but keep ints", {
  n_state_full <- n_state <- 100
  res <- test_cuda_pars(0, 2000, 2000,
                        n_state, n_state_full,
                        200, 50000, 32,
                        40000)
  expect_true(res$run$shared_int)
  expect_false(res$run$shared_real)
  expect_true(res$compare$shared_int)
  expect_false(res$compare$shared_real)
  expect_equal(res$run$shared_size_bytes, 800) # i.e., 200 * 4
  expect_equal(res$compare$shared_size_bytes, 800) # i.e., 200 * 4
})


test_that("Will spill a really large model out of shared", {
  n_state_full <- n_state <- 100
  res <- test_cuda_pars(0, 2000, 2000,
                        n_state, n_state_full,
                        20000, 10000, 0,
                        40000)
  expect_false(res$run$shared_int)
  expect_false(res$run$shared_real)
  expect_false(res$compare$shared_int)
  expect_false(res$compare$shared_real)
  expect_equal(res$run$shared_size_bytes, 0)
  expect_equal(res$compare$shared_size_bytes, 0)
})


test_that("Can tune block size", {
  n_state <- 100
  n_state_full <- 202
  config <- list(device_id = 0, run_block_size = 512)
  res <- test_cuda_pars(config, 2000, 2000,
                        n_state, n_state_full,
                        20, 30, 0,
                        40000)
  expect_true(res$run$shared_int)
  expect_true(res$run$shared_real)
  expect_true(res$compare$shared_int)
  expect_true(res$compare$shared_real)
  ## 200 is 20 * 4 + 30 * 4
  expect_equal(res$run$shared_size_bytes, 200)
  expect_equal(res$compare$shared_size_bytes, 200)

  expect_equal(res$run$block_size, 512)
  expect_equal(res$run$block_count, 4)
  expect_equal(res$compare$block_size, 128)
  expect_equal(res$compare$block_count, 16)

  expect_equal(res$reorder, create_launch_control(128, 3157))
  expect_equal(res$scatter, create_launch_control(64, 6313))
  expect_equal(res$index_scatter, create_launch_control(64, 3125))
  expect_equal(res$interval, create_launch_control(128, 3157))
})


test_that("correctly organise blocks with multiple parameters", {
  f <- function(n_pars, n_particles, block_size = 128) {
    res <- test_cuda_pars(list(device_id = 0, run_block_size = block_size),
                          n_particles * n_pars, n_particles, n_pars,
                          4, 1, 1, 0, 100)
    c(res$run$block_count, res$run$block_size)
  }

  expect_equal(f(4, 32), c(4, 32))
  expect_equal(f(4, 34), c(4, 64))
  expect_equal(f(4, 64), c(4, 64))
  expect_equal(f(4, 68), c(4, 96))
  expect_equal(f(4, 128), c(4, 128))
  expect_equal(f(4, 130), c(8, 128))
  expect_equal(f(4, 130, 256), c(4, 160))
})


test_that("Can validate block size", {
  n_state <- 100
  n_state_full <- 202
  config <- list(device_id = 0, run_block_size = -512)
  expect_error(
    test_cuda_pars(config, 2000, 2000,
                   n_state, n_state_full,
                   20, 30, 0,
                   40000),
    "'run_block_size' must be positive (but was -512)",
    fixed = TRUE)

  config$run_block_size <- 513
  expect_error(
    test_cuda_pars(config, 2000, 2000,
                   n_state, n_state_full,
                   20, 30, 0,
                   40000),
    "'run_block_size' must be a multiple of 32 (but was 513)",
    fixed = TRUE)
})


test_that("Can't run deterministically on the gpu", {
  np <- 100
  len <- 20
  gen <- dust_example("variable")
  expect_error(
    gen$new(list(len = len), 0, np, seed = 1L, deterministic = TRUE,
            gpu_config = 0L),
    "Deterministic models not supported on gpu")
})


test_that("can update parameters, leaving state alone", {
  np <- 100
  len <- 20
  gen <- dust_example("variable")
  pa <- list(len = len, sd = 1)
  pb <- list(len = len, sd = 2)

  mod1 <- gen$new(pa, 0, np, seed = 1L, gpu_config = NULL)
  mod2 <- gen$new(pa, 0, np, seed = 1L, gpu_config = 0L)

  mod1$run(5)
  mod2$run(5)

  mod1$update_state(pars = pb)
  mod2$update_state(pars = pb)
  expect_identical(mod1$state(), mod2$state())

  y1 <- mod1$run(10)
  y2 <- mod2$run(10)
  expect_identical(y1, y2)
})


test_that("Can reset time", {
  np <- 100
  len <- 20
  gen <- dust_example("variable")
  mod1 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = NULL)
  mod2 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)

  y1 <- mod1$run(5)
  y2 <- mod2$run(5)

  mod1$update_state(time = 0)
  mod2$update_state(time = 0)
  expect_identical(mod1$state(), mod2$state())
  expect_identical(mod2$time(), 0L)

  y1 <- mod1$run(10)
  y2 <- mod2$run(10)
  expect_identical(y1, y2)
})


test_that("can update parameters and time, resetting state", {
  np <- 5
  len <- 3
  gen <- dust_example("variable")
  pa <- list(len = len, sd = 1)
  pb <- list(len = len, sd = 2)
  mod1 <- gen$new(pa, 0, np, seed = 1L, gpu_config = NULL)
  mod2 <- gen$new(pa, 0, np, seed = 1L, gpu_config = 0L)
  y1 <- mod1$run(5)
  y2 <- mod2$run(5)
  expect_equal(y1, y2)

  ## Doing this totally breaks the stepping...
  mod1$update_state(pars = pb, time = 0)
  mod2$update_state(pars = pb, time = 0)

  expect_identical(mod1$state(), mod2$state())
  expect_equal(mod2$time(), 0)
  y1 <- mod1$run(20)
  y2 <- mod2$run(20)
  expect_identical(y1, y2)
})


test_that("Can set state", {
  np <- 5
  len <- 3
  gen <- dust_example("variable")

  mod1 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = NULL)
  mod2 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)

  s1 <- matrix(runif(len * np), len, np)
  mod1$update_state(state = s1)
  mod2$update_state(state = s1)
  expect_equal(mod2$state(), s1)
  y1 <- mod1$run(10)
  y2 <- mod2$run(10)
  expect_equal(y1, y2)

  s2 <- matrix(runif(len * np), len, np)
  mod1$update_state(state = s2)
  mod2$update_state(state = s2)
  expect_equal(mod2$state(), s2)
  y1 <- mod1$run(19)
  y2 <- mod2$run(19)
  expect_equal(y1, y2)
})


test_that("Can extract partial state", {
  np <- 5
  len <- 20
  gen <- dust_example("variable")
  mod <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)
  m <- matrix(as.numeric(seq_len(np * len)), len)
  mod$update_state(state = m)
  idx <- sample(len, 8)
  expect_equal(mod$state(idx), m[idx, ])
})


test_that("can extract and reset rng state", {
  np <- 5
  len <- 20
  gen <- dust_example("variable")
  mod1 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = NULL)
  mod2 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)

  r1 <- mod1$rng_state()
  r2 <- mod2$rng_state()
  expect_identical(r1, r2)

  y1 <- mod1$run(10)
  y2 <- mod2$run(10)
  expect_identical(y1, y2)

  expect_identical(mod1$rng_state(), mod1$rng_state())

  mod1$set_rng_state(rev(r1))
  mod2$set_rng_state(rev(r1))

  y1 <- mod1$run(20)
  y2 <- mod2$run(20)
  expect_identical(y1, y2)

  expect_identical(mod1$rng_state(), mod1$rng_state())
})


test_that("Can update number of threads", {
  np <- 100
  len <- 20
  gen <- dust_example("variable")
  mod <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)
  expect_equal(mod$set_n_threads(2), 1)
  expect_equal(mod$n_threads(), 2)
})


test_that("Can resample particles", {
  ## This one suggests a bug in how we're doing the fiddly index
  ## calculation (actual resample does look correct though)
  res <- dust_example("variable")
  obj <- res$new(list(len = 5), 0, 7, seed = 1L, gpu_config = 0L)
  m <- matrix(as.numeric(1:35), 5, 7)
  obj$update_state(state = m)
  rng <- dust_rng$new(obj$rng_state(last_only = TRUE))
  u <- rng$random_real(1)
  w <- runif(obj$n_particles())

  idx <- obj$resample(w)

  expect_equal(idx, resample_index(w, u))
  expect_equal(obj$state(), m[, idx])
  expect_equal(rng$state(), obj$rng_state(last_only = TRUE))
})


test_that("Detect inconsistent parameter set size", {
  res <- dust_example("variable")
  p <- list(list(len = 3, sd = 1), list(len = 10, sd = 1))
  msg <- paste("'pars' created inconsistent state size:",
               "expected length 3 but parameter set 2 created length 10")
  expect_error(
    res$new(p, 0, 5, seed = 1L, pars_multi = TRUE),
    msg)
  expect_error(
    res$new(p, 0, 5, seed = 1L, pars_multi = TRUE, gpu_config = 0L),
    msg)
})


test_that("Can't set vector of times into gpu object", {
  np <- 100
  len <- 20
  gen <- dust_example("variable")
  mod <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)
  expect_error(
    mod$update_state(time = seq_len(np)),
    "Expected 'time' to be scalar")
})


test_that("Set parameters in multiparameter object without updating state", {
  np <- 100
  len <- 20
  gen <- dust_example("variable")
  pa <- list(list(len = len, sd = 1), list(len = len, sd = 10))
  pb <- list(list(len = len, sd = 2), list(len = len, sd = 20))

  mod1 <- gen$new(pa, 0, np, seed = 1L, pars_multi = TRUE, gpu_config = NULL)
  mod2 <- gen$new(pa, 0, np, seed = 1L, pars_multi = TRUE, gpu_config = 0L)

  y1 <- mod1$run(5)
  y2 <- mod2$run(5)
  expect_equal(y1, y1)

  mod1$update_state(pars = pb)
  mod2$update_state(pars = pb)
  expect_identical(mod2$state(), y2)

  y1 <- mod1$run(10)
  y2 <- mod2$run(10)
  expect_identical(y1, y2)
})


test_that("can update parameters and time, resetting state", {
  np <- 5
  len <- 3
  gen <- dust_example("variable")
  pa <- list(list(len = len, sd = 1), list(len = len, sd = 10))
  pb <- list(list(len = len, sd = 2), list(len = len, sd = 20))

  mod1 <- gen$new(pa, 0, np, seed = 1L, pars_multi = TRUE, gpu_config = NULL)
  mod2 <- gen$new(pa, 0, np, seed = 1L, pars_multi = TRUE, gpu_config = 0L)

  y1 <- mod1$run(5)
  y2 <- mod2$run(5)
  expect_equal(y1, y2)

  ## Doing this totally breaks the stepping...
  mod1$update_state(pars = pb, time = 0)
  mod2$update_state(pars = pb, time = 0)

  expect_identical(mod1$state(), mod2$state())
  expect_equal(mod2$time(), 0)
  y1 <- mod1$run(20)
  y2 <- mod2$run(20)
  expect_identical(y1, y2)
})


test_that("Particles are initialised based on time", {
  skip_for_compilation()
  path <- "examples/init.cpp"
  res <- dust(path, quiet = TRUE)

  mod <- res$new(list(sd = 1), 0, 5, gpu_config = 0L)
  expect_equal(mod$state(), matrix(0, 1, 5))
  mod$update_state(list(sd = 1), time = 2)
  expect_equal(mod$state(), matrix(2, 1, 5))
  mod$update_state(list(sd = 1), time = 3)
  expect_equal(mod$state(), matrix(3, 1, 5))
  mod <- res$new(list(sd = 1), 4, 5, gpu_config = 0L)
  expect_equal(mod$state(), matrix(4, 1, 5))
})


test_that("Can partially run filter for the gpu model", {
  dat <- example_sirs()

  np <- 10

  mod_h <- dat$model$new(list(), 0, np, seed = 10L)
  mod_h$set_data(dat$dat_dust)
  ans_h1 <- mod_h$filter(100,
                         save_trajectories = TRUE,
                         time_snapshot = c(4, 16))
  ans_h2 <- mod_h$filter(400,
                         save_trajectories = TRUE)

  mod_d <- dat$model$new(list(), 0, np, seed = 10L, gpu_config = 0L)
  mod_d$set_data(dat$dat_dust)
  ans_d1 <- mod_d$filter(100,
                         save_trajectories = TRUE,
                         time_snapshot = c(4, 16))
  ans_d2 <- mod_d$filter(400,
                         save_trajectories = TRUE)

  expect_equal(ans_h1$log_likelihood, ans_d1$log_likelihood)
  expect_equal(ans_h2$log_likelihood, ans_d2$log_likelihood)
  expect_identical(ans_h1$trajectories, ans_d1$trajectories)
  expect_identical(ans_h2$trajectories, ans_d2$trajectories)
  expect_identical(ans_h1$snapshots, ans_d1$snapshots)
  expect_identical(ans_h2$snapshots, ans_d2$snapshots) # NULL
})


test_that("Can run on device with different data sets", {
  dat <- example_sirs()

  np <- 10
  pars <- list(list(beta = 0.2, I0 = 5), list(beta = 0.1, I0 = 20))

  d <- rbind(
    data_frame(time = dat$dat$time,
               incidence = dat$dat$incidence,
               group = "a"),
    data_frame(time = dat$dat$time,
               incidence = round(dat$dat$incidence * 1.2),
               group = "b"))
  d$group <- factor(d$group)

  mod_h <- dat$model$new(pars, 0, np, seed = 10L, pars_multi = TRUE)
  mod_h$set_data(dust_data(d, multi = "group"))
  ans_h <- mod_h$filter(save_trajectories = TRUE,
                        time_snapshot = c(4, 16))

  mod_d <- dat$model$new(pars, 0, np, seed = 10L, pars_multi = TRUE,
                         gpu_config = 0L)
  mod_d$set_data(dust_data(d, multi = "group"))
  ans_d <- mod_d$filter(save_trajectories = TRUE,
                        time_snapshot = c(4, 16))

  expect_equal(ans_h$log_likelihood, ans_d$log_likelihood)
  expect_identical(ans_h$trajectories, ans_d$trajectories)
  expect_identical(ans_h$snapshots, ans_d$snapshots)
})


test_that("Can run multiple particle filters with shared data on the GPU", {
  dat <- example_sirs()

  np <- 10
  pars <- list(list(beta = 0.2, I0 = 5), list(beta = 0.1, I0 = 20))

  mod_cmp <- dat$model$new(pars, 0, np, seed = 10L, pars_multi = TRUE)
  mod_cmp$set_data(dust_data(dat$dat, multi = 2))
  ans_cmp <- mod_cmp$filter(save_trajectories = TRUE,
                            time_snapshot = c(4, 16))

  mod_h <- dat$model$new(pars, 0, np, seed = 10L, pars_multi = TRUE)
  mod_h$set_data(dust_data(dat$dat), shared = TRUE)
  ans_h <- mod_h$filter(save_trajectories = TRUE,
                        time_snapshot = c(4, 16))

  mod_d <- dat$model$new(pars, 0, np, seed = 10L, pars_multi = TRUE,
                         gpu_config = 0L)
  mod_d$set_data(dust_data(dat$dat), shared = TRUE)
  ans_d <- mod_d$filter(save_trajectories = TRUE,
                        time_snapshot = c(4, 16))

  expect_equal(ans_h$log_likelihood, ans_cmp$log_likelihood)
  expect_identical(ans_h$trajectories, ans_cmp$trajectories)
  expect_identical(ans_h$snapshots, ans_cmp$snapshots)

  expect_equal(ans_h$log_likelihood, ans_d$log_likelihood)
  expect_identical(ans_h$trajectories, ans_d$trajectories)
  expect_identical(ans_h$snapshots, ans_d$snapshots)
})
