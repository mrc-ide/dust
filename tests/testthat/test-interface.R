test_that("Basic interface use", {
  skip_for_compilation()
  filename <- dust_file("examples/walk.cpp")
  res <- dust(filename, quiet = TRUE)
  expect_s3_class(res, "dust_generator")
  obj <- res$new(list(sd = 1), 0L, 100L)
  expect_s3_class(obj, "dust")
})


test_that("Interface passes arguments as expected", {
  skip_if_not_installed("mockery")
  filename <- dust_file("examples/walk.cpp")
  mock_compile_and_load <- mockery::mock(NULL)
  workdir <- tempfile()

  mockery::stub(dust, "compile_and_load", mock_compile_and_load)
  dust(filename, TRUE, workdir)

  mockery::expect_called(mock_compile_and_load, 1L)
  expect_equal(
    mockery::mock_args(mock_compile_and_load)[[1]],
    list(filename, TRUE, workdir, NULL, FALSE))
})


test_that("dust_workdir uses tempdir() if NULL", {
  p <- dust_workdir(NULL)
  expect_equal(normalizePath(dirname(p)), normalizePath(tempdir()))
  expect_false(file.exists(p))
})


test_that("dust_workdir passes nonexistant directories", {
  p <- tempfile()
  expect_equal(dust_workdir(p), p)
  expect_false(file.exists(p))
})


test_that("dust_workdir allows existing empty directories", {
  p <- tempfile()
  dir.create(p, FALSE, TRUE)
  expect_equal(dust_workdir(p), p)
  expect_true(file.exists(p))
  expect_equal(dir(p), character(0))
})


test_that("dust_workdir allows existing dusty directories", {
  p <- tempfile()
  dir.create(p, FALSE, TRUE)
  dir.create(file.path(p, "src"))
  dir.create(file.path(p, "R"))
  files <- c("DESCRIPTION", "NAMESPACE",
             "src/Makevars", "src/dust.cpp", "src/cpp11.cpp",
             "R/dust.R", "R/cpp11.R")
  for (f in files) {
    file.create(file.path(p, f))
  }
  expect_equal(dust_workdir(p), p)

  ## Also allow compilation artefacts
  file.create(file.path(p, "src/interface.o"))
  file.create(file.path(p, "src/interface.dll"))
  file.create(file.path(p, "src/interface.so"))
  expect_equal(dust_workdir(p), p)
})


test_that("dust_workdir will error in directory with extra files", {
  p <- tempfile()
  dir.create(p, FALSE, TRUE)
  dir.create(file.path(p, "src"))
  files <- c("DESCRIPTION", "NAMESPACE", "src/Makevars", "README.md")
  for (f in files) {
    file.create(file.path(p, f))
  }
  expect_error(
    dust_workdir(p),
    "Path '.+' does not look like a dust directory")
})


test_that("dust_workdir will error if path is not a directory", {
  p <- tempfile()
  file.create(p)
  expect_error(
    dust_workdir(p),
    "Path '.+' already exists but is not a directory")
})


test_that("validate interface", {
  skip_for_compilation()
  res <- dust(dust_file("examples/walk.cpp"), quiet = TRUE)
  cmp <- dust_generator

  expect_setequal(names(res$public_methods),
                  names(cmp$public_methods))
  for (m in names(res$public_methods)) {
    expect_identical(formals(res$public_methods[[m]]),
                     formals(cmp$public_methods[[m]]))
  }
})


test_that("validate package interface", {
  tmp <- tempfile(fileext = ".R")
  template <- read_lines(dust_file("template/dust.R.template"))
  writeLines(glue_whisker(template, list(name = "testing", param = "NULL")),
             tmp)
  env <- new.env()
  sys.source(tmp, env)
  res <- env$testing
  cmp <- dust_generator

  expect_setequal(names(res$public_methods),
                  names(cmp$public_methods))
  for (m in names(res$public_methods)) {
    expect_identical(formals(res$public_methods[[m]]),
                     formals(cmp$public_methods[[m]]))
  }
})


test_that("get rng state", {
  res <- dust_example("walk")
  seed <- 1
  np <- 10
  obj <- res$new(list(sd = 1), 0L, np, seed = seed)
  s <- dust_rng$new(seed, np + 1)$state()
  expect_identical(obj$rng_state(), s)
  expect_identical(obj$rng_state(first_only = TRUE),
                   s[seq_len(32)])
  expect_identical(obj$rng_state(last_only = TRUE),
                   s[(np * 32 + 1):((np + 1) * 32)])
  expect_error(obj$rng_state(TRUE, TRUE),
               "Only one of 'first_only' or 'last_only' may be TRUE")
})


test_that("set rng state", {
  res <- dust_example("walk")
  seed <- 1
  np <- 10
  obj <- res$new(list(sd = 1), 0L, np, seed = seed)
  state <- obj$rng_state()

  ans <- obj$run(20)
  obj$set_rng_state(state)

  expect_equal(obj$run(40), 2 * ans)
})


test_that("set rng state", {
  res <- dust_example("walk")
  seed <- 1
  np <- 10
  obj <- res$new(list(sd = 1), 0L, np, seed = seed)
  expect_error(obj$set_rng_state(1)) # cpp11 error, don't test the message
  expect_error(
    obj$set_rng_state(as.raw(1)),
    "'rng_state' must be a raw vector of length 352 (but was 1)",
    fixed = TRUE)
  expect_error(
    obj$set_rng_state(as.raw(0:255)),
    "'rng_state' must be a raw vector of length 352 (but was 256)",
    fixed = TRUE)
  expect_error(
    obj$set_rng_state(raw(1000)),
    "'rng_state' must be a raw vector of length 352 (but was 1000)",
    fixed = TRUE)
})


test_that("setting a named index returns names", {
  skip_for_compilation()
  res <- dust(dust_file("examples/sirs.cpp"), quiet = TRUE)
  mod <- res$new(list(), 0, 10)

  mod$set_index(3:1)
  expect_identical(
    mod$run(0),
    rbind(rep(0, 10), rep(10, 10), rep(1000, 10)))

  mod$set_index(c(S = 1L, I = 2L, R = 3L))
  expect_identical(
    mod$run(0),
    rbind(S = rep(1000, 10), I = rep(10, 10), R = rep(0, 10)))

  mod$set_index(seq_len(3))
  expect_identical(
    mod$run(0),
    rbind(rep(1000, 10), rep(10, 10), rep(0, 10)))

})


test_that("resetting preserves index names", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 5)

  mod$set_index(setNames(c(1, 3, 5), c("x", "y", "z")))
  expect_equal(
    mod$run(0),
    matrix(c(1, 3, 5), 3, 5, dimnames = list(c("x", "y", "z"), NULL)))

  mod$update_state(pars = list(len = 10), step = 0)
  expect_equal(
    mod$run(0),
    matrix(c(1, 3, 5), 3, 5, dimnames = list(c("x", "y", "z"), NULL)))
})


test_that("Can't change dimensionality on reset/set_pars", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 10, 5)
  y <- matrix(runif(10 * 5), 10, 5)
  mod$update_state(state = y)

  expect_error(
    mod$update_state(pars = list(len = 5), step = 0),
    paste("'pars' created inconsistent state size:",
          "expected length 10 but created length 5"))
  expect_error(
    mod$update_state(list(len = 5), set_initial_state = FALSE),
    paste("'pars' created inconsistent state size:",
          "expected length 10 but created length 5"))

  ## No change to model state
  expect_identical(mod$state(), y)
  expect_identical(mod$step(), 10L)
})


test_that("names are copied when using state()", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 5)
  expect_equal(
    mod$state(4:5),
    matrix(4:5, 2, 5))
  expect_equal(
    mod$state(c(x = 4L, y = 5L)),
    matrix(4:5, 2, 5, dimnames = list(c("x", "y"), NULL)))
})


test_that("can return the number of threads initialised with", {
  res <- dust_example("walk")
  expect_equal(res$new(list(sd = 1), 0, 5)$n_threads(), 1)
  expect_equal(res$new(list(sd = 1), 0, 5, n_threads = 2)$n_threads(), 2)
})


test_that("can change the number of threads after initialisation", {
  mod <- dust_example("walk")$new(list(sd = 1), 0, 5)
  expect_equal(withVisible(mod$set_n_threads(2)),
               list(value = 1L, visible = FALSE))
  expect_equal(mod$n_threads(), 2L)
  expect_equal(withVisible(mod$set_n_threads(1)),
               list(value = 2L, visible = FALSE))
})


test_that("can't change to an impossible thread count", {
  mod <- dust_example("walk")$new(list(sd = 1), 0, 5)
  expect_error(mod$set_n_threads(0),
               "'n_threads' must be positive")
  expect_error(mod$set_n_threads(-1),
               "'n_threads' must be positive")
})


test_that("number of threads must be positive", {
  res <- dust_example("walk")
  expect_error(
    res$new(list(sd = 1), 0, 5, n_threads = 0),
    "'n_threads' must be positive")
  expect_error(
    res$new(list(sd = 1), 0, 5, n_threads = -1),
    "'n_threads' must be positive")
})


test_that("step must be nonnegative", {
  res <- dust_example("walk")
  expect_error(
    res$new(list(), -1, 4),
    "'step' must be non-negative")
})


test_that("Can get parameters from generators", {
  skip_for_compilation()
  res <- dust(dust_file("examples/sirs.cpp"), quiet = TRUE)
  expect_s3_class(res, "dust_generator")
  expect_equal(coef(res), parse_metadata(dust_file("examples/sirs.cpp"))$param)
  mod <- res$new(list(), 1, 1)
  expect_equal(coef(mod), coef(res))
  expect_equal(coef(mod), mod$param())
})


test_that("can change pars", {
  res <- dust_example("walk")

  obj <- res$new(list(sd = 1), 0L, 10L, seed = 1L)
  y1 <- obj$run(1)

  obj$update_state(pars = list(sd = 2), set_initial_state = FALSE)
  expect_equal(obj$state(), y1)
  expect_equal(obj$step(), 1)
  expect_equal(obj$pars(), list(sd = 2))

  y2 <- obj$run(2)

  ## Then the comparison:
  cmp <- dust_rng$new(seed = 1, 10)
  expect_equal(cmp$normal(10, 0, 1), drop(y1))
  expect_equal(y1 + cmp$normal(10, 0, 2), y2)
})


test_that("Validate changing pars leaves particles in sensible state", {
  res <- dust_example("variable")

  obj <- res$new(list(len = 5, mean = 0, sd = 1), 0, 10, seed = 1L)
  y1 <- obj$run(1)

  expect_error(
    obj$update_state(pars = list(len = 6, mean = 10, sd = 10),
                     set_initial_state = FALSE),
    paste("'pars' created inconsistent state size:",
          "expected length 5 but created length 6"))
  expect_identical(obj$state(), y1)

  y2 <- obj$run(2)
  expect_identical(
    y2,
    res$new(list(len = 5, mean = 0, sd = 1), 0, 10, seed = 1L)$run(2))
})


test_that("rewrite types", {
  res <- dust_rewrite_real(dust_file("examples/sir.cpp"), "float")
  expect_equal(tools::file_ext(res), "cpp")
  expect_length(grep("^  typedef float real_t;$", readLines(res)), 1)
})


test_that("informative error message if expected string not found", {
  ## This will break the regular expression
  tmp <- dust_rewrite_real(dust_file("examples/sir.cpp"), "++float++")
  expect_error(
    dust_rewrite_real(tmp, "float"),
    "did not find real_t declaration in '.+\\.cpp'")
})


test_that("create temporary package", {
  skip_on_cran()
  filename <- dust_file("examples/walk.cpp")
  path <- dust_generate(filename, quiet = TRUE, mangle = TRUE)
  expect_match(
    read.dcf(file.path(path, "DESCRIPTION"))[, "Package"],
    "^walk[[:xdigit:]]{8}$")
  pkg <- pkgload::load_all(path, quiet = TRUE, export_all = FALSE)
  expect_s3_class(pkg$env$walk, "dust_generator")
  obj <- pkg$env$walk$new(list(sd = 1), 0L, 100L)
  expect_s3_class(obj, "dust")
})


test_that("Don't mangle name in generated package", {
  skip_on_cran()
  filename <- dust_file("examples/walk.cpp")
  path <- dust_generate(filename, quiet = TRUE)
  expect_equal(
    unname(read.dcf(file.path(path, "DESCRIPTION"))[, "Package"]),
    "walk")
})
