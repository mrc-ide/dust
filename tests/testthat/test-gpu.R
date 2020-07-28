context("gpu")

test_that("can generate GPU code", {
  cache <- new.env(parent = emptyenv())

  res <- generate_dust(dust_file("examples/gpu/sirs.cpp"), "sirs", "sirs",
                       gpu = TRUE, quiet = TRUE, workdir = NULL, cache = cache)
  expect_match(res$key, "^sirsgpu[[:xdigit:]]+$")

  path_src <- file.path(res$path, "src")
  expect_setequal(dir(path_src), c("cpp11.cpp", "dust.cu", "dust.hpp",
                                   "Makevars"))
})

test_that("sirs smoke test", {
  skip_if_no_nvcc()
  gen_g <- dust(dust_file("examples/gpu/sirs.cpp"), quiet = TRUE, gpu = TRUE)

  skip_if_no_gpu()

  gen_c <- dust(dust_file("examples/sirs.cpp"), quiet = TRUE, gpu = FALSE)

  mod_c <- gen_c$new(list(), 0, 10)
  mod_g <- gen_g$new(list(), 0, 10)

  y_c <- mod_c$run(10)
  y_g <- mod_g$run(10)

  expect_equal(y_c, y_g)
})


test_that("volatility smoke test", {
  skip_if_no_nvcc()
  gen_g <- dust(dust_file("examples/gpu/volatility.cpp"),
                quiet = TRUE, gpu = TRUE)

  skip_if_no_gpu()

  gen_c <- dust(dust_file("examples/volatility.cpp"),
                quiet = TRUE, gpu = FALSE)

  mod_c <- gen_c$new(list(), 0, 100)
  mod_g <- gen_g$new(list(), 0, 100)

  y_c <- mod_c$run(10)
  y_g <- mod_g$run(10)

  expect_equal(y_c, y_g, tolerance = 1e-6)
})


## This test really does nothing interesting asid
test_that("Create gpu package", {
  path <- create_test_package(
    "gpupkg",
    examples = c("gpu/sirs.cpp", "gpu/volatility.cpp"))
  path <- dust_package(path, gpu = TRUE)
  expect_setequal(
    dir(file.path(path, "src")),
    c("Makevars", "cpp11.cpp",
      "sirs.cu", "sirs.hpp",
      "volatility.cu", "volatility.hpp"))

  skip_if_no_nvcc()

  pkgbuild::compile_dll(path, quiet = TRUE)

  skip_if_no_gpu()

  res <- pkgload::load_all(path)

  mod <- res$env$volatility$new(list(), 0, 100)
  expect_equal(mod$state(), matrix(0, 1, 100))
})
