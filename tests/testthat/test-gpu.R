context("gpu")

test_that("can generate GPU code", {
  cache <- new.env(parent = emptyenv())

  res <- generate_dust(dust_file("examples/sirs.cpp"), "sirs", "sirs",
                       gpu = TRUE, quiet = TRUE, workdir = NULL, cache = cache)
  expect_match(res$key, "^sirsgpu[[:xdigit:]]+$")

  path_src <- file.path(res$path, "src")
  expect_setequal(dir(path_src), c("cpp11.cpp", "dust.cu", "dust.hpp",
                                   "Makevars"))
})

test_that("sirs smoke test", {
  skip_if_no_gpu()

  gen_c <- dust(dust_file("examples/sirs.cpp"), quiet = TRUE, gpu = FALSE)
  gen_g <- dust(dust_file("examples/sirs.cpp"), quiet = TRUE, gpu = TRUE)

  mod_c <- gen_c$new(list(), 0, 10)
  mod_g <- gen_g$new(list(), 0, 10)

  y_c <- mod_c$run(10)
  y_g <- mod_g$run(10)

  expect_equal(y_c, y_g)
})
