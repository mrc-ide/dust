context("metadata")

test_that("parse sir model metadata", {
  meta <- parse_metadata(dust_file("examples/sir.cpp"))
  expect_equal(meta$class, "sir")
  expect_equal(meta$name, "sir")
  expect_equal(
    meta$param,
    list(beta = list(required = FALSE),
         gamma = list(required = FALSE)))
})


test_that("Can allow two classes if [[dust::class()]] used", {
  meta <- parse_metadata(dust_file("examples/twoclass.cpp"))
  expect_equal(meta$name, "walk")
  expect_equal(meta$class, "walk")
})


test_that("Cannot allow two classes if [[dust::class()]] missing", {
  code <- readLines(dust_file("examples/twoclass.cpp"))
  tmp <- tempfile()
  writeLines(code[!grepl("[[dust::", code, fixed = TRUE)], tmp)
  expect_error(
    parse_metadata(tmp),
    "Could not automatically detect class name; add [[dust::class()]]",
    fixed = TRUE)
})


test_that("Can override explicit class name with [[dust::name()]]", {
  code <- readLines(dust_file("examples/twoclass.cpp"))
  tmp <- tempfile()
  writeLines(c("// [[dust::name(model)]]", code), tmp)
  meta <- parse_metadata(tmp)
  expect_equal(meta$name, "model")
  expect_equal(meta$class, "walk")
})


test_that("Can override implicit class name with [[dust::name()]]", {
  code <- readLines(dust_file("examples/walk.cpp"))
  tmp <- tempfile()
  writeLines(c("// [[dust::name(model)]]", code), tmp)
  meta <- parse_metadata(tmp)
  expect_equal(meta$name, "model")
  expect_equal(meta$class, "walk")
})


test_that("Cannot specify [[dust::class()]] twice", {
  code <- readLines(dust_file("examples/twoclass.cpp"))
  tmp <- tempfile(fileext = ".cpp")
  writeLines(c("// [[dust::class(a)]]", code), tmp)

  expect_error(
    parse_metadata(tmp),
    sprintf("More than one [[dust::class()]] attribute found %s:(1, 5)",
            basename(tmp)),
    fixed = TRUE)
})


test_that("Cannot specify [[dust::class()]] without an argument", {
  tmp <- helper_metadata("// [[dust::class()]]")
  on.exit(unlink(tmp))
  expect_error(
    parse_metadata(tmp),
    sprintf("Expected [[dust::class()]] to have one argument %s:1",
            basename(tmp)),
    fixed = TRUE)
})


test_that("Cannot specify [[dust::class()]] with more than one argument", {
  tmp <- helper_metadata("// [[dust::class(a, use = TRUE)]]")
  on.exit(unlink(tmp))
  expect_error(
    parse_metadata(tmp),
    sprintf("Expected [[dust::class()]] to have one argument %s:1",
            basename(tmp)),
    fixed = TRUE)
})


test_that("Cannot specify [[dust::class()]] with named argument", {
  tmp <- helper_metadata("// [[dust::class(name = a)]]")
  on.exit(unlink(tmp))
  expect_error(
    parse_metadata(tmp),
    sprintf("Invalid format for [[dust::class()]] attribute %s:1",
            basename(tmp)),
    fixed = TRUE)
})


test_that("Cannot duplicate parameter names", {
  tmp <- helper_metadata("// [[dust::param(a)]]", "// [[dust::param(a)]]",
                         base = dust_file("examples/sir.cpp"))
  on.exit(unlink(tmp))
  expect_error(
    parse_metadata(tmp),
    sprintf("Duplicated [[dust::param()]] attributes: 'a' %s:(1, 2)",
            basename(tmp)),
    fixed = TRUE)
})

test_that("Cannot duplicate parameter names", {
  tmp <- helper_metadata("// [[dust::param(beta)]]", "// [[dust::param(a)]]",
                         base = dust_file("examples/sir.cpp"))
  on.exit(unlink(tmp))
  expect_error(
    parse_metadata(tmp),
    sprintf("Duplicated [[dust::param()]] attributes: 'beta' %s:(1,",
            basename(tmp)),
    fixed = TRUE)
})


test_that("Cannot duplicate parameter names", {
  tmp <- helper_metadata("// [[dust::param(beta)]]",
                         "// [[dust::param(gamma)]]",
                         base = dust_file("examples/sir.cpp"))
  on.exit(unlink(tmp))
  expect_error(
    parse_metadata(tmp),
    sprintf("Duplicated [[dust::param()]] attributes: 'beta', 'gamma' %s:(1, 2",
            basename(tmp)),
    fixed = TRUE)
})


test_that("dust::param requires an argument", {
  tmp <- helper_metadata("// [[dust::param()]]")
  on.exit(unlink(tmp))
  expect_error(
    parse_metadata(tmp),
    sprintf("At least one argument required to [[dust::param()]] %s:1",
            basename(tmp)),
    fixed = TRUE)
})


test_that("dust::param requires first argument is unnamed", {
  tmp <- helper_metadata("// [[dust::param(name = x)]]")
  on.exit(unlink(tmp))
  expect_error(
    parse_metadata(tmp),
    sprintf("First argument of [[dust::param()]] must be unnamed %s:1",
            basename(tmp)),
    fixed = TRUE)
})


test_that("dust::param requires subsequent arguments are named", {
  tmp <- helper_metadata("// [[dust::param(x, y, z = 1)]]")
  on.exit(unlink(tmp))
  expect_error(
    parse_metadata(tmp),
    "Arguments 2 and following of of [[dust::param]] must be named",
    fixed = TRUE)
})
