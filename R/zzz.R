##' @useDynLib dust, .registration = TRUE
.onLoad <- function(...) {           # nolint
  cache$models <- simple_cache$new() # nocov
}

cache <- new.env(parent = emptyenv())
