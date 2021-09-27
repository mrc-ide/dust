##' @useDynLib dust, .registration = TRUE
.onLoad <- function(...) {
  cache$models <- simple_cache$new() # nocov
}
NULL

cache <- new.env(parent = emptyenv())
