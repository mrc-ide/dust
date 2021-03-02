##' @useDynLib dust, .registration = TRUE
NULL

cache <- new.env(parent = emptyenv())
cache$models <- list()
