r6_private <- function(x) {
  environment(x$initialize)$private
}


has_openmp <- function() {
  !is.na(openmp_info()$max_threads)
}
