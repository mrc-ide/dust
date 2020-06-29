r6_private <- function(x) {
  environment(x$initialize)$private
}


has_openmp <- function() {
  openmp_info()$has_openmp
}
