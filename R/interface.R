dust <- function(alloc, run, classname = "dust") {
  R6::R6Class(
    classname,
    cloneable = FALSE,

    private = list(
      cpp_alloc = alloc,
      cpp_run = run,
      ptr = NULL
    ),

    public = list(
      initialize = function(data, n_particles, seed) {
        private$ptr <- .Call(private$cpp_alloc, data, n_particles, seed)
      },

      run = function(step_end) {
        .Call(private$cpp_run, private$ptr, step_end)
      }
    ))
}


## The checking code looks for the objects in the package namespace, so defining
## dll here removes the following NOTE
## Registration problem:
##   Evaluating 'private$cpp_alloc' during check gives error
## 'object 'private' not found':
##    .Call(private$cpp_alloc, ...)
## See https://github.com/wch/r-source/blob/d4e8fc9832f35f3c63f2201e7a35fbded5b5e14c/src/library/tools/R/QC.R#L1950-L1980
private <- list(cpp_alloc = structure(list(), class = "NativeSymbolInfo"),
                cpp_run = structure(list(), class = "NativeSymbolInfo"))
