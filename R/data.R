##' Prepare data for use with the `$set_data()` method. This is not
##' required for use but tries to simplify the most common use case
##' where you have a [data.frame] with some column indicating "model
##' step" (`name_step`), and other columns that might be use in your
##' `data_compare` function. Each row will be turned into a named R
##' list, which your `dust_data` function can then work with to get
##' this time-steps values.
##'
##' The data object as accepted by `data_set` must be a [list] and
##' each element must itself be a list with two elements; the model
##' `step` at which the data applies and any R object that corresponds
##' to data at that point. We expect that most of the time this second
##' element will be a key-value list with scalar keys, but more
##' flexibility may be required.
##'
##' @title Process data for dust
##'
##' @param object An object, at this point must be a [data.frame]
##'
##' @param name_step The name of the data column within `object`; this
##'   column must be integer-like and every integer must be
##'   nonnegative and unique
##'
##' @export
##' @examples
##' d <- data.frame(step = seq(0, 50, by = 10), a = runif(6), b = runif(6))
##' dust::dust_data(d)
dust_data <- function(object, name_step = "step") {
  assert_is(object, "data.frame")
  steps <- object[[name_step]]
  if (is.null(steps)) {
    stop(sprintf("'%s' is not a column in %s",
                 name_step, deparse(substitute(object))))
  }
  isteps <- as.integer(round(steps))
  if (any(isteps < 0)) {
    stop(sprintf("All elements in column '%s' must be nonnegative", name_step))
  }
  if (any(abs(steps - isteps) > sqrt(.Machine$double.eps))) {
    stop(sprintf("All elements in column '%s' must be integer-like", name_step))
  }
  if (any(duplicated(isteps))) {
    stop(sprintf("All elements in column '%s' must be unique", name_step))
  }
  rows <- lapply(seq_len(nrow(object)), function(i) as.list(object[i, ]))
  Map(list, isteps, rows)
}
