## This is a direct R translation of the Rust implementation of the
## HIN/H2PE algorithm described in the paper, set up so that the
## source of uniform random numbers can be controlled. With this we
## can crossreference this translation against R's version of the same
## algorithm (see ?rhyper) and then against our C++ implementation.
hypergeometric_r <- function(random_real) {

  hypergeometric_hin <- function(random_real, n1, n2, n, k) {
    if (k < n2) {
      p <- fraction_of_products_of_factorials(n2, n - k, n, n2 - k)
      x <- 0
    } else {
      ## We only hit this branch I think where n1 == n2 == k (and m <
      ## 10) so this is not well travelled.
      p <- fraction_of_products_of_factorials(n1, k, n, k - n2)
      x <- (k - n2)
    }

    u <- random_real()
    while (u > p && x < k) {
      ## the paper erroneously uses `until n < p`, which doesn't make any sense
      u <- u - p
      p <- p * ((n1 - x) * (k - x))
      p <- p / ((x + 1) * (n2 - k + 1 + x))
      x <- x + 1
    }
    x
  }

  hypergeometric_h2pe <- function(random_real, n1, n2, n, k, m) {
    a <- lfactorial(m) +
      lfactorial(n1 - m) +
      lfactorial(k - m) +
      lfactorial((n2 - k) + m)

    numerator <- (n - k) * k * n1 * n2
    denominator <- (n - 1) * n * n
    d <- floor(1.5 * sqrt(numerator / denominator)) + 0.5

    x_l <- m - d + 0.5
    x_r <- m + d + 0.5

    k_l <- exp(a -
                 lfactorial(x_l) -
                 lfactorial(n1 - x_l) -
                 lfactorial(k - x_l) -
                 lfactorial((n2 - k) + x_l))
    k_r <- exp(a -
                 lfactorial(x_r - 1.0) -
                 lfactorial(n1 - x_r + 1.0) -
                 lfactorial(k - x_r + 1.0) -
                 lfactorial((n2 - k) + x_r - 1.0))

    numerator <- x_l * ((n2 - k) + x_l)
    denominator <- (n1 - x_l + 1.0) * (k - x_l + 1.0)
    lambda_l <- -log(numerator / denominator)

    numerator <- (n1 - x_r + 1.0) * (k - x_r + 1.0)
    denominator <- x_r * ((n2 - k) + x_r)
    lambda_r <- -log(numerator / denominator)

    ## the paper literally gives `p2 + kL/lambdaL` where it (probably)
    ## should have been `p2 <- p1 + kL/lambdaL` another print error?!
    p1 <- 2.0 * d
    p2 <- p1 + k_l / lambda_l
    p3 <- p2 + k_r / lambda_r

    repeat {
      ## this loop aims to set v, y
      repeat {
        ## Here, it looks like I have the correct number for v
        u <- random_real() * p3 # U(0, p3) for region selection
        v <- random_real()      # U(0, 1)  for accept/reject
        if (u <= p1) {
          ## Region 1, central bell
          y <- floor(x_l + u)
          break
        } else if (u <= p2) {
          ## Region 2, left exponential tail
          y <- floor(x_l + log(v) / lambda_l)
          if (y >= max(0, k - n2)) {
            v <- v * (u - p1) * lambda_l
            break
          }
        } else {
          ## Region 3, right exponential tail
          y <- floor(x_r - log(v) / lambda_r)
          if (y <= min(n1, k)) {
            v <- v * (u - p2) * lambda_r
            break
          }
        }
      }

      ## this block aims to set y as x
      if (m < 100.0 || y <= 50.0) {
        f <- 1.0
        if (m < y) {
          for (i in seq(m + 1, y)) {
            f <- f * (n1 - i + 1) * (k - i + 1) / ((n2 - k + i) * i)
          }
        } else if (m > y) {
          for (i in seq(y + 1, m)) {
            ## The rust version does not have the + 1 on both parts of
            ## the denominator, added in the R version and a fixable
            ## bug in the Rust version.
            f <- f * i * (n2 - k + i) / ((n1 - i + 1) * (k - i + 1))
          }
        }

        if (v <= f) {
          x <- y # done here
          break
        }
      } else {
        ## Step 4.2: Squeezing
        y1 <- y + 1.0
        ym <- y - m
        yn <- n1 - y + 1.0
        yk <- k - y + 1.0
        nk <- n2 - k + y1
        r <- -ym / y1
        s <- ym / yn
        t <- ym / yk
        e <- -ym / nk
        g <- yn * yk / (y1 * nk) - 1.0
        dg <- if (g < 0.0) 1 + g else 1

        gu <- g * (1.0 + g * (-0.5 + g / 3.0))
        gl <- gu - quad(g) / (4.0 * dg) # different but equivalent
        xm <- m + 0.5
        xn <- n1 - m + 0.5
        xk <- k - m + 0.5
        nm <- n2 - k + xm
        ub <-
          xm * r * (1.0 + r * (-0.5 + r / 3.0)) +
            xn * s * (1.0 + s * (-0.5 + s / 3.0)) +
            xk * t * (1.0 + t * (-0.5 + t / 3.0)) +
            nm * e * (1.0 + e * (-0.5 + e / 3.0)) +
            y * gu - m * gl + 0.0034
        av <- log(v)
        if (av > ub) {
          next
        }

        dr <- if (r < 0) xm * quad(r) / (1.0 + r) else xm * quad(r)
        ds <- if (s < 0) xn * quad(s) / (1.0 + s) else xn * quad(s)
        dt <- if (t < 0) xk * quad(t) / (1.0 + t) else xk * quad(t)
        de <- if (e < 0) nm * quad(e) / (1.0 + e) else nm * quad(e)

        ok <- av < ub - 0.25 * (dr + ds + dt + de) + (y + m) * (gl - gu) -
          0.0078
        if (ok) {
          x <- y
          break
        }

        ## Step 4.3: Final Acceptance/Rejection Test
        av_critical <- a -
          lfactorial(y) -
          lfactorial(n1 - y) -
          lfactorial(k - y) -
          lfactorial((n2 - k) + y)
        if (log(v) <= av_critical) {
          x <- y
          break
        }
      }
    }
    x
  }

  quad <- function(x) {
    x * x * x * x
  }

  fraction_of_products_of_factorials <- function(a, b, c, d) {
    exp(lfactorial(a) + lfactorial(b) -
          lfactorial(c) -
          lfactorial(d))
  }

  function(n1, n2, k) {
    n <- n1 + n2
    if (n1 < 0 || n2 < 0 || k > n) {
      stop("Invalid parameters")
    }
    if (n1 > n2 || k > n / 2) {
      stop("Incorrect ordering")
    }
    hin_threshold <- 10.0
    m <- floor((k + 1) * (n1 + 1) / (n + 2))
    if (m < hin_threshold) {
      x <- hypergeometric_hin(random_real, n1, n2, n, k)
    } else {
      x <- hypergeometric_h2pe(random_real, n1, n2, n, k, m)
    }

    x
  }
}
