# dust <img src='man/figures/logo.png' align="right" height="139" />

<!-- badges: start -->
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Build Status](https://travis-ci.com/mrc-ide/dust.svg?branch=master)](https://travis-ci.com/mrc-ide/dust)
[![AppVeyor build status](https://ci.appveyor.com/api/projects/status/github/mrc-ide/dust?branch=master&svg=true)](https://ci.appveyor.com/project/mrc-ide/dust)
[![CodeFactor](https://www.codefactor.io/repository/github/mrc-ide/dust/badge)](https://www.codefactor.io/repository/github/mrc-ide/dust)
[![codecov.io](https://codecov.io/github/mrc-ide/dust/coverage.svg?branch=master)](https://codecov.io/github/mrc-ide/dust?branch=master)
<!-- badges: end -->

Fast and simple iteration of stochastic models. Designed to work as a component of a particle filter.

## Background

Stochastic models appear in many domains as they are easy to write out, but hard to analyse without running many realisations of the process. `dust` provides a light interface to run models that are written in C++ in parallel from R. It provides very little functionality aside from a random number generator that is designed to be run in parallel, and is mostly interested in providing an _interface_, with which more powerful tools can be developed.

See `vignette("dust")` for instructions for using `dust` to create basic models. See the [`mcstate`](https://mrc-ide.github.io/mcstate) for a package that uses `dust` to implement a particle filter and particle MCMC.

As a (much) higher-level interface, the [`odin.dust`](https://mrc-ide.github.io/odin.dust) provides a way of compiling stochastic [`odin`](https://mrc-ide.github.io/odin) models to work with `dust`.

## Installation

```r
# install.packages("drat") # -- if you don't have drat installed
drat:::add("mrc-ide")
install.packages("dust")
```

You will need a compiler to install dependencies for the package, and to build any models with dust.  `dust` uses `pkgbuild` to build its shared libraries so use `pkgbuild::check_build_tools()` to see if your system is ok to use.

The development version of the package can be installed directly from GitHub if you prefer with:

```r
devtools::install_github("mrc-ide/dust", upgrade = FALSE)
```

## License

MIT © Imperial College of Science, Technology and Medicine
