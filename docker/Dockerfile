FROM rocker/r-ver:4.0.0

RUN apt-get update &&  apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        valgrind \
        zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN install2.r --error \
        R6 \
        Rcpp \
        devtools \
        glue \
        pkgbuild \
        roxygen2 \
        testthat
