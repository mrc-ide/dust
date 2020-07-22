FROM rocker/r-ver:4.0.0

RUN apt-get update &&  apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        valgrind \
        zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Without this, we are unable to pick up more recent packages
COPY docker/Rprofile.site /usr/local/lib/R/etc/Rprofile.site

RUN install2.r --error \
        R6 \
        bench \
        brio \
        decor \
        devtools \
        glue \
        pkgbuild \
        pkgload \
        remotes \
        roxygen2 \
        testthat

RUN R CMD INSTALL .
