FROM rocker/r-ver:4.0.0

RUN apt-get update &&  apt-get install -y --no-install-recommends \
        gnupg2 \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        valgrind \
        wget \
        zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Install cuda
RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && dpkg -i cuda-repo-ubuntu1804_10.2.89-1_amd64.deb && \
    && rm cuda-repo-ubuntu1804_10.2.89-1_amd64.deb \
    && apt-get -y update \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
        cuda-compiler-10-2  \
        cuda-cudart-dev-10-2  \
        cuda-curand-dev-10-2 \
        libcublas-dev \
    && ln -s /usr/local/cuda-10.2 /usr/local/cuda

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH}

# Without this, we are unable to pick up more recent packages
COPY docker/Rprofile.site /usr/local/lib/R/etc/Rprofile.site

RUN install2.r --error \
        R6 \
        bench \
        brio \
        cpp11 \
        decor \
        devtools \
        glue \
        pkgbuild \
        pkgload \
        remotes \
        roxygen2 \
        testthat

COPY . /src
RUN R CMD INSTALL /src && rm -rf /src
