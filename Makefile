PACKAGE := $(shell grep '^Package:' DESCRIPTION | sed -E 's/^Package:[[:space:]]+//')
RSCRIPT = Rscript

all: install

test:
	${RSCRIPT} -e 'library(methods); devtools::test()'

test_leaks: .valgrind_ignore
	R -d 'valgrind --leak-check=full --suppressions=.valgrind_ignore' -e 'devtools::test()'

.valgrind_ignore:
	R -d 'valgrind --leak-check=full --gen-suppressions=all --log-file=$@' -e 'library(testthat)'
	sed -i.bak '/^=/ d' $@
	$(RM) $@.bak

roxygen:
	./scripts/update_dust_generator
	@mkdir -p man
	${RSCRIPT} -e "library(methods); devtools::document()"

install:
	R CMD INSTALL .

build:
	R CMD build .

check:
	_R_CHECK_CRAN_INCOMING_=FALSE make check_all

check_all:
	${RSCRIPT} -e "rcmdcheck::rcmdcheck(args = c('--as-cran', '--no-manual'))"

README.md: README.Rmd
	Rscript -e "options(warnPartialMatchArgs=FALSE); knitr::knit('$<')"
	sed -i.bak 's/[[:space:]]*$$//' README.md
	$(RM) $@.bak


pkgdown:
	${RSCRIPT} -e "library(methods); pkgdown::build_site()"

website: pkgdown
	./scripts/update_web.sh

clean:
	$(RM) src/*.o src/dust.so src/dust.dll \
		tests/testthat/example/*.o tests/testthat/example/*.so \
		src/*.gcov src/*.gcda src/*.gcno

vignettes/gpu.Rmd: vignettes_src/gpu.Rmd
	./scripts/build_vignette gpu

vignettes/rng_distributed.Rmd: vignettes_src/rng_distributed.Rmd
	./scripts/build_vignette rng_distributed

vignettes/rng_package.Rmd: vignettes_src/rng_package.Rmd
	./scripts/build_vignette rng_package

vignettes: vignettes/dust.Rmd vignettes/rng_package.Rmd vignettes/rng_distributed.Rmd
	${RSCRIPT} -e 'tools::buildVignettes(dir = ".")'
	mkdir -p inst/doc
	cp vignettes/*.html vignettes/*.Rmd inst/doc

.PHONY: all test roxygen install vignettes pkgdown
