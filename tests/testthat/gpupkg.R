## Generate out code:
devtools::load_all()
unlink("gpupkg", recursive = TRUE)
path <- create_test_package("gpupkg", "gpupkg",
                            examples = c("gpu/sirs.cpp", "gpu/volatility.cpp"))
dust::dust_package(path, gpu = TRUE)
