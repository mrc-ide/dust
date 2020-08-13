dust_example <- function(name) {
  switch(name,
         sir = sir,
         variable = variable,
         volatility = volatility,
         walk = walk,
         stop(sprintf("Unknown example '%s'", name)))
}
