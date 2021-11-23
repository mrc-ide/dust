#include <iostream>
#include <dust/random/random.hpp>

int main() {
  // Check version compatibility at compile time or runtime
  static_assert(DUST_VERSION_CODE > 1110, // requires at least 0.11.10
                "Your version of dust is too old, please upgrade");
  std::cout << DUST_VERSION_STRING << std::endl;
}
