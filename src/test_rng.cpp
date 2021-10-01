#include <cpp11.hpp>
#include <vector>
#include <string>
#include <sstream>

#include <dust/xoshiro/xoshiro.hpp>

template <typename T>
std::string to_string(const T& t) {
  std::ostringstream ss;
  ss << t;
  return ss.str();
}

template <typename T>
std::vector<std::string> test_xoshiro_run1() {
  T state;
  rng_seed(state, 42);

  constexpr int n = 10;

  std::vector<std::string> ret;
  for (int i = 0; i < 3 * n; ++i) {
    if (i == n - 1) {
      rng_jump(state);
    } else if (i == 2 * n - 1) {
      rng_long_jump(state);
    }
    auto x = rng_next(state);
    ret.push_back(to_string(x));
  }

  return ret;
}

[[cpp11::register]]
std::vector<std::string> test_xoshiro_run(std::string name) {
  std::vector<std::string> ret;
  if (name == "xoshiro256starstar") {
    ret = test_xoshiro_run1<xoshiro256starstar_state>();
  } else if (name == "xoshiro256plusplus") {
    ret = test_xoshiro_run1<xoshiro256plusplus_state>();
  } else if (name == "xoshiro256plus") {
    ret = test_xoshiro_run1<xoshiro256plus_state>();
  } else if (name == "xoshiro128starstar") {
    ret = test_xoshiro_run1<xoshiro128starstar_state>();
  } else if (name == "xoshiro128plusplus") {
    ret = test_xoshiro_run1<xoshiro128plusplus_state>();
  } else if (name == "xoshiro128plus") {
    ret = test_xoshiro_run1<xoshiro128plus_state>();
  } else if (name == "xoroshiro128starstar") {
    ret = test_xoshiro_run1<xoroshiro128starstar_state>();
  } else if (name == "xoroshiro128plusplus") {
    ret = test_xoshiro_run1<xoroshiro128plusplus_state>();
  } else if (name == "xoroshiro128plus") {
    ret = test_xoshiro_run1<xoroshiro128plus_state>();
  } else {
    cpp11::stop("Unknown type %s", name.c_str());
  }

  return ret;
}
