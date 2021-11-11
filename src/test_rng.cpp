#include <vector>
#include <string>
#include <sstream>

#include <cpp11.hpp>

#include <dust/random/generator.hpp>
#include <dust/r/random.hpp>
template <typename T>
std::string to_string(const T& t) {
  std::ostringstream ss;
  ss << t;
  return ss.str();
}

template <typename T>
std::vector<std::string> test_xoshiro_run1() {
  T state = dust::random::seed<T>(42);

  constexpr int n = 10;

  std::vector<std::string> ret;
  for (int i = 0; i < 3 * n; ++i) {
    if (i == n - 1) {
      dust::random::jump(state);
    } else if (i == 2 * n - 1) {
      dust::random::long_jump(state);
    }
    auto x = dust::random::next(state);
    ret.push_back(to_string(x));
  }

  return ret;
}

[[cpp11::register]]
std::vector<std::string> test_xoshiro_run(std::string name) {
  std::vector<std::string> ret;
  if (name == "xoshiro256starstar") {
    ret = test_xoshiro_run1<dust::random::xoshiro256starstar_state>();
  } else if (name == "xoshiro256plusplus") {
    ret = test_xoshiro_run1<dust::random::xoshiro256plusplus_state>();
  } else if (name == "xoshiro256plus") {
    ret = test_xoshiro_run1<dust::random::xoshiro256plus_state>();
  } else if (name == "xoshiro128starstar") {
    ret = test_xoshiro_run1<dust::random::xoshiro128starstar_state>();
  } else if (name == "xoshiro128plusplus") {
    ret = test_xoshiro_run1<dust::random::xoshiro128plusplus_state>();
  } else if (name == "xoshiro128plus") {
    ret = test_xoshiro_run1<dust::random::xoshiro128plus_state>();
  } else if (name == "xoroshiro128starstar") {
    ret = test_xoshiro_run1<dust::random::xoroshiro128starstar_state>();
  } else if (name == "xoroshiro128plusplus") {
    ret = test_xoshiro_run1<dust::random::xoroshiro128plusplus_state>();
  } else if (name == "xoroshiro128plus") {
    ret = test_xoshiro_run1<dust::random::xoroshiro128plus_state>();
  } else if (name == "xoshiro512starstar") {
    ret = test_xoshiro_run1<dust::random::xoshiro512starstar_state>();
  } else if (name == "xoshiro512plusplus") {
    ret = test_xoshiro_run1<dust::random::xoshiro512plusplus_state>();
  } else if (name == "xoshiro512plus") {
    ret = test_xoshiro_run1<dust::random::xoshiro512plus_state>();
  }

  return ret;
}


// This is copied over from our example, we'll tidy this up later, but
// we need something that can be easily tested for now.
[[cpp11::register]]
double pi_dust(int n, cpp11::sexp ptr) {
  auto rng =
    dust::random::r::rng_pointer_get<dust::random::xoshiro256plus_state>(ptr);
  auto& state = rng->state(0);
  int tot = 0;
  for (int i = 0; i < n; ++i) {
    const double u1 = dust::random::random_real<double>(state);
    const double u2 = dust::random::random_real<double>(state);
    if (u1 * u1 + u2 * u2 < 1) {
      tot++;
    }
  }
  return tot / static_cast<double>(n) * 4.0;
}
