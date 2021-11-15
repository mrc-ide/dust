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
std::vector<std::string> test_xoshiro_run1(cpp11::environment ptr) {
  auto rng = dust::random::r::rng_pointer_get<T>(ptr, 1);
  auto& state = rng->state(0);

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
std::vector<std::string> test_xoshiro_run(cpp11::environment obj) {
  const auto algorithm = cpp11::as_cpp<std::string>(obj["algorithm"]);
  std::vector<std::string> ret;
  if (algorithm == "xoshiro256starstar") {
    ret = test_xoshiro_run1<dust::random::xoshiro256starstar>(obj);
  } else if (algorithm == "xoshiro256plusplus") {
    ret = test_xoshiro_run1<dust::random::xoshiro256plusplus>(obj);
  } else if (algorithm == "xoshiro256plus") {
    ret = test_xoshiro_run1<dust::random::xoshiro256plus>(obj);
  } else if (algorithm == "xoshiro128starstar") {
    ret = test_xoshiro_run1<dust::random::xoshiro128starstar>(obj);
  } else if (algorithm == "xoshiro128plusplus") {
    ret = test_xoshiro_run1<dust::random::xoshiro128plusplus>(obj);
  } else if (algorithm == "xoshiro128plus") {
    ret = test_xoshiro_run1<dust::random::xoshiro128plus>(obj);
  } else if (algorithm == "xoroshiro128starstar") {
    ret = test_xoshiro_run1<dust::random::xoroshiro128starstar>(obj);
  } else if (algorithm == "xoroshiro128plusplus") {
    ret = test_xoshiro_run1<dust::random::xoroshiro128plusplus>(obj);
  } else if (algorithm == "xoroshiro128plus") {
    ret = test_xoshiro_run1<dust::random::xoroshiro128plus>(obj);
  } else if (algorithm == "xoshiro512starstar") {
    ret = test_xoshiro_run1<dust::random::xoshiro512starstar>(obj);
  } else if (algorithm == "xoshiro512plusplus") {
    ret = test_xoshiro_run1<dust::random::xoshiro512plusplus>(obj);
  } else if (algorithm == "xoshiro512plus") {
    ret = test_xoshiro_run1<dust::random::xoshiro512plus>(obj);
  }

  return ret;
}
