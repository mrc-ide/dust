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
  // TODO: consider making algorithm a r-o field in self, and
  // algorithm_ a field in private?
  cpp11::environment env_enclos =
    cpp11::as_cpp<cpp11::environment>(obj[".__enclos_env__"]);
  cpp11::environment env =
    cpp11::as_cpp<cpp11::environment>(env_enclos["private"]);

  const auto algorithm = cpp11::as_cpp<std::string>(env["algorithm_"]);
  std::vector<std::string> ret;
  if (algorithm == "xoshiro256starstar") {
    ret = test_xoshiro_run1<dust::random::xoshiro256starstar_state>(obj);
  } else if (algorithm == "xoshiro256plusplus") {
    ret = test_xoshiro_run1<dust::random::xoshiro256plusplus_state>(obj);
  } else if (algorithm == "xoshiro256plus") {
    ret = test_xoshiro_run1<dust::random::xoshiro256plus_state>(obj);
  } else if (algorithm == "xoshiro128starstar") {
    ret = test_xoshiro_run1<dust::random::xoshiro128starstar_state>(obj);
  } else if (algorithm == "xoshiro128plusplus") {
    ret = test_xoshiro_run1<dust::random::xoshiro128plusplus_state>(obj);
  } else if (algorithm == "xoshiro128plus") {
    ret = test_xoshiro_run1<dust::random::xoshiro128plus_state>(obj);
  } else if (algorithm == "xoroshiro128starstar") {
    ret = test_xoshiro_run1<dust::random::xoroshiro128starstar_state>(obj);
  } else if (algorithm == "xoroshiro128plusplus") {
    ret = test_xoshiro_run1<dust::random::xoroshiro128plusplus_state>(obj);
  } else if (algorithm == "xoroshiro128plus") {
    ret = test_xoshiro_run1<dust::random::xoroshiro128plus_state>(obj);
  } else if (algorithm == "xoshiro512starstar") {
    ret = test_xoshiro_run1<dust::random::xoshiro512starstar_state>(obj);
  } else if (algorithm == "xoshiro512plusplus") {
    ret = test_xoshiro_run1<dust::random::xoshiro512plusplus_state>(obj);
  } else if (algorithm == "xoshiro512plus") {
    ret = test_xoshiro_run1<dust::random::xoshiro512plus_state>(obj);
  }

  return ret;
}
