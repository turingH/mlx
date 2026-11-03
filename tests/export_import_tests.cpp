// Copyright © 2024 Apple Inc.

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

#include "mlx/export.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {
std::string get_temp_file(const std::string& name) {
  return std::filesystem::temp_directory_path().append(name).string();
}
} // namespace

TEST_CASE("test export basic functions") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](std::vector<array> x) -> std::vector<array> {
    return {negative(exp(x[0]))};
  };

  export_function(file_path, fun, {array({1.0, 2.0})});

  auto imported_fun = import_function(file_path);

  // Check num inputs mismatch throws
  CHECK_THROWS_AS(
      imported_fun({array({1.0}), array({2.0})}), std::invalid_argument);

  // Check shape mismatch throws
  CHECK_THROWS_AS(imported_fun({array({1.0})}), std::invalid_argument);

  // Check type mismatch throws
  CHECK_THROWS_AS(imported_fun({array({1.0}, float16)}), std::invalid_argument);

  auto expected = fun({array({1.0, -1.0})});
  auto out = imported_fun({array({1.0, -1.0})});
  CHECK(allclose(expected[0], out[0]).item<bool>());
}

TEST_CASE("test export function with no inputs") {
  auto fun = [](std::vector<array> x) -> std::vector<array> {
    return {zeros({2, 2})};
  };

  std::string file_path = get_temp_file("model.mlxfn");

  export_function(file_path, fun, {});

  auto imported_fun = import_function(file_path);

  auto expected = fun({});
  auto out = imported_fun({});
  CHECK(allclose(expected[0], out[0]).item<bool>());
}

TEST_CASE("test export multi output primitives") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](std::vector<array> x) -> std::vector<array> {
    return {divmod(x[0], x[1])};
  };

  auto inputs = std::vector<array>{array({5.0, -10.0}), array({3.0, -2.0})};
  export_function(file_path, fun, inputs);

  auto imported_fun = import_function(file_path);

  auto expected = fun(inputs);
  auto out = imported_fun(inputs);
  CHECK(allclose(expected[0], out[0]).item<bool>());
  CHECK(allclose(expected[1], out[1]).item<bool>());
}

TEST_CASE("test export primitives with state") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](std::vector<array> x) -> std::vector<array> {
    return {argpartition(x[0], 2, 0)};
  };

  auto x = array({1, 3, 2, 4, 5, 7, 6, 8}, {4, 2});
  export_function(file_path, fun, {x});

  auto imported_fun = import_function(file_path);

  auto expected = fun({x});
  auto out = imported_fun({x});
  CHECK(allclose(expected[0], out[0]).item<bool>());
}

TEST_CASE("test export functions with kwargs") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](const Kwargs& kwargs) -> std::vector<array> {
    return {kwargs.at("x") + kwargs.at("y")};
  };

  export_function(file_path, fun, {{"x", array(1)}, {"y", array(2)}});
  auto fn = import_function(file_path);

  // Must use kwargs
  CHECK_THROWS(fn({array(1), array(2)}));

  // Wrong number of keys
  CHECK_THROWS(fn({{"x", array(1)}, {"y", array(2)}, {"z", array(3)}}));

  // Wrong keys
  CHECK_THROWS(fn({{"a", array(1)}, {"b", array(2)}}));

  // Works
  auto out = fn({{"x", array(1)}, {"y", array(2)}})[0];
  CHECK_EQ(out.item<int>(), 3);
  out = fn({}, {{"x", array(1)}, {"y", array(2)}})[0];
  CHECK_EQ(out.item<int>(), 3);
}

TEST_CASE("test export function with variable inputs") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](const std::vector<array>& args) -> std::vector<array> {
    auto out = array({1, 1, 1, 1});
    for (auto x : args) {
      out = out + x;
    }
    return {out};
  };

  {
    auto fn_exporter = exporter(file_path, fun);
    fn_exporter({array(0), array(0)});
    fn_exporter({array(0), array(0), array(0)});
  }

  auto imported_fun = import_function(file_path);

  // Call with two inputs
  auto out = imported_fun({array(1), array(2)})[0];

  CHECK(array_equal(out, array({4, 4, 4, 4})).item<bool>());

  // Call with three inputs
  out = imported_fun({array(1), array(2), array(3)})[0];
  CHECK(array_equal(out, array({7, 7, 7, 7})).item<bool>());
}

TEST_CASE("test export function on different stream") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](const std::vector<array>& args) -> std::vector<array> {
    return {abs(args[0], Stream(1000, Device::cpu))};
  };

  export_function(file_path, fun, {array({0, 1, 2})});

  // Should make a new stream that we can run computation on
  eval(import_function(file_path)({array({0, 1, 2})}));
}

TEST_CASE("test export import with metadata") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](const std::vector<array>& args) -> std::vector<array> {
    return {abs(args[0])};
  };

  std::string metadata = "{\"name\": \"model\", \"params\": 7000000000}";

  export_function(file_path, fun, {array({0, 1, 2})}, false, metadata);

  auto imported = import_function(file_path);
  CHECK(imported.metadata() == metadata);
  eval(imported({array({0, 1, 2})}));

  // No metadata gives an empty string
  export_function(file_path, fun, {array({0, 1, 2})});
  CHECK(import_function(file_path).metadata().empty());

  // Metadata survives the per-trace header rewrite of a multi-trace export
  {
    auto fn_exporter = exporter(file_path, fun, false, metadata);
    fn_exporter({array({0, 1})});
    fn_exporter({array({0, 1, 2})});
  }
  CHECK(import_function(file_path).metadata() == metadata);
}

namespace {
std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<char>(
      (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void write_file(const std::string& path, const std::vector<char>& buf) {
  std::ofstream out(path, std::ios::binary);
  out.write(buf.data(), buf.size());
}

void push_u64(std::vector<char>& v, uint64_t x) {
  for (int i = 0; i < 8; i++) {
    v.push_back((x >> (8 * i)) & 0xff);
  }
}

void push_i32(std::vector<char>& v, int32_t x) {
  for (int i = 0; i < 4; i++) {
    v.push_back((x >> (8 * i)) & 0xff);
  }
}

void patch(std::vector<char>& buf, int at, int32_t value) {
  std::memcpy(buf.data() + at, &value, sizeof(value));
}

int count_patterns(const std::vector<char>& buf, const std::vector<char>& pat) {
  int count = 0;
  for (size_t i = 0; i + pat.size() <= buf.size(); ++i) {
    if (std::equal(pat.begin(), pat.end(), buf.begin() + i)) {
      count++;
    }
  }
  return count;
}

int find_pattern(const std::vector<char>& buf, const std::vector<char>& pat) {
  for (size_t i = 0; i + pat.size() <= buf.size(); ++i) {
    if (std::equal(pat.begin(), pat.end(), buf.begin() + i)) {
      return static_cast<int>(i);
    }
  }
  return -1;
}
} // namespace

TEST_CASE("test import round trips raw slice indices") {
  // The serialized slice state keeps the raw python-style indices: the end
  // may be the -1 sentinel, may exceed the dimension, or may be -n-1 with a
  // negative stride; a start past the beginning yields an empty output.
  // Importing and evaluating these must work unchanged.
  std::string path = get_temp_file("raw_slice.mlxfn");
  std::vector<float> data{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::vector<int>> cases{
      {0, -1, 1}, // x[:-1]
      {2, 100, 1}, // x[2:100]
      {7, -9, -1}, // x[::-1]
      {-100, -9, -1}, // x[-100::-1], empty output
  };
  for (const auto& c : cases) {
    auto fun = [&c](const std::vector<array>& x) -> std::vector<array> {
      return {slice(x[0], {c[0]}, {c[1]}, {c[2]})};
    };
    auto x = array(data.data(), {8}, float32);
    export_function(path, fun, {x});
    auto expected = fun({x});
    auto out = import_function(path)({x});
    CHECK_EQ(out[0].shape(), expected[0].shape());
    CHECK(allclose(out[0], expected[0]).item<bool>());
  }
}

TEST_CASE("test import rejects patched slice state") {
  std::string path = get_temp_file("evil_slice.mlxfn");
  auto fun = [](const std::vector<array>& x) -> std::vector<array> {
    return {slice(x[0], {2}, {4}, {1})};
  };
  std::vector<float> data{0, 1, 2, 3, 4, 5, 6, 7};
  auto x = array(data.data(), {8}, float32);
  export_function(path, fun, {x});

  // The benign file imports and runs.
  {
    auto out = import_function(path)({x});
    CHECK_EQ(out[0].shape(), Shape{2});
  }

  // The serialized state is starts={2}, ends={4}, strides={1}; patch the
  // start. A large value reads past the end of the input buffer and a very
  // negative one before its beginning once the graph is evaluated.
  auto buf = read_file(path);
  std::vector<char> pat;
  push_u64(pat, 1);
  push_i32(pat, 2);
  push_u64(pat, 1);
  push_i32(pat, 4);
  push_u64(pat, 1);
  push_i32(pat, 1);
  REQUIRE_EQ(count_patterns(buf, pat), 1);
  int found = find_pattern(buf, pat);
  REQUIRE(found >= 0);
  for (int32_t evil : {int32_t(0x40000000), int32_t(-0x40000000)}) {
    auto evil_buf = buf;
    patch(evil_buf, found + 8, evil);
    std::string evil_path = get_temp_file("evil_slice_patched.mlxfn");
    write_file(evil_path, evil_buf);
    CHECK_THROWS_AS(import_function(evil_path), std::runtime_error);
  }
}

TEST_CASE("test import rejects patched slice update state") {
  std::string path = get_temp_file("evil_slice_update.mlxfn");
  auto fun = [](const std::vector<array>& x) -> std::vector<array> {
    return {slice_update(x[0], x[1], {1}, {3}, {1})};
  };
  std::vector<float> data(8, 1.0f);
  std::vector<float> upd{9.0f, 9.0f};
  auto src = array(data.data(), {8}, float32);
  auto u = array(upd.data(), {2}, float32);
  export_function(path, fun, {src, u});

  // The benign file imports and runs.
  {
    auto out = import_function(path)({src, u});
    CHECK_EQ(out[0].shape(), Shape{8});
  }

  // The serialized state is reduce=None (4), starts={1}, ends={3},
  // strides={1}; patch the start so the update writes at an offset far
  // outside the destination buffer.
  auto buf = read_file(path);
  std::vector<char> pat;
  push_i32(pat, 4);
  push_u64(pat, 1);
  push_i32(pat, 1);
  push_u64(pat, 1);
  push_i32(pat, 3);
  push_u64(pat, 1);
  push_i32(pat, 1);
  REQUIRE_EQ(count_patterns(buf, pat), 1);
  int found = find_pattern(buf, pat);
  REQUIRE(found >= 0);
  patch(buf, found + 4 + 8, 0x40000000);
  std::string evil_path = get_temp_file("evil_slice_update_patched.mlxfn");
  write_file(evil_path, buf);
  CHECK_THROWS_AS(import_function(evil_path), std::runtime_error);
}

TEST_CASE("test import rejects patched output shape") {
  // A deflated concatenate output shape makes the kernel write 8 floats
  // into a 2-float buffer (heap overflow); the file must instead be
  // rejected at import.
  std::string path = get_temp_file("evil_cat.mlxfn");
  auto fun = [](const std::vector<array>& x) -> std::vector<array> {
    return {concatenate({x[0], x[1]}, 0)};
  };
  std::vector<float> d1(8, 1.0f), d2(8, 2.0f);
  auto x1 = array(d1.data(), {8}, float32);
  auto x2 = array(d2.data(), {8}, float32);
  export_function(path, fun, {x1, x2});

  // The benign file imports and runs.
  {
    auto out = import_function(path)({x1, x2});
    CHECK_EQ(out[0].shape(), Shape{16});
  }

  // The tape's recorded output shape is the only {16} in the file; shrink
  // it to {2}.
  auto buf = read_file(path);
  std::vector<char> pat;
  push_u64(pat, 1);
  push_i32(pat, 16);
  REQUIRE_EQ(count_patterns(buf, pat), 1);
  int found = find_pattern(buf, pat);
  REQUIRE(found >= 0);
  patch(buf, found + 8, 2);
  std::string evil_path = get_temp_file("evil_cat_patched.mlxfn");
  write_file(evil_path, buf);
  CHECK_THROWS_AS(import_function(evil_path), std::runtime_error);
}
