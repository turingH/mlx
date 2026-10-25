// Copyright © 2024 Apple Inc.

#include <filesystem>
#include <fstream>
#include <cstring>
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

TEST_CASE("test import rejects patched slice state") {
  std::string path = get_temp_file("evil_slice.mlxfn");
  auto fun = [](const std::vector<array>& x) -> std::vector<array> {
    return {slice(x[0], {2}, {4}, {1})};
  };
  std::vector<float> data{0, 1, 2, 3, 4, 5, 6, 7};
  export_function(path, fun, {array(data.data(), {8}, float32)});

  // The benign file imports and runs.
  {
    auto f = import_function(path);
    auto out = f(std::vector<array>{array(data.data(), {8}, float32)});
    CHECK_EQ(out[0].shape(), Shape{2});
  }

  // Locate the serialized Slice state (starts={2}, ends={4}, strides={1})
  // and patch start to a huge out-of-bounds value.
  std::ifstream f(path, std::ios::binary);
  std::vector<char> buf((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
  std::vector<char> pat;
  auto push_u64 = [&](uint64_t v) {
    for (int i = 0; i < 8; i++) pat.push_back((v >> (8 * i)) & 0xff);
  };
  auto push_i32 = [&](int32_t v) {
    for (int i = 0; i < 4; i++) pat.push_back((v >> (8 * i)) & 0xff);
  };
  push_u64(1);
  push_i32(2);
  push_u64(1);
  push_i32(4);
  push_u64(1);
  push_i32(1);
  int found = -1;
  for (size_t i = 0; i + pat.size() <= buf.size(); ++i) {
    if (std::equal(pat.begin(), pat.end(), buf.begin() + i)) {
      found = (int)i;
      break;
    }
  }
  REQUIRE(found >= 0);
  int32_t evil = 0x40000000;
  std::memcpy(buf.data() + found + 8, &evil, 4);
  std::string evil_path = get_temp_file("evil_slice_patched.mlxfn");
  {
    std::ofstream out(evil_path, std::ios::binary);
    out.write(buf.data(), buf.size());
  }

  // The patched file must be rejected instead of producing an array whose
  // values read out of bounds.
  CHECK_THROWS_AS(import_function(evil_path), std::runtime_error);
}

TEST_CASE("test import rejects patched output shape") {
  // A deflated Concat output shape makes the concat kernel write 8 floats
  // into a 2-float buffer (heap overflow, ASAN-proven). The file's shapes
  // must instead be rejected at import.
  std::string path = get_temp_file("evil_cat.mlxfn");
  auto fun = [](const std::vector<array>& x) -> std::vector<array> {
    return {concatenate({x[0], x[1]}, 0)};
  };
  std::vector<float> d1(8, 1.0f), d2(8, 2.0f);
  export_function(path, fun,
                  {array(d1.data(), {8}, float32), array(d2.data(), {8}, float32)});

  std::ifstream f(path, std::ios::binary);
  std::vector<char> buf((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
  // The tape output record is the only shape {16} in the file.
  std::vector<char> pat;
  auto push_u64 = [&](uint64_t v) {
    for (int i = 0; i < 8; i++) pat.push_back((v >> (8 * i)) & 0xff);
  };
  auto push_i32 = [&](int32_t v) {
    for (int i = 0; i < 4; i++) pat.push_back((v >> (8 * i)) & 0xff);
  };
  push_u64(1);
  push_i32(16);
  int found = -1, count = 0;
  for (size_t i = 0; i + 12 <= buf.size(); ++i) {
    if (std::equal(pat.begin(), pat.end(), buf.begin() + i)) {
      found = (int)i;
      count++;
    }
  }
  REQUIRE(found >= 0);
  REQUIRE_EQ(count, 1);
  int32_t evil = 2;
  std::memcpy(buf.data() + found + 8, &evil, 4);
  std::string evil_path = get_temp_file("evil_cat_patched.mlxfn");
  {
    std::ofstream out(evil_path, std::ios::binary);
    out.write(buf.data(), buf.size());
  }
  CHECK_THROWS_AS(import_function(evil_path), std::runtime_error);
}
