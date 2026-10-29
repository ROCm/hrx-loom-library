// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Model-owned AIE2P panel arithmetic for the prepared 128-token composition.
// Public helpers describe fixed BF16/F32 view layouts; callers own disjoint
// inputs, outputs, scratch ranges, and synchronization. Native BFP dot products
// consume N8/K8 records and their FIFO lookahead. Compensated BF16 arithmetic,
// RMS normalization, rotary transforms, causal sequence-masked softmax, and
// SiLU preserve the round points of the qualified resident representation.
// Portable prefix normalization and native packing are separate motif inputs.

#include <loomcxx/low.h>
#include <loomcxx/numeric.h>
#include <loomcxx/scalar.h>
#include <loomcxx/vector.h>

#include <stdfloat>

using BFloat16x16 = std::bfloat16_t __attribute__((ext_vector_type(16)));
using BFloat16x32 = std::bfloat16_t __attribute__((ext_vector_type(32)));
using BFloat16x64 = std::bfloat16_t __attribute__((ext_vector_type(64)));
using BFloat16x8 = std::bfloat16_t __attribute__((ext_vector_type(8)));
using BFloat16x8x8 = loom::type::vector<loom::type::shape<8, 8>, std::bfloat16_t>;
using Boolx16 = loom::type::vector<loom::type::shape<16>, bool>;
using Float32x16 = float __attribute__((ext_vector_type(16)));
using Float32x64 = float __attribute__((ext_vector_type(64)));
using Int32x16 = int __attribute__((ext_vector_type(16)));
using UInt16x16 = unsigned short __attribute__((ext_vector_type(16)));
using UInt32x16 = unsigned __attribute__((ext_vector_type(16)));

struct [[loom::representation("amd.xdna.aie2p.core")]] Aie2pCore {};

struct qwen3_embedding_dot128_result {
  Float32x64 value0;
  Float32x64 value1;
  Float32x64 value2;
  Float32x64 value3;
};

[[loom::symbol("qwen3_embedding.silu_product"), loom::force_inline]]
static Float32x16 qwen3_embedding_silu_product(BFloat16x16 left, BFloat16x16 right);

[[loom::symbol("qwen3_embedding.prepare_silu_lookup"), loom::force_inline]]
void qwen3_embedding_prepare_silu_lookup(loom::type::view<loom::type::shape<4096>, std::bfloat16_t> input, loom::type::view<loom::type::shape<8192>, std::bfloat16_t> output_ab, loom::type::view<loom::type::shape<8192>, std::bfloat16_t> output_cd);

[[loom::symbol("qwen3_embedding.identity"), loom::force_inline]]
void qwen3_embedding_identity(loom::type::view<loom::type::shape<8, 8>, std::bfloat16_t> output);

[[loom::symbol("qwen3_embedding.copy_output"), loom::force_inline]]
void qwen3_embedding_copy_output(loom::type::view<loom::type::shape<4096>, std::bfloat16_t> input, loom::type::view<loom::type::shape<4096>, std::bfloat16_t> output);

[[loom::symbol("qwen3_embedding.block_right"), loom::force_inline]]
void qwen3_embedding_block_right(loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> input, loom::type::view<loom::type::shape<4096>, std::bfloat16_t> output);

[[loom::symbol("qwen3_embedding.block_left"), loom::force_inline]]
void qwen3_embedding_block_left(loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> input, loom::type::view<loom::type::shape<4096>, std::bfloat16_t> output);

[[loom::symbol("qwen3_embedding.three_products"), loom::force_inline]]
void qwen3_embedding_three_products(loom::type::buffer<float> storage, loom::type::view<loom::type::shape<8, 8>, float> temporary, loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> output);

[[loom::symbol("qwen3_embedding.exp"), loom::force_inline]]
static Float32x16 qwen3_embedding_exp(Float32x16 input);

[[loom::symbol("qwen3_embedding.scale"), loom::force_inline]]
static BFloat16x16 qwen3_embedding_scale(BFloat16x16 input, float factor);

[[loom::symbol("qwen3_embedding.softmax"), loom::force_inline]]
void qwen3_embedding_softmax(loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> matrix, loom::type::view<loom::type::shape<128>, float> scores, loom::type::view<loom::type::shape<128>, unsigned> sequences, loom::type::index query_base);

[[loom::symbol("qwen3_embedding.normalize_dense_head"), loom::force_inline]]
static void qwen3_embedding_normalize_dense_head(loom::type::index length, loom::type::view<loom::type::shape<128>, std::bfloat16_t> input, loom::type::view<loom::type::shape<128>, std::bfloat16_t> scale, loom::type::view<loom::type::shape<128>, std::bfloat16_t> output);

[[loom::symbol("qwen3_embedding.rotate"), loom::force_inline]]
static void qwen3_embedding_rotate(loom::type::view<loom::type::shape<2, 64>, std::bfloat16_t> rope, loom::type::view<loom::type::shape<128>, std::bfloat16_t> output_head);

[[loom::symbol("qwen3_embedding.dense_accumulate"), loom::force_inline]]
void qwen3_embedding_dense_accumulate(loom::type::buffer<unsigned char> storage, loom::type::view<loom::type::shape<16, 4, 64>, float> output, bool first_chunk);

[[loom::symbol("qwen3_embedding.dense_rows"), loom::force_inline]]
void qwen3_embedding_dense_rows(loom::type::view<loom::type::shape<16, 4, 64>, float> input, loom::type::view<loom::type::shape<16, 4, 64>, float> secondary, bool first, loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> output);

[[loom::symbol("qwen3_embedding.dense_heads"), loom::force_inline]]
void qwen3_embedding_dense_heads(loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> input, loom::type::view<loom::type::shape<128>, std::bfloat16_t> scale, loom::type::view<loom::type::shape<32, 2, 64>, std::bfloat16_t> rope);

[[loom::symbol("qwen3_embedding.dense_transpose_values"), loom::force_inline]]
void qwen3_embedding_dense_transpose_values(loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> input, loom::type::view<loom::type::shape<128, 32>, std::bfloat16_t> output);

[[loom::symbol("qwen3_embedding.normalize_shared"), loom::force_inline]]
void qwen3_embedding_normalize_shared(loom::type::index length, loom::type::view<loom::type::shape<1024>, std::bfloat16_t> input, loom::type::view<loom::type::shape<1024>, std::bfloat16_t> scale, loom::type::view<loom::type::shape<1024>, std::bfloat16_t> output);

[[loom::symbol("qwen3_embedding.block_four_rows"), loom::force_inline]]
void qwen3_embedding_block_four_rows(loom::type::view<loom::type::shape<4, 1024>, std::bfloat16_t> input, loom::type::view<loom::type::shape<128, 4, 8>, std::bfloat16_t> output);

[[loom::symbol("qwen3_embedding.three_products_blocked"), loom::force_inline]]
void qwen3_embedding_three_products_blocked(loom::type::buffer<float> storage, loom::type::view<loom::type::shape<8, 8>, float> temporary, loom::type::view<loom::type::shape<16, 4, 64>, std::bfloat16_t> output);

[[loom::symbol("qwen3_embedding.swiglu_dense"), loom::force_inline]]
void qwen3_embedding_swiglu_dense(loom::type::view<loom::type::shape<4096>, float> gate, loom::type::view<loom::type::shape<4096>, float> up, loom::type::buffer<unsigned char> table_ab, loom::type::buffer<unsigned char> table_cd, loom::type::view<loom::type::shape<4096>, std::bfloat16_t> output);

[[loom::symbol("qwen3_embedding.dense_residual"), loom::force_inline]]
void qwen3_embedding_dense_residual(loom::type::view<loom::type::shape<4096>, std::bfloat16_t> input, loom::type::view<loom::type::shape<4096>, std::bfloat16_t> skip);

[[loom::symbol("qwen3_embedding.retain_residual_panel"), loom::force_inline]]
void qwen3_embedding_retain_residual_panel(loom::type::view<loom::type::shape<4096>, std::bfloat16_t> input, loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> output, loom::type::index record, loom::type::index feature);

[[loom::symbol("qwen3_embedding.block_dense_result"), loom::force_inline]]
void qwen3_embedding_block_dense_result(loom::type::view<loom::type::shape<16, 4, 64>, std::bfloat16_t> input, loom::type::view<loom::type::shape<16, 4, 64>, std::bfloat16_t> output);

[[loom::symbol("qwen3_embedding.dot128"), loom::force_inline]]
static qwen3_embedding_dot128_result qwen3_embedding_dot128(loom::type::buffer<unsigned char> weights, loom::type::offset weight_offset, loom::type::buffer<unsigned char> activations, loom::type::offset activation_offset, Float32x64 initial0, Float32x64 initial1, Float32x64 initial2, Float32x64 initial3);

[[loom::symbol("qwen3_embedding.silu_lookup"), loom::force_inline]]
static BFloat16x16 qwen3_embedding_silu_lookup(loom::type::buffer<unsigned char> table_ab, loom::type::buffer<unsigned char> table_cd, UInt32x16 indices);

[[loom::symbol("qwen3_embedding.silu_product"), loom::force_inline]]
static Float32x16 qwen3_embedding_silu_product(BFloat16x16 left, BFloat16x16 right) {
  BFloat16x16 inactive_lanes = BFloat16x16{};
  Float32x16 initial = Float32x16{};
  BFloat16x32 left_pairs = loom::vector::interleave<0, BFloat16x32>(left, inactive_lanes);
  BFloat16x32 right_pairs = loom::vector::interleave<0, BFloat16x32>(right, inactive_lanes);
  Float32x16 product = loom::vector::dot2f(left_pairs, right_pairs, initial);
  return product;
}

[[loom::symbol("qwen3_embedding.prepare_silu_lookup"), loom::force_inline]]
void qwen3_embedding_prepare_silu_lookup(loom::type::view<loom::type::shape<4096>, std::bfloat16_t> input, loom::type::view<loom::type::shape<8192>, std::bfloat16_t> output_ab, loom::type::view<loom::type::shape<8192>, std::bfloat16_t> output_cd) {
  for (int block = 0; block < 256; ++block) {
    loom::type::index relative = loom::index::cast(block) * loom::type::index{16};
    loom::type::index index = loom::type::index{4080} - relative;
    BFloat16x16 value = loom::vector::load<BFloat16x16>(input, loom::type::indices{index});
    BFloat16x32 duplicate = loom::vector::concat<0, BFloat16x32>(value, value);
    loom::type::index first = index * loom::type::index{2};
    loom::vector::store(duplicate, output_ab, loom::type::indices{first});
    loom::vector::store(duplicate, output_cd, loom::type::indices{first});
  }
  return;
}

[[loom::symbol("qwen3_embedding.identity"), loom::force_inline]]
void qwen3_embedding_identity(loom::type::view<loom::type::shape<8, 8>, std::bfloat16_t> output) {
  BFloat16x8x8 c0 = loom::vector::splat<BFloat16x8x8>({}, std::bfloat16_t{});
  loom::vector::store(c0, output, {0, 0});
  for (int position = 0; position < 8; ++position) {
    loom::view::store(static_cast<std::bfloat16_t>(1.0f), output, position, position);
  }
  return;
}

[[loom::symbol("qwen3_embedding.copy_output"), loom::force_inline]]
void qwen3_embedding_copy_output(loom::type::view<loom::type::shape<4096>, std::bfloat16_t> input, loom::type::view<loom::type::shape<4096>, std::bfloat16_t> output) {
  for (int position = 0; position < 4096; position += 32) {
    BFloat16x32 values = loom::vector::load<BFloat16x32>(input, {position});
    loom::vector::store(values, output, {position});
  }
  return;
}

[[loom::symbol("qwen3_embedding.block_right"), loom::force_inline]]
void qwen3_embedding_block_right(loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> input, loom::type::view<loom::type::shape<4096>, std::bfloat16_t> output) {
  for (int reduction = 0; reduction < 16; ++reduction) {
    loom::type::index column = loom::index::cast(reduction) * loom::type::index{8};
    loom::type::index reduction_base = loom::index::cast(reduction) * loom::type::index{1};
    for (int block = 0; block < 1; ++block) {
      loom::type::index row_base = loom::index::cast(block) * loom::type::index{8};
      loom::type::index tile = reduction_base + loom::index::cast(block);
      loom::type::index tile_base = tile * loom::type::index{64};
      for (int lane = 0; lane < 8; ++lane) {
        loom::type::index row = row_base + loom::index::cast(lane);
        BFloat16x8 values = loom::vector::load<BFloat16x8>(input, loom::type::indices{row, column});
        loom::type::index lane_offset = loom::index::cast(lane) * loom::type::index{8};
        loom::type::index destination = tile_base + lane_offset;
        loom::vector::store(values, output, loom::type::indices{destination});
      }
    }
  }
  return;
}

[[loom::symbol("qwen3_embedding.block_left"), loom::force_inline]]
void qwen3_embedding_block_left(loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> input, loom::type::view<loom::type::shape<4096>, std::bfloat16_t> output) {
  for (int reduction = 0; reduction < 16; ++reduction) {
    loom::type::index column = loom::index::cast(reduction) * loom::type::index{8};
    loom::type::index reduction_base = loom::index::cast(reduction) * loom::type::index{4};
    for (int block = 0; block < 4; ++block) {
      loom::type::index row_base = loom::index::cast(block) * loom::type::index{8};
      loom::type::index tile = reduction_base + loom::index::cast(block);
      loom::type::index tile_base = tile * loom::type::index{64};
      for (int lane = 0; lane < 8; ++lane) {
        loom::type::index row = row_base + loom::index::cast(lane);
        BFloat16x8 values = loom::vector::load<BFloat16x8>(input, loom::type::indices{row, column});
        loom::type::index lane_offset = loom::index::cast(lane) * loom::type::index{8};
        loom::type::index destination = tile_base + lane_offset;
        loom::vector::store(values, output, loom::type::indices{destination});
      }
    }
  }
  return;
}

[[loom::symbol("qwen3_embedding.three_products"), loom::force_inline]]
void qwen3_embedding_three_products(loom::type::buffer<float> storage, loom::type::view<loom::type::shape<8, 8>, float> temporary, loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> output) {
  auto partials = loom::buffer::view<4, 64>(storage, loom::type::offset{9344}, {});
  Float32x64 c0 = Float32x64{};
  Float32x64 initial = loom::vector::fragment::attach<loom::vector::fragment::role::init>(c0, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
  for (int panel = 0; panel < 16; ++panel) {
    loom::type::offset relative = loom::index::scale(loom::index::cast(panel), loom::type::offset{1152});
    loom::type::offset high_offset = loom::type::offset{27136} + relative;
    loom::type::offset low_offset = loom::type::offset{45632} + relative;
    loom::type::index column = loom::index::cast(panel) * loom::type::index{8};
    auto partial_span0 = loom::view::subview<1, 64>(partials, {0, 0}, {});
    auto partial_tile0 = loom::view::bitcast<loom::type::view<loom::type::shape<8, 8>, float>>(partial_span0);
    auto partial_span1 = loom::view::subview<1, 64>(partials, {1, 0}, {});
    auto partial_tile1 = loom::view::bitcast<loom::type::view<loom::type::shape<8, 8>, float>>(partial_span1);
    auto partial_span2 = loom::view::subview<1, 64>(partials, {2, 0}, {});
    auto partial_tile2 = loom::view::bitcast<loom::type::view<loom::type::shape<8, 8>, float>>(partial_span2);
    auto partial_span3 = loom::view::subview<1, 64>(partials, {3, 0}, {});
    auto partial_tile3 = loom::view::bitcast<loom::type::view<loom::type::shape<8, 8>, float>>(partial_span3);
    for (int product_phase = 0; product_phase < 3; ++product_phase) {
      bool first_product = loom::index::cast(product_phase) == loom::type::index{0};
      bool left_residual_product = loom::index::cast(product_phase) == loom::type::index{1};
      bool right_residual_product = loom::index::cast(product_phase) == loom::type::index{2};
      loom::type::offset product_right = right_residual_product ? low_offset : high_offset;
      loom::type::offset product_left = left_residual_product ? loom::type::offset{4672} : loom::type::offset{0};
      qwen3_embedding_dot128_result call_result = qwen3_embedding_dot128(loom::buffer::cast<unsigned char>(storage), product_right, loom::buffer::cast<unsigned char>(storage), product_left, initial, initial, initial, initial);
      Float32x64 product0 = call_result.value0;
      Float32x64 product1 = call_result.value1;
      Float32x64 product2 = call_result.value2;
      Float32x64 product3 = call_result.value3;
      if (first_product) {
        loom::vector::fragment::store<loom::vector::fragment::role::result>(product0, partial_tile0, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        loom::vector::fragment::store<loom::vector::fragment::role::result>(product1, partial_tile1, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        loom::vector::fragment::store<loom::vector::fragment::role::result>(product2, partial_tile2, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        loom::vector::fragment::store<loom::vector::fragment::role::result>(product3, partial_tile3, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
      } else {
        Float32x64 previous0 = loom::vector::load<Float32x64>(partials, {0, 0});
        Float32x64 fragment0 = loom::vector::fragment::attach<loom::vector::fragment::role::init>(previous0, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x64 added0 = fragment0 + product0;
        loom::vector::fragment::store<loom::vector::fragment::role::result>(added0, partial_tile0, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        Float32x64 previous1 = loom::vector::load<Float32x64>(partials, {1, 0});
        Float32x64 fragment1 = loom::vector::fragment::attach<loom::vector::fragment::role::init>(previous1, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x64 added1 = fragment1 + product1;
        loom::vector::fragment::store<loom::vector::fragment::role::result>(added1, partial_tile1, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        Float32x64 previous2 = loom::vector::load<Float32x64>(partials, {2, 0});
        Float32x64 fragment2 = loom::vector::fragment::attach<loom::vector::fragment::role::init>(previous2, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x64 added2 = fragment2 + product2;
        loom::vector::fragment::store<loom::vector::fragment::role::result>(added2, partial_tile2, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        Float32x64 previous3 = loom::vector::load<Float32x64>(partials, {3, 0});
        Float32x64 fragment3 = loom::vector::fragment::attach<loom::vector::fragment::role::init>(previous3, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x64 added3 = fragment3 + product3;
        loom::vector::fragment::store<loom::vector::fragment::role::result>(added3, partial_tile3, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
      }
    }
    for (int result_block = 0; result_block < 4; ++result_block) {
      loom::type::index result_base = loom::index::cast(result_block) * loom::type::index{8};
      Float32x64 wide = loom::vector::load<Float32x64>(partials, {result_block, 0});
      BFloat16x64 flat = loom::convert<BFloat16x64, loom::conversion::subnormal::daz>(wide);
      BFloat16x8x8 narrow = loom::vector::bitcast<BFloat16x8x8>(flat);
      BFloat16x8x8 transposed = loom::vector::transpose<BFloat16x8x8, 1, 0>(narrow);
      loom::type::index row0 = result_base + loom::type::index{0};
      BFloat16x8 values0 = loom::vector::extract<BFloat16x8, 0>(transposed);
      loom::vector::store(values0, output, loom::type::indices{row0, column});
      loom::type::index row1 = result_base + loom::type::index{1};
      BFloat16x8 values1 = loom::vector::extract<BFloat16x8, 1>(transposed);
      loom::vector::store(values1, output, loom::type::indices{row1, column});
      loom::type::index row2 = result_base + loom::type::index{2};
      BFloat16x8 values2 = loom::vector::extract<BFloat16x8, 2>(transposed);
      loom::vector::store(values2, output, loom::type::indices{row2, column});
      loom::type::index row3 = result_base + loom::type::index{3};
      BFloat16x8 values3 = loom::vector::extract<BFloat16x8, 3>(transposed);
      loom::vector::store(values3, output, loom::type::indices{row3, column});
      loom::type::index row4 = result_base + loom::type::index{4};
      BFloat16x8 values4 = loom::vector::extract<BFloat16x8, 4>(transposed);
      loom::vector::store(values4, output, loom::type::indices{row4, column});
      loom::type::index row5 = result_base + loom::type::index{5};
      BFloat16x8 values5 = loom::vector::extract<BFloat16x8, 5>(transposed);
      loom::vector::store(values5, output, loom::type::indices{row5, column});
      loom::type::index row6 = result_base + loom::type::index{6};
      BFloat16x8 values6 = loom::vector::extract<BFloat16x8, 6>(transposed);
      loom::vector::store(values6, output, loom::type::indices{row6, column});
      loom::type::index row7 = result_base + loom::type::index{7};
      BFloat16x8 values7 = loom::vector::extract<BFloat16x8, 7>(transposed);
      loom::vector::store(values7, output, loom::type::indices{row7, column});
    }
  }
  return;
}

[[loom::symbol("qwen3_embedding.exp"), loom::force_inline]]
static Float32x16 qwen3_embedding_exp(Float32x16 input) {
  Float32x16 floor = -86.9f;
  [[loom::fastmath("nnan", "nsz")]] Float32x16 bounded = loom::vector::maxnumf(input, floor);
  Float32x16 c0 = Float32x16{};
  BFloat16x16 narrow_zero = BFloat16x16{};
  BFloat16x16 high = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(bounded);
  Float32x16 high_f32 = loom::convert<Float32x16, loom::conversion::subnormal::daz>(high);
  Float32x16 remainder = bounded - high_f32;
  BFloat16x16 middle = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(remainder);
  Float32x16 middle_f32 = loom::convert<Float32x16, loom::conversion::subnormal::daz>(middle);
  Float32x16 tail_f32 = remainder - middle_f32;
  BFloat16x16 tail = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(tail_f32);
  BFloat16x16 coefficient_high = static_cast<std::bfloat16_t>(1.4453125f);
  BFloat16x16 coefficient_middle = static_cast<std::bfloat16_t>(-0.00262451171875f);
  BFloat16x16 coefficient_tail = static_cast<std::bfloat16_t>(7.063150405883789e-06f);
  BFloat16x32 input0 = loom::vector::interleave<0, BFloat16x32>(high, middle);
  BFloat16x32 factor0 = loom::vector::interleave<0, BFloat16x32>(coefficient_high, coefficient_high);
  Float32x16 partial0 = loom::vector::dot2f(input0, factor0, c0);
  BFloat16x32 input1 = loom::vector::interleave<0, BFloat16x32>(tail, high);
  BFloat16x32 factor1 = loom::vector::interleave<0, BFloat16x32>(coefficient_high, coefficient_middle);
  Float32x16 partial1 = loom::vector::dot2f(input1, factor1, partial0);
  BFloat16x32 input2 = loom::vector::interleave<0, BFloat16x32>(middle, high);
  BFloat16x32 factor2 = loom::vector::interleave<0, BFloat16x32>(coefficient_middle, coefficient_tail);
  Float32x16 scaled = loom::vector::dot2f(input2, factor2, partial1);
  Float32x16 magic = 12582912.0f;
  Float32x16 biased = scaled + magic;
  Float32x16 integer = biased - magic;
  Float32x16 fraction = scaled - integer;
  BFloat16x16 fraction_high = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(fraction);
  Float32x16 fraction_high_f32 = loom::convert<Float32x16, loom::conversion::subnormal::daz>(fraction_high);
  Float32x16 fraction_tail = fraction - fraction_high_f32;
  BFloat16x16 fraction_low = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(fraction_tail);
  BFloat16x32 fraction_pair = loom::vector::interleave<0, BFloat16x32>(fraction_high, fraction_low);
  BFloat16x32 fraction_zero_pair = loom::vector::interleave<0, BFloat16x32>(fraction_high, narrow_zero);
  BFloat16x16 coefficient4 = static_cast<std::bfloat16_t>(0.00958251953125f);
  BFloat16x32 coefficient4_pair = loom::vector::interleave<0, BFloat16x32>(coefficient4, narrow_zero);
  Float32x16 coefficient3 = 0.0559063815f;
  Float32x16 polynomial0 = loom::vector::dot2f(fraction_zero_pair, coefficient4_pair, coefficient3);
  BFloat16x16 polynomial0_high = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(polynomial0);
  Float32x16 polynomial0_high_f32 = loom::convert<Float32x16, loom::conversion::subnormal::daz>(polynomial0_high);
  Float32x16 polynomial0_tail = polynomial0 - polynomial0_high_f32;
  BFloat16x16 polynomial0_low = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(polynomial0_tail);
  BFloat16x32 polynomial0_pair = loom::vector::interleave<0, BFloat16x32>(polynomial0_high, polynomial0_high);
  BFloat16x32 polynomial0_zero_pair = loom::vector::interleave<0, BFloat16x32>(polynomial0_low, narrow_zero);
  Float32x16 constant0 = 0.240241051f;
  Float32x16 polynomial0_partial = loom::vector::dot2f(fraction_pair, polynomial0_pair, constant0);
  Float32x16 polynomial1 = loom::vector::dot2f(fraction_zero_pair, polynomial0_zero_pair, polynomial0_partial);
  BFloat16x16 polynomial1_high = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(polynomial1);
  Float32x16 polynomial1_high_f32 = loom::convert<Float32x16, loom::conversion::subnormal::daz>(polynomial1_high);
  Float32x16 polynomial1_tail = polynomial1 - polynomial1_high_f32;
  BFloat16x16 polynomial1_low = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(polynomial1_tail);
  BFloat16x32 polynomial1_pair = loom::vector::interleave<0, BFloat16x32>(polynomial1_high, polynomial1_high);
  BFloat16x32 polynomial1_zero_pair = loom::vector::interleave<0, BFloat16x32>(polynomial1_low, narrow_zero);
  Float32x16 constant1 = 0.693124175f;
  Float32x16 polynomial1_partial = loom::vector::dot2f(fraction_pair, polynomial1_pair, constant1);
  Float32x16 polynomial2 = loom::vector::dot2f(fraction_zero_pair, polynomial1_zero_pair, polynomial1_partial);
  BFloat16x16 polynomial2_high = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(polynomial2);
  Float32x16 polynomial2_high_f32 = loom::convert<Float32x16, loom::conversion::subnormal::daz>(polynomial2_high);
  Float32x16 polynomial2_tail = polynomial2 - polynomial2_high_f32;
  BFloat16x16 polynomial2_low = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(polynomial2_tail);
  BFloat16x32 polynomial2_pair = loom::vector::interleave<0, BFloat16x32>(polynomial2_high, polynomial2_high);
  BFloat16x32 polynomial2_zero_pair = loom::vector::interleave<0, BFloat16x32>(polynomial2_low, narrow_zero);
  Float32x16 constant2 = 1.0f;
  Float32x16 polynomial2_partial = loom::vector::dot2f(fraction_pair, polynomial2_pair, constant2);
  Float32x16 polynomial3 = loom::vector::dot2f(fraction_zero_pair, polynomial2_zero_pair, polynomial2_partial);
  UInt32x16 biased_bits = __builtin_bit_cast(UInt32x16, biased);
  UInt32x16 shift = 23u;
  UInt32x16 exponent_bits = biased_bits << shift;
  UInt32x16 polynomial_bits = __builtin_bit_cast(UInt32x16, polynomial3);
  UInt32x16 result_bits = polynomial_bits + exponent_bits;
  Float32x16 result = __builtin_bit_cast(Float32x16, result_bits);
  Boolx16 below = loom::vector::cmpf::olt<Boolx16>(input, floor);
  Float32x16 finite = loom::vector::select(below, c0, result);
  BFloat16x16 narrow = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(finite);
  Float32x16 rounded = loom::convert<Float32x16, loom::conversion::subnormal::daz>(narrow);
  return rounded;
}

[[loom::symbol("qwen3_embedding.scale"), loom::force_inline]]
static BFloat16x16 qwen3_embedding_scale(BFloat16x16 input, float factor) {
  Float32x16 factor_vector = factor;
  BFloat16x16 high = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(factor_vector);
  Float32x16 high_f32 = loom::convert<Float32x16, loom::conversion::subnormal::daz>(high);
  Float32x16 remainder = factor_vector - high_f32;
  BFloat16x16 middle = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(remainder);
  Float32x16 middle_f32 = loom::convert<Float32x16, loom::conversion::subnormal::daz>(middle);
  Float32x16 tail_f32 = remainder - middle_f32;
  BFloat16x16 tail = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(tail_f32);
  Float32x16 c0 = Float32x16{};
  BFloat16x16 narrow_zero = BFloat16x16{};
  BFloat16x32 factor_pair = loom::vector::interleave<0, BFloat16x32>(high, middle);
  BFloat16x32 tail_pair = loom::vector::interleave<0, BFloat16x32>(tail, narrow_zero);
  BFloat16x32 repeated = loom::vector::interleave<0, BFloat16x32>(input, input);
  Float32x16 leading = loom::vector::dot2f(repeated, factor_pair, c0);
  Float32x16 product = loom::vector::dot2f(repeated, tail_pair, leading);
  BFloat16x16 result = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(product);
  return result;
}

[[loom::symbol("qwen3_embedding.softmax"), loom::force_inline]]
void qwen3_embedding_softmax(loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> matrix, loom::type::view<loom::type::shape<128>, float> scores, loom::type::view<loom::type::shape<128>, unsigned> sequences, loom::type::index query_base) {
  Float32x16 c0_vector_16_f32 = Float32x16{};
  Float32x16 infinities = __builtin_bit_cast(float, 0xff800000u);
  UInt32x16 lanes = loom::vector::iota<UInt32x16>(static_cast<unsigned>(0), 1u);
  for (int query = 0; query < 32; ++query) {
    loom::type::index absolute_query = query_base + loom::index::cast(query);
    int query_sequence = loom::view::load(sequences, absolute_query);
    bool valid = query_sequence >= 0;
    Boolx16 valid_lanes = loom::vector::splat<Boolx16>({}, valid);
    Int32x16 query_sequences = query_sequence;
    unsigned query_integer = loom::index::cast<unsigned>(absolute_query);
    UInt32x16 query_indices = query_integer;
    Float32x16 maxima = infinities;
    for (int position = 0; position < 128; position += 16) {
      BFloat16x16 input = loom::vector::load<BFloat16x16>(matrix, {query, position});
      BFloat16x16 scaled = qwen3_embedding_scale(input, 0.08838834764831843f);
      Float32x16 wide = loom::convert<Float32x16, loom::conversion::subnormal::daz>(scaled);
      UInt32x16 key_sequences = loom::vector::load<UInt32x16>(sequences, {position});
      Boolx16 same = loom::vector::cmpi::eq<Boolx16>(key_sequences, query_sequences);
      unsigned position_integer = loom::index::cast<unsigned>(loom::index::cast(position));
      UInt32x16 positions = position_integer;
      UInt32x16 keys = positions + lanes;
      Boolx16 causal = loom::vector::cmpi::ule<Boolx16>(keys, query_indices);
      Boolx16 same_causal = same & causal;
      Boolx16 visible = same_causal & valid_lanes;
      Float32x16 masked = loom::vector::select(visible, wide, infinities);
      loom::vector::store(masked, scores, {position});
      [[loom::fastmath("nnan", "nsz")]] maxima = loom::vector::maxnumf(maxima, masked);
    }
    [[loom::fastmath("reassoc", "nnan", "nsz")]] float maximum_1 = loom::vector::reduce::maxnumf(maxima, __builtin_bit_cast(float, 0xff800000u));
    float finite_maximum = valid ? maximum_1 : 0.0f;
    Float32x16 maximum_lanes = finite_maximum;
    Float32x16 sums = Float32x16{};
    for (int position_1 = 0; position_1 < 128; position_1 += 16) {
      Float32x16 input_1 = loom::vector::load<Float32x16>(scores, {position_1});
      [[loom::fastmath("nnan", "nsz")]] Float32x16 shifted = input_1 - maximum_lanes;
      Float32x16 exponents = qwen3_embedding_exp(shifted);
      BFloat16x16 narrow = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(exponents);
      loom::vector::store(narrow, matrix, {query, position_1});
      [[loom::fastmath("reassoc", "nnan", "ninf", "nsz")]] sums = sums + exponents;
    }
    [[loom::fastmath("reassoc", "nnan", "ninf", "nsz")]] float sum_1 = loom::vector::reduce::addf(sums, 0.0f);
    float denominator = valid ? sum_1 : 1.0f;
    [[loom::fastmath("arcp")]] float inverse = 1.0f / denominator;
    for (int position_2 = 0; position_2 < 128; position_2 += 16) {
      BFloat16x16 input_2 = loom::vector::load<BFloat16x16>(matrix, {query, position_2});
      BFloat16x16 normalized = qwen3_embedding_scale(input_2, inverse);
      loom::vector::store(normalized, matrix, {query, position_2});
    }
  }
  return;
}

[[loom::symbol("qwen3_embedding.normalize_dense_head"), loom::force_inline]]
static void qwen3_embedding_normalize_dense_head(loom::type::index length, loom::type::view<loom::type::shape<128>, std::bfloat16_t> input, loom::type::view<loom::type::shape<128>, std::bfloat16_t> scale, loom::type::view<loom::type::shape<128>, std::bfloat16_t> output) {
  bool is_head = length == loom::type::index{128};
  BFloat16x16 head_inverse = static_cast<std::bfloat16_t>(0.0078125f);
  BFloat16x16 row_inverse = static_cast<std::bfloat16_t>(0.0009765625f);
  BFloat16x16 width_inverse = is_head ? head_inverse : row_inverse;
  Float32x16 c0 = Float32x16{};
  BFloat16x16 narrow_zero = BFloat16x16{};
  Float32x16 squares = Float32x16{};
  for (loom::type::index position = loom::type::index{0}; position < length; position = position + loom::type::index{32}) {
    BFloat16x32 values = loom::vector::load<BFloat16x32>(input, loom::type::indices{position});
    squares = loom::vector::dot2f(values, values, squares);
  }
  BFloat16x16 squares_bf16 = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(squares);
  Float32x16 squares_high = loom::convert<Float32x16, loom::conversion::subnormal::daz>(squares_bf16);
  [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x16 squares_remainder = squares - squares_high;
  BFloat16x16 squares_low = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(squares_remainder);
  Float32x16 squares_middle = loom::convert<Float32x16, loom::conversion::subnormal::daz>(squares_low);
  [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x16 squares_residual = squares_remainder - squares_middle;
  BFloat16x16 squares_tail = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(squares_residual);
  BFloat16x32 squares_tail_pair = loom::vector::interleave<0, BFloat16x32>(squares_tail, narrow_zero);
  BFloat16x32 squares_pair = loom::vector::interleave<0, BFloat16x32>(squares_bf16, squares_low);
  BFloat16x32 width_pair = loom::vector::interleave<0, BFloat16x32>(width_inverse, width_inverse);
  Float32x16 leading_means = loom::vector::dot2f(squares_pair, width_pair, c0);
  Float32x16 means = loom::vector::dot2f(squares_tail_pair, width_pair, leading_means);
  [[loom::fastmath("reassoc", "nnan", "ninf", "nsz")]] float mean = loom::vector::reduce::addf(means, 0.0f);
  Float32x16 mean_vector = mean;
  Float32x16 epsilon = 1e-06f;
  [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x16 variance_vector = mean_vector + epsilon;
  float variance = variance_vector[0];
  float inverse = loom::scalar::approximate::rsqrtf(variance);
  Float32x16 inverse_vector = inverse;
  BFloat16x16 inverse_bf16 = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(inverse_vector);
  Float32x16 inverse_high = loom::convert<Float32x16, loom::conversion::subnormal::daz>(inverse_bf16);
  [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x16 inverse_remainder = inverse_vector - inverse_high;
  BFloat16x16 inverse_low = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(inverse_remainder);
  Float32x16 inverse_middle = loom::convert<Float32x16, loom::conversion::subnormal::daz>(inverse_low);
  [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x16 inverse_residual = inverse_remainder - inverse_middle;
  BFloat16x16 inverse_tail = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(inverse_residual);
  BFloat16x32 inverse_tail_pair = loom::vector::interleave<0, BFloat16x32>(inverse_tail, narrow_zero);
  BFloat16x32 inverse_pair = loom::vector::interleave<0, BFloat16x32>(inverse_bf16, inverse_low);
  for (loom::type::index position_1 = loom::type::index{0}; position_1 < length; position_1 = position_1 + loom::type::index{16}) {
    BFloat16x16 input_value = loom::vector::load<BFloat16x16>(input, loom::type::indices{position_1});
    BFloat16x32 input_pair = loom::vector::interleave<0, BFloat16x32>(input_value, input_value);
    Float32x16 leading_normalized = loom::vector::dot2f(input_pair, inverse_pair, c0);
    Float32x16 normalized = loom::vector::dot2f(input_pair, inverse_tail_pair, leading_normalized);
    BFloat16x16 normalized_bf16 = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(normalized);
    BFloat16x32 normalized_pair = loom::vector::interleave<0, BFloat16x32>(normalized_bf16, narrow_zero);
    BFloat16x16 scale_value = loom::vector::load<BFloat16x16>(scale, loom::type::indices{position_1});
    BFloat16x32 scale_pair = loom::vector::interleave<0, BFloat16x32>(scale_value, narrow_zero);
    Float32x16 scaled = loom::vector::dot2f(normalized_pair, scale_pair, c0);
    BFloat16x16 result = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(scaled);
    loom::vector::store(result, output, loom::type::indices{position_1});
  }
  return;
}

[[loom::symbol("qwen3_embedding.rotate"), loom::force_inline]]
static void qwen3_embedding_rotate(loom::type::view<loom::type::shape<2, 64>, std::bfloat16_t> rope, loom::type::view<loom::type::shape<128>, std::bfloat16_t> output_head) {
  Float32x16 c0 = Float32x16{};
  BFloat16x16 narrow_zero = BFloat16x16{};
  for (int position = 0; position < 64; position += 16) {
    loom::type::index upper = loom::index::cast(position) + loom::type::index{64};
    BFloat16x16 low = loom::vector::load<BFloat16x16>(output_head, {position});
    BFloat16x16 high = loom::vector::load<BFloat16x16>(output_head, loom::type::indices{upper});
    BFloat16x16 cosine = loom::vector::load<BFloat16x16>(rope, {0, position});
    BFloat16x16 sine = loom::vector::load<BFloat16x16>(rope, {1, position});
    BFloat16x32 low_pair = loom::vector::interleave<0, BFloat16x32>(low, narrow_zero);
    BFloat16x32 high_pair = loom::vector::interleave<0, BFloat16x32>(high, narrow_zero);
    BFloat16x32 cosine_pair = loom::vector::interleave<0, BFloat16x32>(cosine, narrow_zero);
    BFloat16x32 sine_pair = loom::vector::interleave<0, BFloat16x32>(sine, narrow_zero);
    Float32x16 low_cosine_f32 = loom::vector::dot2f(low_pair, cosine_pair, c0);
    BFloat16x16 low_cosine_bf16 = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(low_cosine_f32);
    Float32x16 high_sine_f32 = loom::vector::dot2f(high_pair, sine_pair, c0);
    BFloat16x16 high_sine_bf16 = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(high_sine_f32);
    Float32x16 low_cosine = loom::convert<Float32x16, loom::conversion::subnormal::daz>(low_cosine_bf16);
    Float32x16 high_sine = loom::convert<Float32x16, loom::conversion::subnormal::daz>(high_sine_bf16);
    [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x16 rotated_low_f32 = low_cosine - high_sine;
    BFloat16x16 rotated_low = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(rotated_low_f32);
    Float32x16 high_cosine_f32 = loom::vector::dot2f(high_pair, cosine_pair, c0);
    BFloat16x16 high_cosine_bf16 = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(high_cosine_f32);
    Float32x16 low_sine_f32 = loom::vector::dot2f(low_pair, sine_pair, c0);
    BFloat16x16 low_sine_bf16 = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(low_sine_f32);
    Float32x16 high_cosine = loom::convert<Float32x16, loom::conversion::subnormal::daz>(high_cosine_bf16);
    Float32x16 low_sine = loom::convert<Float32x16, loom::conversion::subnormal::daz>(low_sine_bf16);
    [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x16 rotated_high_f32 = high_cosine + low_sine;
    BFloat16x16 rotated_high = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(rotated_high_f32);
    loom::vector::store(rotated_low, output_head, {position});
    loom::vector::store(rotated_high, output_head, loom::type::indices{upper});
  }
  return;
}

[[loom::symbol("qwen3_embedding.dense_accumulate"), loom::force_inline]]
void qwen3_embedding_dense_accumulate(loom::type::buffer<unsigned char> storage, loom::type::view<loom::type::shape<16, 4, 64>, float> output, bool first_chunk) {
  for (int panel = 0; panel < 16; ++panel) {
    loom::type::offset weight_offset = loom::index::scale(loom::index::cast(panel), loom::type::offset{1152});
    Float32x64 initial0 = Float32x64{};
    Float32x64 initial1 = Float32x64{};
    Float32x64 initial2 = Float32x64{};
    Float32x64 initial3 = Float32x64{};
    if (first_chunk) {
      Float32x64 c0 = Float32x64{};
      Float32x64 initial = loom::vector::fragment::attach<loom::vector::fragment::role::init>(c0, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
      auto next_initial0 = initial;
      auto next_initial1 = initial;
      auto next_initial2 = initial;
      auto next_initial3 = initial;
      initial0 = next_initial0;
      initial1 = next_initial1;
      initial2 = next_initial2;
      initial3 = next_initial3;
    } else {
      Float32x64 previous0 = loom::vector::load<Float32x64>(output, {panel, 0, 0});
      initial0 = loom::vector::fragment::attach<loom::vector::fragment::role::init>(previous0, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
      Float32x64 previous1 = loom::vector::load<Float32x64>(output, {panel, 1, 0});
      initial1 = loom::vector::fragment::attach<loom::vector::fragment::role::init>(previous1, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
      Float32x64 previous2 = loom::vector::load<Float32x64>(output, {panel, 2, 0});
      initial2 = loom::vector::fragment::attach<loom::vector::fragment::role::init>(previous2, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
      Float32x64 previous3 = loom::vector::load<Float32x64>(output, {panel, 3, 0});
      initial3 = loom::vector::fragment::attach<loom::vector::fragment::role::init>(previous3, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
    }
    qwen3_embedding_dot128_result call_result = qwen3_embedding_dot128(storage, weight_offset, storage, loom::type::offset{27008}, initial0, initial1, initial2, initial3);
    Float32x64 sum0 = call_result.value0;
    Float32x64 sum1 = call_result.value1;
    Float32x64 sum2 = call_result.value2;
    Float32x64 sum3 = call_result.value3;
    auto panel_span = loom::view::subview<1, 4, 64>(output, {panel, 0, 0}, {});
    auto fragments = loom::view::bitcast<loom::type::view<loom::type::shape<4, 64>, float>>(panel_span);
    auto span0 = loom::view::subview<1, 64>(fragments, {0, 0}, {});
    auto tile0 = loom::view::bitcast<loom::type::view<loom::type::shape<8, 8>, float>>(span0);
    loom::vector::fragment::store<loom::vector::fragment::role::result>(sum0, tile0, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
    auto span1 = loom::view::subview<1, 64>(fragments, {1, 0}, {});
    auto tile1 = loom::view::bitcast<loom::type::view<loom::type::shape<8, 8>, float>>(span1);
    loom::vector::fragment::store<loom::vector::fragment::role::result>(sum1, tile1, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
    auto span2 = loom::view::subview<1, 64>(fragments, {2, 0}, {});
    auto tile2 = loom::view::bitcast<loom::type::view<loom::type::shape<8, 8>, float>>(span2);
    loom::vector::fragment::store<loom::vector::fragment::role::result>(sum2, tile2, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
    auto span3 = loom::view::subview<1, 64>(fragments, {3, 0}, {});
    auto tile3 = loom::view::bitcast<loom::type::view<loom::type::shape<8, 8>, float>>(span3);
    loom::vector::fragment::store<loom::vector::fragment::role::result>(sum3, tile3, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
  }
  return;
}

[[loom::symbol("qwen3_embedding.dense_rows"), loom::force_inline]]
void qwen3_embedding_dense_rows(loom::type::view<loom::type::shape<16, 4, 64>, float> input, loom::type::view<loom::type::shape<16, 4, 64>, float> secondary, bool first, loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> output) {
  for (int panel = 0; panel < 16; ++panel) {
    loom::type::index feature_base = loom::index::cast(panel) * loom::type::index{8};
    for (int block = 0; block < 4; ++block) {
      loom::type::index token_base = loom::index::cast(block) * loom::type::index{8};
      Float32x64 wide = Float32x64{};
      if (first) {
        wide = loom::vector::load<Float32x64>(input, {panel, block, 0});
      } else {
        wide = loom::vector::load<Float32x64>(secondary, {panel, block, 0});
      }
      BFloat16x64 narrow = loom::convert<BFloat16x64, loom::conversion::subnormal::daz>(wide);
      BFloat16x8x8 tile = loom::vector::bitcast<BFloat16x8x8>(narrow);
      BFloat16x8x8 transposed = loom::vector::transpose<BFloat16x8x8, 1, 0>(tile);
      loom::type::index row0 = token_base + loom::type::index{0};
      BFloat16x8 values0 = loom::vector::extract<BFloat16x8, 0>(transposed);
      loom::vector::store(values0, output, loom::type::indices{row0, feature_base});
      loom::type::index row1 = token_base + loom::type::index{1};
      BFloat16x8 values1 = loom::vector::extract<BFloat16x8, 1>(transposed);
      loom::vector::store(values1, output, loom::type::indices{row1, feature_base});
      loom::type::index row2 = token_base + loom::type::index{2};
      BFloat16x8 values2 = loom::vector::extract<BFloat16x8, 2>(transposed);
      loom::vector::store(values2, output, loom::type::indices{row2, feature_base});
      loom::type::index row3 = token_base + loom::type::index{3};
      BFloat16x8 values3 = loom::vector::extract<BFloat16x8, 3>(transposed);
      loom::vector::store(values3, output, loom::type::indices{row3, feature_base});
      loom::type::index row4 = token_base + loom::type::index{4};
      BFloat16x8 values4 = loom::vector::extract<BFloat16x8, 4>(transposed);
      loom::vector::store(values4, output, loom::type::indices{row4, feature_base});
      loom::type::index row5 = token_base + loom::type::index{5};
      BFloat16x8 values5 = loom::vector::extract<BFloat16x8, 5>(transposed);
      loom::vector::store(values5, output, loom::type::indices{row5, feature_base});
      loom::type::index row6 = token_base + loom::type::index{6};
      BFloat16x8 values6 = loom::vector::extract<BFloat16x8, 6>(transposed);
      loom::vector::store(values6, output, loom::type::indices{row6, feature_base});
      loom::type::index row7 = token_base + loom::type::index{7};
      BFloat16x8 values7 = loom::vector::extract<BFloat16x8, 7>(transposed);
      loom::vector::store(values7, output, loom::type::indices{row7, feature_base});
    }
  }
  return;
}

[[loom::symbol("qwen3_embedding.dense_heads"), loom::force_inline]]
void qwen3_embedding_dense_heads(loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> input, loom::type::view<loom::type::shape<128>, std::bfloat16_t> scale, loom::type::view<loom::type::shape<32, 2, 64>, std::bfloat16_t> rope) {
  for (int row = 0; row < 32; ++row) {
    auto input_span = loom::view::subview<1, 128>(input, {row, 0}, {});
    auto values = loom::view::bitcast<loom::type::view<loom::type::shape<128>, std::bfloat16_t>>(input_span);
    auto rope_span = loom::view::subview<1, 2, 64>(rope, {row, 0, 0}, {});
    auto angles = loom::view::bitcast<loom::type::view<loom::type::shape<2, 64>, std::bfloat16_t>>(rope_span);
    qwen3_embedding_normalize_dense_head(loom::type::index{128}, values, scale, values);
    qwen3_embedding_rotate(angles, values);
  }
  return;
}

[[loom::symbol("qwen3_embedding.dense_transpose_values"), loom::force_inline]]
void qwen3_embedding_dense_transpose_values(loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> input, loom::type::view<loom::type::shape<128, 32>, std::bfloat16_t> output) {
  for (int feature_block = 0; feature_block < 16; ++feature_block) {
    loom::type::index feature_base = loom::index::cast(feature_block) * loom::type::index{8};
    for (int token_block = 0; token_block < 4; ++token_block) {
      loom::type::index token_base = loom::index::cast(token_block) * loom::type::index{8};
      loom::type::index source_row0 = token_base + loom::type::index{0};
      BFloat16x8 row0 = loom::vector::load<BFloat16x8>(input, loom::type::indices{source_row0, feature_base});
      loom::type::index source_row1 = token_base + loom::type::index{1};
      BFloat16x8 row1 = loom::vector::load<BFloat16x8>(input, loom::type::indices{source_row1, feature_base});
      loom::type::index source_row2 = token_base + loom::type::index{2};
      BFloat16x8 row2 = loom::vector::load<BFloat16x8>(input, loom::type::indices{source_row2, feature_base});
      loom::type::index source_row3 = token_base + loom::type::index{3};
      BFloat16x8 row3 = loom::vector::load<BFloat16x8>(input, loom::type::indices{source_row3, feature_base});
      loom::type::index source_row4 = token_base + loom::type::index{4};
      BFloat16x8 row4 = loom::vector::load<BFloat16x8>(input, loom::type::indices{source_row4, feature_base});
      loom::type::index source_row5 = token_base + loom::type::index{5};
      BFloat16x8 row5 = loom::vector::load<BFloat16x8>(input, loom::type::indices{source_row5, feature_base});
      loom::type::index source_row6 = token_base + loom::type::index{6};
      BFloat16x8 row6 = loom::vector::load<BFloat16x8>(input, loom::type::indices{source_row6, feature_base});
      loom::type::index source_row7 = token_base + loom::type::index{7};
      BFloat16x8 row7 = loom::vector::load<BFloat16x8>(input, loom::type::indices{source_row7, feature_base});
      BFloat16x16 join8_0 = loom::vector::concat<0, BFloat16x16>(row0, row1);
      BFloat16x16 join8_2 = loom::vector::concat<0, BFloat16x16>(row2, row3);
      BFloat16x16 join8_4 = loom::vector::concat<0, BFloat16x16>(row4, row5);
      BFloat16x16 join8_6 = loom::vector::concat<0, BFloat16x16>(row6, row7);
      BFloat16x32 join16_0 = loom::vector::concat<0, BFloat16x32>(join8_0, join8_2);
      BFloat16x32 join16_2 = loom::vector::concat<0, BFloat16x32>(join8_4, join8_6);
      BFloat16x64 join32_0 = loom::vector::concat<0, BFloat16x64>(join16_0, join16_2);
      BFloat16x8x8 tile = loom::vector::bitcast<BFloat16x8x8>(join32_0);
      BFloat16x8x8 transposed = loom::vector::transpose<BFloat16x8x8, 1, 0>(tile);
      loom::type::index destination_row0 = feature_base + loom::type::index{0};
      BFloat16x8 values0 = loom::vector::extract<BFloat16x8, 0>(transposed);
      loom::vector::store(values0, output, loom::type::indices{destination_row0, token_base});
      loom::type::index destination_row1 = feature_base + loom::type::index{1};
      BFloat16x8 values1 = loom::vector::extract<BFloat16x8, 1>(transposed);
      loom::vector::store(values1, output, loom::type::indices{destination_row1, token_base});
      loom::type::index destination_row2 = feature_base + loom::type::index{2};
      BFloat16x8 values2 = loom::vector::extract<BFloat16x8, 2>(transposed);
      loom::vector::store(values2, output, loom::type::indices{destination_row2, token_base});
      loom::type::index destination_row3 = feature_base + loom::type::index{3};
      BFloat16x8 values3 = loom::vector::extract<BFloat16x8, 3>(transposed);
      loom::vector::store(values3, output, loom::type::indices{destination_row3, token_base});
      loom::type::index destination_row4 = feature_base + loom::type::index{4};
      BFloat16x8 values4 = loom::vector::extract<BFloat16x8, 4>(transposed);
      loom::vector::store(values4, output, loom::type::indices{destination_row4, token_base});
      loom::type::index destination_row5 = feature_base + loom::type::index{5};
      BFloat16x8 values5 = loom::vector::extract<BFloat16x8, 5>(transposed);
      loom::vector::store(values5, output, loom::type::indices{destination_row5, token_base});
      loom::type::index destination_row6 = feature_base + loom::type::index{6};
      BFloat16x8 values6 = loom::vector::extract<BFloat16x8, 6>(transposed);
      loom::vector::store(values6, output, loom::type::indices{destination_row6, token_base});
      loom::type::index destination_row7 = feature_base + loom::type::index{7};
      BFloat16x8 values7 = loom::vector::extract<BFloat16x8, 7>(transposed);
      loom::vector::store(values7, output, loom::type::indices{destination_row7, token_base});
    }
  }
  return;
}

[[loom::symbol("qwen3_embedding.normalize_shared"), loom::force_inline]]
void qwen3_embedding_normalize_shared(loom::type::index length, loom::type::view<loom::type::shape<1024>, std::bfloat16_t> input, loom::type::view<loom::type::shape<1024>, std::bfloat16_t> scale, loom::type::view<loom::type::shape<1024>, std::bfloat16_t> output) {
  bool is_head = length == loom::type::index{128};
  BFloat16x16 head_inverse = static_cast<std::bfloat16_t>(0.0078125f);
  BFloat16x16 row_inverse = static_cast<std::bfloat16_t>(0.0009765625f);
  BFloat16x16 width_inverse = is_head ? head_inverse : row_inverse;
  Float32x16 c0 = Float32x16{};
  BFloat16x16 narrow_zero = BFloat16x16{};
  Float32x16 squares = Float32x16{};
  Float32x16 sum_error = Float32x16{};
  for (loom::type::index position = loom::type::index{0}; position < length; position = position + loom::type::index{32}) {
    BFloat16x32 values = loom::vector::load<BFloat16x32>(input, loom::type::indices{position});
    Float32x16 products = loom::vector::dot2f(values, values, c0);
    [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x16 adjusted = products - sum_error;
    [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x16 next = squares + adjusted;
    [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x16 increment = next - squares;
    [[loom::fastmath("nnan", "ninf", "nsz")]] sum_error = increment - adjusted;
    squares = next;
  }
  BFloat16x16 squares_bf16 = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(squares);
  Float32x16 squares_high = loom::convert<Float32x16, loom::conversion::subnormal::daz>(squares_bf16);
  [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x16 squares_remainder = squares - squares_high;
  BFloat16x16 squares_low = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(squares_remainder);
  Float32x16 squares_middle = loom::convert<Float32x16, loom::conversion::subnormal::daz>(squares_low);
  [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x16 squares_residual = squares_remainder - squares_middle;
  BFloat16x16 squares_tail = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(squares_residual);
  BFloat16x32 squares_tail_pair = loom::vector::interleave<0, BFloat16x32>(squares_tail, narrow_zero);
  BFloat16x32 squares_pair = loom::vector::interleave<0, BFloat16x32>(squares_bf16, squares_low);
  BFloat16x32 width_pair = loom::vector::interleave<0, BFloat16x32>(width_inverse, width_inverse);
  Float32x16 leading_means = loom::vector::dot2f(squares_pair, width_pair, c0);
  Float32x16 means = loom::vector::dot2f(squares_tail_pair, width_pair, leading_means);
  [[loom::fastmath("reassoc", "nnan", "ninf", "nsz")]] float mean = loom::vector::reduce::addf(means, 0.0f);
  Float32x16 mean_vector = mean;
  Float32x16 epsilon = 1e-06f;
  [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x16 variance_vector = mean_vector + epsilon;
  float variance = variance_vector[0];
  float inverse = loom::scalar::approximate::rsqrtf(variance);
  Float32x16 inverse_vector = inverse;
  BFloat16x16 inverse_bf16 = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(inverse_vector);
  Float32x16 inverse_high = loom::convert<Float32x16, loom::conversion::subnormal::daz>(inverse_bf16);
  [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x16 inverse_remainder = inverse_vector - inverse_high;
  BFloat16x16 inverse_low = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(inverse_remainder);
  Float32x16 inverse_middle = loom::convert<Float32x16, loom::conversion::subnormal::daz>(inverse_low);
  [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x16 inverse_residual = inverse_remainder - inverse_middle;
  BFloat16x16 inverse_tail = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(inverse_residual);
  BFloat16x32 inverse_tail_pair = loom::vector::interleave<0, BFloat16x32>(inverse_tail, narrow_zero);
  BFloat16x32 inverse_pair = loom::vector::interleave<0, BFloat16x32>(inverse_bf16, inverse_low);
  for (loom::type::index position_1 = loom::type::index{0}; position_1 < length; position_1 = position_1 + loom::type::index{16}) {
    BFloat16x16 input_value = loom::vector::load<BFloat16x16>(input, loom::type::indices{position_1});
    BFloat16x32 input_pair = loom::vector::interleave<0, BFloat16x32>(input_value, input_value);
    Float32x16 leading_normalized = loom::vector::dot2f(input_pair, inverse_pair, c0);
    Float32x16 normalized = loom::vector::dot2f(input_pair, inverse_tail_pair, leading_normalized);
    BFloat16x16 normalized_bf16 = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(normalized);
    BFloat16x32 normalized_pair = loom::vector::interleave<0, BFloat16x32>(normalized_bf16, narrow_zero);
    BFloat16x16 scale_value = loom::vector::load<BFloat16x16>(scale, loom::type::indices{position_1});
    BFloat16x32 scale_pair = loom::vector::interleave<0, BFloat16x32>(scale_value, narrow_zero);
    Float32x16 scaled = loom::vector::dot2f(normalized_pair, scale_pair, c0);
    BFloat16x16 result = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(scaled);
    loom::vector::store(result, output, loom::type::indices{position_1});
  }
  return;
}

[[loom::symbol("qwen3_embedding.block_four_rows"), loom::force_inline]]
void qwen3_embedding_block_four_rows(loom::type::view<loom::type::shape<4, 1024>, std::bfloat16_t> input, loom::type::view<loom::type::shape<128, 4, 8>, std::bfloat16_t> output) {
  for (int group = 0; group < 128; ++group) {
    loom::type::index feature = loom::index::cast(group) * loom::type::index{8};
    for (int row = 0; row < 4; ++row) {
      BFloat16x8 values = loom::vector::load<BFloat16x8>(input, loom::type::indices{loom::index::cast(row), feature});
      loom::vector::store(values, output, {group, row, 0});
    }
  }
  return;
}

[[loom::symbol("qwen3_embedding.three_products_blocked"), loom::force_inline]]
void qwen3_embedding_three_products_blocked(loom::type::buffer<float> storage, loom::type::view<loom::type::shape<8, 8>, float> temporary, loom::type::view<loom::type::shape<16, 4, 64>, std::bfloat16_t> output) {
  auto partials = loom::buffer::view<4, 64>(storage, loom::type::offset{9344}, {});
  Float32x64 c0 = Float32x64{};
  Float32x64 initial = loom::vector::fragment::attach<loom::vector::fragment::role::init>(c0, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
  for (int panel = 0; panel < 16; ++panel) {
    loom::type::offset relative = loom::index::scale(loom::index::cast(panel), loom::type::offset{1152});
    loom::type::offset high_offset = loom::type::offset{27136} + relative;
    loom::type::offset low_offset = loom::type::offset{45632} + relative;
    loom::type::index column = loom::index::cast(panel) * loom::type::index{8};
    auto partial_span0 = loom::view::subview<1, 64>(partials, {0, 0}, {});
    auto partial_tile0 = loom::view::bitcast<loom::type::view<loom::type::shape<8, 8>, float>>(partial_span0);
    auto partial_span1 = loom::view::subview<1, 64>(partials, {1, 0}, {});
    auto partial_tile1 = loom::view::bitcast<loom::type::view<loom::type::shape<8, 8>, float>>(partial_span1);
    auto partial_span2 = loom::view::subview<1, 64>(partials, {2, 0}, {});
    auto partial_tile2 = loom::view::bitcast<loom::type::view<loom::type::shape<8, 8>, float>>(partial_span2);
    auto partial_span3 = loom::view::subview<1, 64>(partials, {3, 0}, {});
    auto partial_tile3 = loom::view::bitcast<loom::type::view<loom::type::shape<8, 8>, float>>(partial_span3);
    for (int product_phase = 0; product_phase < 3; ++product_phase) {
      bool first_product = loom::index::cast(product_phase) == loom::type::index{0};
      bool left_residual_product = loom::index::cast(product_phase) == loom::type::index{1};
      bool right_residual_product = loom::index::cast(product_phase) == loom::type::index{2};
      loom::type::offset product_right = right_residual_product ? low_offset : high_offset;
      loom::type::offset product_left = left_residual_product ? loom::type::offset{4672} : loom::type::offset{0};
      qwen3_embedding_dot128_result call_result = qwen3_embedding_dot128(loom::buffer::cast<unsigned char>(storage), product_right, loom::buffer::cast<unsigned char>(storage), product_left, initial, initial, initial, initial);
      Float32x64 product0 = call_result.value0;
      Float32x64 product1 = call_result.value1;
      Float32x64 product2 = call_result.value2;
      Float32x64 product3 = call_result.value3;
      if (first_product) {
        loom::vector::fragment::store<loom::vector::fragment::role::result>(product0, partial_tile0, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        loom::vector::fragment::store<loom::vector::fragment::role::result>(product1, partial_tile1, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        loom::vector::fragment::store<loom::vector::fragment::role::result>(product2, partial_tile2, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        loom::vector::fragment::store<loom::vector::fragment::role::result>(product3, partial_tile3, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
      } else {
        Float32x64 previous0 = loom::vector::load<Float32x64>(partials, {0, 0});
        Float32x64 fragment0 = loom::vector::fragment::attach<loom::vector::fragment::role::init>(previous0, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x64 added0 = fragment0 + product0;
        loom::vector::fragment::store<loom::vector::fragment::role::result>(added0, partial_tile0, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        Float32x64 previous1 = loom::vector::load<Float32x64>(partials, {1, 0});
        Float32x64 fragment1 = loom::vector::fragment::attach<loom::vector::fragment::role::init>(previous1, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x64 added1 = fragment1 + product1;
        loom::vector::fragment::store<loom::vector::fragment::role::result>(added1, partial_tile1, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        Float32x64 previous2 = loom::vector::load<Float32x64>(partials, {2, 0});
        Float32x64 fragment2 = loom::vector::fragment::attach<loom::vector::fragment::role::init>(previous2, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x64 added2 = fragment2 + product2;
        loom::vector::fragment::store<loom::vector::fragment::role::result>(added2, partial_tile2, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        Float32x64 previous3 = loom::vector::load<Float32x64>(partials, {3, 0});
        Float32x64 fragment3 = loom::vector::fragment::attach<loom::vector::fragment::role::init>(previous3, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
        [[loom::fastmath("nnan", "ninf", "nsz")]] Float32x64 added3 = fragment3 + product3;
        loom::vector::fragment::store<loom::vector::fragment::role::result>(added3, partial_tile3, {0, 0}, loom::vector::fragment::shape{loom::type::index{8}, loom::type::index{8}});
      }
    }
    for (int result_block = 0; result_block < 4; ++result_block) {
      loom::type::index result_base = loom::index::cast(result_block) * loom::type::index{8};
      Float32x64 wide = loom::vector::load<Float32x64>(partials, {result_block, 0});
      BFloat16x64 flat = loom::convert<BFloat16x64, loom::conversion::subnormal::daz>(wide);
      BFloat16x8x8 narrow = loom::vector::bitcast<BFloat16x8x8>(flat);
      BFloat16x8x8 transposed = loom::vector::transpose<BFloat16x8x8, 1, 0>(narrow);
      BFloat16x64 output_values = loom::vector::bitcast<BFloat16x64>(transposed);
      loom::vector::store(output_values, output, {panel, result_block, 0});
    }
  }
  return;
}

[[loom::symbol("qwen3_embedding.swiglu_dense"), loom::force_inline]]
void qwen3_embedding_swiglu_dense(loom::type::view<loom::type::shape<4096>, float> gate, loom::type::view<loom::type::shape<4096>, float> up, loom::type::buffer<unsigned char> table_ab, loom::type::buffer<unsigned char> table_cd, loom::type::view<loom::type::shape<4096>, std::bfloat16_t> output) {
  UInt32x16 c0 = UInt32x16{};
  UInt32x16 c1 = 1u;
  UInt32x16 c4 = 4u;
  UInt32x16 half_exponent = 128u;
  UInt32x16 normal_threshold = 256u;
  UInt32x16 lower = 15232u;
  UInt32x16 upper = 17152u;
  UInt32x16 sign_mask = 32768u;
  UInt32x16 magnitude_mask = 32767u;
  Float32x16 c0_vector_16_f32 = Float32x16{};
  BFloat16x16 narrow_zero = BFloat16x16{};
  for (int position = 0; position < 4096; position += 16) {
    Float32x16 wide_values = loom::vector::load<Float32x16>(gate, {position});
    BFloat16x16 narrow_values = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(wide_values);
    UInt16x16 bits16 = __builtin_bit_cast(UInt16x16, narrow_values);
    UInt32x16 bits = loom::convert<UInt32x16>(bits16);
    UInt32x16 sign = bits & sign_mask;
    UInt32x16 magnitude = bits & magnitude_mask;
    UInt32x16 at_least_lower = loom::vector::maxui(magnitude, lower);
    UInt32x16 bounded_magnitude = loom::vector::minui(at_least_lower, upper);
    UInt32x16 base = sign >> c4;
    UInt32x16 relative = bounded_magnitude - lower;
    UInt32x16 offsets = base + relative;
    BFloat16x16 lookup = qwen3_embedding_silu_lookup(table_ab, table_cd, offsets);
    UInt16x16 lookup16 = __builtin_bit_cast(UInt16x16, lookup);
    UInt32x16 lookup_bits = loom::convert<UInt32x16>(lookup16);
    UInt32x16 normal_half = magnitude - half_exponent;
    UInt32x16 shifted = magnitude >> c1;
    UInt32x16 low = magnitude & c1;
    UInt32x16 odd = shifted & c1;
    UInt32x16 round = low & odd;
    UInt32x16 subnormal_half = shifted + round;
    Boolx16 normal = loom::vector::cmpi::uge<Boolx16>(magnitude, normal_threshold);
    UInt32x16 half = loom::vector::select(normal, normal_half, subnormal_half);
    UInt32x16 signed_half = half | sign;
    Boolx16 nonnegative = loom::vector::cmpi::eq<Boolx16>(sign, c0);
    UInt32x16 saturated = loom::vector::select(nonnegative, bits, sign);
    Boolx16 large = loom::vector::cmpi::ugt<Boolx16>(magnitude, upper);
    UInt32x16 table_or_saturated = loom::vector::select(large, saturated, lookup_bits);
    Boolx16 small = loom::vector::cmpi::ult<Boolx16>(magnitude, lower);
    UInt32x16 selected_bits = loom::vector::select(small, signed_half, table_or_saturated);
    UInt16x16 selected16 = loom::convert<UInt16x16>(selected_bits);
    BFloat16x16 silu = __builtin_bit_cast(BFloat16x16, selected16);
    Float32x16 up_wide = loom::vector::load<Float32x16>(up, {position});
    BFloat16x16 up_narrow = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(up_wide);
    Float32x16 product = qwen3_embedding_silu_product(silu, up_narrow);
    BFloat16x16 result = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(product);
    loom::vector::store(result, output, {position});
  }
  return;
}

[[loom::symbol("qwen3_embedding.dense_residual"), loom::force_inline]]
void qwen3_embedding_dense_residual(loom::type::view<loom::type::shape<4096>, std::bfloat16_t> input, loom::type::view<loom::type::shape<4096>, std::bfloat16_t> skip) {
  for (int position = 0; position < 4096; position += 16) {
    BFloat16x16 left = loom::vector::load<BFloat16x16>(input, {position});
    BFloat16x16 right = loom::vector::load<BFloat16x16>(skip, {position});
    Float32x16 left_wide = loom::convert<Float32x16, loom::conversion::subnormal::daz>(left);
    Float32x16 right_wide = loom::convert<Float32x16, loom::conversion::subnormal::daz>(right);
    Float32x16 sum = left_wide + right_wide;
    BFloat16x16 value = loom::convert<BFloat16x16, loom::conversion::subnormal::daz>(sum);
    loom::vector::store(value, input, {position});
  }
  return;
}

[[loom::symbol("qwen3_embedding.retain_residual_panel"), loom::force_inline]]
void qwen3_embedding_retain_residual_panel(loom::type::view<loom::type::shape<4096>, std::bfloat16_t> input, loom::type::view<loom::type::shape<32, 128>, std::bfloat16_t> output, loom::type::index record, loom::type::index feature) {
  loom::type::index output_base = record * loom::type::index{4};
  for (int row = 0; row < 4; ++row) {
    loom::type::index input_row = loom::index::cast(row) * loom::type::index{1024};
    loom::type::index input_base = input_row + feature;
    loom::type::index output_row = output_base + loom::index::cast(row);
    for (int word = 0; word < 128; word += 32) {
      loom::type::index input_position = input_base + loom::index::cast(word);
      BFloat16x32 value = loom::vector::load<BFloat16x32>(input, loom::type::indices{input_position});
      loom::vector::store(value, output, loom::type::indices{output_row, loom::index::cast(word)});
    }
  }
  return;
}

[[loom::symbol("qwen3_embedding.block_dense_result"), loom::force_inline]]
void qwen3_embedding_block_dense_result(loom::type::view<loom::type::shape<16, 4, 64>, std::bfloat16_t> input, loom::type::view<loom::type::shape<16, 4, 64>, std::bfloat16_t> output) {
  for (int panel = 0; panel < 16; ++panel) {
    for (int block = 0; block < 4; ++block) {
      BFloat16x64 flat = loom::vector::load<BFloat16x64>(input, {panel, block, 0});
      BFloat16x8x8 tile = loom::vector::bitcast<BFloat16x8x8>(flat);
      BFloat16x8x8 transposed = loom::vector::transpose<BFloat16x8x8, 1, 0>(tile);
      BFloat16x64 result = loom::vector::bitcast<BFloat16x64>(transposed);
      loom::vector::store(result, output, {panel, block, 0});
    }
  }
  return;
}

[[loom::symbol("qwen3_embedding.dot128"), loom::force_inline]]
static qwen3_embedding_dot128_result qwen3_embedding_dot128(loom::type::buffer<unsigned char> weights, loom::type::offset weight_offset, loom::type::buffer<unsigned char> activations, loom::type::offset activation_offset, Float32x64 initial0, Float32x64 initial1, Float32x64 initial2, Float32x64 initial3) {
  return loom::low::assembly<Aie2pCore, qwen3_embedding_dot128_result>(R"loom(
(%weights: reg<aie2p.ep>, %weight_offset: reg<aie2p.er>, %activations: reg<aie2p.ep>, %activation_offset: reg<aie2p.er>, %initial0: reg<aie2p.mbms x4>, %initial1: reg<aie2p.mbms x4>, %initial2: reg<aie2p.mbms x4>, %initial3: reg<aie2p.mbms x4>) -> (reg<aie2p.mbms x4>, reg<aie2p.mbms x4>, reg<aie2p.mbms x4>, reg<aie2p.mbms x4>) {
  %control = mova.i32 780
  %zero = mov.short 0
  %one = mov.short 1
  %reduction_halves = mov.short 2
  %w_modifier = mov.modifier %weight_offset
  %w_base = padda.modifier %weights, %w_modifier
  %w_pointer = copy %w_base : reg<aie2p.ep> -> reg<aie2p.eps>
  %w_low = vlda.load-fifo.low512 %w_base, 0
  %w_fifo = vlda.load-fifo.high512 %w_base, %w_low, 64
  %w_position = mova.fifo.position 0
  %a_modifier = mov.modifier %activation_offset
  %a_base = padda.modifier %activations, %a_modifier
  %a_pointer = copy %a_base : reg<aie2p.ep> -> reg<aie2p.eps>
  %a_low = vlda.load-fifo.low512 %a_base, 0
  %a_fifo = vlda.load-fifo.high512 %a_base, %a_low, 64
  %a_position = mova.fifo.position 0
  %first_w_pointer, %first_w_fifo, %first_w_position = vlda.fill.512 %w_pointer, %w_fifo, %w_position
  %first_a_pointer, %first_a_fifo, %first_a_position = vlda.fill.512 %a_pointer, %a_fifo, %a_position
  low.br ^loop(%zero: reg<aie2p.er>, %first_w_pointer: reg<aie2p.eps>, %first_w_fifo: reg<aie2p.eldfiforeg>, %first_w_position: reg<aie2p.erf2>, %first_a_pointer: reg<aie2p.eps>, %first_a_fifo: reg<aie2p.eldfiforeg>, %first_a_position: reg<aie2p.erf2>, %initial0: reg<aie2p.mbms x4>, %initial1: reg<aie2p.mbms x4>, %initial2: reg<aie2p.mbms x4>, %initial3: reg<aie2p.mbms x4>)
^loop(%group: reg<aie2p.er>, %w: reg<aie2p.eps>, %wf: reg<aie2p.eldfiforeg>, %wp: reg<aie2p.erf2>, %a: reg<aie2p.eps>, %af: reg<aie2p.eldfiforeg>, %ap: reg<aie2p.erf2>, %c0: reg<aie2p.mbms x4>, %c1: reg<aie2p.mbms x4>, %c2: reg<aie2p.mbms x4>, %c3: reg<aie2p.mbms x4>):
  %w_0_0, %w_pointer_0_0, %w_fifo_0_0, %w_position_0_0 = vlda.pop.bfp16ebs8 %w, %wf, %wp
  %a_0_0, %a_pointer_0_0, %a_fifo_0_0, %a_position_0_0 = vlda.pop.bfp16ebs8 %a, %af, %ap
  %a_0_1, %a_pointer_0_1, %a_fifo_0_1, %a_position_0_1 = vlda.pop.bfp16ebs8 %a_pointer_0_0, %a_fifo_0_0, %a_position_0_0
  %a_0_2, %a_pointer_0_2, %a_fifo_0_2, %a_position_0_2 = vlda.pop.bfp16ebs8 %a_pointer_0_1, %a_fifo_0_1, %a_position_0_1
  %a_0_3, %a_pointer_0_3, %a_fifo_0_3, %a_position_0_3 = vlda.pop.bfp16ebs8 %a_pointer_0_2, %a_fifo_0_2, %a_position_0_2
  %c0_0 = mma.bfp16ebs8.m8n8k8 %c0, %w_0_0, %a_0_0, %control
  %c1_0 = mma.bfp16ebs8.m8n8k8 %c1, %w_0_0, %a_0_1, %control
  %c2_0 = mma.bfp16ebs8.m8n8k8 %c2, %w_0_0, %a_0_2, %control
  %c3_0 = mma.bfp16ebs8.m8n8k8 %c3, %w_0_0, %a_0_3, %control
  %w_1_0, %w_pointer_1_0, %w_fifo_1_0, %w_position_1_0 = vlda.pop.bfp16ebs8 %w_pointer_0_0, %w_fifo_0_0, %w_position_0_0
  %a_1_0, %a_pointer_1_0, %a_fifo_1_0, %a_position_1_0 = vlda.pop.bfp16ebs8 %a_pointer_0_3, %a_fifo_0_3, %a_position_0_3
  %a_1_1, %a_pointer_1_1, %a_fifo_1_1, %a_position_1_1 = vlda.pop.bfp16ebs8 %a_pointer_1_0, %a_fifo_1_0, %a_position_1_0
  %a_1_2, %a_pointer_1_2, %a_fifo_1_2, %a_position_1_2 = vlda.pop.bfp16ebs8 %a_pointer_1_1, %a_fifo_1_1, %a_position_1_1
  %a_1_3, %a_pointer_1_3, %a_fifo_1_3, %a_position_1_3 = vlda.pop.bfp16ebs8 %a_pointer_1_2, %a_fifo_1_2, %a_position_1_2
  %c0_1 = mma.bfp16ebs8.m8n8k8 %c0_0, %w_1_0, %a_1_0, %control
  %c1_1 = mma.bfp16ebs8.m8n8k8 %c1_0, %w_1_0, %a_1_1, %control
  %c2_1 = mma.bfp16ebs8.m8n8k8 %c2_0, %w_1_0, %a_1_2, %control
  %c3_1 = mma.bfp16ebs8.m8n8k8 %c3_0, %w_1_0, %a_1_3, %control
  %a_pointer_fill2, %a_fifo_fill2, %a_position_fill2 = vlda.fill.512 %a_pointer_1_3, %a_fifo_1_3, %a_position_1_3
  %w_2_0, %w_pointer_2_0, %w_fifo_2_0, %w_position_2_0 = vlda.pop.bfp16ebs8 %w_pointer_1_0, %w_fifo_1_0, %w_position_1_0
  %a_2_0, %a_pointer_2_0, %a_fifo_2_0, %a_position_2_0 = vlda.pop.bfp16ebs8 %a_pointer_fill2, %a_fifo_fill2, %a_position_fill2
  %a_2_1, %a_pointer_2_1, %a_fifo_2_1, %a_position_2_1 = vlda.pop.bfp16ebs8 %a_pointer_2_0, %a_fifo_2_0, %a_position_2_0
  %a_2_2, %a_pointer_2_2, %a_fifo_2_2, %a_position_2_2 = vlda.pop.bfp16ebs8 %a_pointer_2_1, %a_fifo_2_1, %a_position_2_1
  %a_2_3, %a_pointer_2_3, %a_fifo_2_3, %a_position_2_3 = vlda.pop.bfp16ebs8 %a_pointer_2_2, %a_fifo_2_2, %a_position_2_2
  %c0_2 = mma.bfp16ebs8.m8n8k8 %c0_1, %w_2_0, %a_2_0, %control
  %c1_2 = mma.bfp16ebs8.m8n8k8 %c1_1, %w_2_0, %a_2_1, %control
  %c2_2 = mma.bfp16ebs8.m8n8k8 %c2_1, %w_2_0, %a_2_2, %control
  %c3_2 = mma.bfp16ebs8.m8n8k8 %c3_1, %w_2_0, %a_2_3, %control
  %w_3_0, %w_pointer_3_0, %w_fifo_3_0, %w_position_3_0 = vlda.pop.bfp16ebs8 %w_pointer_2_0, %w_fifo_2_0, %w_position_2_0
  %a_3_0, %a_pointer_3_0, %a_fifo_3_0, %a_position_3_0 = vlda.pop.bfp16ebs8 %a_pointer_2_3, %a_fifo_2_3, %a_position_2_3
  %a_3_1, %a_pointer_3_1, %a_fifo_3_1, %a_position_3_1 = vlda.pop.bfp16ebs8 %a_pointer_3_0, %a_fifo_3_0, %a_position_3_0
  %a_3_2, %a_pointer_3_2, %a_fifo_3_2, %a_position_3_2 = vlda.pop.bfp16ebs8 %a_pointer_3_1, %a_fifo_3_1, %a_position_3_1
  %a_3_3, %a_pointer_3_3, %a_fifo_3_3, %a_position_3_3 = vlda.pop.bfp16ebs8 %a_pointer_3_2, %a_fifo_3_2, %a_position_3_2
  %c0_3 = mma.bfp16ebs8.m8n8k8 %c0_2, %w_3_0, %a_3_0, %control
  %c1_3 = mma.bfp16ebs8.m8n8k8 %c1_2, %w_3_0, %a_3_1, %control
  %c2_3 = mma.bfp16ebs8.m8n8k8 %c2_2, %w_3_0, %a_3_2, %control
  %c3_3 = mma.bfp16ebs8.m8n8k8 %c3_2, %w_3_0, %a_3_3, %control
  %a_pointer_fill4, %a_fifo_fill4, %a_position_fill4 = vlda.fill.512 %a_pointer_3_3, %a_fifo_3_3, %a_position_3_3
  %w_4_0, %w_pointer_4_0, %w_fifo_4_0, %w_position_4_0 = vlda.pop.bfp16ebs8 %w_pointer_3_0, %w_fifo_3_0, %w_position_3_0
  %a_4_0, %a_pointer_4_0, %a_fifo_4_0, %a_position_4_0 = vlda.pop.bfp16ebs8 %a_pointer_fill4, %a_fifo_fill4, %a_position_fill4
  %a_4_1, %a_pointer_4_1, %a_fifo_4_1, %a_position_4_1 = vlda.pop.bfp16ebs8 %a_pointer_4_0, %a_fifo_4_0, %a_position_4_0
  %a_4_2, %a_pointer_4_2, %a_fifo_4_2, %a_position_4_2 = vlda.pop.bfp16ebs8 %a_pointer_4_1, %a_fifo_4_1, %a_position_4_1
  %a_4_3, %a_pointer_4_3, %a_fifo_4_3, %a_position_4_3 = vlda.pop.bfp16ebs8 %a_pointer_4_2, %a_fifo_4_2, %a_position_4_2
  %c0_4 = mma.bfp16ebs8.m8n8k8 %c0_3, %w_4_0, %a_4_0, %control
  %c1_4 = mma.bfp16ebs8.m8n8k8 %c1_3, %w_4_0, %a_4_1, %control
  %c2_4 = mma.bfp16ebs8.m8n8k8 %c2_3, %w_4_0, %a_4_2, %control
  %c3_4 = mma.bfp16ebs8.m8n8k8 %c3_3, %w_4_0, %a_4_3, %control
  %w_5_0, %w_pointer_5_0, %w_fifo_5_0, %w_position_5_0 = vlda.pop.bfp16ebs8 %w_pointer_4_0, %w_fifo_4_0, %w_position_4_0
  %a_5_0, %a_pointer_5_0, %a_fifo_5_0, %a_position_5_0 = vlda.pop.bfp16ebs8 %a_pointer_4_3, %a_fifo_4_3, %a_position_4_3
  %a_5_1, %a_pointer_5_1, %a_fifo_5_1, %a_position_5_1 = vlda.pop.bfp16ebs8 %a_pointer_5_0, %a_fifo_5_0, %a_position_5_0
  %a_5_2, %a_pointer_5_2, %a_fifo_5_2, %a_position_5_2 = vlda.pop.bfp16ebs8 %a_pointer_5_1, %a_fifo_5_1, %a_position_5_1
  %a_5_3, %a_pointer_5_3, %a_fifo_5_3, %a_position_5_3 = vlda.pop.bfp16ebs8 %a_pointer_5_2, %a_fifo_5_2, %a_position_5_2
  %c0_5 = mma.bfp16ebs8.m8n8k8 %c0_4, %w_5_0, %a_5_0, %control
  %c1_5 = mma.bfp16ebs8.m8n8k8 %c1_4, %w_5_0, %a_5_1, %control
  %c2_5 = mma.bfp16ebs8.m8n8k8 %c2_4, %w_5_0, %a_5_2, %control
  %c3_5 = mma.bfp16ebs8.m8n8k8 %c3_4, %w_5_0, %a_5_3, %control
  %a_pointer_fill6, %a_fifo_fill6, %a_position_fill6 = vlda.fill.512 %a_pointer_5_3, %a_fifo_5_3, %a_position_5_3
  %w_6_0, %w_pointer_6_0, %w_fifo_6_0, %w_position_6_0 = vlda.pop.bfp16ebs8 %w_pointer_5_0, %w_fifo_5_0, %w_position_5_0
  %a_6_0, %a_pointer_6_0, %a_fifo_6_0, %a_position_6_0 = vlda.pop.bfp16ebs8 %a_pointer_fill6, %a_fifo_fill6, %a_position_fill6
  %a_6_1, %a_pointer_6_1, %a_fifo_6_1, %a_position_6_1 = vlda.pop.bfp16ebs8 %a_pointer_6_0, %a_fifo_6_0, %a_position_6_0
  %a_6_2, %a_pointer_6_2, %a_fifo_6_2, %a_position_6_2 = vlda.pop.bfp16ebs8 %a_pointer_6_1, %a_fifo_6_1, %a_position_6_1
  %a_6_3, %a_pointer_6_3, %a_fifo_6_3, %a_position_6_3 = vlda.pop.bfp16ebs8 %a_pointer_6_2, %a_fifo_6_2, %a_position_6_2
  %c0_6 = mma.bfp16ebs8.m8n8k8 %c0_5, %w_6_0, %a_6_0, %control
  %c1_6 = mma.bfp16ebs8.m8n8k8 %c1_5, %w_6_0, %a_6_1, %control
  %c2_6 = mma.bfp16ebs8.m8n8k8 %c2_5, %w_6_0, %a_6_2, %control
  %c3_6 = mma.bfp16ebs8.m8n8k8 %c3_5, %w_6_0, %a_6_3, %control
  %w_7_0, %w_pointer_7_0, %w_fifo_7_0, %w_position_7_0 = vlda.pop.bfp16ebs8 %w_pointer_6_0, %w_fifo_6_0, %w_position_6_0
  %a_7_0, %a_pointer_7_0, %a_fifo_7_0, %a_position_7_0 = vlda.pop.bfp16ebs8 %a_pointer_6_3, %a_fifo_6_3, %a_position_6_3
  %a_7_1, %a_pointer_7_1, %a_fifo_7_1, %a_position_7_1 = vlda.pop.bfp16ebs8 %a_pointer_7_0, %a_fifo_7_0, %a_position_7_0
  %a_7_2, %a_pointer_7_2, %a_fifo_7_2, %a_position_7_2 = vlda.pop.bfp16ebs8 %a_pointer_7_1, %a_fifo_7_1, %a_position_7_1
  %a_7_3, %a_pointer_7_3, %a_fifo_7_3, %a_position_7_3 = vlda.pop.bfp16ebs8 %a_pointer_7_2, %a_fifo_7_2, %a_position_7_2
  %c0_7 = mma.bfp16ebs8.m8n8k8 %c0_6, %w_7_0, %a_7_0, %control
  %c1_7 = mma.bfp16ebs8.m8n8k8 %c1_6, %w_7_0, %a_7_1, %control
  %c2_7 = mma.bfp16ebs8.m8n8k8 %c2_6, %w_7_0, %a_7_2, %control
  %c3_7 = mma.bfp16ebs8.m8n8k8 %c3_6, %w_7_0, %a_7_3, %control
  %next_group = add.rr %group, %one
  %condition = lt %next_group, %reduction_halves
  low.cond_br %condition, ^refill, ^finish : reg<aie2p.er>
^refill:
  %refilled_w_pointer, %refilled_w_fifo, %refilled_w_position = vlda.fill.512 %w_pointer_7_0, %w_fifo_7_0, %w_position_7_0
  %refilled_a_pointer, %refilled_a_fifo, %refilled_a_position = vlda.fill.512 %a_pointer_7_3, %a_fifo_7_3, %a_position_7_3
  low.br ^loop(%next_group: reg<aie2p.er>, %refilled_w_pointer: reg<aie2p.eps>, %refilled_w_fifo: reg<aie2p.eldfiforeg>, %refilled_w_position: reg<aie2p.erf2>, %refilled_a_pointer: reg<aie2p.eps>, %refilled_a_fifo: reg<aie2p.eldfiforeg>, %refilled_a_position: reg<aie2p.erf2>, %c0_7: reg<aie2p.mbms x4>, %c1_7: reg<aie2p.mbms x4>, %c2_7: reg<aie2p.mbms x4>, %c3_7: reg<aie2p.mbms x4>)
^finish:
  return %c0_7, %c1_7, %c2_7, %c3_7
}
  )loom", weights, weight_offset, activations, activation_offset, initial0, initial1, initial2, initial3);
}

[[loom::symbol("qwen3_embedding.silu_lookup"), loom::force_inline]]
static BFloat16x16 qwen3_embedding_silu_lookup(loom::type::buffer<unsigned char> table_ab, loom::type::buffer<unsigned char> table_cd, UInt32x16 indices) {
  return loom::low::assembly<Aie2pCore, BFloat16x16>(R"loom(
(%table_ab: reg<aie2p.ep>, %table_cd: reg<aie2p.ep>, %indices: reg<aie2p.vec256 x2>) -> (reg<aie2p.vec256 x2>) {
  %address_ab = mov.address-to-scalar %table_ab
  %address_cd = mov.address-to-scalar %table_cd
  %base_ab = vbcst.32 %address_ab
  %base_cd = vbcst.32 %address_cd
  %selection_bits = mov.i32 52428
  %selection = copy %selection_bits : reg<aie2p.er> -> reg<aie2p.ers16>
  %base = vsel.32 %base_ab, %base_cd, %selection
  %twice = vadd.32 %indices, %indices
  %four_times = vadd.32 %twice, %twice
  %pointers = vadd.32 %base, %four_times
  %first = slice %pointers[0] : reg<aie2p.vec256 x2> -> reg<aie2p.vec256>
  %second = slice %pointers[1] : reg<aie2p.vec256 x2> -> reg<aie2p.vec256>
  %values0 = vldb.4x16.lo %first
  %values1 = vldb.4x16.hi %first
  %values2 = vldb.4x16.lo %second
  %values3 = vldb.4x16.hi %second
  %low = concat(%values0, %values1) : (reg<aie2p.vec256>, reg<aie2p.vec256>) -> reg<aie2p.vec256 x2>
  %high = concat(%values2, %values3) : (reg<aie2p.vec256>, reg<aie2p.vec256>) -> reg<aie2p.vec256 x2>
  %mode = mova.i32 24
  %values = vshuffle %low, %high, %mode
  return %values
}
  )loom", table_ab, table_cd, indices);
}
