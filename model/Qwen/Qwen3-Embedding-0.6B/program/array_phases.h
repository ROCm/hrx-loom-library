// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

// Composable dense phases. Child FIFO cursors restart on entry;
// their backing allocations and the shared row channels belong to the root.
// The serial caller joins all cores and DMA before reusing child-owned state.

#ifndef QWEN3_EMBEDDING_ARRAY_PHASES_H_
#define QWEN3_EMBEDDING_ARRAY_PHASES_H_

#include "array_resources.h"
#include "array_workers.h"

#include <loomcxx/predicate.h>

[[loom::symbol("qwen3_embedding.fill_weight_column"), loom::inline]]
static void qwen3_embedding_fill_weight_column(loomt::buffer<unsigned> input0,
                                               WeightStream incoming0,
                                               loomt::offset base,
                                               loomt::index count,
                                               bool refill) {
  for (loomt::index record = loomt::index{0}; record < count;
       record = record + loomt::index{1}) {
    loomt::offset relative = loom::index::scale(record, loomt::offset{18496});
    loomt::offset offset = base + relative;
    auto external0 = loom::buffer::view<4624>(
        loom::buffer::cast<const unsigned>(input0), offset, {});
    auto reservation = loom::channel::reserve(incoming0);
    if (refill) {
      auto copy0 = loom::kernel::async::copy<
          loom::kernel::async::direction::global_to_workgroup,
          loom::cache::scope::workgroup, loom::cache::temporal::regular>(
          external0, reservation.payload);
      auto copy0_group = loom::kernel::async::group(copy0);
      loom::kernel::async::wait<0>(copy0_group);
    }
    loom::channel::publish(reservation.write);
  }
  return;
}

[[loom::pipeline, loom::symbol("qwen3_embedding.distribute_weights16")]]
static void qwen3_embedding_distribute_weights16(
    [[loom::launch]] WeightStream weight_ingress0_16, WeightStream weights0,
    WeightStream weights1, WeightStream weights2, WeightStream weights3,
    loomt::index panels, WeightStream weight_ingress1_16, WeightStream weights4,
    WeightStream weights5, WeightStream weights6, WeightStream weights7,
    WeightStream weight_ingress2_16, WeightStream weights8,
    WeightStream weights9, WeightStream weights10, WeightStream weights11,
    WeightStream weight_ingress3_16, WeightStream weights12,
    WeightStream weights13, WeightStream weights14, WeightStream weights15,
    WeightStream weight_ingress4_16, WeightStream weights16,
    WeightStream weights17, WeightStream weights18, WeightStream weights19,
    WeightStream weight_ingress5_16, WeightStream weights20,
    WeightStream weights21, WeightStream weights22, WeightStream weights23,
    WeightStream weight_ingress6_16, WeightStream weights24,
    WeightStream weights25, WeightStream weights26, WeightStream weights27,
    WeightStream weight_ingress7_16, WeightStream weights28,
    WeightStream weights29, WeightStream weights30, WeightStream weights31) {
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights16(weight_ingress0_16, weights0, weights1,
                                     weights2, weights3, panels);
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights16(weight_ingress1_16, weights4, weights5,
                                     weights6, weights7, panels);
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights16(weight_ingress2_16, weights8, weights9,
                                     weights10, weights11, panels);
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights16(weight_ingress3_16, weights12, weights13,
                                     weights14, weights15, panels);
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights16(weight_ingress4_16, weights16, weights17,
                                     weights18, weights19, panels);
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights16(weight_ingress5_16, weights20, weights21,
                                     weights22, weights23, panels);
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights16(weight_ingress6_16, weights24, weights25,
                                     weights26, weights27, panels);
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights16(weight_ingress7_16, weights28, weights29,
                                     weights30, weights31, panels);
  }
}

[[loom::pipeline, loom::symbol("qwen3_embedding.distribute_weights24")]]
static void qwen3_embedding_distribute_weights24(
    [[loom::launch]] WeightStream weight_ingress0_24, WeightStream weights0,
    WeightStream weights1, WeightStream weights2, WeightStream weights3,
    loomt::index panels, WeightStream weight_ingress1_24, WeightStream weights4,
    WeightStream weights5, WeightStream weights6, WeightStream weights7,
    WeightStream weight_ingress2_24, WeightStream weights8,
    WeightStream weights9, WeightStream weights10, WeightStream weights11,
    WeightStream weight_ingress3_24, WeightStream weights12,
    WeightStream weights13, WeightStream weights14, WeightStream weights15,
    WeightStream weight_ingress4_24, WeightStream weights16,
    WeightStream weights17, WeightStream weights18, WeightStream weights19,
    WeightStream weight_ingress5_24, WeightStream weights20,
    WeightStream weights21, WeightStream weights22, WeightStream weights23,
    WeightStream weight_ingress6_24, WeightStream weights24,
    WeightStream weights25, WeightStream weights26, WeightStream weights27,
    WeightStream weight_ingress7_24, WeightStream weights28,
    WeightStream weights29, WeightStream weights30, WeightStream weights31) {
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights24(weight_ingress0_24, weights0, weights1,
                                     weights2, weights3, panels);
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights24(weight_ingress1_24, weights4, weights5,
                                     weights6, weights7, panels);
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights24(weight_ingress2_24, weights8, weights9,
                                     weights10, weights11, panels);
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights24(weight_ingress3_24, weights12, weights13,
                                     weights14, weights15, panels);
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights24(weight_ingress4_24, weights16, weights17,
                                     weights18, weights19, panels);
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights24(weight_ingress5_24, weights20, weights21,
                                     weights22, weights23, panels);
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights24(weight_ingress6_24, weights24, weights25,
                                     weights26, weights27, panels);
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_replay_weights24(weight_ingress7_24, weights28, weights29,
                                     weights30, weights31, panels);
  }
}

[[loom::pipeline, loom::symbol("qwen3_embedding.dense_group")]]
static void qwen3_embedding_dense_group(
    [[loom::launch]] Bytes sequence0_storage, Bytes storage0, Bytes storage1,
    Bytes gather_cache0, Bytes activations, ActivationStream activations0,
    Bytes constants, Bytes embedding, Bytes state, BarrierStream arrival0,
    BarrierStream departure0, FeatureStream query_output0, Bytes output,
    WeightStream weights0, FeatureStream probability0, FeatureStream result0,
    BarrierStream phase_complete, Bytes dimensions1_storage,
    ActivationStream activations1, BarrierStream arrival1,
    BarrierStream departure1, WeightStream weights1, Bytes sequence2_storage,
    Bytes storage2, Bytes storage3, ActivationStream activations2,
    BarrierStream arrival2, BarrierStream departure2,
    FeatureStream query_output2, WeightStream weights2,
    FeatureStream probability2, FeatureStream result2,
    Bytes dimensions3_storage, ActivationStream activations3,
    BarrierStream arrival3, BarrierStream departure3, WeightStream weights3,
    Bytes sequence4_storage, Bytes storage4, Bytes storage5,
    ActivationStream activations4, BarrierStream arrival4,
    BarrierStream arrival5, BarrierStream departure5, BarrierStream departure4,
    FeatureStream query_output4, WeightStream weights4,
    FeatureStream probability4, FeatureStream result4,
    Bytes dimensions5_storage, ActivationStream activations5,
    BarrierStream arrival6, BarrierStream departure6, WeightStream weights5,
    Bytes sequence6_storage, Bytes storage6, Bytes storage7,
    ActivationStream activations6, BarrierStream arrival7,
    BarrierStream departure7, FeatureStream query_output6,
    WeightStream weights6, FeatureStream probability6, FeatureStream result6,
    Bytes dimensions7_storage, ActivationStream activations7,
    WeightStream weights7, Bytes sequence8_storage, Bytes dimensions8_storage,
    Bytes storage8, Bytes storage9, ActivationStream activations8,
    BarrierStream arrival8, BarrierStream departure8,
    FeatureStream normal_output8, WeightStream weights8,
    FeatureStream query_output8, FeatureStream probability8,
    Bytes gather_cache9, ActivationStream activations9, BarrierStream arrival9,
    BarrierStream departure9, WeightStream weights9, FeatureStream result8,
    Bytes sequence10_storage, Bytes dimensions10_storage, Bytes storage10,
    Bytes storage11, ActivationStream activations10, BarrierStream arrival10,
    BarrierStream departure10, FeatureStream normal_output10,
    WeightStream weights10, FeatureStream probability10,
    ActivationStream activations11, BarrierStream arrival11,
    BarrierStream departure11, WeightStream weights11,
    FeatureStream query_output10, FeatureStream result10,
    Bytes sequence12_storage, Bytes storage12, Bytes storage13,
    ActivationStream activations12, BarrierStream arrival12,
    BarrierStream arrival13, BarrierStream departure13,
    BarrierStream departure12, FeatureStream query_output12,
    WeightStream weights12, FeatureStream probability12, FeatureStream result12,
    Bytes dimensions13_storage, ActivationStream activations13,
    BarrierStream arrival14, BarrierStream departure14, WeightStream weights13,
    Bytes sequence14_storage, Bytes storage14, Bytes storage15,
    ActivationStream activations14, BarrierStream arrival15,
    BarrierStream departure15, FeatureStream query_output14,
    WeightStream weights14, FeatureStream probability14, FeatureStream result14,
    Bytes dimensions15_storage, ActivationStream activations15,
    WeightStream weights15, Bytes sequence16_storage, Bytes storage16,
    Bytes storage17, ActivationStream activations16, BarrierStream arrival16,
    BarrierStream departure16, FeatureStream query_output16,
    WeightStream weights16, FeatureStream probability16, FeatureStream result16,
    Bytes dimensions17_storage, ActivationStream activations17,
    BarrierStream arrival17, BarrierStream departure17, WeightStream weights17,
    Bytes sequence18_storage, Bytes storage18, Bytes storage19,
    Bytes gather_cache18, ActivationStream activations18,
    BarrierStream arrival18, BarrierStream departure18,
    FeatureStream query_output18, WeightStream weights18,
    FeatureStream probability18, FeatureStream result18,
    Bytes dimensions19_storage, ActivationStream activations19,
    BarrierStream arrival19, BarrierStream departure19, WeightStream weights19,
    Bytes sequence20_storage, Bytes storage20, Bytes storage21,
    ActivationStream activations20, BarrierStream arrival20,
    BarrierStream arrival21, BarrierStream departure21,
    BarrierStream departure20, FeatureStream query_output20,
    WeightStream weights20, FeatureStream probability20, FeatureStream result20,
    Bytes dimensions21_storage, ActivationStream activations21,
    BarrierStream arrival22, BarrierStream departure22, WeightStream weights21,
    Bytes sequence22_storage, Bytes storage22, Bytes storage23,
    ActivationStream activations22, BarrierStream arrival23,
    BarrierStream departure23, FeatureStream query_output22,
    WeightStream weights22, FeatureStream probability22, FeatureStream result22,
    Bytes dimensions23_storage, ActivationStream activations23,
    WeightStream weights23, Bytes sequence24_storage,
    Bytes dimensions24_storage, Bytes storage24, Bytes storage25,
    ActivationStream activations24, BarrierStream arrival24,
    BarrierStream departure24, FeatureStream normal_output24,
    WeightStream weights24, FeatureStream probability24,
    ActivationStream activations25, BarrierStream arrival25,
    BarrierStream departure25, WeightStream weights25,
    FeatureStream query_output24, FeatureStream result24,
    Bytes sequence26_storage, Bytes dimensions26_storage, Bytes storage26,
    Bytes storage27, ActivationStream activations26, BarrierStream arrival26,
    BarrierStream departure26, FeatureStream normal_output26,
    WeightStream weights26, FeatureStream query_output26,
    FeatureStream probability26, Bytes gather_cache27,
    ActivationStream activations27, BarrierStream arrival27,
    BarrierStream departure27, WeightStream weights27, FeatureStream result26,
    Bytes sequence28_storage, Bytes storage28, Bytes storage29,
    ActivationStream activations28, BarrierStream arrival29,
    BarrierStream departure29, FeatureStream query_output28,
    WeightStream weights28, FeatureStream probability28, FeatureStream result28,
    Bytes dimensions29_storage, ActivationStream activations29,
    BarrierStream arrival30, BarrierStream departure30, WeightStream weights29,
    Bytes sequence30_storage, Bytes storage30, Bytes storage31,
    ActivationStream activations30, BarrierStream arrival31,
    BarrierStream departure31, FeatureStream query_output30,
    WeightStream weights30, FeatureStream probability30, FeatureStream result30,
    Bytes dimensions31_storage, ActivationStream activations31,
    WeightStream weights31, Bytes projection0, Bytes weight_ingress0_16_storage,
    Bytes projection1, Bytes weight_ingress1_16_storage, Bytes projection2,
    Bytes weight_ingress2_16_storage, Bytes projection3,
    Bytes weight_ingress3_16_storage, Bytes projection4,
    Bytes weight_ingress4_16_storage, Bytes projection5,
    Bytes weight_ingress5_16_storage, Bytes projection6,
    Bytes weight_ingress6_16_storage, Bytes projection7,
    Bytes weight_ingress7_16_storage, Bytes weight_ingress0_24_storage,
    Bytes weight_ingress1_24_storage, Bytes weight_ingress2_24_storage,
    Bytes weight_ingress3_24_storage, Bytes weight_ingress4_24_storage,
    Bytes weight_ingress5_24_storage, Bytes weight_ingress6_24_storage,
    Bytes weight_ingress7_24_storage, loomt::index layer,
    loomt::index dense_group, loomt::index panel_index, bool reload_weights) {
  auto weight_ingress0_16 = bind_weight_ingress<16>(weight_ingress0_16_storage);
  auto weight_ingress1_16 = bind_weight_ingress<16>(weight_ingress1_16_storage);
  auto weight_ingress2_16 = bind_weight_ingress<16>(weight_ingress2_16_storage);
  auto weight_ingress3_16 = bind_weight_ingress<16>(weight_ingress3_16_storage);
  auto weight_ingress4_16 = bind_weight_ingress<16>(weight_ingress4_16_storage);
  auto weight_ingress5_16 = bind_weight_ingress<16>(weight_ingress5_16_storage);
  auto weight_ingress6_16 = bind_weight_ingress<16>(weight_ingress6_16_storage);
  auto weight_ingress7_16 = bind_weight_ingress<16>(weight_ingress7_16_storage);
  auto weight_ingress0_24 = bind_weight_ingress<24>(weight_ingress0_24_storage);
  auto weight_ingress1_24 = bind_weight_ingress<24>(weight_ingress1_24_storage);
  auto weight_ingress2_24 = bind_weight_ingress<24>(weight_ingress2_24_storage);
  auto weight_ingress3_24 = bind_weight_ingress<24>(weight_ingress3_24_storage);
  auto weight_ingress4_24 = bind_weight_ingress<24>(weight_ingress4_24_storage);
  auto weight_ingress5_24 = bind_weight_ingress<24>(weight_ingress5_24_storage);
  auto weight_ingress6_24 = bind_weight_ingress<24>(weight_ingress6_24_storage);
  auto weight_ingress7_24 = bind_weight_ingress<24>(weight_ingress7_24_storage);
  bool is_down = dense_group == loomt::index{6};
  bool is_model_layer = loom::index::ult(layer, loomt::index{28});
  [[loom::serial]] {
    if (is_model_layer) {
      if (is_down) {
        loom::pipeline::compose<qwen3_embedding_distribute_weights24>(
            weight_ingress0_24, weights0, weights1, weights2, weights3,
            loomt::index{1}, weight_ingress1_24, weights4, weights5, weights6,
            weights7, weight_ingress2_24, weights8, weights9, weights10,
            weights11, weight_ingress3_24, weights12, weights13, weights14,
            weights15, weight_ingress4_24, weights16, weights17, weights18,
            weights19, weight_ingress5_24, weights20, weights21, weights22,
            weights23, weight_ingress6_24, weights24, weights25, weights26,
            weights27, weight_ingress7_24, weights28, weights29, weights30,
            weights31);
      } else {
        loom::pipeline::compose<qwen3_embedding_distribute_weights16>(
            weight_ingress0_16, weights0, weights1, weights2, weights3,
            loomt::index{1}, weight_ingress1_16, weights4, weights5, weights6,
            weights7, weight_ingress2_16, weights8, weights9, weights10,
            weights11, weight_ingress3_16, weights12, weights13, weights14,
            weights15, weight_ingress4_16, weights16, weights17, weights18,
            weights19, weight_ingress5_16, weights20, weights21, weights22,
            weights23, weight_ingress6_16, weights24, weights25, weights26,
            weights27, weight_ingress7_16, weights28, weights29, weights30,
            weights31);
      }
    }
  }
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_ingress_root(
        sequence0_storage, loomt::offset{0}, storage0, loomt::offset{10624},
        storage1, gather_cache0, activations, activations0, loomt::index{0},
        loomt::index{1}, constants, embedding, state, arrival0, departure0,
        query_output0, output, weights0, loomt::offset{18816}, probability0,
        result0, phase_complete, layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{3}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(
        false, dimensions1_storage, loomt::offset{0}, storage1,
        loomt::offset{10624}, storage0, activations1, loomt::index{0},
        loomt::index{1}, arrival1, arrival0, departure0, departure1,
        query_output0, weights1, loomt::offset{18816}, probability0, result0,
        loomt::index{0}, loomt::index{0}, loomt::index{0}, loomt::index{0},
        layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{4}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    if (is_model_layer) {
      loom::assume(loom::predicate::range(layer, 0, 27));
      loomt::offset layer_base =
          loom::index::scale(layer, loomt::offset{2219520});
      loomt::offset group_relative =
          loom::index::scale(dense_group, loomt::offset{295936});
      loomt::offset group_base = layer_base + group_relative;
      if (is_down) {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection0), weight_ingress0_24,
            group_base, loomt::index{24}, reload_weights);
      } else {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection0), weight_ingress0_16,
            group_base, loomt::index{16}, reload_weights);
      }
    }
    qwen3_embedding_query_key_egress(
        sequence2_storage, loomt::offset{0}, storage2, loomt::offset{10624},
        storage3, activations2, loomt::index{0}, loomt::index{1}, arrival2,
        arrival1, departure1, departure2, query_output2, output, state,
        weights2, loomt::offset{18816}, probability2, result2, loomt::index{0},
        loomt::index{0}, loomt::offset{262144}, loomt::offset{1073152},
        loomt::index{0}, loomt::offset{548864}, loomt::offset{24576},
        loomt::offset{271360}, loomt::offset{1859584}, loomt::offset{2760704},
        loomt::offset{1351680}, layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{5}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(
        false, dimensions3_storage, loomt::offset{0}, storage3,
        loomt::offset{10624}, storage2, activations3, loomt::index{0},
        loomt::index{1}, arrival3, arrival2, departure2, departure3,
        query_output2, weights3, loomt::offset{18816}, probability2, result2,
        loomt::index{0}, loomt::index{64}, loomt::index{0}, loomt::index{0},
        layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    if (is_model_layer) {
      loom::assume(loom::predicate::range(layer, 0, 27));
      loomt::offset layer_base_1 =
          loom::index::scale(layer, loomt::offset{2219520});
      loomt::offset group_relative_1 =
          loom::index::scale(dense_group, loomt::offset{295936});
      loomt::offset group_base_1 = layer_base_1 + group_relative_1;
      if (is_down) {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection1), weight_ingress1_24,
            group_base_1, loomt::index{24}, reload_weights);
      } else {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection1), weight_ingress1_16,
            group_base_1, loomt::index{16}, reload_weights);
      }
    }
    qwen3_embedding_query_key_egress(
        sequence4_storage, loomt::offset{0}, storage4, loomt::offset{10624},
        storage5, activations4, loomt::index{0}, loomt::index{1}, arrival4,
        arrival5, departure5, departure4, query_output4, output, state,
        weights4, loomt::offset{18816}, probability4, result4,
        loomt::index{512}, loomt::index{1}, loomt::offset{16384},
        loomt::offset{1056832}, loomt::index{128}, loomt::offset{794624},
        loomt::offset{40960}, loomt::offset{303104}, loomt::offset{1875968},
        loomt::offset{2629888}, loomt::offset{1384448}, layer, dense_group,
        panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{3}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(
        false, dimensions5_storage, loomt::offset{0}, storage5,
        loomt::offset{10624}, storage4, activations5, loomt::index{0},
        loomt::index{1}, arrival5, arrival6, departure6, departure5,
        query_output4, weights5, loomt::offset{18816}, probability4, result4,
        loomt::index{512}, loomt::index{4}, loomt::index{1}, loomt::index{128},
        layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{4}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(
        sequence6_storage, loomt::offset{0}, storage6, loomt::offset{10624},
        storage7, activations6, loomt::index{0}, loomt::index{1}, arrival6,
        arrival7, departure7, departure6, query_output6, output, state,
        weights6, loomt::offset{18816}, probability6, result6,
        loomt::index{512}, loomt::index{1}, loomt::offset{278528},
        loomt::offset{1073216}, loomt::index{128}, loomt::offset{811008},
        loomt::offset{57344}, loomt::offset{304128}, loomt::offset{1892352},
        loomt::offset{2760960}, loomt::offset{1417216}, layer, dense_group,
        panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{5}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(
        false, dimensions7_storage, loomt::offset{0}, storage7,
        loomt::offset{10624}, storage6, activations7, loomt::index{0},
        loomt::index{1}, arrival7, arrival3, departure3, departure7,
        query_output6, weights7, loomt::offset{18816}, probability6, result6,
        loomt::index{512}, loomt::index{68}, loomt::index{1}, loomt::index{128},
        layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_epilogue(
        sequence8_storage, loomt::offset{0}, dimensions8_storage, storage8,
        loomt::offset{10624}, storage9, activations8, loomt::index{0},
        loomt::index{1}, arrival8, arrival4, departure4, departure8,
        normal_output8, weights8, loomt::offset{18816}, query_output8,
        probability8, loomt::index{1024}, loomt::index{8}, loomt::index{2},
        loomt::index{256}, layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{3}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value_ingress(
        dimensions8_storage, storage9, loomt::offset{0}, loomt::offset{10624},
        storage8, gather_cache9, activations, activations9, loomt::index{0},
        loomt::index{1}, constants, embedding, state, arrival9, arrival8,
        departure8, departure9, normal_output8, output, weights9,
        loomt::offset{18816}, query_output8, probability8, sequence8_storage,
        result8, loomt::index{1024}, loomt::index{32}, loomt::index{2},
        loomt::offset{2695168}, loomt::offset{32768}, loomt::offset{1056896},
        loomt::offset{1064960}, loomt::offset{1327104}, loomt::offset{1851392},
        loomt::offset{2695168}, loomt::offset{2899968}, loomt::index{256},
        loomt::offset{565248}, loomt::offset{73728}, loomt::offset{335872},
        loomt::offset{1908736}, loomt::offset{2630144}, loomt::offset{1449984},
        layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{4}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_epilogue(
        sequence10_storage, loomt::offset{0}, dimensions10_storage, storage10,
        loomt::offset{10624}, storage11, activations10, loomt::index{0},
        loomt::index{1}, arrival10, arrival9, departure9, departure10,
        normal_output10, weights10, loomt::offset{18816}, query_output10,
        probability10, loomt::index{1024}, loomt::index{72}, loomt::index{2},
        loomt::index{256}, layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{5}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    if (is_model_layer) {
      loom::assume(loom::predicate::range(layer, 0, 27));
      loomt::offset layer_base_2 =
          loom::index::scale(layer, loomt::offset{2219520});
      loomt::offset group_relative_2 =
          loom::index::scale(dense_group, loomt::offset{295936});
      loomt::offset group_base_2 = layer_base_2 + group_relative_2;
      if (is_down) {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection2), weight_ingress2_24,
            group_base_2, loomt::index{24}, reload_weights);
      } else {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection2), weight_ingress2_16,
            group_base_2, loomt::index{16}, reload_weights);
      }
    }
    qwen3_embedding_probability_value_egress(
        dimensions10_storage, storage11, loomt::offset{0}, loomt::offset{10624},
        storage10, activations11, loomt::index{0}, loomt::index{1}, arrival11,
        arrival10, departure10, departure11, normal_output10, output, state,
        weights11, loomt::offset{18816}, query_output10, probability10,
        result10, loomt::index{1024}, loomt::index{2}, loomt::offset{294912},
        loomt::offset{1073280}, loomt::index{256}, loomt::offset{581632},
        loomt::offset{90112}, loomt::offset{336896}, loomt::offset{1925120},
        loomt::offset{2761216}, loomt::offset{1482752}, layer, dense_group,
        panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    if (is_model_layer) {
      loom::assume(loom::predicate::range(layer, 0, 27));
      loomt::offset layer_base_3 =
          loom::index::scale(layer, loomt::offset{2219520});
      loomt::offset group_relative_3 =
          loom::index::scale(dense_group, loomt::offset{295936});
      loomt::offset group_base_3 = layer_base_3 + group_relative_3;
      if (is_down) {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection3), weight_ingress3_24,
            group_base_3, loomt::index{24}, reload_weights);
      } else {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection3), weight_ingress3_16,
            group_base_3, loomt::index{16}, reload_weights);
      }
    }
    qwen3_embedding_query_key_egress(
        sequence12_storage, loomt::offset{0}, storage12, loomt::offset{10624},
        storage13, activations12, loomt::index{0}, loomt::index{1}, arrival12,
        arrival13, departure13, departure12, query_output12, output, state,
        weights12, loomt::offset{18816}, probability12, result12,
        loomt::index{1536}, loomt::index{3}, loomt::offset{49152},
        loomt::offset{1056960}, loomt::index{384}, loomt::offset{827392},
        loomt::offset{106496}, loomt::offset{368640}, loomt::offset{1941504},
        loomt::offset{2630400}, loomt::offset{1515520}, layer, dense_group,
        panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{3}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(
        false, dimensions13_storage, loomt::offset{0}, storage13,
        loomt::offset{10624}, storage12, activations13, loomt::index{0},
        loomt::index{1}, arrival13, arrival14, departure14, departure13,
        query_output12, weights13, loomt::offset{18816}, probability12,
        result12, loomt::index{1536}, loomt::index{12}, loomt::index{3},
        loomt::index{384}, layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{4}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(
        sequence14_storage, loomt::offset{0}, storage14, loomt::offset{10624},
        storage15, activations14, loomt::index{0}, loomt::index{1}, arrival14,
        arrival15, departure15, departure14, query_output14, output, state,
        weights14, loomt::offset{18816}, probability14, result14,
        loomt::index{1536}, loomt::index{3}, loomt::offset{311296},
        loomt::offset{1073344}, loomt::index{384}, loomt::offset{843776},
        loomt::offset{122880}, loomt::offset{369664}, loomt::offset{1957888},
        loomt::offset{2761472}, loomt::offset{1548288}, layer, dense_group,
        panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{5}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(
        false, dimensions15_storage, loomt::offset{0}, storage15,
        loomt::offset{10624}, storage14, activations15, loomt::index{0},
        loomt::index{1}, arrival15, arrival11, departure11, departure15,
        query_output14, weights15, loomt::offset{18816}, probability14,
        result14, loomt::index{1536}, loomt::index{76}, loomt::index{3},
        loomt::index{384}, layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    if (is_model_layer) {
      loom::assume(loom::predicate::range(layer, 0, 27));
      loomt::offset layer_base_4 =
          loom::index::scale(layer, loomt::offset{2219520});
      loomt::offset group_relative_4 =
          loom::index::scale(dense_group, loomt::offset{295936});
      loomt::offset group_base_4 = layer_base_4 + group_relative_4;
      if (is_down) {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection4), weight_ingress4_24,
            group_base_4, loomt::index{24}, reload_weights);
      } else {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection4), weight_ingress4_16,
            group_base_4, loomt::index{16}, reload_weights);
      }
    }
    qwen3_embedding_query_key_egress(
        sequence16_storage, loomt::offset{0}, storage16, loomt::offset{10624},
        storage17, activations16, loomt::index{0}, loomt::index{1}, arrival16,
        arrival12, departure12, departure16, query_output16, output, state,
        weights16, loomt::offset{18816}, probability16, result16,
        loomt::index{2048}, loomt::index{4}, loomt::offset{65536},
        loomt::offset{1057024}, loomt::index{512}, loomt::offset{598016},
        loomt::offset{139264}, loomt::offset{401408}, loomt::offset{1974272},
        loomt::offset{2630656}, loomt::offset{1581056}, layer, dense_group,
        panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{3}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(
        false, dimensions17_storage, loomt::offset{0}, storage17,
        loomt::offset{10624}, storage16, activations17, loomt::index{0},
        loomt::index{1}, arrival17, arrival16, departure16, departure17,
        query_output16, weights17, loomt::offset{18816}, probability16,
        result16, loomt::index{2048}, loomt::index{16}, loomt::index{4},
        loomt::index{512}, layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{4}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_ingress(
        sequence18_storage, loomt::offset{0}, storage18, loomt::offset{10624},
        storage19, gather_cache18, activations, activations18, loomt::index{0},
        loomt::index{1}, constants, embedding, state, arrival18, arrival17,
        departure17, departure18, query_output18, output, weights18,
        loomt::offset{18816}, probability18, result18, layer, dense_group,
        panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{5}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(
        false, dimensions19_storage, loomt::offset{0}, storage19,
        loomt::offset{10624}, storage18, activations19, loomt::index{0},
        loomt::index{1}, arrival19, arrival18, departure18, departure19,
        query_output18, weights19, loomt::offset{18816}, probability18,
        result18, loomt::index{2048}, loomt::index{80}, loomt::index{4},
        loomt::index{512}, layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    if (is_model_layer) {
      loom::assume(loom::predicate::range(layer, 0, 27));
      loomt::offset layer_base_5 =
          loom::index::scale(layer, loomt::offset{2219520});
      loomt::offset group_relative_5 =
          loom::index::scale(dense_group, loomt::offset{295936});
      loomt::offset group_base_5 = layer_base_5 + group_relative_5;
      if (is_down) {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection5), weight_ingress5_24,
            group_base_5, loomt::index{24}, reload_weights);
      } else {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection5), weight_ingress5_16,
            group_base_5, loomt::index{16}, reload_weights);
      }
    }
    qwen3_embedding_query_key_egress(
        sequence20_storage, loomt::offset{0}, storage20, loomt::offset{10624},
        storage21, activations20, loomt::index{0}, loomt::index{1}, arrival20,
        arrival21, departure21, departure20, query_output20, output, state,
        weights20, loomt::offset{18816}, probability20, result20,
        loomt::index{2560}, loomt::index{5}, loomt::offset{81920},
        loomt::offset{1057088}, loomt::index{640}, loomt::offset{860160},
        loomt::offset{172032}, loomt::offset{434176}, loomt::offset{2007040},
        loomt::offset{2630912}, loomt::offset{1646592}, layer, dense_group,
        panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{3}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(
        false, dimensions21_storage, loomt::offset{0}, storage21,
        loomt::offset{10624}, storage20, activations21, loomt::index{0},
        loomt::index{1}, arrival21, arrival22, departure22, departure21,
        query_output20, weights21, loomt::offset{18816}, probability20,
        result20, loomt::index{2560}, loomt::index{20}, loomt::index{5},
        loomt::index{640}, layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{4}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(
        sequence22_storage, loomt::offset{0}, storage22, loomt::offset{10624},
        storage23, activations22, loomt::index{0}, loomt::index{1}, arrival22,
        arrival23, departure23, departure22, query_output22, output, state,
        weights22, loomt::offset{18816}, probability22, result22,
        loomt::index{2560}, loomt::index{5}, loomt::offset{344064},
        loomt::offset{1073472}, loomt::index{640}, loomt::offset{876544},
        loomt::offset{188416}, loomt::offset{435200}, loomt::offset{2023424},
        loomt::offset{2761984}, loomt::offset{1679360}, layer, dense_group,
        panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{5}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(
        false, dimensions23_storage, loomt::offset{0}, storage23,
        loomt::offset{10624}, storage22, activations23, loomt::index{0},
        loomt::index{1}, arrival23, arrival19, departure19, departure23,
        query_output22, weights23, loomt::offset{18816}, probability22,
        result22, loomt::index{2560}, loomt::index{84}, loomt::index{5},
        loomt::index{640}, layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_epilogue(
        sequence24_storage, loomt::offset{0}, dimensions24_storage, storage24,
        loomt::offset{10624}, storage25, activations24, loomt::index{0},
        loomt::index{1}, arrival24, arrival20, departure20, departure24,
        normal_output24, weights24, loomt::offset{18816}, query_output24,
        probability24, loomt::index{3072}, loomt::index{24}, loomt::index{6},
        loomt::index{768}, layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{3}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    if (is_model_layer) {
      loom::assume(loom::predicate::range(layer, 0, 27));
      loomt::offset layer_base_6 =
          loom::index::scale(layer, loomt::offset{2219520});
      loomt::offset group_relative_6 =
          loom::index::scale(dense_group, loomt::offset{295936});
      loomt::offset group_base_6 = layer_base_6 + group_relative_6;
      if (is_down) {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection6), weight_ingress6_24,
            group_base_6, loomt::index{24}, reload_weights);
      } else {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection6), weight_ingress6_16,
            group_base_6, loomt::index{16}, reload_weights);
      }
    }
    qwen3_embedding_probability_value_egress(
        dimensions24_storage, storage25, loomt::offset{0}, loomt::offset{10624},
        storage24, activations25, loomt::index{0}, loomt::index{1}, arrival25,
        arrival24, departure24, departure25, normal_output24, output, state,
        weights25, loomt::offset{18816}, query_output24, probability24,
        result24, loomt::index{3072}, loomt::index{6}, loomt::offset{98304},
        loomt::offset{1057152}, loomt::index{768}, loomt::offset{630784},
        loomt::offset{204800}, loomt::offset{466944}, loomt::offset{2039808},
        loomt::offset{2631168}, loomt::offset{1712128}, layer, dense_group,
        panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{4}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_epilogue(
        sequence26_storage, loomt::offset{0}, dimensions26_storage, storage26,
        loomt::offset{10624}, storage27, activations26, loomt::index{0},
        loomt::index{1}, arrival26, arrival25, departure25, departure26,
        normal_output26, weights26, loomt::offset{18816}, query_output26,
        probability26, loomt::index{3072}, loomt::index{88}, loomt::index{6},
        loomt::index{768}, layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{5}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value_ingress(
        dimensions26_storage, storage27, loomt::offset{0}, loomt::offset{10624},
        storage26, gather_cache27, activations, activations27, loomt::index{0},
        loomt::index{1}, constants, embedding, state, arrival27, arrival26,
        departure26, departure27, normal_output26, output, weights27,
        loomt::offset{18816}, query_output26, probability26, sequence26_storage,
        result26, loomt::index{3072}, loomt::index{96}, loomt::index{6},
        loomt::offset{2826240}, loomt::offset{360448}, loomt::offset{1073536},
        loomt::offset{1081344}, loomt::offset{1343488}, loomt::offset{1867776},
        loomt::offset{2826240}, loomt::offset{2916352}, loomt::index{768},
        loomt::offset{647168}, loomt::offset{221184}, loomt::offset{467968},
        loomt::offset{2056192}, loomt::offset{2762240}, loomt::offset{1744896},
        layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    if (is_model_layer) {
      loom::assume(loom::predicate::range(layer, 0, 27));
      loomt::offset layer_base_7 =
          loom::index::scale(layer, loomt::offset{2219520});
      loomt::offset group_relative_7 =
          loom::index::scale(dense_group, loomt::offset{295936});
      loomt::offset group_base_7 = layer_base_7 + group_relative_7;
      if (is_down) {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection7), weight_ingress7_24,
            group_base_7, loomt::index{24}, reload_weights);
      } else {
        qwen3_embedding_fill_weight_column(
            loom::buffer::cast<unsigned>(projection7), weight_ingress7_16,
            group_base_7, loomt::index{16}, reload_weights);
      }
    }
    qwen3_embedding_query_key_egress_terminal(
        sequence28_storage, loomt::offset{0}, storage28, loomt::offset{10624},
        storage29, activations28, loomt::index{0}, loomt::index{1}, arrival29,
        departure29, query_output28, output, state, weights28,
        loomt::offset{18816}, probability28, result28, layer, dense_group,
        panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{3}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(
        false, dimensions29_storage, loomt::offset{0}, storage29,
        loomt::offset{10624}, storage28, activations29, loomt::index{0},
        loomt::index{1}, arrival29, arrival30, departure30, departure29,
        query_output28, weights29, loomt::offset{18816}, probability28,
        result28, loomt::index{3584}, loomt::index{28}, loomt::index{7},
        loomt::index{896}, layer, dense_group, panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{4}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(
        sequence30_storage, loomt::offset{0}, storage30, loomt::offset{10624},
        storage31, activations30, loomt::index{0}, loomt::index{1}, arrival30,
        arrival31, departure31, departure30, query_output30, output, state,
        weights30, loomt::offset{18816}, probability30, result30,
        loomt::index{3584}, loomt::index{7}, loomt::offset{376832},
        loomt::offset{1073600}, loomt::index{896}, loomt::offset{909312},
        loomt::offset{253952}, loomt::offset{500736}, loomt::offset{2088960},
        loomt::offset{2762496}, loomt::offset{1810432}, layer, dense_group,
        panel_index);
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{5}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(
        false, dimensions31_storage, loomt::offset{0}, storage31,
        loomt::offset{10624}, storage30, activations31, loomt::index{0},
        loomt::index{1}, arrival31, arrival27, departure27, departure31,
        query_output30, weights31, loomt::offset{18816}, probability30,
        result30, loomt::index{3584}, loomt::index{92}, loomt::index{7},
        loomt::index{896}, layer, dense_group, panel_index);
  }
}

#endif  // QWEN3_EMBEDDING_ARRAY_PHASES_H_
