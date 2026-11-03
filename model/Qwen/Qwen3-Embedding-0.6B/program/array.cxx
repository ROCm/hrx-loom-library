// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

// Eight-column by four-core-row Strix Halo composition. One resident program
// executes all 28 layers, reusing each projection group across two token
// panels. Root-owned SRAM and channels persist across joined dense/attention
// phases.

#include "array_phases.h"

[[loom::symbol("qwen3_embedding.array")]]
extern const loom::target::declaration array_target;

[[loom::pipeline("kernel"), loom::symbol("qwen3_embedding.cohort"),
  loom::retain, loom::target(array_target)]]
void qwen3_embedding_cohort([[loom::launch]] Bytes projection0_arg,
                            Bytes projection1_arg, Bytes projection2_arg,
                            Bytes projection3_arg, Bytes projection4_arg,
                            Bytes projection5_arg, Bytes projection6_arg,
                            Bytes projection7_arg, Bytes constants_arg,
                            Bytes embedding_arg, Bytes activations_arg,
                            Bytes output_arg, Bytes state_arg) {
  auto projection0_unique = loom::buffer::assume::noalias(projection0_arg);
  auto projection1_unique = loom::buffer::assume::noalias(projection1_arg);
  auto projection2_unique = loom::buffer::assume::noalias(projection2_arg);
  auto projection3_unique = loom::buffer::assume::noalias(projection3_arg);
  auto projection4_unique = loom::buffer::assume::noalias(projection4_arg);
  auto projection5_unique = loom::buffer::assume::noalias(projection5_arg);
  auto projection6_unique = loom::buffer::assume::noalias(projection6_arg);
  auto projection7_unique = loom::buffer::assume::noalias(projection7_arg);
  auto constants_unique = loom::buffer::assume::noalias(constants_arg);
  auto embedding_unique = loom::buffer::assume::noalias(embedding_arg);
  auto activations_unique = loom::buffer::assume::noalias(activations_arg);
  auto output_unique = loom::buffer::assume::noalias(output_arg);
  auto state_unique = loom::buffer::assume::noalias(state_arg);
  auto projection0 = loom::buffer::assume::alignment<64>(projection0_unique);
  auto projection1 = loom::buffer::assume::alignment<64>(projection1_unique);
  auto projection2 = loom::buffer::assume::alignment<64>(projection2_unique);
  auto projection3 = loom::buffer::assume::alignment<64>(projection3_unique);
  auto projection4 = loom::buffer::assume::alignment<64>(projection4_unique);
  auto projection5 = loom::buffer::assume::alignment<64>(projection5_unique);
  auto projection6 = loom::buffer::assume::alignment<64>(projection6_unique);
  auto projection7 = loom::buffer::assume::alignment<64>(projection7_unique);
  auto constants = loom::buffer::assume::alignment<64>(constants_unique);
  auto embedding = loom::buffer::assume::alignment<64>(embedding_unique);
  auto activations = loom::buffer::assume::alignment<64>(activations_unique);
  auto output = loom::buffer::assume::alignment<64>(output_unique);
  auto state = loom::buffer::assume::alignment<64>(state_unique);
  auto memory_tile0 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{0}, loomt::index{1}});
  auto *weight_ingress0_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{443904}, memory_tile0);
  auto memory_tile1 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{1}, loomt::index{1}});
  auto *weight_ingress1_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{443904}, memory_tile1);
  auto memory_tile2 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{2}, loomt::index{1}});
  auto *weight_ingress2_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{443904}, memory_tile2);
  auto memory_tile3 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{3}, loomt::index{1}});
  auto *weight_ingress3_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{443904}, memory_tile3);
  auto memory_tile4 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{4}, loomt::index{1}});
  auto *weight_ingress4_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{443904}, memory_tile4);
  auto memory_tile5 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{5}, loomt::index{1}});
  auto *weight_ingress5_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{443904}, memory_tile5);
  auto memory_tile6 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{6}, loomt::index{1}});
  auto *weight_ingress6_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{443904}, memory_tile6);
  auto memory_tile7 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{7}, loomt::index{1}});
  auto *weight_ingress7_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{443904}, memory_tile7);
  auto tile0 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{0}, loomt::index{2}});
  auto core0 = allocate_core(tile0);
  auto tile1 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{0}, loomt::index{3}});
  auto core1 = allocate_core(tile1);
  auto tile2 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{0}, loomt::index{4}});
  auto core2 = allocate_core(tile2);
  auto tile3 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{0}, loomt::index{5}});
  auto core3 = allocate_core(tile3);
  auto tile4 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{1}, loomt::index{2}});
  auto core4 = allocate_core(tile4);
  auto tile5 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{1}, loomt::index{3}});
  auto core5 = allocate_core(tile5);
  auto tile6 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{1}, loomt::index{4}});
  auto core6 = allocate_core(tile6);
  auto tile7 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{1}, loomt::index{5}});
  auto core7 = allocate_core(tile7);
  auto tile8 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{2}, loomt::index{2}});
  auto core8 = allocate_core(tile8);
  auto tile9 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{2}, loomt::index{3}});
  auto core9 = allocate_core(tile9);
  auto tile10 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{2}, loomt::index{4}});
  auto core10 = allocate_core(tile10);
  auto tile11 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{2}, loomt::index{5}});
  auto core11 = allocate_core(tile11);
  auto tile12 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{3}, loomt::index{2}});
  auto core12 = allocate_core(tile12);
  auto tile13 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{3}, loomt::index{3}});
  auto core13 = allocate_core(tile13);
  auto tile14 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{3}, loomt::index{4}});
  auto core14 = allocate_core(tile14);
  auto tile15 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{3}, loomt::index{5}});
  auto core15 = allocate_core(tile15);
  auto tile16 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{4}, loomt::index{2}});
  auto core16 = allocate_core(tile16);
  auto tile17 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{4}, loomt::index{3}});
  auto core17 = allocate_core(tile17);
  auto tile18 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{4}, loomt::index{4}});
  auto core18 = allocate_core(tile18);
  auto tile19 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{4}, loomt::index{5}});
  auto core19 = allocate_core(tile19);
  auto tile20 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{5}, loomt::index{2}});
  auto core20 = allocate_core(tile20);
  auto tile21 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{5}, loomt::index{3}});
  auto core21 = allocate_core(tile21);
  auto tile22 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{5}, loomt::index{4}});
  auto core22 = allocate_core(tile22);
  auto tile23 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{5}, loomt::index{5}});
  auto core23 = allocate_core(tile23);
  auto tile24 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{6}, loomt::index{2}});
  auto core24 = allocate_core(tile24);
  auto tile25 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{6}, loomt::index{3}});
  auto core25 = allocate_core(tile25);
  auto tile26 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{6}, loomt::index{4}});
  auto core26 = allocate_core(tile26);
  auto tile27 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{6}, loomt::index{5}});
  auto core27 = allocate_core(tile27);
  auto tile28 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{7}, loomt::index{2}});
  auto core28 = allocate_core(tile28);
  auto tile29 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{7}, loomt::index{3}});
  auto core29 = allocate_core(tile29);
  auto tile30 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{7}, loomt::index{4}});
  auto core30 = allocate_core(tile30);
  auto tile31 = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{7}, loomt::index{5}});
  auto core31 = allocate_core(tile31);
  auto streams0 = bind_core_streams(core0.storage);
  auto streams1 = bind_core_streams(core1.storage);
  auto streams2 = bind_core_streams(core2.storage);
  auto streams3 = bind_core_streams(core3.storage);
  auto streams4 = bind_core_streams(core4.storage);
  auto streams5 = bind_core_streams(core5.storage);
  auto streams6 = bind_core_streams(core6.storage);
  auto streams7 = bind_core_streams(core7.storage);
  auto streams8 = bind_core_streams(core8.storage);
  auto streams9 = bind_core_streams(core9.storage);
  auto streams10 = bind_core_streams(core10.storage);
  auto streams11 = bind_core_streams(core11.storage);
  auto streams12 = bind_core_streams(core12.storage);
  auto streams13 = bind_core_streams(core13.storage);
  auto streams14 = bind_core_streams(core14.storage);
  auto streams15 = bind_core_streams(core15.storage);
  auto streams16 = bind_core_streams(core16.storage);
  auto streams17 = bind_core_streams(core17.storage);
  auto streams18 = bind_core_streams(core18.storage);
  auto streams19 = bind_core_streams(core19.storage);
  auto streams20 = bind_core_streams(core20.storage);
  auto streams21 = bind_core_streams(core21.storage);
  auto streams22 = bind_core_streams(core22.storage);
  auto streams23 = bind_core_streams(core23.storage);
  auto streams24 = bind_core_streams(core24.storage);
  auto streams25 = bind_core_streams(core25.storage);
  auto streams26 = bind_core_streams(core26.storage);
  auto streams27 = bind_core_streams(core27.storage);
  auto streams28 = bind_core_streams(core28.storage);
  auto streams29 = bind_core_streams(core29.storage);
  auto streams30 = bind_core_streams(core30.storage);
  auto streams31 = bind_core_streams(core31.storage);
  auto barrier0 = make_barriers(tile0);
  auto barrier1 = make_barriers(tile1);
  auto barrier2 = make_barriers(tile2);
  auto barrier3 = make_barriers(tile3);
  auto barrier7 = make_barriers(tile7);
  auto barrier6 = make_barriers(tile6);
  auto barrier5 = make_barriers(tile5);
  auto barrier4 = make_barriers(tile4);
  auto barrier8 = make_barriers(tile8);
  auto barrier9 = make_barriers(tile9);
  auto barrier10 = make_barriers(tile10);
  auto barrier11 = make_barriers(tile11);
  auto barrier15 = make_barriers(tile15);
  auto barrier14 = make_barriers(tile14);
  auto barrier13 = make_barriers(tile13);
  auto barrier12 = make_barriers(tile12);
  auto barrier16 = make_barriers(tile16);
  auto barrier17 = make_barriers(tile17);
  auto barrier18 = make_barriers(tile18);
  auto barrier19 = make_barriers(tile19);
  auto barrier23 = make_barriers(tile23);
  auto barrier22 = make_barriers(tile22);
  auto barrier21 = make_barriers(tile21);
  auto barrier20 = make_barriers(tile20);
  auto barrier24 = make_barriers(tile24);
  auto barrier25 = make_barriers(tile25);
  auto barrier26 = make_barriers(tile26);
  auto barrier27 = make_barriers(tile27);
  auto barrier31 = make_barriers(tile31);
  auto barrier30 = make_barriers(tile30);
  auto barrier29 = make_barriers(tile29);
  auto result0 = make_feature_stream(core0.storage);
  auto result2 = make_feature_stream(core2.storage);
  auto result4 = make_feature_stream(core4.storage);
  auto result6 = make_feature_stream(core6.storage);
  auto result8 = make_feature_stream(core9.storage);
  auto result10 = make_feature_stream(core11.storage);
  auto result12 = make_feature_stream(core12.storage);
  auto result14 = make_feature_stream(core14.storage);
  auto result16 = make_feature_stream(core16.storage);
  auto result18 = make_feature_stream(core18.storage);
  auto result20 = make_feature_stream(core20.storage);
  auto result22 = make_feature_stream(core22.storage);
  auto result24 = make_feature_stream(core25.storage);
  auto result26 = make_feature_stream(core27.storage);
  auto result28 = make_feature_stream(core28.storage);
  auto result30 = make_feature_stream(core30.storage);
  auto probability0 = make_feature_stream(core0.storage);
  auto *sequence0_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile0);
  auto probability2 = make_feature_stream(core2.storage);
  auto *sequence2_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile2);
  auto probability4 = make_feature_stream(core4.storage);
  auto *sequence4_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile4);
  auto probability6 = make_feature_stream(core6.storage);
  auto *sequence6_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile6);
  auto probability8 = make_feature_stream(core8.storage);
  auto *sequence8_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile8);
  auto probability10 = make_feature_stream(core10.storage);
  auto *sequence10_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile10);
  auto probability12 = make_feature_stream(core12.storage);
  auto *sequence12_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile12);
  auto probability14 = make_feature_stream(core14.storage);
  auto *sequence14_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile14);
  auto probability16 = make_feature_stream(core16.storage);
  auto *sequence16_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile16);
  auto probability18 = make_feature_stream(core18.storage);
  auto *sequence18_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile18);
  auto probability20 = make_feature_stream(core20.storage);
  auto *sequence20_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile20);
  auto probability22 = make_feature_stream(core22.storage);
  auto *sequence22_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile22);
  auto probability24 = make_feature_stream(core24.storage);
  auto *sequence24_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile24);
  auto probability26 = make_feature_stream(core26.storage);
  auto *sequence26_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile26);
  auto probability28 = make_feature_stream(core28.storage);
  auto *sequence28_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile28);
  auto probability30 = make_feature_stream(core30.storage);
  auto *sequence30_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          loomt::offset{512}, tile30);
  auto query0 = make_query_output(core0.storage);
  auto query2 = make_query_output(core2.storage);
  auto query4 = make_query_output(core4.storage);
  auto query6 = make_query_output(core6.storage);
  auto query8 = make_query_output(core9.storage);
  auto query10 = make_query_output(core11.storage);
  auto query12 = make_query_output(core12.storage);
  auto query14 = make_query_output(core14.storage);
  auto query16 = make_query_output(core16.storage);
  auto query18 = make_query_output(core18.storage);
  auto query20 = make_query_output(core20.storage);
  auto query22 = make_query_output(core22.storage);
  auto query24 = make_query_output(core25.storage);
  auto query26 = make_query_output(core27.storage);
  auto query28 = make_query_output(core28.storage);
  auto query30 = make_query_output(core30.storage);
  auto normal_output8 = loom::channel::bind<loom::channel::discipline::fifo>(
      query8.slots, loomt::index{2});
  auto normal_output10 = loom::channel::bind<loom::channel::discipline::fifo>(
      query10.slots, loomt::index{2});
  auto normal_output24 = loom::channel::bind<loom::channel::discipline::fifo>(
      query24.slots, loomt::index{2});
  auto normal_output26 = loom::channel::bind<loom::channel::discipline::fifo>(
      query26.slots, loomt::index{2});
  auto *dimensions1_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile1);
  auto *dimensions3_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile3);
  auto *dimensions5_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile5);
  auto *dimensions7_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile7);
  auto *dimensions8_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile8);
  auto *dimensions10_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile10);
  auto *dimensions13_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile13);
  auto *dimensions15_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile15);
  auto *dimensions17_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile17);
  auto *dimensions19_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile19);
  auto *dimensions21_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile21);
  auto *dimensions23_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile23);
  auto *dimensions24_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile24);
  auto *dimensions26_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile26);
  auto *dimensions29_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile29);
  auto *dimensions31_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32},
                                                              tile31);
  auto *gather_cache0 = loom::buffer::alloca<loom::memory_space::workgroup, 16>(
      loomt::offset{192}, tile0);
  auto *gather_cache9 = loom::buffer::alloca<loom::memory_space::workgroup, 16>(
      loomt::offset{192}, tile9);
  auto *gather_cache18 =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(
          loomt::offset{192}, tile18);
  auto *gather_cache27 =
      loom::buffer::alloca<loom::memory_space::workgroup, 16>(
          loomt::offset{192}, tile27);
  auto *phase_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(
      loomt::offset{64}, tile1);
  auto phase_slots = loom::buffer::view<1, 16>(
      reinterpret_cast<unsigned *>(
          reinterpret_cast<unsigned char *>(phase_storage)),
      {});
  auto phase_complete = loom::channel::bind<loom::channel::discipline::fifo>(
      phase_slots, loomt::index{1});
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record = 0; record < 6852; ++record) {
      auto acquisition = loom::channel::acquire(streams0.activations);
      auto fanout = loom::channel::fanout<7>(acquisition.read);
      auto reservation = loom::channel::reserve(streams4.activations);
      loom::channel::copy(fanout.first, reservation.write);
      auto reservation_2 = loom::channel::reserve(streams8.activations);
      loom::channel::copy(fanout.remaining.first, reservation_2.write);
      auto reservation_3 = loom::channel::reserve(streams12.activations);
      loom::channel::copy(fanout.remaining.remaining.first,
                          reservation_3.write);
      auto reservation_4 = loom::channel::reserve(streams16.activations);
      loom::channel::copy(fanout.remaining.remaining.remaining.first,
                          reservation_4.write);
      auto reservation_5 = loom::channel::reserve(streams20.activations);
      loom::channel::copy(fanout.remaining.remaining.remaining.remaining.first,
                          reservation_5.write);
      auto reservation_6 = loom::channel::reserve(streams24.activations);
      loom::channel::copy(
          fanout.remaining.remaining.remaining.remaining.remaining.first,
          reservation_6.write);
      auto reservation_7 = loom::channel::reserve(streams28.activations);
      loom::channel::copy(
          fanout.remaining.remaining.remaining.remaining.remaining.second,
          reservation_7.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{3}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_1 = 0; record_1 < 6852; ++record_1) {
      auto acquisition_2 = loom::channel::acquire(streams9.activations);
      auto fanout_2 = loom::channel::fanout<7>(acquisition_2.read);
      auto reservation_8 = loom::channel::reserve(streams1.activations);
      loom::channel::copy(fanout_2.first, reservation_8.write);
      auto reservation_9 = loom::channel::reserve(streams5.activations);
      loom::channel::copy(fanout_2.remaining.first, reservation_9.write);
      auto reservation_10 = loom::channel::reserve(streams13.activations);
      loom::channel::copy(fanout_2.remaining.remaining.first,
                          reservation_10.write);
      auto reservation_11 = loom::channel::reserve(streams17.activations);
      loom::channel::copy(fanout_2.remaining.remaining.remaining.first,
                          reservation_11.write);
      auto reservation_12 = loom::channel::reserve(streams21.activations);
      loom::channel::copy(
          fanout_2.remaining.remaining.remaining.remaining.first,
          reservation_12.write);
      auto reservation_13 = loom::channel::reserve(streams25.activations);
      loom::channel::copy(
          fanout_2.remaining.remaining.remaining.remaining.remaining.first,
          reservation_13.write);
      auto reservation_14 = loom::channel::reserve(streams29.activations);
      loom::channel::copy(
          fanout_2.remaining.remaining.remaining.remaining.remaining.second,
          reservation_14.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{4}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_2 = 0; record_2 < 6852; ++record_2) {
      auto acquisition_3 = loom::channel::acquire(streams18.activations);
      auto fanout_3 = loom::channel::fanout<7>(acquisition_3.read);
      auto reservation_15 = loom::channel::reserve(streams2.activations);
      loom::channel::copy(fanout_3.first, reservation_15.write);
      auto reservation_16 = loom::channel::reserve(streams6.activations);
      loom::channel::copy(fanout_3.remaining.first, reservation_16.write);
      auto reservation_17 = loom::channel::reserve(streams10.activations);
      loom::channel::copy(fanout_3.remaining.remaining.first,
                          reservation_17.write);
      auto reservation_18 = loom::channel::reserve(streams14.activations);
      loom::channel::copy(fanout_3.remaining.remaining.remaining.first,
                          reservation_18.write);
      auto reservation_19 = loom::channel::reserve(streams22.activations);
      loom::channel::copy(
          fanout_3.remaining.remaining.remaining.remaining.first,
          reservation_19.write);
      auto reservation_20 = loom::channel::reserve(streams26.activations);
      loom::channel::copy(
          fanout_3.remaining.remaining.remaining.remaining.remaining.first,
          reservation_20.write);
      auto reservation_21 = loom::channel::reserve(streams30.activations);
      loom::channel::copy(
          fanout_3.remaining.remaining.remaining.remaining.remaining.second,
          reservation_21.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{5}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_3 = 0; record_3 < 6852; ++record_3) {
      auto acquisition_4 = loom::channel::acquire(streams27.activations);
      auto fanout_4 = loom::channel::fanout<7>(acquisition_4.read);
      auto reservation_22 = loom::channel::reserve(streams3.activations);
      loom::channel::copy(fanout_4.first, reservation_22.write);
      auto reservation_23 = loom::channel::reserve(streams7.activations);
      loom::channel::copy(fanout_4.remaining.first, reservation_23.write);
      auto reservation_24 = loom::channel::reserve(streams11.activations);
      loom::channel::copy(fanout_4.remaining.remaining.first,
                          reservation_24.write);
      auto reservation_25 = loom::channel::reserve(streams15.activations);
      loom::channel::copy(fanout_4.remaining.remaining.remaining.first,
                          reservation_25.write);
      auto reservation_26 = loom::channel::reserve(streams19.activations);
      loom::channel::copy(
          fanout_4.remaining.remaining.remaining.remaining.first,
          reservation_26.write);
      auto reservation_27 = loom::channel::reserve(streams23.activations);
      loom::channel::copy(
          fanout_4.remaining.remaining.remaining.remaining.remaining.first,
          reservation_27.write);
      auto reservation_28 = loom::channel::reserve(streams31.activations);
      loom::channel::copy(
          fanout_4.remaining.remaining.remaining.remaining.remaining.second,
          reservation_28.write);
    }
  }
  [[loom::serial]] {
    for (int schedule_position = 0; schedule_position < 225;
         ++schedule_position) {
      loomt::index layer =
          loom::index::cast(schedule_position) / loomt::index{8};
      loomt::index dense_group =
          loom::index::cast(schedule_position) % loomt::index{8};
      bool dense = loom::index::ult(dense_group, loomt::index{7});
      if (dense) {
        loom::pipeline::compose<qwen3_embedding_dense_group>(
            loom::buffer::root(sequence0_storage),
            loom::buffer::root(core0.storage),
            loom::buffer::root(core1.storage),
            loom::buffer::root(gather_cache0), activations,
            streams0.activations, constants, embedding, state, barrier0.arrival,
            barrier0.departure, query0.stream, output, streams0.weights,
            probability0, result0, phase_complete,
            loom::buffer::root(dimensions1_storage), streams1.activations,
            barrier1.arrival, barrier1.departure, streams1.weights,
            loom::buffer::root(sequence2_storage),
            loom::buffer::root(core2.storage),
            loom::buffer::root(core3.storage), streams2.activations,
            barrier2.arrival, barrier2.departure, query2.stream,
            streams2.weights, probability2, result2,
            loom::buffer::root(dimensions3_storage), streams3.activations,
            barrier3.arrival, barrier3.departure, streams3.weights,
            loom::buffer::root(sequence4_storage),
            loom::buffer::root(core4.storage),
            loom::buffer::root(core5.storage), streams4.activations,
            barrier4.arrival, barrier5.arrival, barrier5.departure,
            barrier4.departure, query4.stream, streams4.weights, probability4,
            result4, loom::buffer::root(dimensions5_storage),
            streams5.activations, barrier6.arrival, barrier6.departure,
            streams5.weights, loom::buffer::root(sequence6_storage),
            loom::buffer::root(core6.storage),
            loom::buffer::root(core7.storage), streams6.activations,
            barrier7.arrival, barrier7.departure, query6.stream,
            streams6.weights, probability6, result6,
            loom::buffer::root(dimensions7_storage), streams7.activations,
            streams7.weights, loom::buffer::root(sequence8_storage),
            loom::buffer::root(dimensions8_storage),
            loom::buffer::root(core8.storage),
            loom::buffer::root(core9.storage), streams8.activations,
            barrier8.arrival, barrier8.departure, normal_output8,
            streams8.weights, query8.stream, probability8,
            loom::buffer::root(gather_cache9), streams9.activations,
            barrier9.arrival, barrier9.departure, streams9.weights, result8,
            loom::buffer::root(sequence10_storage),
            loom::buffer::root(dimensions10_storage),
            loom::buffer::root(core10.storage),
            loom::buffer::root(core11.storage), streams10.activations,
            barrier10.arrival, barrier10.departure, normal_output10,
            streams10.weights, probability10, streams11.activations,
            barrier11.arrival, barrier11.departure, streams11.weights,
            query10.stream, result10, loom::buffer::root(sequence12_storage),
            loom::buffer::root(core12.storage),
            loom::buffer::root(core13.storage), streams12.activations,
            barrier12.arrival, barrier13.arrival, barrier13.departure,
            barrier12.departure, query12.stream, streams12.weights,
            probability12, result12, loom::buffer::root(dimensions13_storage),
            streams13.activations, barrier14.arrival, barrier14.departure,
            streams13.weights, loom::buffer::root(sequence14_storage),
            loom::buffer::root(core14.storage),
            loom::buffer::root(core15.storage), streams14.activations,
            barrier15.arrival, barrier15.departure, query14.stream,
            streams14.weights, probability14, result14,
            loom::buffer::root(dimensions15_storage), streams15.activations,
            streams15.weights, loom::buffer::root(sequence16_storage),
            loom::buffer::root(core16.storage),
            loom::buffer::root(core17.storage), streams16.activations,
            barrier16.arrival, barrier16.departure, query16.stream,
            streams16.weights, probability16, result16,
            loom::buffer::root(dimensions17_storage), streams17.activations,
            barrier17.arrival, barrier17.departure, streams17.weights,
            loom::buffer::root(sequence18_storage),
            loom::buffer::root(core18.storage),
            loom::buffer::root(core19.storage),
            loom::buffer::root(gather_cache18), streams18.activations,
            barrier18.arrival, barrier18.departure, query18.stream,
            streams18.weights, probability18, result18,
            loom::buffer::root(dimensions19_storage), streams19.activations,
            barrier19.arrival, barrier19.departure, streams19.weights,
            loom::buffer::root(sequence20_storage),
            loom::buffer::root(core20.storage),
            loom::buffer::root(core21.storage), streams20.activations,
            barrier20.arrival, barrier21.arrival, barrier21.departure,
            barrier20.departure, query20.stream, streams20.weights,
            probability20, result20, loom::buffer::root(dimensions21_storage),
            streams21.activations, barrier22.arrival, barrier22.departure,
            streams21.weights, loom::buffer::root(sequence22_storage),
            loom::buffer::root(core22.storage),
            loom::buffer::root(core23.storage), streams22.activations,
            barrier23.arrival, barrier23.departure, query22.stream,
            streams22.weights, probability22, result22,
            loom::buffer::root(dimensions23_storage), streams23.activations,
            streams23.weights, loom::buffer::root(sequence24_storage),
            loom::buffer::root(dimensions24_storage),
            loom::buffer::root(core24.storage),
            loom::buffer::root(core25.storage), streams24.activations,
            barrier24.arrival, barrier24.departure, normal_output24,
            streams24.weights, probability24, streams25.activations,
            barrier25.arrival, barrier25.departure, streams25.weights,
            query24.stream, result24, loom::buffer::root(sequence26_storage),
            loom::buffer::root(dimensions26_storage),
            loom::buffer::root(core26.storage),
            loom::buffer::root(core27.storage), streams26.activations,
            barrier26.arrival, barrier26.departure, normal_output26,
            streams26.weights, query26.stream, probability26,
            loom::buffer::root(gather_cache27), streams27.activations,
            barrier27.arrival, barrier27.departure, streams27.weights, result26,
            loom::buffer::root(sequence28_storage),
            loom::buffer::root(core28.storage),
            loom::buffer::root(core29.storage), streams28.activations,
            barrier29.arrival, barrier29.departure, query28.stream,
            streams28.weights, probability28, result28,
            loom::buffer::root(dimensions29_storage), streams29.activations,
            barrier30.arrival, barrier30.departure, streams29.weights,
            loom::buffer::root(sequence30_storage),
            loom::buffer::root(core30.storage),
            loom::buffer::root(core31.storage), streams30.activations,
            barrier31.arrival, barrier31.departure, query30.stream,
            streams30.weights, probability30, result30,
            loom::buffer::root(dimensions31_storage), streams31.activations,
            streams31.weights, projection0,
            loom::buffer::root(weight_ingress0_storage), projection1,
            loom::buffer::root(weight_ingress1_storage), projection2,
            loom::buffer::root(weight_ingress2_storage), projection3,
            loom::buffer::root(weight_ingress3_storage), projection4,
            loom::buffer::root(weight_ingress4_storage), projection5,
            loom::buffer::root(weight_ingress5_storage), projection6,
            loom::buffer::root(weight_ingress6_storage), projection7,
            loom::buffer::root(weight_ingress7_storage),
            loom::buffer::root(weight_ingress0_storage),
            loom::buffer::root(weight_ingress1_storage),
            loom::buffer::root(weight_ingress2_storage),
            loom::buffer::root(weight_ingress3_storage),
            loom::buffer::root(weight_ingress4_storage),
            loom::buffer::root(weight_ingress5_storage),
            loom::buffer::root(weight_ingress6_storage),
            loom::buffer::root(weight_ingress7_storage), layer, dense_group);
      }
      bool attention = dense_group == loomt::index{1};
      if (attention) {
        loom::pipeline::compose<qwen3_embedding_attention_phase>(
            loom::buffer::root(sequence0_storage),
            loom::buffer::root(core0.storage),
            loom::buffer::root(core1.storage),
            loom::buffer::root(gather_cache0), activations,
            streams0.activations, constants, embedding, state, barrier0.arrival,
            barrier0.departure, query0.stream, output, streams0.weights,
            probability0, result0, phase_complete,
            loom::buffer::root(dimensions1_storage), streams1.activations,
            barrier1.arrival, barrier1.departure, streams1.weights,
            loom::buffer::root(sequence2_storage),
            loom::buffer::root(core2.storage),
            loom::buffer::root(core3.storage), streams2.activations,
            barrier2.arrival, barrier2.departure, query2.stream,
            streams2.weights, probability2, result2,
            loom::buffer::root(dimensions3_storage), streams3.activations,
            barrier3.arrival, barrier3.departure, streams3.weights,
            loom::buffer::root(sequence4_storage),
            loom::buffer::root(core4.storage),
            loom::buffer::root(core5.storage), streams4.activations,
            barrier4.arrival, barrier5.arrival, barrier5.departure,
            barrier4.departure, query4.stream, streams4.weights, probability4,
            result4, loom::buffer::root(dimensions5_storage),
            streams5.activations, barrier6.arrival, barrier6.departure,
            streams5.weights, loom::buffer::root(sequence6_storage),
            loom::buffer::root(core6.storage),
            loom::buffer::root(core7.storage), streams6.activations,
            barrier7.arrival, barrier7.departure, query6.stream,
            streams6.weights, probability6, result6,
            loom::buffer::root(dimensions7_storage), streams7.activations,
            streams7.weights, loom::buffer::root(sequence8_storage),
            loom::buffer::root(dimensions8_storage),
            loom::buffer::root(core8.storage),
            loom::buffer::root(core9.storage), streams8.activations,
            barrier8.arrival, barrier8.departure, normal_output8,
            streams8.weights, query8.stream, probability8,
            loom::buffer::root(gather_cache9), streams9.activations,
            barrier9.arrival, barrier9.departure, streams9.weights, result8,
            loom::buffer::root(sequence10_storage),
            loom::buffer::root(dimensions10_storage),
            loom::buffer::root(core10.storage),
            loom::buffer::root(core11.storage), streams10.activations,
            barrier10.arrival, barrier10.departure, normal_output10,
            streams10.weights, probability10, streams11.activations,
            barrier11.arrival, barrier11.departure, streams11.weights,
            query10.stream, result10, loom::buffer::root(sequence12_storage),
            loom::buffer::root(core12.storage),
            loom::buffer::root(core13.storage), streams12.activations,
            barrier12.arrival, barrier13.arrival, barrier13.departure,
            barrier12.departure, query12.stream, streams12.weights,
            probability12, result12, loom::buffer::root(dimensions13_storage),
            streams13.activations, barrier14.arrival, barrier14.departure,
            streams13.weights, loom::buffer::root(sequence14_storage),
            loom::buffer::root(core14.storage),
            loom::buffer::root(core15.storage), streams14.activations,
            barrier15.arrival, barrier15.departure, query14.stream,
            streams14.weights, probability14, result14,
            loom::buffer::root(dimensions15_storage), streams15.activations,
            streams15.weights, loom::buffer::root(sequence16_storage),
            loom::buffer::root(core16.storage),
            loom::buffer::root(core17.storage), streams16.activations,
            barrier16.arrival, barrier16.departure, query16.stream,
            streams16.weights, probability16, result16,
            loom::buffer::root(dimensions17_storage), streams17.activations,
            barrier17.arrival, barrier17.departure, streams17.weights,
            loom::buffer::root(sequence18_storage),
            loom::buffer::root(core18.storage),
            loom::buffer::root(core19.storage),
            loom::buffer::root(gather_cache18), streams18.activations,
            barrier18.arrival, barrier18.departure, query18.stream,
            streams18.weights, probability18, result18,
            loom::buffer::root(dimensions19_storage), streams19.activations,
            barrier19.arrival, barrier19.departure, streams19.weights,
            loom::buffer::root(sequence20_storage),
            loom::buffer::root(core20.storage),
            loom::buffer::root(core21.storage), streams20.activations,
            barrier20.arrival, barrier21.arrival, barrier21.departure,
            barrier20.departure, query20.stream, streams20.weights,
            probability20, result20, loom::buffer::root(dimensions21_storage),
            streams21.activations, barrier22.arrival, barrier22.departure,
            streams21.weights, loom::buffer::root(sequence22_storage),
            loom::buffer::root(core22.storage),
            loom::buffer::root(core23.storage), streams22.activations,
            barrier23.arrival, barrier23.departure, query22.stream,
            streams22.weights, probability22, result22,
            loom::buffer::root(dimensions23_storage), streams23.activations,
            streams23.weights, loom::buffer::root(sequence24_storage),
            loom::buffer::root(dimensions24_storage),
            loom::buffer::root(core24.storage),
            loom::buffer::root(core25.storage), streams24.activations,
            barrier24.arrival, barrier24.departure, normal_output24,
            streams24.weights, probability24, streams25.activations,
            barrier25.arrival, barrier25.departure, streams25.weights,
            query24.stream, result24, loom::buffer::root(sequence26_storage),
            loom::buffer::root(dimensions26_storage),
            loom::buffer::root(core26.storage),
            loom::buffer::root(core27.storage), streams26.activations,
            barrier26.arrival, barrier26.departure, normal_output26,
            streams26.weights, query26.stream, probability26,
            loom::buffer::root(gather_cache27), streams27.activations,
            barrier27.arrival, barrier27.departure, streams27.weights, result26,
            loom::buffer::root(sequence28_storage),
            loom::buffer::root(core28.storage),
            loom::buffer::root(core29.storage), streams28.activations,
            barrier29.arrival, barrier29.departure, query28.stream,
            streams28.weights, probability28, result28,
            loom::buffer::root(dimensions29_storage), streams29.activations,
            barrier30.arrival, barrier30.departure, streams29.weights,
            loom::buffer::root(sequence30_storage),
            loom::buffer::root(core30.storage),
            loom::buffer::root(core31.storage), streams30.activations,
            barrier31.arrival, barrier31.departure, query30.stream,
            streams30.weights, probability30, result30,
            loom::buffer::root(dimensions31_storage), streams31.activations,
            streams31.weights, projection0,
            loom::buffer::root(weight_ingress0_storage), projection1,
            loom::buffer::root(weight_ingress1_storage), projection2,
            loom::buffer::root(weight_ingress2_storage), projection3,
            loom::buffer::root(weight_ingress3_storage), projection4,
            loom::buffer::root(weight_ingress4_storage), projection5,
            loom::buffer::root(weight_ingress5_storage), projection6,
            loom::buffer::root(weight_ingress6_storage), projection7,
            loom::buffer::root(weight_ingress7_storage),
            loom::buffer::root(weight_ingress0_storage),
            loom::buffer::root(weight_ingress1_storage),
            loom::buffer::root(weight_ingress2_storage),
            loom::buffer::root(weight_ingress3_storage),
            loom::buffer::root(weight_ingress4_storage),
            loom::buffer::root(weight_ingress5_storage),
            loom::buffer::root(weight_ingress6_storage),
            loom::buffer::root(weight_ingress7_storage), layer);
      }
    }
  }
}
