// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

// Fixed 8-column by 4-core-row Strix Halo composition for one 128-token
// cohort. The pipeline owns concrete placement, local memory banks, FIFO
// storage, stream fanout, and barrier endpoints. Workers remain independently
// linkable providers and receive every owned resource explicitly.

#include <loomcxx/kernel.h>
#include <loomcxx/pipeline.h>
#include <loomcxx/target.h>

#include <stdfloat>

namespace loomt = loom::type;

using Bytes = loomt::buffer<unsigned char>;
using ActivationStream = loomt::channel<loomt::tile<loomt::shape<2048>, unsigned>>;
using BarrierStream = loomt::channel<loomt::tile<loomt::shape<16>, unsigned>>;
using FeatureStream = loomt::channel<loomt::tile<loomt::shape<4096>, std::bfloat16_t>>;
using QueryOutputSlots = loomt::view<loomt::shape<2, 4096>, std::bfloat16_t, loomt::encoding<loom::encoding::role::layout, 2>>;
using WeightStream = loomt::channel<loomt::tile<loomt::shape<4624>, unsigned>>;

struct CoreStorage {
  loomt::pool memory;
  unsigned char* storage;
};

struct CoreStreams {
  WeightStream weights;
  ActivationStream activations;
};

struct BarrierResources {
  BarrierStream arrival;
  BarrierStream departure;
};

struct QueryOutputResources {
  QueryOutputSlots slots;
  FeatureStream stream;
};

[[loom::force_inline]] static WeightStream make_weight_ingress(loomt::pool memory) {
  auto* storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{18496}, memory);
  auto slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(storage), {});
  return loom::channel::bind<loom::channel::discipline::fifo>(slots, loomt::index{1});
}

[[loom::force_inline]] static CoreStorage allocate_core(loomt::pool memory) {
  auto* storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, memory);
  return {.memory = memory, .storage = storage};
}

[[loom::force_inline]] static CoreStreams bind_core_streams(unsigned char* storage) {
  auto weight_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(storage), {});
  auto weights = loom::channel::bind<loom::channel::discipline::fifo>(weight_slots, loomt::index{1});
  auto activation_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(storage + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations = loom::channel::bind<loom::channel::discipline::fifo>(activation_slots, loomt::index{1});
  return {.weights = weights, .activations = activations};
}

[[loom::force_inline]] static BarrierResources make_barriers(loomt::pool memory) {
  auto* arrival_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, memory);
  auto arrival_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(arrival_storage), {});
  auto arrival = loom::channel::bind<loom::channel::discipline::fifo>(arrival_slots, loomt::index{1});
  auto* departure_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, memory);
  auto departure_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(departure_storage), {});
  auto departure = loom::channel::bind<loom::channel::discipline::fifo>(departure_slots, loomt::index{1});
  return {.arrival = arrival, .departure = departure};
}

[[loom::force_inline]] static FeatureStream make_feature_stream(unsigned char* storage) {
  auto slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(storage + static_cast<unsigned long long>(loomt::offset{10624})), {});
  return loom::channel::bind<loom::channel::discipline::fifo>(slots, loomt::index{1});
}

[[loom::force_inline]] static QueryOutputResources make_query_output(unsigned char* storage) {
  auto slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(storage + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto stream = loom::channel::bind<loom::channel::discipline::fifo>(slots, loomt::index{2});
  return {.slots = slots, .stream = stream};
}

[[loom::symbol("qwen3_embedding.array")]]
extern const loom::target::declaration array_target;

[[loom::symbol("qwen3_embedding.query_key_ingress_root"), loom::force_inline]]
void qwen3_embedding_query_key_ingress_root(Bytes sequence_storage, loomt::offset origin, Bytes local_storage, loomt::offset result_offset, Bytes peer_storage, Bytes gather_storage, Bytes activations, ActivationStream activation_stream, loomt::index zero, loomt::index one, Bytes constants, Bytes embedding, Bytes state, BarrierStream incoming_arrival, BarrierStream outgoing_departure, FeatureStream projection_stream, Bytes output, WeightStream weight_stream, loomt::offset activation_offset, FeatureStream probability_stream, FeatureStream result_stream);

[[loom::symbol("qwen3_embedding.query_key_egress"), loom::force_inline]]
void qwen3_embedding_query_key_egress(Bytes sequence_storage, loomt::offset origin, Bytes local_storage, loomt::offset result_offset, Bytes peer_storage, ActivationStream activation_stream, loomt::index zero, loomt::index one, BarrierStream incoming_arrival, BarrierStream outgoing_arrival, BarrierStream incoming_departure, BarrierStream outgoing_departure, FeatureStream projection_stream, Bytes output, Bytes state, WeightStream weight_stream, loomt::offset activation_offset, FeatureStream probability_stream, FeatureStream result_stream, loomt::index column_offset, loomt::index row_column, loomt::offset finish_initial, loomt::offset normalized_initial, loomt::index skip_column, loomt::offset query_initial, loomt::offset key_initial, loomt::offset value_initial, loomt::offset mlp_initial, loomt::offset residual_initial, loomt::offset output_initial);

[[loom::symbol("qwen3_embedding.probability_value"), loom::force_inline]]
void qwen3_embedding_probability_value(bool normalize_peer_rows, Bytes dimension_storage, loomt::offset origin, Bytes local_storage, loomt::offset result_offset, Bytes peer_storage, ActivationStream activation_stream, loomt::index zero, loomt::index one, BarrierStream incoming_arrival, BarrierStream outgoing_arrival, BarrierStream incoming_departure, BarrierStream outgoing_departure, FeatureStream projection_stream, WeightStream weight_stream, loomt::offset activation_offset, FeatureStream probability_stream, FeatureStream result_stream, loomt::index column_offset, loomt::index dimension_row_base, loomt::index row_column, loomt::index skip_column);

[[loom::symbol("qwen3_embedding.query_key_epilogue"), loom::force_inline]]
void qwen3_embedding_query_key_epilogue(Bytes sequence_storage, loomt::offset origin, Bytes dimension_storage, Bytes local_storage, loomt::offset result_offset, Bytes peer_storage, ActivationStream activation_stream, loomt::index zero, loomt::index one, BarrierStream incoming_arrival, BarrierStream outgoing_arrival, BarrierStream incoming_departure, BarrierStream outgoing_departure, FeatureStream normalization_stream, WeightStream weight_stream, loomt::offset activation_offset, FeatureStream projection_stream, FeatureStream probability_stream, loomt::index column_offset, loomt::index dimension_row_base, loomt::index row_column, loomt::index skip_column);

[[loom::symbol("qwen3_embedding.probability_value_ingress"), loom::force_inline]]
void qwen3_embedding_probability_value_ingress(Bytes local_storage, loomt::offset origin, loomt::offset result_offset, Bytes peer_storage, Bytes gather_storage, Bytes activations, ActivationStream activation_stream, loomt::index zero, loomt::index one, Bytes constants, Bytes embedding, Bytes state, BarrierStream incoming_arrival, BarrierStream outgoing_arrival, BarrierStream incoming_departure, BarrierStream outgoing_departure, FeatureStream normalization_stream, Bytes output, WeightStream weight_stream, loomt::offset activation_offset, FeatureStream projection_stream, FeatureStream probability_stream, Bytes sequence_storage, FeatureStream result_stream, loomt::index column_offset, loomt::index gather_row_base, loomt::index row_column, loomt::offset row_input_initial, loomt::offset finish_initial, loomt::offset normalized_initial, loomt::offset dense_normalized_initial, loomt::offset dense_context_initial, loomt::offset dense_mlp_initial, loomt::offset skip_initial, loomt::offset rotary_address, loomt::index skip_column, loomt::offset query_initial, loomt::offset key_initial, loomt::offset value_initial, loomt::offset mlp_initial, loomt::offset residual_initial, loomt::offset output_initial);

[[loom::symbol("qwen3_embedding.query_key_normalize"), loom::force_inline]]
void qwen3_embedding_query_key_normalize(Bytes sequence_storage, loomt::offset origin, Bytes dimension_storage, Bytes local_storage, loomt::offset result_offset, Bytes peer_storage, ActivationStream activation_stream, loomt::index zero, loomt::index one, BarrierStream incoming_arrival, BarrierStream outgoing_arrival, BarrierStream incoming_departure, BarrierStream outgoing_departure, FeatureStream normalization_stream, WeightStream weight_stream, loomt::offset activation_offset, FeatureStream probability_stream, loomt::index column_offset, loomt::index dimension_row_base, loomt::index row_column, loomt::index skip_column);

[[loom::symbol("qwen3_embedding.probability_value_egress"), loom::force_inline]]
void qwen3_embedding_probability_value_egress(Bytes local_storage, loomt::offset origin, loomt::offset result_offset, Bytes peer_storage, ActivationStream activation_stream, loomt::index zero, loomt::index one, BarrierStream incoming_arrival, BarrierStream outgoing_arrival, BarrierStream incoming_departure, BarrierStream outgoing_departure, FeatureStream normalization_stream, Bytes output, Bytes state, WeightStream weight_stream, loomt::offset activation_offset, FeatureStream projection_stream, FeatureStream probability_stream, FeatureStream result_stream, loomt::index column_offset, loomt::index row_column, loomt::offset finish_initial, loomt::offset normalized_initial, loomt::index skip_column, loomt::offset query_initial, loomt::offset key_initial, loomt::offset value_initial, loomt::offset mlp_initial, loomt::offset residual_initial, loomt::offset output_initial);

[[loom::symbol("qwen3_embedding.query_key_ingress"), loom::force_inline]]
void qwen3_embedding_query_key_ingress(Bytes sequence_storage, loomt::offset origin, Bytes local_storage, loomt::offset result_offset, Bytes peer_storage, Bytes gather_storage, Bytes activations, ActivationStream activation_stream, loomt::index zero, loomt::index one, Bytes constants, Bytes embedding, Bytes state, BarrierStream incoming_arrival, BarrierStream outgoing_arrival, BarrierStream incoming_departure, BarrierStream outgoing_departure, FeatureStream projection_stream, Bytes output, WeightStream weight_stream, loomt::offset activation_offset, FeatureStream probability_stream, FeatureStream result_stream);

[[loom::symbol("qwen3_embedding.query_key_egress_terminal"), loom::force_inline]]
void qwen3_embedding_query_key_egress_terminal(Bytes sequence_storage, loomt::offset origin, Bytes local_storage, loomt::offset result_offset, Bytes peer_storage, ActivationStream activation_stream, loomt::index zero, loomt::index one, BarrierStream outgoing_arrival, BarrierStream incoming_departure, FeatureStream projection_stream, Bytes output, Bytes state, WeightStream weight_stream, loomt::offset activation_offset, FeatureStream probability_stream, FeatureStream result_stream);

[[loom::pipeline("kernel"), loom::symbol("qwen3_embedding.cohort"), loom::retain, loom::target(array_target)]]
void qwen3_embedding_cohort([[loom::launch]] Bytes projection0_arg, Bytes projection1_arg, Bytes projection2_arg, Bytes projection3_arg, Bytes projection4_arg, Bytes projection5_arg, Bytes projection6_arg, Bytes projection7_arg, Bytes constants_arg, Bytes embedding_arg, Bytes activations_arg, Bytes output_arg, Bytes state_arg) {
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
  auto weight_ingress0_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{0}, loomt::index{1}});
  auto weight_ingress0 = make_weight_ingress(weight_ingress0_pool);
  auto weight_ingress1_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{1}, loomt::index{1}});
  auto weight_ingress1 = make_weight_ingress(weight_ingress1_pool);
  auto weight_ingress2_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{2}, loomt::index{1}});
  auto weight_ingress2 = make_weight_ingress(weight_ingress2_pool);
  auto weight_ingress3_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{3}, loomt::index{1}});
  auto weight_ingress3 = make_weight_ingress(weight_ingress3_pool);
  auto weight_ingress4_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{4}, loomt::index{1}});
  auto weight_ingress4 = make_weight_ingress(weight_ingress4_pool);
  auto weight_ingress5_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{5}, loomt::index{1}});
  auto weight_ingress5 = make_weight_ingress(weight_ingress5_pool);
  auto weight_ingress6_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{6}, loomt::index{1}});
  auto weight_ingress6 = make_weight_ingress(weight_ingress6_pool);
  auto weight_ingress7_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{7}, loomt::index{1}});
  auto weight_ingress7 = make_weight_ingress(weight_ingress7_pool);
  auto core0_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{0}, loomt::index{2}});
  auto core0 = allocate_core(core0_pool);
  auto core1_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{0}, loomt::index{3}});
  auto core1 = allocate_core(core1_pool);
  auto core2_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{0}, loomt::index{4}});
  auto core2 = allocate_core(core2_pool);
  auto core3_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{0}, loomt::index{5}});
  auto core3 = allocate_core(core3_pool);
  auto core4_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{1}, loomt::index{2}});
  auto core4 = allocate_core(core4_pool);
  auto core5_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{1}, loomt::index{3}});
  auto core5 = allocate_core(core5_pool);
  auto core6_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{1}, loomt::index{4}});
  auto core6 = allocate_core(core6_pool);
  auto core7_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{1}, loomt::index{5}});
  auto core7 = allocate_core(core7_pool);
  auto core8_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{2}, loomt::index{2}});
  auto core8 = allocate_core(core8_pool);
  auto core9_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{2}, loomt::index{3}});
  auto core9 = allocate_core(core9_pool);
  auto core10_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{2}, loomt::index{4}});
  auto core10 = allocate_core(core10_pool);
  auto core11_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{2}, loomt::index{5}});
  auto core11 = allocate_core(core11_pool);
  auto core12_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{3}, loomt::index{2}});
  auto core12 = allocate_core(core12_pool);
  auto core13_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{3}, loomt::index{3}});
  auto core13 = allocate_core(core13_pool);
  auto core14_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{3}, loomt::index{4}});
  auto core14 = allocate_core(core14_pool);
  auto core15_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{3}, loomt::index{5}});
  auto core15 = allocate_core(core15_pool);
  auto core16_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{4}, loomt::index{2}});
  auto core16 = allocate_core(core16_pool);
  auto core17_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{4}, loomt::index{3}});
  auto core17 = allocate_core(core17_pool);
  auto core18_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{4}, loomt::index{4}});
  auto core18 = allocate_core(core18_pool);
  auto core19_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{4}, loomt::index{5}});
  auto core19 = allocate_core(core19_pool);
  auto core20_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{5}, loomt::index{2}});
  auto core20 = allocate_core(core20_pool);
  auto core21_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{5}, loomt::index{3}});
  auto core21 = allocate_core(core21_pool);
  auto core22_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{5}, loomt::index{4}});
  auto core22 = allocate_core(core22_pool);
  auto core23_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{5}, loomt::index{5}});
  auto core23 = allocate_core(core23_pool);
  auto core24_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{6}, loomt::index{2}});
  auto core24 = allocate_core(core24_pool);
  auto core25_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{6}, loomt::index{3}});
  auto core25 = allocate_core(core25_pool);
  auto core26_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{6}, loomt::index{4}});
  auto core26 = allocate_core(core26_pool);
  auto core27_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{6}, loomt::index{5}});
  auto core27 = allocate_core(core27_pool);
  auto core28_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{7}, loomt::index{2}});
  auto core28 = allocate_core(core28_pool);
  auto core29_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{7}, loomt::index{3}});
  auto core29 = allocate_core(core29_pool);
  auto core30_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{7}, loomt::index{4}});
  auto core30 = allocate_core(core30_pool);
  auto core31_pool = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{7}, loomt::index{5}});
  auto core31 = allocate_core(core31_pool);
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

  auto barriers0 = make_barriers(core0.memory);
  auto barriers1 = make_barriers(core1.memory);
  auto barriers2 = make_barriers(core2.memory);
  auto barriers3 = make_barriers(core3.memory);
  auto barriers7 = make_barriers(core7.memory);
  auto barriers6 = make_barriers(core6.memory);
  auto barriers5 = make_barriers(core5.memory);
  auto barriers4 = make_barriers(core4.memory);
  auto barriers8 = make_barriers(core8.memory);
  auto barriers9 = make_barriers(core9.memory);
  auto barriers10 = make_barriers(core10.memory);
  auto barriers11 = make_barriers(core11.memory);
  auto barriers15 = make_barriers(core15.memory);
  auto barriers14 = make_barriers(core14.memory);
  auto barriers13 = make_barriers(core13.memory);
  auto barriers12 = make_barriers(core12.memory);
  auto barriers16 = make_barriers(core16.memory);
  auto barriers17 = make_barriers(core17.memory);
  auto barriers18 = make_barriers(core18.memory);
  auto barriers19 = make_barriers(core19.memory);
  auto barriers23 = make_barriers(core23.memory);
  auto barriers22 = make_barriers(core22.memory);
  auto barriers21 = make_barriers(core21.memory);
  auto barriers20 = make_barriers(core20.memory);
  auto barriers24 = make_barriers(core24.memory);
  auto barriers25 = make_barriers(core25.memory);
  auto barriers26 = make_barriers(core26.memory);
  auto barriers27 = make_barriers(core27.memory);
  auto barriers31 = make_barriers(core31.memory);
  auto barriers30 = make_barriers(core30.memory);
  auto barriers29 = make_barriers(core29.memory);
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
  auto* sequence0_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core0.memory);
  auto probability2 = make_feature_stream(core2.storage);
  auto* sequence2_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core2.memory);
  auto probability4 = make_feature_stream(core4.storage);
  auto* sequence4_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core4.memory);
  auto probability6 = make_feature_stream(core6.storage);
  auto* sequence6_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core6.memory);
  auto probability8 = make_feature_stream(core8.storage);
  auto* sequence8_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core8.memory);
  auto probability10 = make_feature_stream(core10.storage);
  auto* sequence10_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core10.memory);
  auto probability12 = make_feature_stream(core12.storage);
  auto* sequence12_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core12.memory);
  auto probability14 = make_feature_stream(core14.storage);
  auto* sequence14_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core14.memory);
  auto probability16 = make_feature_stream(core16.storage);
  auto* sequence16_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core16.memory);
  auto probability18 = make_feature_stream(core18.storage);
  auto* sequence18_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core18.memory);
  auto probability20 = make_feature_stream(core20.storage);
  auto* sequence20_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core20.memory);
  auto probability22 = make_feature_stream(core22.storage);
  auto* sequence22_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core22.memory);
  auto probability24 = make_feature_stream(core24.storage);
  auto* sequence24_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core24.memory);
  auto probability26 = make_feature_stream(core26.storage);
  auto* sequence26_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core26.memory);
  auto probability28 = make_feature_stream(core28.storage);
  auto* sequence28_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core28.memory);
  auto probability30 = make_feature_stream(core30.storage);
  auto* sequence30_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, core30.memory);
  auto query_output0 = make_query_output(core0.storage);
  auto query_output2 = make_query_output(core2.storage);
  auto query_output4 = make_query_output(core4.storage);
  auto query_output6 = make_query_output(core6.storage);
  auto query_output8 = make_query_output(core9.storage);
  auto query_output10 = make_query_output(core11.storage);
  auto query_output12 = make_query_output(core12.storage);
  auto query_output14 = make_query_output(core14.storage);
  auto query_output16 = make_query_output(core16.storage);
  auto query_output18 = make_query_output(core18.storage);
  auto query_output20 = make_query_output(core20.storage);
  auto query_output22 = make_query_output(core22.storage);
  auto query_output24 = make_query_output(core25.storage);
  auto query_output26 = make_query_output(core27.storage);
  auto query_output28 = make_query_output(core28.storage);
  auto query_output30 = make_query_output(core30.storage);
  auto normal_output8 = loom::channel::bind<loom::channel::discipline::fifo>(query_output8.slots, loomt::index{2});
  auto normal_output10 = loom::channel::bind<loom::channel::discipline::fifo>(query_output10.slots, loomt::index{2});
  auto normal_output24 = loom::channel::bind<loom::channel::discipline::fifo>(query_output24.slots, loomt::index{2});
  auto normal_output26 = loom::channel::bind<loom::channel::discipline::fifo>(query_output26.slots, loomt::index{2});
  auto* dimensions1_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core1.memory);
  auto* dimensions3_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core3.memory);
  auto* dimensions5_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core5.memory);
  auto* dimensions7_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core7.memory);
  auto* dimensions8_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core8.memory);
  auto* dimensions10_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core10.memory);
  auto* dimensions13_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core13.memory);
  auto* dimensions15_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core15.memory);
  auto* dimensions17_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core17.memory);
  auto* dimensions19_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core19.memory);
  auto* dimensions21_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core21.memory);
  auto* dimensions23_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core23.memory);
  auto* dimensions24_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core24.memory);
  auto* dimensions26_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core26.memory);
  auto* dimensions29_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core29.memory);
  auto* dimensions31_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, core31.memory);
  auto* gather_cache0 = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{192}, core0.memory);
  auto* gather_cache9 = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{192}, core9.memory);
  auto* gather_cache18 = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{192}, core18.memory);
  auto* gather_cache27 = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{192}, core27.memory);
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record = 0; record < 3360; ++record) {
      loomt::offset relative = loom::index::scale(loom::index::cast(record), loomt::offset{18496});
      loomt::offset offset = loomt::offset{0} + relative;
      auto external = loom::buffer::view<4624>(loom::buffer::cast<const unsigned>(projection0), offset, {});
      auto reservation = loom::channel::reserve(weight_ingress0);
      auto ingress_copy = loom::kernel::async::copy<loom::kernel::async::direction::global_to_workgroup, loom::cache::scope::workgroup, loom::cache::temporal::regular>(external, reservation.payload);
      auto ingress_copy_group = loom::kernel::async::group(ingress_copy);
      loom::kernel::async::wait<0>(ingress_copy_group);
      loom::channel::publish(reservation.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_1 = 0; record_1 < 3360; ++record_1) {
      auto acquisition = loom::channel::acquire(weight_ingress0);
      auto fanout = loom::channel::fanout<4>(acquisition.read);
      auto reservation_2 = loom::channel::reserve(streams0.weights);
      loom::channel::copy(fanout.first, reservation_2.write);
      auto reservation_3 = loom::channel::reserve(streams1.weights);
      loom::channel::copy(fanout.remaining.first, reservation_3.write);
      auto reservation_4 = loom::channel::reserve(streams2.weights);
      loom::channel::copy(fanout.remaining.remaining.first, reservation_4.write);
      auto reservation_5 = loom::channel::reserve(streams3.weights);
      loom::channel::copy(fanout.remaining.remaining.second, reservation_5.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_2 = 0; record_2 < 3360; ++record_2) {
      loomt::offset relative_1 = loom::index::scale(loom::index::cast(record_2), loomt::offset{18496});
      loomt::offset offset_1 = loomt::offset{0} + relative_1;
      auto external_1 = loom::buffer::view<4624>(loom::buffer::cast<const unsigned>(projection1), offset_1, {});
      auto reservation_6 = loom::channel::reserve(weight_ingress1);
      auto ingress_copy_1 = loom::kernel::async::copy<loom::kernel::async::direction::global_to_workgroup, loom::cache::scope::workgroup, loom::cache::temporal::regular>(external_1, reservation_6.payload);
      auto ingress_copy_group_1 = loom::kernel::async::group(ingress_copy_1);
      loom::kernel::async::wait<0>(ingress_copy_group_1);
      loom::channel::publish(reservation_6.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_3 = 0; record_3 < 3360; ++record_3) {
      auto acquisition_2 = loom::channel::acquire(weight_ingress1);
      auto fanout_2 = loom::channel::fanout<4>(acquisition_2.read);
      auto reservation_7 = loom::channel::reserve(streams4.weights);
      loom::channel::copy(fanout_2.first, reservation_7.write);
      auto reservation_8 = loom::channel::reserve(streams5.weights);
      loom::channel::copy(fanout_2.remaining.first, reservation_8.write);
      auto reservation_9 = loom::channel::reserve(streams6.weights);
      loom::channel::copy(fanout_2.remaining.remaining.first, reservation_9.write);
      auto reservation_10 = loom::channel::reserve(streams7.weights);
      loom::channel::copy(fanout_2.remaining.remaining.second, reservation_10.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_4 = 0; record_4 < 3360; ++record_4) {
      loomt::offset relative_2 = loom::index::scale(loom::index::cast(record_4), loomt::offset{18496});
      loomt::offset offset_2 = loomt::offset{0} + relative_2;
      auto external_2 = loom::buffer::view<4624>(loom::buffer::cast<const unsigned>(projection2), offset_2, {});
      auto reservation_11 = loom::channel::reserve(weight_ingress2);
      auto ingress_copy_2 = loom::kernel::async::copy<loom::kernel::async::direction::global_to_workgroup, loom::cache::scope::workgroup, loom::cache::temporal::regular>(external_2, reservation_11.payload);
      auto ingress_copy_group_2 = loom::kernel::async::group(ingress_copy_2);
      loom::kernel::async::wait<0>(ingress_copy_group_2);
      loom::channel::publish(reservation_11.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_5 = 0; record_5 < 3360; ++record_5) {
      auto acquisition_3 = loom::channel::acquire(weight_ingress2);
      auto fanout_3 = loom::channel::fanout<4>(acquisition_3.read);
      auto reservation_12 = loom::channel::reserve(streams8.weights);
      loom::channel::copy(fanout_3.first, reservation_12.write);
      auto reservation_13 = loom::channel::reserve(streams9.weights);
      loom::channel::copy(fanout_3.remaining.first, reservation_13.write);
      auto reservation_14 = loom::channel::reserve(streams10.weights);
      loom::channel::copy(fanout_3.remaining.remaining.first, reservation_14.write);
      auto reservation_15 = loom::channel::reserve(streams11.weights);
      loom::channel::copy(fanout_3.remaining.remaining.second, reservation_15.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_6 = 0; record_6 < 3360; ++record_6) {
      loomt::offset relative_3 = loom::index::scale(loom::index::cast(record_6), loomt::offset{18496});
      loomt::offset offset_3 = loomt::offset{0} + relative_3;
      auto external_3 = loom::buffer::view<4624>(loom::buffer::cast<const unsigned>(projection3), offset_3, {});
      auto reservation_16 = loom::channel::reserve(weight_ingress3);
      auto ingress_copy_3 = loom::kernel::async::copy<loom::kernel::async::direction::global_to_workgroup, loom::cache::scope::workgroup, loom::cache::temporal::regular>(external_3, reservation_16.payload);
      auto ingress_copy_group_3 = loom::kernel::async::group(ingress_copy_3);
      loom::kernel::async::wait<0>(ingress_copy_group_3);
      loom::channel::publish(reservation_16.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_7 = 0; record_7 < 3360; ++record_7) {
      auto acquisition_4 = loom::channel::acquire(weight_ingress3);
      auto fanout_4 = loom::channel::fanout<4>(acquisition_4.read);
      auto reservation_17 = loom::channel::reserve(streams12.weights);
      loom::channel::copy(fanout_4.first, reservation_17.write);
      auto reservation_18 = loom::channel::reserve(streams13.weights);
      loom::channel::copy(fanout_4.remaining.first, reservation_18.write);
      auto reservation_19 = loom::channel::reserve(streams14.weights);
      loom::channel::copy(fanout_4.remaining.remaining.first, reservation_19.write);
      auto reservation_20 = loom::channel::reserve(streams15.weights);
      loom::channel::copy(fanout_4.remaining.remaining.second, reservation_20.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_8 = 0; record_8 < 3360; ++record_8) {
      loomt::offset relative_4 = loom::index::scale(loom::index::cast(record_8), loomt::offset{18496});
      loomt::offset offset_4 = loomt::offset{0} + relative_4;
      auto external_4 = loom::buffer::view<4624>(loom::buffer::cast<const unsigned>(projection4), offset_4, {});
      auto reservation_21 = loom::channel::reserve(weight_ingress4);
      auto ingress_copy_4 = loom::kernel::async::copy<loom::kernel::async::direction::global_to_workgroup, loom::cache::scope::workgroup, loom::cache::temporal::regular>(external_4, reservation_21.payload);
      auto ingress_copy_group_4 = loom::kernel::async::group(ingress_copy_4);
      loom::kernel::async::wait<0>(ingress_copy_group_4);
      loom::channel::publish(reservation_21.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_9 = 0; record_9 < 3360; ++record_9) {
      auto acquisition_5 = loom::channel::acquire(weight_ingress4);
      auto fanout_5 = loom::channel::fanout<4>(acquisition_5.read);
      auto reservation_22 = loom::channel::reserve(streams16.weights);
      loom::channel::copy(fanout_5.first, reservation_22.write);
      auto reservation_23 = loom::channel::reserve(streams17.weights);
      loom::channel::copy(fanout_5.remaining.first, reservation_23.write);
      auto reservation_24 = loom::channel::reserve(streams18.weights);
      loom::channel::copy(fanout_5.remaining.remaining.first, reservation_24.write);
      auto reservation_25 = loom::channel::reserve(streams19.weights);
      loom::channel::copy(fanout_5.remaining.remaining.second, reservation_25.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_10 = 0; record_10 < 3360; ++record_10) {
      loomt::offset relative_5 = loom::index::scale(loom::index::cast(record_10), loomt::offset{18496});
      loomt::offset offset_5 = loomt::offset{0} + relative_5;
      auto external_5 = loom::buffer::view<4624>(loom::buffer::cast<const unsigned>(projection5), offset_5, {});
      auto reservation_26 = loom::channel::reserve(weight_ingress5);
      auto ingress_copy_5 = loom::kernel::async::copy<loom::kernel::async::direction::global_to_workgroup, loom::cache::scope::workgroup, loom::cache::temporal::regular>(external_5, reservation_26.payload);
      auto ingress_copy_group_5 = loom::kernel::async::group(ingress_copy_5);
      loom::kernel::async::wait<0>(ingress_copy_group_5);
      loom::channel::publish(reservation_26.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_11 = 0; record_11 < 3360; ++record_11) {
      auto acquisition_6 = loom::channel::acquire(weight_ingress5);
      auto fanout_6 = loom::channel::fanout<4>(acquisition_6.read);
      auto reservation_27 = loom::channel::reserve(streams20.weights);
      loom::channel::copy(fanout_6.first, reservation_27.write);
      auto reservation_28 = loom::channel::reserve(streams21.weights);
      loom::channel::copy(fanout_6.remaining.first, reservation_28.write);
      auto reservation_29 = loom::channel::reserve(streams22.weights);
      loom::channel::copy(fanout_6.remaining.remaining.first, reservation_29.write);
      auto reservation_30 = loom::channel::reserve(streams23.weights);
      loom::channel::copy(fanout_6.remaining.remaining.second, reservation_30.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_12 = 0; record_12 < 3360; ++record_12) {
      loomt::offset relative_6 = loom::index::scale(loom::index::cast(record_12), loomt::offset{18496});
      loomt::offset offset_6 = loomt::offset{0} + relative_6;
      auto external_6 = loom::buffer::view<4624>(loom::buffer::cast<const unsigned>(projection6), offset_6, {});
      auto reservation_31 = loom::channel::reserve(weight_ingress6);
      auto ingress_copy_6 = loom::kernel::async::copy<loom::kernel::async::direction::global_to_workgroup, loom::cache::scope::workgroup, loom::cache::temporal::regular>(external_6, reservation_31.payload);
      auto ingress_copy_group_6 = loom::kernel::async::group(ingress_copy_6);
      loom::kernel::async::wait<0>(ingress_copy_group_6);
      loom::channel::publish(reservation_31.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_13 = 0; record_13 < 3360; ++record_13) {
      auto acquisition_7 = loom::channel::acquire(weight_ingress6);
      auto fanout_7 = loom::channel::fanout<4>(acquisition_7.read);
      auto reservation_32 = loom::channel::reserve(streams24.weights);
      loom::channel::copy(fanout_7.first, reservation_32.write);
      auto reservation_33 = loom::channel::reserve(streams25.weights);
      loom::channel::copy(fanout_7.remaining.first, reservation_33.write);
      auto reservation_34 = loom::channel::reserve(streams26.weights);
      loom::channel::copy(fanout_7.remaining.remaining.first, reservation_34.write);
      auto reservation_35 = loom::channel::reserve(streams27.weights);
      loom::channel::copy(fanout_7.remaining.remaining.second, reservation_35.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_14 = 0; record_14 < 3360; ++record_14) {
      loomt::offset relative_7 = loom::index::scale(loom::index::cast(record_14), loomt::offset{18496});
      loomt::offset offset_7 = loomt::offset{0} + relative_7;
      auto external_7 = loom::buffer::view<4624>(loom::buffer::cast<const unsigned>(projection7), offset_7, {});
      auto reservation_36 = loom::channel::reserve(weight_ingress7);
      auto ingress_copy_7 = loom::kernel::async::copy<loom::kernel::async::direction::global_to_workgroup, loom::cache::scope::workgroup, loom::cache::temporal::regular>(external_7, reservation_36.payload);
      auto ingress_copy_group_7 = loom::kernel::async::group(ingress_copy_7);
      loom::kernel::async::wait<0>(ingress_copy_group_7);
      loom::channel::publish(reservation_36.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_15 = 0; record_15 < 3360; ++record_15) {
      auto acquisition_8 = loom::channel::acquire(weight_ingress7);
      auto fanout_8 = loom::channel::fanout<4>(acquisition_8.read);
      auto reservation_37 = loom::channel::reserve(streams28.weights);
      loom::channel::copy(fanout_8.first, reservation_37.write);
      auto reservation_38 = loom::channel::reserve(streams29.weights);
      loom::channel::copy(fanout_8.remaining.first, reservation_38.write);
      auto reservation_39 = loom::channel::reserve(streams30.weights);
      loom::channel::copy(fanout_8.remaining.remaining.first, reservation_39.write);
      auto reservation_40 = loom::channel::reserve(streams31.weights);
      loom::channel::copy(fanout_8.remaining.remaining.second, reservation_40.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_16 = 0; record_16 < 5190; ++record_16) {
      auto acquisition_9 = loom::channel::acquire(streams0.activations);
      auto fanout_9 = loom::channel::fanout<7>(acquisition_9.read);
      auto reservation_41 = loom::channel::reserve(streams4.activations);
      loom::channel::copy(fanout_9.first, reservation_41.write);
      auto reservation_42 = loom::channel::reserve(streams8.activations);
      loom::channel::copy(fanout_9.remaining.first, reservation_42.write);
      auto reservation_43 = loom::channel::reserve(streams12.activations);
      loom::channel::copy(fanout_9.remaining.remaining.first, reservation_43.write);
      auto reservation_44 = loom::channel::reserve(streams16.activations);
      loom::channel::copy(fanout_9.remaining.remaining.remaining.first, reservation_44.write);
      auto reservation_45 = loom::channel::reserve(streams20.activations);
      loom::channel::copy(fanout_9.remaining.remaining.remaining.remaining.first, reservation_45.write);
      auto reservation_46 = loom::channel::reserve(streams24.activations);
      loom::channel::copy(fanout_9.remaining.remaining.remaining.remaining.remaining.first, reservation_46.write);
      auto reservation_47 = loom::channel::reserve(streams28.activations);
      loom::channel::copy(fanout_9.remaining.remaining.remaining.remaining.remaining.second, reservation_47.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_17 = 0; record_17 < 4294; ++record_17) {
      auto acquisition_10 = loom::channel::acquire(streams9.activations);
      auto fanout_10 = loom::channel::fanout<7>(acquisition_10.read);
      auto reservation_48 = loom::channel::reserve(streams1.activations);
      loom::channel::copy(fanout_10.first, reservation_48.write);
      auto reservation_49 = loom::channel::reserve(streams5.activations);
      loom::channel::copy(fanout_10.remaining.first, reservation_49.write);
      auto reservation_50 = loom::channel::reserve(streams13.activations);
      loom::channel::copy(fanout_10.remaining.remaining.first, reservation_50.write);
      auto reservation_51 = loom::channel::reserve(streams17.activations);
      loom::channel::copy(fanout_10.remaining.remaining.remaining.first, reservation_51.write);
      auto reservation_52 = loom::channel::reserve(streams21.activations);
      loom::channel::copy(fanout_10.remaining.remaining.remaining.remaining.first, reservation_52.write);
      auto reservation_53 = loom::channel::reserve(streams25.activations);
      loom::channel::copy(fanout_10.remaining.remaining.remaining.remaining.remaining.first, reservation_53.write);
      auto reservation_54 = loom::channel::reserve(streams29.activations);
      loom::channel::copy(fanout_10.remaining.remaining.remaining.remaining.remaining.second, reservation_54.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_18 = 0; record_18 < 5190; ++record_18) {
      auto acquisition_11 = loom::channel::acquire(streams18.activations);
      auto fanout_11 = loom::channel::fanout<7>(acquisition_11.read);
      auto reservation_55 = loom::channel::reserve(streams2.activations);
      loom::channel::copy(fanout_11.first, reservation_55.write);
      auto reservation_56 = loom::channel::reserve(streams6.activations);
      loom::channel::copy(fanout_11.remaining.first, reservation_56.write);
      auto reservation_57 = loom::channel::reserve(streams10.activations);
      loom::channel::copy(fanout_11.remaining.remaining.first, reservation_57.write);
      auto reservation_58 = loom::channel::reserve(streams14.activations);
      loom::channel::copy(fanout_11.remaining.remaining.remaining.first, reservation_58.write);
      auto reservation_59 = loom::channel::reserve(streams22.activations);
      loom::channel::copy(fanout_11.remaining.remaining.remaining.remaining.first, reservation_59.write);
      auto reservation_60 = loom::channel::reserve(streams26.activations);
      loom::channel::copy(fanout_11.remaining.remaining.remaining.remaining.remaining.first, reservation_60.write);
      auto reservation_61 = loom::channel::reserve(streams30.activations);
      loom::channel::copy(fanout_11.remaining.remaining.remaining.remaining.remaining.second, reservation_61.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_19 = 0; record_19 < 4294; ++record_19) {
      auto acquisition_12 = loom::channel::acquire(streams27.activations);
      auto fanout_12 = loom::channel::fanout<7>(acquisition_12.read);
      auto reservation_62 = loom::channel::reserve(streams3.activations);
      loom::channel::copy(fanout_12.first, reservation_62.write);
      auto reservation_63 = loom::channel::reserve(streams7.activations);
      loom::channel::copy(fanout_12.remaining.first, reservation_63.write);
      auto reservation_64 = loom::channel::reserve(streams11.activations);
      loom::channel::copy(fanout_12.remaining.remaining.first, reservation_64.write);
      auto reservation_65 = loom::channel::reserve(streams15.activations);
      loom::channel::copy(fanout_12.remaining.remaining.remaining.first, reservation_65.write);
      auto reservation_66 = loom::channel::reserve(streams19.activations);
      loom::channel::copy(fanout_12.remaining.remaining.remaining.remaining.first, reservation_66.write);
      auto reservation_67 = loom::channel::reserve(streams23.activations);
      loom::channel::copy(fanout_12.remaining.remaining.remaining.remaining.remaining.first, reservation_67.write);
      auto reservation_68 = loom::channel::reserve(streams31.activations);
      loom::channel::copy(fanout_12.remaining.remaining.remaining.remaining.remaining.second, reservation_68.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_ingress_root(loom::buffer::root(sequence0_storage), loomt::offset{0}, loom::buffer::root(core0.storage), loomt::offset{10624}, loom::buffer::root(core1.storage), loom::buffer::root(gather_cache0), activations, streams0.activations, loomt::index{0}, loomt::index{1}, constants, embedding, state, barriers0.arrival, barriers0.departure, query_output0.stream, output, streams0.weights, loomt::offset{18816}, probability0, result0);
  }
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(true, loom::buffer::root(dimensions1_storage), loomt::offset{0}, loom::buffer::root(core1.storage), loomt::offset{10624}, loom::buffer::root(core0.storage), streams1.activations, loomt::index{0}, loomt::index{1}, barriers1.arrival, barriers0.arrival, barriers0.departure, barriers1.departure, query_output0.stream, streams1.weights, loomt::offset{18816}, probability0, result0, loomt::index{0}, loomt::index{0}, loomt::index{0}, loomt::index{0});
  }
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence2_storage), loomt::offset{0}, loom::buffer::root(core2.storage), loomt::offset{10624}, loom::buffer::root(core3.storage), streams2.activations, loomt::index{0}, loomt::index{1}, barriers2.arrival, barriers1.arrival, barriers1.departure, barriers2.departure, query_output2.stream, output, state, streams2.weights, loomt::offset{18816}, probability2, result2, loomt::index{0}, loomt::index{0}, loomt::offset{262144}, loomt::offset{1073152}, loomt::index{0}, loomt::offset{663552}, loomt::offset{139264}, loomt::offset{270464}, loomt::offset{1859584}, loomt::offset{2760704}, loomt::offset{1351680});
  }
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions3_storage), loomt::offset{0}, loom::buffer::root(core3.storage), loomt::offset{10624}, loom::buffer::root(core2.storage), streams3.activations, loomt::index{0}, loomt::index{1}, barriers3.arrival, barriers2.arrival, barriers2.departure, barriers3.departure, query_output2.stream, streams3.weights, loomt::offset{18816}, probability2, result2, loomt::index{0}, loomt::index{64}, loomt::index{0}, loomt::index{0});
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence4_storage), loomt::offset{0}, loom::buffer::root(core4.storage), loomt::offset{10624}, loom::buffer::root(core5.storage), streams4.activations, loomt::index{0}, loomt::index{1}, barriers4.arrival, barriers5.arrival, barriers5.departure, barriers4.departure, query_output4.stream, output, state, streams4.weights, loomt::offset{18816}, probability4, result4, loomt::index{512}, loomt::index{1}, loomt::offset{16384}, loomt::offset{1056832}, loomt::index{128}, loomt::offset{794624}, loomt::offset{9216}, loomt::offset{271360}, loomt::offset{1875968}, loomt::offset{2629888}, loomt::offset{1384448});
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions5_storage), loomt::offset{0}, loom::buffer::root(core5.storage), loomt::offset{10624}, loom::buffer::root(core4.storage), streams5.activations, loomt::index{0}, loomt::index{1}, barriers5.arrival, barriers6.arrival, barriers6.departure, barriers5.departure, query_output4.stream, streams5.weights, loomt::offset{18816}, probability4, result4, loomt::index{512}, loomt::index{4}, loomt::index{1}, loomt::index{128});
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence6_storage), loomt::offset{0}, loom::buffer::root(core6.storage), loomt::offset{10624}, loom::buffer::root(core7.storage), streams6.activations, loomt::index{0}, loomt::index{1}, barriers6.arrival, barriers7.arrival, barriers7.departure, barriers6.departure, query_output6.stream, output, state, streams6.weights, loomt::offset{18816}, probability6, result6, loomt::index{512}, loomt::index{1}, loomt::offset{278528}, loomt::offset{1073216}, loomt::index{128}, loomt::offset{925696}, loomt::offset{140288}, loomt::offset{271488}, loomt::offset{1892352}, loomt::offset{2760960}, loomt::offset{1417216});
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions7_storage), loomt::offset{0}, loom::buffer::root(core7.storage), loomt::offset{10624}, loom::buffer::root(core6.storage), streams7.activations, loomt::index{0}, loomt::index{1}, barriers7.arrival, barriers3.arrival, barriers3.departure, barriers7.departure, query_output6.stream, streams7.weights, loomt::offset{18816}, probability6, result6, loomt::index{512}, loomt::index{68}, loomt::index{1}, loomt::index{128});
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_epilogue(loom::buffer::root(sequence8_storage), loomt::offset{0}, loom::buffer::root(dimensions8_storage), loom::buffer::root(core8.storage), loomt::offset{10624}, loom::buffer::root(core9.storage), streams8.activations, loomt::index{0}, loomt::index{1}, barriers8.arrival, barriers4.arrival, barriers4.departure, barriers8.departure, normal_output8, streams8.weights, loomt::offset{18816}, query_output8.stream, probability8, loomt::index{1024}, loomt::index{8}, loomt::index{2}, loomt::index{256});
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value_ingress(loom::buffer::root(core9.storage), loomt::offset{0}, loomt::offset{10624}, loom::buffer::root(core8.storage), loom::buffer::root(gather_cache9), activations, streams9.activations, loomt::index{0}, loomt::index{1}, constants, embedding, state, barriers9.arrival, barriers8.arrival, barriers8.departure, barriers9.departure, normal_output8, output, streams9.weights, loomt::offset{18816}, query_output8.stream, probability8, loom::buffer::root(sequence8_storage), result8, loomt::index{1024}, loomt::index{32}, loomt::index{2}, loomt::offset{2695168}, loomt::offset{32768}, loomt::offset{1056896}, loomt::offset{1064960}, loomt::offset{1327104}, loomt::offset{1851392}, loomt::offset{2695168}, loomt::offset{2899968}, loomt::index{256}, loomt::offset{533504}, loomt::offset{10240}, loomt::offset{272384}, loomt::offset{1908736}, loomt::offset{2630144}, loomt::offset{1449984});
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_normalize(loom::buffer::root(sequence10_storage), loomt::offset{0}, loom::buffer::root(dimensions10_storage), loom::buffer::root(core10.storage), loomt::offset{10624}, loom::buffer::root(core11.storage), streams10.activations, loomt::index{0}, loomt::index{1}, barriers10.arrival, barriers9.arrival, barriers9.departure, barriers10.departure, normal_output10, streams10.weights, loomt::offset{18816}, probability10, loomt::index{1024}, loomt::index{72}, loomt::index{2}, loomt::index{256});
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value_egress(loom::buffer::root(core11.storage), loomt::offset{0}, loomt::offset{10624}, loom::buffer::root(core10.storage), streams11.activations, loomt::index{0}, loomt::index{1}, barriers11.arrival, barriers10.arrival, barriers10.departure, barriers11.departure, normal_output10, output, state, streams11.weights, loomt::offset{18816}, query_output10.stream, probability10, result10, loomt::index{1024}, loomt::index{2}, loomt::offset{294912}, loomt::offset{1073280}, loomt::index{256}, loomt::offset{664576}, loomt::offset{141312}, loomt::offset{272512}, loomt::offset{1925120}, loomt::offset{2761216}, loomt::offset{1482752});
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence12_storage), loomt::offset{0}, loom::buffer::root(core12.storage), loomt::offset{10624}, loom::buffer::root(core13.storage), streams12.activations, loomt::index{0}, loomt::index{1}, barriers12.arrival, barriers13.arrival, barriers13.departure, barriers12.departure, query_output12.stream, output, state, streams12.weights, loomt::offset{18816}, probability12, result12, loomt::index{1536}, loomt::index{3}, loomt::offset{49152}, loomt::offset{1056960}, loomt::index{384}, loomt::offset{795648}, loomt::offset{11264}, loomt::offset{273408}, loomt::offset{1941504}, loomt::offset{2630400}, loomt::offset{1515520});
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions13_storage), loomt::offset{0}, loom::buffer::root(core13.storage), loomt::offset{10624}, loom::buffer::root(core12.storage), streams13.activations, loomt::index{0}, loomt::index{1}, barriers13.arrival, barriers14.arrival, barriers14.departure, barriers13.departure, query_output12.stream, streams13.weights, loomt::offset{18816}, probability12, result12, loomt::index{1536}, loomt::index{12}, loomt::index{3}, loomt::index{384});
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence14_storage), loomt::offset{0}, loom::buffer::root(core14.storage), loomt::offset{10624}, loom::buffer::root(core15.storage), streams14.activations, loomt::index{0}, loomt::index{1}, barriers14.arrival, barriers15.arrival, barriers15.departure, barriers14.departure, query_output14.stream, output, state, streams14.weights, loomt::offset{18816}, probability14, result14, loomt::index{1536}, loomt::index{3}, loomt::offset{311296}, loomt::offset{1073344}, loomt::index{384}, loomt::offset{926720}, loomt::offset{142336}, loomt::offset{273536}, loomt::offset{1957888}, loomt::offset{2761472}, loomt::offset{1548288});
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions15_storage), loomt::offset{0}, loom::buffer::root(core15.storage), loomt::offset{10624}, loom::buffer::root(core14.storage), streams15.activations, loomt::index{0}, loomt::index{1}, barriers15.arrival, barriers11.arrival, barriers11.departure, barriers15.departure, query_output14.stream, streams15.weights, loomt::offset{18816}, probability14, result14, loomt::index{1536}, loomt::index{76}, loomt::index{3}, loomt::index{384});
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence16_storage), loomt::offset{0}, loom::buffer::root(core16.storage), loomt::offset{10624}, loom::buffer::root(core17.storage), streams16.activations, loomt::index{0}, loomt::index{1}, barriers16.arrival, barriers12.arrival, barriers12.departure, barriers16.departure, query_output16.stream, output, state, streams16.weights, loomt::offset{18816}, probability16, result16, loomt::index{2048}, loomt::index{4}, loomt::offset{65536}, loomt::offset{1057024}, loomt::index{512}, loomt::offset{534528}, loomt::offset{12288}, loomt::offset{274432}, loomt::offset{1974272}, loomt::offset{2630656}, loomt::offset{1581056});
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions17_storage), loomt::offset{0}, loom::buffer::root(core17.storage), loomt::offset{10624}, loom::buffer::root(core16.storage), streams17.activations, loomt::index{0}, loomt::index{1}, barriers17.arrival, barriers16.arrival, barriers16.departure, barriers17.departure, query_output16.stream, streams17.weights, loomt::offset{18816}, probability16, result16, loomt::index{2048}, loomt::index{16}, loomt::index{4}, loomt::index{512});
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_ingress(loom::buffer::root(sequence18_storage), loomt::offset{0}, loom::buffer::root(core18.storage), loomt::offset{10624}, loom::buffer::root(core19.storage), loom::buffer::root(gather_cache18), activations, streams18.activations, loomt::index{0}, loomt::index{1}, constants, embedding, state, barriers18.arrival, barriers17.arrival, barriers17.departure, barriers18.departure, query_output18.stream, output, streams18.weights, loomt::offset{18816}, probability18, result18);
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(true, loom::buffer::root(dimensions19_storage), loomt::offset{0}, loom::buffer::root(core19.storage), loomt::offset{10624}, loom::buffer::root(core18.storage), streams19.activations, loomt::index{0}, loomt::index{1}, barriers19.arrival, barriers18.arrival, barriers18.departure, barriers19.departure, query_output18.stream, streams19.weights, loomt::offset{18816}, probability18, result18, loomt::index{2048}, loomt::index{80}, loomt::index{4}, loomt::index{512});
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence20_storage), loomt::offset{0}, loom::buffer::root(core20.storage), loomt::offset{10624}, loom::buffer::root(core21.storage), streams20.activations, loomt::index{0}, loomt::index{1}, barriers20.arrival, barriers21.arrival, barriers21.departure, barriers20.departure, query_output20.stream, output, state, streams20.weights, loomt::offset{18816}, probability20, result20, loomt::index{2560}, loomt::index{5}, loomt::offset{81920}, loomt::offset{1057088}, loomt::index{640}, loomt::offset{796672}, loomt::offset{13312}, loomt::offset{275456}, loomt::offset{2007040}, loomt::offset{2630912}, loomt::offset{1646592});
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions21_storage), loomt::offset{0}, loom::buffer::root(core21.storage), loomt::offset{10624}, loom::buffer::root(core20.storage), streams21.activations, loomt::index{0}, loomt::index{1}, barriers21.arrival, barriers22.arrival, barriers22.departure, barriers21.departure, query_output20.stream, streams21.weights, loomt::offset{18816}, probability20, result20, loomt::index{2560}, loomt::index{20}, loomt::index{5}, loomt::index{640});
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence22_storage), loomt::offset{0}, loom::buffer::root(core22.storage), loomt::offset{10624}, loom::buffer::root(core23.storage), streams22.activations, loomt::index{0}, loomt::index{1}, barriers22.arrival, barriers23.arrival, barriers23.departure, barriers22.departure, query_output22.stream, output, state, streams22.weights, loomt::offset{18816}, probability22, result22, loomt::index{2560}, loomt::index{5}, loomt::offset{344064}, loomt::offset{1073472}, loomt::index{640}, loomt::offset{927744}, loomt::offset{144384}, loomt::offset{275584}, loomt::offset{2023424}, loomt::offset{2761984}, loomt::offset{1679360});
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions23_storage), loomt::offset{0}, loom::buffer::root(core23.storage), loomt::offset{10624}, loom::buffer::root(core22.storage), streams23.activations, loomt::index{0}, loomt::index{1}, barriers23.arrival, barriers19.arrival, barriers19.departure, barriers23.departure, query_output22.stream, streams23.weights, loomt::offset{18816}, probability22, result22, loomt::index{2560}, loomt::index{84}, loomt::index{5}, loomt::index{640});
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_normalize(loom::buffer::root(sequence24_storage), loomt::offset{0}, loom::buffer::root(dimensions24_storage), loom::buffer::root(core24.storage), loomt::offset{10624}, loom::buffer::root(core25.storage), streams24.activations, loomt::index{0}, loomt::index{1}, barriers24.arrival, barriers20.arrival, barriers20.departure, barriers24.departure, normal_output24, streams24.weights, loomt::offset{18816}, probability24, loomt::index{3072}, loomt::index{24}, loomt::index{6}, loomt::index{768});
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value_egress(loom::buffer::root(core25.storage), loomt::offset{0}, loomt::offset{10624}, loom::buffer::root(core24.storage), streams25.activations, loomt::index{0}, loomt::index{1}, barriers25.arrival, barriers24.arrival, barriers24.departure, barriers25.departure, normal_output24, output, state, streams25.weights, loomt::offset{18816}, query_output24.stream, probability24, result24, loomt::index{3072}, loomt::index{6}, loomt::offset{98304}, loomt::offset{1057152}, loomt::index{768}, loomt::offset{535552}, loomt::offset{14336}, loomt::offset{276480}, loomt::offset{2039808}, loomt::offset{2631168}, loomt::offset{1712128});
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_epilogue(loom::buffer::root(sequence26_storage), loomt::offset{0}, loom::buffer::root(dimensions26_storage), loom::buffer::root(core26.storage), loomt::offset{10624}, loom::buffer::root(core27.storage), streams26.activations, loomt::index{0}, loomt::index{1}, barriers26.arrival, barriers25.arrival, barriers25.departure, barriers26.departure, normal_output26, streams26.weights, loomt::offset{18816}, query_output26.stream, probability26, loomt::index{3072}, loomt::index{88}, loomt::index{6}, loomt::index{768});
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value_ingress(loom::buffer::root(core27.storage), loomt::offset{0}, loomt::offset{10624}, loom::buffer::root(core26.storage), loom::buffer::root(gather_cache27), activations, streams27.activations, loomt::index{0}, loomt::index{1}, constants, embedding, state, barriers27.arrival, barriers26.arrival, barriers26.departure, barriers27.departure, normal_output26, output, streams27.weights, loomt::offset{18816}, query_output26.stream, probability26, loom::buffer::root(sequence26_storage), result26, loomt::index{3072}, loomt::index{96}, loomt::index{6}, loomt::offset{2826240}, loomt::offset{360448}, loomt::offset{1073536}, loomt::offset{1081344}, loomt::offset{1343488}, loomt::offset{1867776}, loomt::offset{2826240}, loomt::offset{2916352}, loomt::index{768}, loomt::offset{666624}, loomt::offset{145408}, loomt::offset{276608}, loomt::offset{2056192}, loomt::offset{2762240}, loomt::offset{1744896});
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress_terminal(loom::buffer::root(sequence28_storage), loomt::offset{0}, loom::buffer::root(core28.storage), loomt::offset{10624}, loom::buffer::root(core29.storage), streams28.activations, loomt::index{0}, loomt::index{1}, barriers29.arrival, barriers29.departure, query_output28.stream, output, state, streams28.weights, loomt::offset{18816}, probability28, result28);
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions29_storage), loomt::offset{0}, loom::buffer::root(core29.storage), loomt::offset{10624}, loom::buffer::root(core28.storage), streams29.activations, loomt::index{0}, loomt::index{1}, barriers29.arrival, barriers30.arrival, barriers30.departure, barriers29.departure, query_output28.stream, streams29.weights, loomt::offset{18816}, probability28, result28, loomt::index{3584}, loomt::index{28}, loomt::index{7}, loomt::index{896});
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence30_storage), loomt::offset{0}, loom::buffer::root(core30.storage), loomt::offset{10624}, loom::buffer::root(core31.storage), streams30.activations, loomt::index{0}, loomt::index{1}, barriers30.arrival, barriers31.arrival, barriers31.departure, barriers30.departure, query_output30.stream, output, state, streams30.weights, loomt::offset{18816}, probability30, result30, loomt::index{3584}, loomt::index{7}, loomt::offset{376832}, loomt::offset{1073600}, loomt::index{896}, loomt::offset{928768}, loomt::offset{146432}, loomt::offset{277632}, loomt::offset{2088960}, loomt::offset{2762496}, loomt::offset{1810432});
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions31_storage), loomt::offset{0}, loom::buffer::root(core31.storage), loomt::offset{10624}, loom::buffer::root(core30.storage), streams31.activations, loomt::index{0}, loomt::index{1}, barriers31.arrival, barriers27.arrival, barriers27.departure, barriers31.departure, query_output30.stream, streams31.weights, loomt::offset{18816}, probability30, result30, loomt::index{3584}, loomt::index{92}, loomt::index{7}, loomt::index{896});
  }
}
