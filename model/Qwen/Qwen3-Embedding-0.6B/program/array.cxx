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
using WeightStream = loomt::channel<loomt::tile<loomt::shape<4624>, unsigned>>;

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
  auto memory_tile0 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{0}, loomt::index{1}});
  auto* weight_ingress0_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{18496}, memory_tile0);
  auto weight_ingress0_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(weight_ingress0_storage)), {});
  auto weight_ingress0 = loom::channel::bind<loom::channel::discipline::fifo>(weight_ingress0_slots, loomt::index{1});
  auto memory_tile1 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{1}, loomt::index{1}});
  auto* weight_ingress1_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{18496}, memory_tile1);
  auto weight_ingress1_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(weight_ingress1_storage)), {});
  auto weight_ingress1 = loom::channel::bind<loom::channel::discipline::fifo>(weight_ingress1_slots, loomt::index{1});
  auto memory_tile2 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{2}, loomt::index{1}});
  auto* weight_ingress2_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{18496}, memory_tile2);
  auto weight_ingress2_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(weight_ingress2_storage)), {});
  auto weight_ingress2 = loom::channel::bind<loom::channel::discipline::fifo>(weight_ingress2_slots, loomt::index{1});
  auto memory_tile3 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{3}, loomt::index{1}});
  auto* weight_ingress3_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{18496}, memory_tile3);
  auto weight_ingress3_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(weight_ingress3_storage)), {});
  auto weight_ingress3 = loom::channel::bind<loom::channel::discipline::fifo>(weight_ingress3_slots, loomt::index{1});
  auto memory_tile4 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{4}, loomt::index{1}});
  auto* weight_ingress4_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{18496}, memory_tile4);
  auto weight_ingress4_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(weight_ingress4_storage)), {});
  auto weight_ingress4 = loom::channel::bind<loom::channel::discipline::fifo>(weight_ingress4_slots, loomt::index{1});
  auto memory_tile5 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{5}, loomt::index{1}});
  auto* weight_ingress5_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{18496}, memory_tile5);
  auto weight_ingress5_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(weight_ingress5_storage)), {});
  auto weight_ingress5 = loom::channel::bind<loom::channel::discipline::fifo>(weight_ingress5_slots, loomt::index{1});
  auto memory_tile6 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{6}, loomt::index{1}});
  auto* weight_ingress6_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{18496}, memory_tile6);
  auto weight_ingress6_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(weight_ingress6_storage)), {});
  auto weight_ingress6 = loom::channel::bind<loom::channel::discipline::fifo>(weight_ingress6_slots, loomt::index{1});
  auto memory_tile7 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{7}, loomt::index{1}});
  auto* weight_ingress7_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{18496}, memory_tile7);
  auto weight_ingress7_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(weight_ingress7_storage)), {});
  auto weight_ingress7 = loom::channel::bind<loom::channel::discipline::fifo>(weight_ingress7_slots, loomt::index{1});
  auto tile0 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{0}, loomt::index{2}});
  auto* storage0 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile0);
  auto tile1 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{0}, loomt::index{3}});
  auto* storage1 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile1);
  auto tile2 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{0}, loomt::index{4}});
  auto* storage2 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile2);
  auto tile3 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{0}, loomt::index{5}});
  auto* storage3 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile3);
  auto tile4 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{1}, loomt::index{2}});
  auto* storage4 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile4);
  auto tile5 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{1}, loomt::index{3}});
  auto* storage5 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile5);
  auto tile6 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{1}, loomt::index{4}});
  auto* storage6 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile6);
  auto tile7 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{1}, loomt::index{5}});
  auto* storage7 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile7);
  auto tile8 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{2}, loomt::index{2}});
  auto* storage8 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile8);
  auto tile9 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{2}, loomt::index{3}});
  auto* storage9 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile9);
  auto tile10 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{2}, loomt::index{4}});
  auto* storage10 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile10);
  auto tile11 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{2}, loomt::index{5}});
  auto* storage11 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile11);
  auto tile12 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{3}, loomt::index{2}});
  auto* storage12 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile12);
  auto tile13 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{3}, loomt::index{3}});
  auto* storage13 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile13);
  auto tile14 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{3}, loomt::index{4}});
  auto* storage14 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile14);
  auto tile15 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{3}, loomt::index{5}});
  auto* storage15 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile15);
  auto tile16 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{4}, loomt::index{2}});
  auto* storage16 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile16);
  auto tile17 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{4}, loomt::index{3}});
  auto* storage17 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile17);
  auto tile18 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{4}, loomt::index{4}});
  auto* storage18 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile18);
  auto tile19 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{4}, loomt::index{5}});
  auto* storage19 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile19);
  auto tile20 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{5}, loomt::index{2}});
  auto* storage20 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile20);
  auto tile21 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{5}, loomt::index{3}});
  auto* storage21 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile21);
  auto tile22 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{5}, loomt::index{4}});
  auto* storage22 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile22);
  auto tile23 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{5}, loomt::index{5}});
  auto* storage23 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile23);
  auto tile24 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{6}, loomt::index{2}});
  auto* storage24 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile24);
  auto tile25 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{6}, loomt::index{3}});
  auto* storage25 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile25);
  auto tile26 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{6}, loomt::index{4}});
  auto* storage26 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile26);
  auto tile27 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{6}, loomt::index{5}});
  auto* storage27 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile27);
  auto tile28 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{7}, loomt::index{2}});
  auto* storage28 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile28);
  auto tile29 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{7}, loomt::index{3}});
  auto* storage29 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile29);
  auto tile30 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{7}, loomt::index{4}});
  auto* storage30 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile30);
  auto tile31 = loom::pipeline::memory<loom::memory_space::workgroup>(loomt::indices{loomt::index{7}, loomt::index{5}});
  auto* storage31 = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64448}, tile31);
  auto weights0_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage0)), {});
  auto weights0 = loom::channel::bind<loom::channel::discipline::fifo>(weights0_slots, loomt::index{1});
  auto activations0_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage0) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations0 = loom::channel::bind<loom::channel::discipline::fifo>(activations0_slots, loomt::index{1});
  auto weights1_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage1)), {});
  auto weights1 = loom::channel::bind<loom::channel::discipline::fifo>(weights1_slots, loomt::index{1});
  auto activations1_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage1) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations1 = loom::channel::bind<loom::channel::discipline::fifo>(activations1_slots, loomt::index{1});
  auto weights2_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage2)), {});
  auto weights2 = loom::channel::bind<loom::channel::discipline::fifo>(weights2_slots, loomt::index{1});
  auto activations2_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage2) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations2 = loom::channel::bind<loom::channel::discipline::fifo>(activations2_slots, loomt::index{1});
  auto weights3_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage3)), {});
  auto weights3 = loom::channel::bind<loom::channel::discipline::fifo>(weights3_slots, loomt::index{1});
  auto activations3_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage3) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations3 = loom::channel::bind<loom::channel::discipline::fifo>(activations3_slots, loomt::index{1});
  auto weights4_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage4)), {});
  auto weights4 = loom::channel::bind<loom::channel::discipline::fifo>(weights4_slots, loomt::index{1});
  auto activations4_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage4) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations4 = loom::channel::bind<loom::channel::discipline::fifo>(activations4_slots, loomt::index{1});
  auto weights5_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage5)), {});
  auto weights5 = loom::channel::bind<loom::channel::discipline::fifo>(weights5_slots, loomt::index{1});
  auto activations5_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage5) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations5 = loom::channel::bind<loom::channel::discipline::fifo>(activations5_slots, loomt::index{1});
  auto weights6_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage6)), {});
  auto weights6 = loom::channel::bind<loom::channel::discipline::fifo>(weights6_slots, loomt::index{1});
  auto activations6_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage6) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations6 = loom::channel::bind<loom::channel::discipline::fifo>(activations6_slots, loomt::index{1});
  auto weights7_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage7)), {});
  auto weights7 = loom::channel::bind<loom::channel::discipline::fifo>(weights7_slots, loomt::index{1});
  auto activations7_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage7) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations7 = loom::channel::bind<loom::channel::discipline::fifo>(activations7_slots, loomt::index{1});
  auto weights8_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage8)), {});
  auto weights8 = loom::channel::bind<loom::channel::discipline::fifo>(weights8_slots, loomt::index{1});
  auto activations8_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage8) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations8 = loom::channel::bind<loom::channel::discipline::fifo>(activations8_slots, loomt::index{1});
  auto weights9_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage9)), {});
  auto weights9 = loom::channel::bind<loom::channel::discipline::fifo>(weights9_slots, loomt::index{1});
  auto activations9_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage9) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations9 = loom::channel::bind<loom::channel::discipline::fifo>(activations9_slots, loomt::index{1});
  auto weights10_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage10)), {});
  auto weights10 = loom::channel::bind<loom::channel::discipline::fifo>(weights10_slots, loomt::index{1});
  auto activations10_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage10) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations10 = loom::channel::bind<loom::channel::discipline::fifo>(activations10_slots, loomt::index{1});
  auto weights11_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage11)), {});
  auto weights11 = loom::channel::bind<loom::channel::discipline::fifo>(weights11_slots, loomt::index{1});
  auto activations11_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage11) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations11 = loom::channel::bind<loom::channel::discipline::fifo>(activations11_slots, loomt::index{1});
  auto weights12_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage12)), {});
  auto weights12 = loom::channel::bind<loom::channel::discipline::fifo>(weights12_slots, loomt::index{1});
  auto activations12_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage12) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations12 = loom::channel::bind<loom::channel::discipline::fifo>(activations12_slots, loomt::index{1});
  auto weights13_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage13)), {});
  auto weights13 = loom::channel::bind<loom::channel::discipline::fifo>(weights13_slots, loomt::index{1});
  auto activations13_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage13) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations13 = loom::channel::bind<loom::channel::discipline::fifo>(activations13_slots, loomt::index{1});
  auto weights14_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage14)), {});
  auto weights14 = loom::channel::bind<loom::channel::discipline::fifo>(weights14_slots, loomt::index{1});
  auto activations14_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage14) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations14 = loom::channel::bind<loom::channel::discipline::fifo>(activations14_slots, loomt::index{1});
  auto weights15_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage15)), {});
  auto weights15 = loom::channel::bind<loom::channel::discipline::fifo>(weights15_slots, loomt::index{1});
  auto activations15_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage15) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations15 = loom::channel::bind<loom::channel::discipline::fifo>(activations15_slots, loomt::index{1});
  auto weights16_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage16)), {});
  auto weights16 = loom::channel::bind<loom::channel::discipline::fifo>(weights16_slots, loomt::index{1});
  auto activations16_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage16) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations16 = loom::channel::bind<loom::channel::discipline::fifo>(activations16_slots, loomt::index{1});
  auto weights17_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage17)), {});
  auto weights17 = loom::channel::bind<loom::channel::discipline::fifo>(weights17_slots, loomt::index{1});
  auto activations17_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage17) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations17 = loom::channel::bind<loom::channel::discipline::fifo>(activations17_slots, loomt::index{1});
  auto weights18_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage18)), {});
  auto weights18 = loom::channel::bind<loom::channel::discipline::fifo>(weights18_slots, loomt::index{1});
  auto activations18_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage18) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations18 = loom::channel::bind<loom::channel::discipline::fifo>(activations18_slots, loomt::index{1});
  auto weights19_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage19)), {});
  auto weights19 = loom::channel::bind<loom::channel::discipline::fifo>(weights19_slots, loomt::index{1});
  auto activations19_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage19) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations19 = loom::channel::bind<loom::channel::discipline::fifo>(activations19_slots, loomt::index{1});
  auto weights20_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage20)), {});
  auto weights20 = loom::channel::bind<loom::channel::discipline::fifo>(weights20_slots, loomt::index{1});
  auto activations20_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage20) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations20 = loom::channel::bind<loom::channel::discipline::fifo>(activations20_slots, loomt::index{1});
  auto weights21_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage21)), {});
  auto weights21 = loom::channel::bind<loom::channel::discipline::fifo>(weights21_slots, loomt::index{1});
  auto activations21_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage21) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations21 = loom::channel::bind<loom::channel::discipline::fifo>(activations21_slots, loomt::index{1});
  auto weights22_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage22)), {});
  auto weights22 = loom::channel::bind<loom::channel::discipline::fifo>(weights22_slots, loomt::index{1});
  auto activations22_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage22) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations22 = loom::channel::bind<loom::channel::discipline::fifo>(activations22_slots, loomt::index{1});
  auto weights23_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage23)), {});
  auto weights23 = loom::channel::bind<loom::channel::discipline::fifo>(weights23_slots, loomt::index{1});
  auto activations23_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage23) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations23 = loom::channel::bind<loom::channel::discipline::fifo>(activations23_slots, loomt::index{1});
  auto weights24_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage24)), {});
  auto weights24 = loom::channel::bind<loom::channel::discipline::fifo>(weights24_slots, loomt::index{1});
  auto activations24_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage24) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations24 = loom::channel::bind<loom::channel::discipline::fifo>(activations24_slots, loomt::index{1});
  auto weights25_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage25)), {});
  auto weights25 = loom::channel::bind<loom::channel::discipline::fifo>(weights25_slots, loomt::index{1});
  auto activations25_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage25) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations25 = loom::channel::bind<loom::channel::discipline::fifo>(activations25_slots, loomt::index{1});
  auto weights26_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage26)), {});
  auto weights26 = loom::channel::bind<loom::channel::discipline::fifo>(weights26_slots, loomt::index{1});
  auto activations26_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage26) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations26 = loom::channel::bind<loom::channel::discipline::fifo>(activations26_slots, loomt::index{1});
  auto weights27_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage27)), {});
  auto weights27 = loom::channel::bind<loom::channel::discipline::fifo>(weights27_slots, loomt::index{1});
  auto activations27_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage27) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations27 = loom::channel::bind<loom::channel::discipline::fifo>(activations27_slots, loomt::index{1});
  auto weights28_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage28)), {});
  auto weights28 = loom::channel::bind<loom::channel::discipline::fifo>(weights28_slots, loomt::index{1});
  auto activations28_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage28) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations28 = loom::channel::bind<loom::channel::discipline::fifo>(activations28_slots, loomt::index{1});
  auto weights29_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage29)), {});
  auto weights29 = loom::channel::bind<loom::channel::discipline::fifo>(weights29_slots, loomt::index{1});
  auto activations29_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage29) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations29 = loom::channel::bind<loom::channel::discipline::fifo>(activations29_slots, loomt::index{1});
  auto weights30_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage30)), {});
  auto weights30 = loom::channel::bind<loom::channel::discipline::fifo>(weights30_slots, loomt::index{1});
  auto activations30_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage30) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations30 = loom::channel::bind<loom::channel::discipline::fifo>(activations30_slots, loomt::index{1});
  auto weights31_slots = loom::buffer::view<1, 4624>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage31)), {});
  auto weights31 = loom::channel::bind<loom::channel::discipline::fifo>(weights31_slots, loomt::index{1});
  auto activations31_slots = loom::buffer::view<1, 2048>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(storage31) + static_cast<unsigned long long>(loomt::offset{18816})), {});
  auto activations31 = loom::channel::bind<loom::channel::discipline::fifo>(activations31_slots, loomt::index{1});
  auto* arrival0_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile0);
  auto arrival0_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival0_storage)), {});
  auto arrival0 = loom::channel::bind<loom::channel::discipline::fifo>(arrival0_slots, loomt::index{1});
  auto* departure0_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile0);
  auto departure0_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure0_storage)), {});
  auto departure0 = loom::channel::bind<loom::channel::discipline::fifo>(departure0_slots, loomt::index{1});
  auto* arrival1_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile1);
  auto arrival1_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival1_storage)), {});
  auto arrival1 = loom::channel::bind<loom::channel::discipline::fifo>(arrival1_slots, loomt::index{1});
  auto* departure1_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile1);
  auto departure1_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure1_storage)), {});
  auto departure1 = loom::channel::bind<loom::channel::discipline::fifo>(departure1_slots, loomt::index{1});
  auto* arrival2_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile2);
  auto arrival2_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival2_storage)), {});
  auto arrival2 = loom::channel::bind<loom::channel::discipline::fifo>(arrival2_slots, loomt::index{1});
  auto* departure2_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile2);
  auto departure2_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure2_storage)), {});
  auto departure2 = loom::channel::bind<loom::channel::discipline::fifo>(departure2_slots, loomt::index{1});
  auto* arrival3_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile3);
  auto arrival3_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival3_storage)), {});
  auto arrival3 = loom::channel::bind<loom::channel::discipline::fifo>(arrival3_slots, loomt::index{1});
  auto* departure3_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile3);
  auto departure3_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure3_storage)), {});
  auto departure3 = loom::channel::bind<loom::channel::discipline::fifo>(departure3_slots, loomt::index{1});
  auto* arrival7_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile7);
  auto arrival7_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival7_storage)), {});
  auto arrival7 = loom::channel::bind<loom::channel::discipline::fifo>(arrival7_slots, loomt::index{1});
  auto* departure7_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile7);
  auto departure7_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure7_storage)), {});
  auto departure7 = loom::channel::bind<loom::channel::discipline::fifo>(departure7_slots, loomt::index{1});
  auto* arrival6_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile6);
  auto arrival6_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival6_storage)), {});
  auto arrival6 = loom::channel::bind<loom::channel::discipline::fifo>(arrival6_slots, loomt::index{1});
  auto* departure6_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile6);
  auto departure6_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure6_storage)), {});
  auto departure6 = loom::channel::bind<loom::channel::discipline::fifo>(departure6_slots, loomt::index{1});
  auto* arrival5_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile5);
  auto arrival5_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival5_storage)), {});
  auto arrival5 = loom::channel::bind<loom::channel::discipline::fifo>(arrival5_slots, loomt::index{1});
  auto* departure5_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile5);
  auto departure5_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure5_storage)), {});
  auto departure5 = loom::channel::bind<loom::channel::discipline::fifo>(departure5_slots, loomt::index{1});
  auto* arrival4_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile4);
  auto arrival4_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival4_storage)), {});
  auto arrival4 = loom::channel::bind<loom::channel::discipline::fifo>(arrival4_slots, loomt::index{1});
  auto* departure4_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile4);
  auto departure4_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure4_storage)), {});
  auto departure4 = loom::channel::bind<loom::channel::discipline::fifo>(departure4_slots, loomt::index{1});
  auto* arrival8_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile8);
  auto arrival8_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival8_storage)), {});
  auto arrival8 = loom::channel::bind<loom::channel::discipline::fifo>(arrival8_slots, loomt::index{1});
  auto* departure8_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile8);
  auto departure8_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure8_storage)), {});
  auto departure8 = loom::channel::bind<loom::channel::discipline::fifo>(departure8_slots, loomt::index{1});
  auto* arrival9_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile9);
  auto arrival9_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival9_storage)), {});
  auto arrival9 = loom::channel::bind<loom::channel::discipline::fifo>(arrival9_slots, loomt::index{1});
  auto* departure9_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile9);
  auto departure9_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure9_storage)), {});
  auto departure9 = loom::channel::bind<loom::channel::discipline::fifo>(departure9_slots, loomt::index{1});
  auto* arrival10_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile10);
  auto arrival10_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival10_storage)), {});
  auto arrival10 = loom::channel::bind<loom::channel::discipline::fifo>(arrival10_slots, loomt::index{1});
  auto* departure10_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile10);
  auto departure10_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure10_storage)), {});
  auto departure10 = loom::channel::bind<loom::channel::discipline::fifo>(departure10_slots, loomt::index{1});
  auto* arrival11_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile11);
  auto arrival11_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival11_storage)), {});
  auto arrival11 = loom::channel::bind<loom::channel::discipline::fifo>(arrival11_slots, loomt::index{1});
  auto* departure11_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile11);
  auto departure11_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure11_storage)), {});
  auto departure11 = loom::channel::bind<loom::channel::discipline::fifo>(departure11_slots, loomt::index{1});
  auto* arrival15_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile15);
  auto arrival15_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival15_storage)), {});
  auto arrival15 = loom::channel::bind<loom::channel::discipline::fifo>(arrival15_slots, loomt::index{1});
  auto* departure15_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile15);
  auto departure15_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure15_storage)), {});
  auto departure15 = loom::channel::bind<loom::channel::discipline::fifo>(departure15_slots, loomt::index{1});
  auto* arrival14_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile14);
  auto arrival14_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival14_storage)), {});
  auto arrival14 = loom::channel::bind<loom::channel::discipline::fifo>(arrival14_slots, loomt::index{1});
  auto* departure14_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile14);
  auto departure14_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure14_storage)), {});
  auto departure14 = loom::channel::bind<loom::channel::discipline::fifo>(departure14_slots, loomt::index{1});
  auto* arrival13_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile13);
  auto arrival13_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival13_storage)), {});
  auto arrival13 = loom::channel::bind<loom::channel::discipline::fifo>(arrival13_slots, loomt::index{1});
  auto* departure13_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile13);
  auto departure13_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure13_storage)), {});
  auto departure13 = loom::channel::bind<loom::channel::discipline::fifo>(departure13_slots, loomt::index{1});
  auto* arrival12_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile12);
  auto arrival12_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival12_storage)), {});
  auto arrival12 = loom::channel::bind<loom::channel::discipline::fifo>(arrival12_slots, loomt::index{1});
  auto* departure12_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile12);
  auto departure12_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure12_storage)), {});
  auto departure12 = loom::channel::bind<loom::channel::discipline::fifo>(departure12_slots, loomt::index{1});
  auto* arrival16_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile16);
  auto arrival16_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival16_storage)), {});
  auto arrival16 = loom::channel::bind<loom::channel::discipline::fifo>(arrival16_slots, loomt::index{1});
  auto* departure16_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile16);
  auto departure16_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure16_storage)), {});
  auto departure16 = loom::channel::bind<loom::channel::discipline::fifo>(departure16_slots, loomt::index{1});
  auto* arrival17_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile17);
  auto arrival17_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival17_storage)), {});
  auto arrival17 = loom::channel::bind<loom::channel::discipline::fifo>(arrival17_slots, loomt::index{1});
  auto* departure17_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile17);
  auto departure17_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure17_storage)), {});
  auto departure17 = loom::channel::bind<loom::channel::discipline::fifo>(departure17_slots, loomt::index{1});
  auto* arrival18_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile18);
  auto arrival18_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival18_storage)), {});
  auto arrival18 = loom::channel::bind<loom::channel::discipline::fifo>(arrival18_slots, loomt::index{1});
  auto* departure18_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile18);
  auto departure18_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure18_storage)), {});
  auto departure18 = loom::channel::bind<loom::channel::discipline::fifo>(departure18_slots, loomt::index{1});
  auto* arrival19_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile19);
  auto arrival19_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival19_storage)), {});
  auto arrival19 = loom::channel::bind<loom::channel::discipline::fifo>(arrival19_slots, loomt::index{1});
  auto* departure19_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile19);
  auto departure19_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure19_storage)), {});
  auto departure19 = loom::channel::bind<loom::channel::discipline::fifo>(departure19_slots, loomt::index{1});
  auto* arrival23_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile23);
  auto arrival23_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival23_storage)), {});
  auto arrival23 = loom::channel::bind<loom::channel::discipline::fifo>(arrival23_slots, loomt::index{1});
  auto* departure23_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile23);
  auto departure23_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure23_storage)), {});
  auto departure23 = loom::channel::bind<loom::channel::discipline::fifo>(departure23_slots, loomt::index{1});
  auto* arrival22_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile22);
  auto arrival22_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival22_storage)), {});
  auto arrival22 = loom::channel::bind<loom::channel::discipline::fifo>(arrival22_slots, loomt::index{1});
  auto* departure22_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile22);
  auto departure22_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure22_storage)), {});
  auto departure22 = loom::channel::bind<loom::channel::discipline::fifo>(departure22_slots, loomt::index{1});
  auto* arrival21_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile21);
  auto arrival21_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival21_storage)), {});
  auto arrival21 = loom::channel::bind<loom::channel::discipline::fifo>(arrival21_slots, loomt::index{1});
  auto* departure21_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile21);
  auto departure21_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure21_storage)), {});
  auto departure21 = loom::channel::bind<loom::channel::discipline::fifo>(departure21_slots, loomt::index{1});
  auto* arrival20_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile20);
  auto arrival20_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival20_storage)), {});
  auto arrival20 = loom::channel::bind<loom::channel::discipline::fifo>(arrival20_slots, loomt::index{1});
  auto* departure20_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile20);
  auto departure20_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure20_storage)), {});
  auto departure20 = loom::channel::bind<loom::channel::discipline::fifo>(departure20_slots, loomt::index{1});
  auto* arrival24_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile24);
  auto arrival24_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival24_storage)), {});
  auto arrival24 = loom::channel::bind<loom::channel::discipline::fifo>(arrival24_slots, loomt::index{1});
  auto* departure24_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile24);
  auto departure24_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure24_storage)), {});
  auto departure24 = loom::channel::bind<loom::channel::discipline::fifo>(departure24_slots, loomt::index{1});
  auto* arrival25_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile25);
  auto arrival25_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival25_storage)), {});
  auto arrival25 = loom::channel::bind<loom::channel::discipline::fifo>(arrival25_slots, loomt::index{1});
  auto* departure25_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile25);
  auto departure25_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure25_storage)), {});
  auto departure25 = loom::channel::bind<loom::channel::discipline::fifo>(departure25_slots, loomt::index{1});
  auto* arrival26_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile26);
  auto arrival26_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival26_storage)), {});
  auto arrival26 = loom::channel::bind<loom::channel::discipline::fifo>(arrival26_slots, loomt::index{1});
  auto* departure26_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile26);
  auto departure26_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure26_storage)), {});
  auto departure26 = loom::channel::bind<loom::channel::discipline::fifo>(departure26_slots, loomt::index{1});
  auto* arrival27_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile27);
  auto arrival27_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival27_storage)), {});
  auto arrival27 = loom::channel::bind<loom::channel::discipline::fifo>(arrival27_slots, loomt::index{1});
  auto* departure27_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile27);
  auto departure27_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure27_storage)), {});
  auto departure27 = loom::channel::bind<loom::channel::discipline::fifo>(departure27_slots, loomt::index{1});
  auto* arrival31_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile31);
  auto arrival31_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival31_storage)), {});
  auto arrival31 = loom::channel::bind<loom::channel::discipline::fifo>(arrival31_slots, loomt::index{1});
  auto* departure31_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile31);
  auto departure31_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure31_storage)), {});
  auto departure31 = loom::channel::bind<loom::channel::discipline::fifo>(departure31_slots, loomt::index{1});
  auto* arrival30_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile30);
  auto arrival30_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival30_storage)), {});
  auto arrival30 = loom::channel::bind<loom::channel::discipline::fifo>(arrival30_slots, loomt::index{1});
  auto* departure30_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile30);
  auto departure30_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure30_storage)), {});
  auto departure30 = loom::channel::bind<loom::channel::discipline::fifo>(departure30_slots, loomt::index{1});
  auto* arrival29_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile29);
  auto arrival29_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(arrival29_storage)), {});
  auto arrival29 = loom::channel::bind<loom::channel::discipline::fifo>(arrival29_slots, loomt::index{1});
  auto* departure29_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64}, tile29);
  auto departure29_slots = loom::buffer::view<1, 16>(reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(departure29_storage)), {});
  auto departure29 = loom::channel::bind<loom::channel::discipline::fifo>(departure29_slots, loomt::index{1});
  auto result0_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage0) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result0 = loom::channel::bind<loom::channel::discipline::fifo>(result0_slots, loomt::index{1});
  auto result2_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage2) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result2 = loom::channel::bind<loom::channel::discipline::fifo>(result2_slots, loomt::index{1});
  auto result4_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage4) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result4 = loom::channel::bind<loom::channel::discipline::fifo>(result4_slots, loomt::index{1});
  auto result6_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage6) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result6 = loom::channel::bind<loom::channel::discipline::fifo>(result6_slots, loomt::index{1});
  auto result8_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage9) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result8 = loom::channel::bind<loom::channel::discipline::fifo>(result8_slots, loomt::index{1});
  auto result10_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage11) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result10 = loom::channel::bind<loom::channel::discipline::fifo>(result10_slots, loomt::index{1});
  auto result12_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage12) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result12 = loom::channel::bind<loom::channel::discipline::fifo>(result12_slots, loomt::index{1});
  auto result14_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage14) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result14 = loom::channel::bind<loom::channel::discipline::fifo>(result14_slots, loomt::index{1});
  auto result16_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage16) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result16 = loom::channel::bind<loom::channel::discipline::fifo>(result16_slots, loomt::index{1});
  auto result18_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage18) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result18 = loom::channel::bind<loom::channel::discipline::fifo>(result18_slots, loomt::index{1});
  auto result20_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage20) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result20 = loom::channel::bind<loom::channel::discipline::fifo>(result20_slots, loomt::index{1});
  auto result22_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage22) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result22 = loom::channel::bind<loom::channel::discipline::fifo>(result22_slots, loomt::index{1});
  auto result24_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage25) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result24 = loom::channel::bind<loom::channel::discipline::fifo>(result24_slots, loomt::index{1});
  auto result26_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage27) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result26 = loom::channel::bind<loom::channel::discipline::fifo>(result26_slots, loomt::index{1});
  auto result28_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage28) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result28 = loom::channel::bind<loom::channel::discipline::fifo>(result28_slots, loomt::index{1});
  auto result30_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage30) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto result30 = loom::channel::bind<loom::channel::discipline::fifo>(result30_slots, loomt::index{1});
  auto probability0_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage0) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability0 = loom::channel::bind<loom::channel::discipline::fifo>(probability0_slots, loomt::index{1});
  auto* sequence0_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile0);
  auto probability2_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage2) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability2 = loom::channel::bind<loom::channel::discipline::fifo>(probability2_slots, loomt::index{1});
  auto* sequence2_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile2);
  auto probability4_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage4) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability4 = loom::channel::bind<loom::channel::discipline::fifo>(probability4_slots, loomt::index{1});
  auto* sequence4_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile4);
  auto probability6_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage6) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability6 = loom::channel::bind<loom::channel::discipline::fifo>(probability6_slots, loomt::index{1});
  auto* sequence6_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile6);
  auto probability8_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage8) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability8 = loom::channel::bind<loom::channel::discipline::fifo>(probability8_slots, loomt::index{1});
  auto* sequence8_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile8);
  auto probability10_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage10) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability10 = loom::channel::bind<loom::channel::discipline::fifo>(probability10_slots, loomt::index{1});
  auto* sequence10_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile10);
  auto probability12_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage12) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability12 = loom::channel::bind<loom::channel::discipline::fifo>(probability12_slots, loomt::index{1});
  auto* sequence12_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile12);
  auto probability14_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage14) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability14 = loom::channel::bind<loom::channel::discipline::fifo>(probability14_slots, loomt::index{1});
  auto* sequence14_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile14);
  auto probability16_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage16) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability16 = loom::channel::bind<loom::channel::discipline::fifo>(probability16_slots, loomt::index{1});
  auto* sequence16_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile16);
  auto probability18_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage18) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability18 = loom::channel::bind<loom::channel::discipline::fifo>(probability18_slots, loomt::index{1});
  auto* sequence18_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile18);
  auto probability20_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage20) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability20 = loom::channel::bind<loom::channel::discipline::fifo>(probability20_slots, loomt::index{1});
  auto* sequence20_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile20);
  auto probability22_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage22) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability22 = loom::channel::bind<loom::channel::discipline::fifo>(probability22_slots, loomt::index{1});
  auto* sequence22_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile22);
  auto probability24_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage24) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability24 = loom::channel::bind<loom::channel::discipline::fifo>(probability24_slots, loomt::index{1});
  auto* sequence24_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile24);
  auto probability26_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage26) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability26 = loom::channel::bind<loom::channel::discipline::fifo>(probability26_slots, loomt::index{1});
  auto* sequence26_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile26);
  auto probability28_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage28) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability28 = loom::channel::bind<loom::channel::discipline::fifo>(probability28_slots, loomt::index{1});
  auto* sequence28_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile28);
  auto probability30_slots = loom::buffer::view<1, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage30) + static_cast<unsigned long long>(loomt::offset{10624})), {});
  auto probability30 = loom::channel::bind<loom::channel::discipline::fifo>(probability30_slots, loomt::index{1});
  auto* sequence30_storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{512}, tile30);
  auto query_output0_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage0) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output0 = loom::channel::bind<loom::channel::discipline::fifo>(query_output0_slots, loomt::index{2});
  auto query_output2_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage2) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output2 = loom::channel::bind<loom::channel::discipline::fifo>(query_output2_slots, loomt::index{2});
  auto query_output4_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage4) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output4 = loom::channel::bind<loom::channel::discipline::fifo>(query_output4_slots, loomt::index{2});
  auto query_output6_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage6) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output6 = loom::channel::bind<loom::channel::discipline::fifo>(query_output6_slots, loomt::index{2});
  auto query_output8_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage9) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output8 = loom::channel::bind<loom::channel::discipline::fifo>(query_output8_slots, loomt::index{2});
  auto query_output10_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage11) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output10 = loom::channel::bind<loom::channel::discipline::fifo>(query_output10_slots, loomt::index{2});
  auto query_output12_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage12) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output12 = loom::channel::bind<loom::channel::discipline::fifo>(query_output12_slots, loomt::index{2});
  auto query_output14_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage14) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output14 = loom::channel::bind<loom::channel::discipline::fifo>(query_output14_slots, loomt::index{2});
  auto query_output16_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage16) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output16 = loom::channel::bind<loom::channel::discipline::fifo>(query_output16_slots, loomt::index{2});
  auto query_output18_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage18) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output18 = loom::channel::bind<loom::channel::discipline::fifo>(query_output18_slots, loomt::index{2});
  auto query_output20_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage20) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output20 = loom::channel::bind<loom::channel::discipline::fifo>(query_output20_slots, loomt::index{2});
  auto query_output22_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage22) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output22 = loom::channel::bind<loom::channel::discipline::fifo>(query_output22_slots, loomt::index{2});
  auto query_output24_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage25) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output24 = loom::channel::bind<loom::channel::discipline::fifo>(query_output24_slots, loomt::index{2});
  auto query_output26_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage27) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output26 = loom::channel::bind<loom::channel::discipline::fifo>(query_output26_slots, loomt::index{2});
  auto query_output28_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage28) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output28 = loom::channel::bind<loom::channel::discipline::fifo>(query_output28_slots, loomt::index{2});
  auto query_output30_slots = loom::buffer::view<2, 4096>(reinterpret_cast<std::bfloat16_t*>(reinterpret_cast<unsigned char*>(storage30) + static_cast<unsigned long long>(loomt::offset{39872})), {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto query_output30 = loom::channel::bind<loom::channel::discipline::fifo>(query_output30_slots, loomt::index{2});
  auto normal_output8 = loom::channel::bind<loom::channel::discipline::fifo>(query_output8_slots, loomt::index{2});
  auto normal_output10 = loom::channel::bind<loom::channel::discipline::fifo>(query_output10_slots, loomt::index{2});
  auto normal_output24 = loom::channel::bind<loom::channel::discipline::fifo>(query_output24_slots, loomt::index{2});
  auto normal_output26 = loom::channel::bind<loom::channel::discipline::fifo>(query_output26_slots, loomt::index{2});
  auto* dimensions1_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile1);
  auto* dimensions3_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile3);
  auto* dimensions5_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile5);
  auto* dimensions7_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile7);
  auto* dimensions8_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile8);
  auto* dimensions10_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile10);
  auto* dimensions13_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile13);
  auto* dimensions15_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile15);
  auto* dimensions17_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile17);
  auto* dimensions19_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile19);
  auto* dimensions21_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile21);
  auto* dimensions23_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile23);
  auto* dimensions24_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile24);
  auto* dimensions26_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile26);
  auto* dimensions29_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile29);
  auto* dimensions31_storage = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{32}, tile31);
  auto* gather_cache0 = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{192}, tile0);
  auto* gather_cache9 = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{192}, tile9);
  auto* gather_cache18 = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{192}, tile18);
  auto* gather_cache27 = loom::buffer::alloca<loom::memory_space::workgroup, 16>(loomt::offset{192}, tile27);
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
      auto reservation_2 = loom::channel::reserve(weights0);
      loom::channel::copy(fanout.first, reservation_2.write);
      auto reservation_3 = loom::channel::reserve(weights1);
      loom::channel::copy(fanout.remaining.first, reservation_3.write);
      auto reservation_4 = loom::channel::reserve(weights2);
      loom::channel::copy(fanout.remaining.remaining.first, reservation_4.write);
      auto reservation_5 = loom::channel::reserve(weights3);
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
      auto reservation_7 = loom::channel::reserve(weights4);
      loom::channel::copy(fanout_2.first, reservation_7.write);
      auto reservation_8 = loom::channel::reserve(weights5);
      loom::channel::copy(fanout_2.remaining.first, reservation_8.write);
      auto reservation_9 = loom::channel::reserve(weights6);
      loom::channel::copy(fanout_2.remaining.remaining.first, reservation_9.write);
      auto reservation_10 = loom::channel::reserve(weights7);
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
      auto reservation_12 = loom::channel::reserve(weights8);
      loom::channel::copy(fanout_3.first, reservation_12.write);
      auto reservation_13 = loom::channel::reserve(weights9);
      loom::channel::copy(fanout_3.remaining.first, reservation_13.write);
      auto reservation_14 = loom::channel::reserve(weights10);
      loom::channel::copy(fanout_3.remaining.remaining.first, reservation_14.write);
      auto reservation_15 = loom::channel::reserve(weights11);
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
      auto reservation_17 = loom::channel::reserve(weights12);
      loom::channel::copy(fanout_4.first, reservation_17.write);
      auto reservation_18 = loom::channel::reserve(weights13);
      loom::channel::copy(fanout_4.remaining.first, reservation_18.write);
      auto reservation_19 = loom::channel::reserve(weights14);
      loom::channel::copy(fanout_4.remaining.remaining.first, reservation_19.write);
      auto reservation_20 = loom::channel::reserve(weights15);
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
      auto reservation_22 = loom::channel::reserve(weights16);
      loom::channel::copy(fanout_5.first, reservation_22.write);
      auto reservation_23 = loom::channel::reserve(weights17);
      loom::channel::copy(fanout_5.remaining.first, reservation_23.write);
      auto reservation_24 = loom::channel::reserve(weights18);
      loom::channel::copy(fanout_5.remaining.remaining.first, reservation_24.write);
      auto reservation_25 = loom::channel::reserve(weights19);
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
      auto reservation_27 = loom::channel::reserve(weights20);
      loom::channel::copy(fanout_6.first, reservation_27.write);
      auto reservation_28 = loom::channel::reserve(weights21);
      loom::channel::copy(fanout_6.remaining.first, reservation_28.write);
      auto reservation_29 = loom::channel::reserve(weights22);
      loom::channel::copy(fanout_6.remaining.remaining.first, reservation_29.write);
      auto reservation_30 = loom::channel::reserve(weights23);
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
      auto reservation_32 = loom::channel::reserve(weights24);
      loom::channel::copy(fanout_7.first, reservation_32.write);
      auto reservation_33 = loom::channel::reserve(weights25);
      loom::channel::copy(fanout_7.remaining.first, reservation_33.write);
      auto reservation_34 = loom::channel::reserve(weights26);
      loom::channel::copy(fanout_7.remaining.remaining.first, reservation_34.write);
      auto reservation_35 = loom::channel::reserve(weights27);
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
      auto reservation_37 = loom::channel::reserve(weights28);
      loom::channel::copy(fanout_8.first, reservation_37.write);
      auto reservation_38 = loom::channel::reserve(weights29);
      loom::channel::copy(fanout_8.remaining.first, reservation_38.write);
      auto reservation_39 = loom::channel::reserve(weights30);
      loom::channel::copy(fanout_8.remaining.remaining.first, reservation_39.write);
      auto reservation_40 = loom::channel::reserve(weights31);
      loom::channel::copy(fanout_8.remaining.remaining.second, reservation_40.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_16 = 0; record_16 < 5190; ++record_16) {
      auto acquisition_9 = loom::channel::acquire(activations0);
      auto fanout_9 = loom::channel::fanout<7>(acquisition_9.read);
      auto reservation_41 = loom::channel::reserve(activations4);
      loom::channel::copy(fanout_9.first, reservation_41.write);
      auto reservation_42 = loom::channel::reserve(activations8);
      loom::channel::copy(fanout_9.remaining.first, reservation_42.write);
      auto reservation_43 = loom::channel::reserve(activations12);
      loom::channel::copy(fanout_9.remaining.remaining.first, reservation_43.write);
      auto reservation_44 = loom::channel::reserve(activations16);
      loom::channel::copy(fanout_9.remaining.remaining.remaining.first, reservation_44.write);
      auto reservation_45 = loom::channel::reserve(activations20);
      loom::channel::copy(fanout_9.remaining.remaining.remaining.remaining.first, reservation_45.write);
      auto reservation_46 = loom::channel::reserve(activations24);
      loom::channel::copy(fanout_9.remaining.remaining.remaining.remaining.remaining.first, reservation_46.write);
      auto reservation_47 = loom::channel::reserve(activations28);
      loom::channel::copy(fanout_9.remaining.remaining.remaining.remaining.remaining.second, reservation_47.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_17 = 0; record_17 < 4294; ++record_17) {
      auto acquisition_10 = loom::channel::acquire(activations9);
      auto fanout_10 = loom::channel::fanout<7>(acquisition_10.read);
      auto reservation_48 = loom::channel::reserve(activations1);
      loom::channel::copy(fanout_10.first, reservation_48.write);
      auto reservation_49 = loom::channel::reserve(activations5);
      loom::channel::copy(fanout_10.remaining.first, reservation_49.write);
      auto reservation_50 = loom::channel::reserve(activations13);
      loom::channel::copy(fanout_10.remaining.remaining.first, reservation_50.write);
      auto reservation_51 = loom::channel::reserve(activations17);
      loom::channel::copy(fanout_10.remaining.remaining.remaining.first, reservation_51.write);
      auto reservation_52 = loom::channel::reserve(activations21);
      loom::channel::copy(fanout_10.remaining.remaining.remaining.remaining.first, reservation_52.write);
      auto reservation_53 = loom::channel::reserve(activations25);
      loom::channel::copy(fanout_10.remaining.remaining.remaining.remaining.remaining.first, reservation_53.write);
      auto reservation_54 = loom::channel::reserve(activations29);
      loom::channel::copy(fanout_10.remaining.remaining.remaining.remaining.remaining.second, reservation_54.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_18 = 0; record_18 < 5190; ++record_18) {
      auto acquisition_11 = loom::channel::acquire(activations18);
      auto fanout_11 = loom::channel::fanout<7>(acquisition_11.read);
      auto reservation_55 = loom::channel::reserve(activations2);
      loom::channel::copy(fanout_11.first, reservation_55.write);
      auto reservation_56 = loom::channel::reserve(activations6);
      loom::channel::copy(fanout_11.remaining.first, reservation_56.write);
      auto reservation_57 = loom::channel::reserve(activations10);
      loom::channel::copy(fanout_11.remaining.remaining.first, reservation_57.write);
      auto reservation_58 = loom::channel::reserve(activations14);
      loom::channel::copy(fanout_11.remaining.remaining.remaining.first, reservation_58.write);
      auto reservation_59 = loom::channel::reserve(activations22);
      loom::channel::copy(fanout_11.remaining.remaining.remaining.remaining.first, reservation_59.write);
      auto reservation_60 = loom::channel::reserve(activations26);
      loom::channel::copy(fanout_11.remaining.remaining.remaining.remaining.remaining.first, reservation_60.write);
      auto reservation_61 = loom::channel::reserve(activations30);
      loom::channel::copy(fanout_11.remaining.remaining.remaining.remaining.remaining.second, reservation_61.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    for (int record_19 = 0; record_19 < 4294; ++record_19) {
      auto acquisition_12 = loom::channel::acquire(activations27);
      auto fanout_12 = loom::channel::fanout<7>(acquisition_12.read);
      auto reservation_62 = loom::channel::reserve(activations3);
      loom::channel::copy(fanout_12.first, reservation_62.write);
      auto reservation_63 = loom::channel::reserve(activations7);
      loom::channel::copy(fanout_12.remaining.first, reservation_63.write);
      auto reservation_64 = loom::channel::reserve(activations11);
      loom::channel::copy(fanout_12.remaining.remaining.first, reservation_64.write);
      auto reservation_65 = loom::channel::reserve(activations15);
      loom::channel::copy(fanout_12.remaining.remaining.remaining.first, reservation_65.write);
      auto reservation_66 = loom::channel::reserve(activations19);
      loom::channel::copy(fanout_12.remaining.remaining.remaining.remaining.first, reservation_66.write);
      auto reservation_67 = loom::channel::reserve(activations23);
      loom::channel::copy(fanout_12.remaining.remaining.remaining.remaining.remaining.first, reservation_67.write);
      auto reservation_68 = loom::channel::reserve(activations31);
      loom::channel::copy(fanout_12.remaining.remaining.remaining.remaining.remaining.second, reservation_68.write);
    }
  }
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_ingress_root(loom::buffer::root(sequence0_storage), loomt::offset{0}, loom::buffer::root(storage0), loomt::offset{10624}, loom::buffer::root(storage1), loom::buffer::root(gather_cache0), activations, activations0, loomt::index{0}, loomt::index{1}, constants, embedding, state, arrival0, departure0, query_output0, output, weights0, loomt::offset{18816}, probability0, result0);
  }
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(true, loom::buffer::root(dimensions1_storage), loomt::offset{0}, loom::buffer::root(storage1), loomt::offset{10624}, loom::buffer::root(storage0), activations1, loomt::index{0}, loomt::index{1}, arrival1, arrival0, departure0, departure1, query_output0, weights1, loomt::offset{18816}, probability0, result0, loomt::index{0}, loomt::index{0}, loomt::index{0}, loomt::index{0});
  }
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence2_storage), loomt::offset{0}, loom::buffer::root(storage2), loomt::offset{10624}, loom::buffer::root(storage3), activations2, loomt::index{0}, loomt::index{1}, arrival2, arrival1, departure1, departure2, query_output2, output, state, weights2, loomt::offset{18816}, probability2, result2, loomt::index{0}, loomt::index{0}, loomt::offset{262144}, loomt::offset{1073152}, loomt::index{0}, loomt::offset{663552}, loomt::offset{139264}, loomt::offset{270464}, loomt::offset{1859584}, loomt::offset{2760704}, loomt::offset{1351680});
  }
  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions3_storage), loomt::offset{0}, loom::buffer::root(storage3), loomt::offset{10624}, loom::buffer::root(storage2), activations3, loomt::index{0}, loomt::index{1}, arrival3, arrival2, departure2, departure3, query_output2, weights3, loomt::offset{18816}, probability2, result2, loomt::index{0}, loomt::index{64}, loomt::index{0}, loomt::index{0});
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence4_storage), loomt::offset{0}, loom::buffer::root(storage4), loomt::offset{10624}, loom::buffer::root(storage5), activations4, loomt::index{0}, loomt::index{1}, arrival4, arrival5, departure5, departure4, query_output4, output, state, weights4, loomt::offset{18816}, probability4, result4, loomt::index{512}, loomt::index{1}, loomt::offset{16384}, loomt::offset{1056832}, loomt::index{128}, loomt::offset{794624}, loomt::offset{9216}, loomt::offset{271360}, loomt::offset{1875968}, loomt::offset{2629888}, loomt::offset{1384448});
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions5_storage), loomt::offset{0}, loom::buffer::root(storage5), loomt::offset{10624}, loom::buffer::root(storage4), activations5, loomt::index{0}, loomt::index{1}, arrival5, arrival6, departure6, departure5, query_output4, weights5, loomt::offset{18816}, probability4, result4, loomt::index{512}, loomt::index{4}, loomt::index{1}, loomt::index{128});
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence6_storage), loomt::offset{0}, loom::buffer::root(storage6), loomt::offset{10624}, loom::buffer::root(storage7), activations6, loomt::index{0}, loomt::index{1}, arrival6, arrival7, departure7, departure6, query_output6, output, state, weights6, loomt::offset{18816}, probability6, result6, loomt::index{512}, loomt::index{1}, loomt::offset{278528}, loomt::offset{1073216}, loomt::index{128}, loomt::offset{925696}, loomt::offset{140288}, loomt::offset{271488}, loomt::offset{1892352}, loomt::offset{2760960}, loomt::offset{1417216});
  }
  [[loom::strand(loomt::indices{loomt::index{1}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions7_storage), loomt::offset{0}, loom::buffer::root(storage7), loomt::offset{10624}, loom::buffer::root(storage6), activations7, loomt::index{0}, loomt::index{1}, arrival7, arrival3, departure3, departure7, query_output6, weights7, loomt::offset{18816}, probability6, result6, loomt::index{512}, loomt::index{68}, loomt::index{1}, loomt::index{128});
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_epilogue(loom::buffer::root(sequence8_storage), loomt::offset{0}, loom::buffer::root(dimensions8_storage), loom::buffer::root(storage8), loomt::offset{10624}, loom::buffer::root(storage9), activations8, loomt::index{0}, loomt::index{1}, arrival8, arrival4, departure4, departure8, normal_output8, weights8, loomt::offset{18816}, query_output8, probability8, loomt::index{1024}, loomt::index{8}, loomt::index{2}, loomt::index{256});
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value_ingress(loom::buffer::root(storage9), loomt::offset{0}, loomt::offset{10624}, loom::buffer::root(storage8), loom::buffer::root(gather_cache9), activations, activations9, loomt::index{0}, loomt::index{1}, constants, embedding, state, arrival9, arrival8, departure8, departure9, normal_output8, output, weights9, loomt::offset{18816}, query_output8, probability8, loom::buffer::root(sequence8_storage), result8, loomt::index{1024}, loomt::index{32}, loomt::index{2}, loomt::offset{2695168}, loomt::offset{32768}, loomt::offset{1056896}, loomt::offset{1064960}, loomt::offset{1327104}, loomt::offset{1851392}, loomt::offset{2695168}, loomt::offset{2899968}, loomt::index{256}, loomt::offset{533504}, loomt::offset{10240}, loomt::offset{272384}, loomt::offset{1908736}, loomt::offset{2630144}, loomt::offset{1449984});
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_normalize(loom::buffer::root(sequence10_storage), loomt::offset{0}, loom::buffer::root(dimensions10_storage), loom::buffer::root(storage10), loomt::offset{10624}, loom::buffer::root(storage11), activations10, loomt::index{0}, loomt::index{1}, arrival10, arrival9, departure9, departure10, normal_output10, weights10, loomt::offset{18816}, probability10, loomt::index{1024}, loomt::index{72}, loomt::index{2}, loomt::index{256});
  }
  [[loom::strand(loomt::indices{loomt::index{2}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value_egress(loom::buffer::root(storage11), loomt::offset{0}, loomt::offset{10624}, loom::buffer::root(storage10), activations11, loomt::index{0}, loomt::index{1}, arrival11, arrival10, departure10, departure11, normal_output10, output, state, weights11, loomt::offset{18816}, query_output10, probability10, result10, loomt::index{1024}, loomt::index{2}, loomt::offset{294912}, loomt::offset{1073280}, loomt::index{256}, loomt::offset{664576}, loomt::offset{141312}, loomt::offset{272512}, loomt::offset{1925120}, loomt::offset{2761216}, loomt::offset{1482752});
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence12_storage), loomt::offset{0}, loom::buffer::root(storage12), loomt::offset{10624}, loom::buffer::root(storage13), activations12, loomt::index{0}, loomt::index{1}, arrival12, arrival13, departure13, departure12, query_output12, output, state, weights12, loomt::offset{18816}, probability12, result12, loomt::index{1536}, loomt::index{3}, loomt::offset{49152}, loomt::offset{1056960}, loomt::index{384}, loomt::offset{795648}, loomt::offset{11264}, loomt::offset{273408}, loomt::offset{1941504}, loomt::offset{2630400}, loomt::offset{1515520});
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions13_storage), loomt::offset{0}, loom::buffer::root(storage13), loomt::offset{10624}, loom::buffer::root(storage12), activations13, loomt::index{0}, loomt::index{1}, arrival13, arrival14, departure14, departure13, query_output12, weights13, loomt::offset{18816}, probability12, result12, loomt::index{1536}, loomt::index{12}, loomt::index{3}, loomt::index{384});
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence14_storage), loomt::offset{0}, loom::buffer::root(storage14), loomt::offset{10624}, loom::buffer::root(storage15), activations14, loomt::index{0}, loomt::index{1}, arrival14, arrival15, departure15, departure14, query_output14, output, state, weights14, loomt::offset{18816}, probability14, result14, loomt::index{1536}, loomt::index{3}, loomt::offset{311296}, loomt::offset{1073344}, loomt::index{384}, loomt::offset{926720}, loomt::offset{142336}, loomt::offset{273536}, loomt::offset{1957888}, loomt::offset{2761472}, loomt::offset{1548288});
  }
  [[loom::strand(loomt::indices{loomt::index{3}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions15_storage), loomt::offset{0}, loom::buffer::root(storage15), loomt::offset{10624}, loom::buffer::root(storage14), activations15, loomt::index{0}, loomt::index{1}, arrival15, arrival11, departure11, departure15, query_output14, weights15, loomt::offset{18816}, probability14, result14, loomt::index{1536}, loomt::index{76}, loomt::index{3}, loomt::index{384});
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence16_storage), loomt::offset{0}, loom::buffer::root(storage16), loomt::offset{10624}, loom::buffer::root(storage17), activations16, loomt::index{0}, loomt::index{1}, arrival16, arrival12, departure12, departure16, query_output16, output, state, weights16, loomt::offset{18816}, probability16, result16, loomt::index{2048}, loomt::index{4}, loomt::offset{65536}, loomt::offset{1057024}, loomt::index{512}, loomt::offset{534528}, loomt::offset{12288}, loomt::offset{274432}, loomt::offset{1974272}, loomt::offset{2630656}, loomt::offset{1581056});
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions17_storage), loomt::offset{0}, loom::buffer::root(storage17), loomt::offset{10624}, loom::buffer::root(storage16), activations17, loomt::index{0}, loomt::index{1}, arrival17, arrival16, departure16, departure17, query_output16, weights17, loomt::offset{18816}, probability16, result16, loomt::index{2048}, loomt::index{16}, loomt::index{4}, loomt::index{512});
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_ingress(loom::buffer::root(sequence18_storage), loomt::offset{0}, loom::buffer::root(storage18), loomt::offset{10624}, loom::buffer::root(storage19), loom::buffer::root(gather_cache18), activations, activations18, loomt::index{0}, loomt::index{1}, constants, embedding, state, arrival18, arrival17, departure17, departure18, query_output18, output, weights18, loomt::offset{18816}, probability18, result18);
  }
  [[loom::strand(loomt::indices{loomt::index{4}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(true, loom::buffer::root(dimensions19_storage), loomt::offset{0}, loom::buffer::root(storage19), loomt::offset{10624}, loom::buffer::root(storage18), activations19, loomt::index{0}, loomt::index{1}, arrival19, arrival18, departure18, departure19, query_output18, weights19, loomt::offset{18816}, probability18, result18, loomt::index{2048}, loomt::index{80}, loomt::index{4}, loomt::index{512});
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence20_storage), loomt::offset{0}, loom::buffer::root(storage20), loomt::offset{10624}, loom::buffer::root(storage21), activations20, loomt::index{0}, loomt::index{1}, arrival20, arrival21, departure21, departure20, query_output20, output, state, weights20, loomt::offset{18816}, probability20, result20, loomt::index{2560}, loomt::index{5}, loomt::offset{81920}, loomt::offset{1057088}, loomt::index{640}, loomt::offset{796672}, loomt::offset{13312}, loomt::offset{275456}, loomt::offset{2007040}, loomt::offset{2630912}, loomt::offset{1646592});
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions21_storage), loomt::offset{0}, loom::buffer::root(storage21), loomt::offset{10624}, loom::buffer::root(storage20), activations21, loomt::index{0}, loomt::index{1}, arrival21, arrival22, departure22, departure21, query_output20, weights21, loomt::offset{18816}, probability20, result20, loomt::index{2560}, loomt::index{20}, loomt::index{5}, loomt::index{640});
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence22_storage), loomt::offset{0}, loom::buffer::root(storage22), loomt::offset{10624}, loom::buffer::root(storage23), activations22, loomt::index{0}, loomt::index{1}, arrival22, arrival23, departure23, departure22, query_output22, output, state, weights22, loomt::offset{18816}, probability22, result22, loomt::index{2560}, loomt::index{5}, loomt::offset{344064}, loomt::offset{1073472}, loomt::index{640}, loomt::offset{927744}, loomt::offset{144384}, loomt::offset{275584}, loomt::offset{2023424}, loomt::offset{2761984}, loomt::offset{1679360});
  }
  [[loom::strand(loomt::indices{loomt::index{5}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions23_storage), loomt::offset{0}, loom::buffer::root(storage23), loomt::offset{10624}, loom::buffer::root(storage22), activations23, loomt::index{0}, loomt::index{1}, arrival23, arrival19, departure19, departure23, query_output22, weights23, loomt::offset{18816}, probability22, result22, loomt::index{2560}, loomt::index{84}, loomt::index{5}, loomt::index{640});
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_normalize(loom::buffer::root(sequence24_storage), loomt::offset{0}, loom::buffer::root(dimensions24_storage), loom::buffer::root(storage24), loomt::offset{10624}, loom::buffer::root(storage25), activations24, loomt::index{0}, loomt::index{1}, arrival24, arrival20, departure20, departure24, normal_output24, weights24, loomt::offset{18816}, probability24, loomt::index{3072}, loomt::index{24}, loomt::index{6}, loomt::index{768});
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value_egress(loom::buffer::root(storage25), loomt::offset{0}, loomt::offset{10624}, loom::buffer::root(storage24), activations25, loomt::index{0}, loomt::index{1}, arrival25, arrival24, departure24, departure25, normal_output24, output, state, weights25, loomt::offset{18816}, query_output24, probability24, result24, loomt::index{3072}, loomt::index{6}, loomt::offset{98304}, loomt::offset{1057152}, loomt::index{768}, loomt::offset{535552}, loomt::offset{14336}, loomt::offset{276480}, loomt::offset{2039808}, loomt::offset{2631168}, loomt::offset{1712128});
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_epilogue(loom::buffer::root(sequence26_storage), loomt::offset{0}, loom::buffer::root(dimensions26_storage), loom::buffer::root(storage26), loomt::offset{10624}, loom::buffer::root(storage27), activations26, loomt::index{0}, loomt::index{1}, arrival26, arrival25, departure25, departure26, normal_output26, weights26, loomt::offset{18816}, query_output26, probability26, loomt::index{3072}, loomt::index{88}, loomt::index{6}, loomt::index{768});
  }
  [[loom::strand(loomt::indices{loomt::index{6}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value_ingress(loom::buffer::root(storage27), loomt::offset{0}, loomt::offset{10624}, loom::buffer::root(storage26), loom::buffer::root(gather_cache27), activations, activations27, loomt::index{0}, loomt::index{1}, constants, embedding, state, arrival27, arrival26, departure26, departure27, normal_output26, output, weights27, loomt::offset{18816}, query_output26, probability26, loom::buffer::root(sequence26_storage), result26, loomt::index{3072}, loomt::index{96}, loomt::index{6}, loomt::offset{2826240}, loomt::offset{360448}, loomt::offset{1073536}, loomt::offset{1081344}, loomt::offset{1343488}, loomt::offset{1867776}, loomt::offset{2826240}, loomt::offset{2916352}, loomt::index{768}, loomt::offset{666624}, loomt::offset{145408}, loomt::offset{276608}, loomt::offset{2056192}, loomt::offset{2762240}, loomt::offset{1744896});
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{2}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress_terminal(loom::buffer::root(sequence28_storage), loomt::offset{0}, loom::buffer::root(storage28), loomt::offset{10624}, loom::buffer::root(storage29), activations28, loomt::index{0}, loomt::index{1}, arrival29, departure29, query_output28, output, state, weights28, loomt::offset{18816}, probability28, result28);
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{3}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions29_storage), loomt::offset{0}, loom::buffer::root(storage29), loomt::offset{10624}, loom::buffer::root(storage28), activations29, loomt::index{0}, loomt::index{1}, arrival29, arrival30, departure30, departure29, query_output28, weights29, loomt::offset{18816}, probability28, result28, loomt::index{3584}, loomt::index{28}, loomt::index{7}, loomt::index{896});
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{4}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_query_key_egress(loom::buffer::root(sequence30_storage), loomt::offset{0}, loom::buffer::root(storage30), loomt::offset{10624}, loom::buffer::root(storage31), activations30, loomt::index{0}, loomt::index{1}, arrival30, arrival31, departure31, departure30, query_output30, output, state, weights30, loomt::offset{18816}, probability30, result30, loomt::index{3584}, loomt::index{7}, loomt::offset{376832}, loomt::offset{1073600}, loomt::index{896}, loomt::offset{928768}, loomt::offset{146432}, loomt::offset{277632}, loomt::offset{2088960}, loomt::offset{2762496}, loomt::offset{1810432});
  }
  [[loom::strand(loomt::indices{loomt::index{7}, loomt::index{5}}, loomt::indices{loomt::index{1}, loomt::index{1}}, loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    qwen3_embedding_probability_value(false, loom::buffer::root(dimensions31_storage), loomt::offset{0}, loom::buffer::root(storage31), loomt::offset{10624}, loom::buffer::root(storage30), activations31, loomt::index{0}, loomt::index{1}, arrival31, arrival27, departure27, departure31, query_output30, weights31, loomt::offset{18816}, probability30, result30, loomt::index{3584}, loomt::index{92}, loomt::index{7}, loomt::index{896});
  }
}
