// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

// Explicit worker-provider ABI. Phase composition supplies all storage,
// channels, physical row ownership, and runtime layer/group values.

#ifndef QWEN3_EMBEDDING_ARRAY_WORKERS_H_
#define QWEN3_EMBEDDING_ARRAY_WORKERS_H_

#include "array_resources.h"

[[loom::symbol("qwen3_embedding.query_key_ingress_root"), loom::force_inline]]
void qwen3_embedding_query_key_ingress_root(
    Bytes sequence_storage, loomt::offset origin, Bytes local_storage,
    loomt::offset result_offset, Bytes peer_storage, Bytes gather_storage,
    Bytes activations, ActivationStream activation_stream, loomt::index zero,
    loomt::index one, Bytes constants, Bytes embedding, Bytes state,
    BarrierStream incoming_arrival, BarrierStream outgoing_departure,
    FeatureStream projection_stream, Bytes output, WeightStream weight_stream,
    loomt::offset activation_offset, FeatureStream probability_stream,
    FeatureStream result_stream, BarrierStream phase_complete,
    loomt::index layer, loomt::index dense_group);

[[loom::symbol("qwen3_embedding.query_key_ingress_root.attention"),
  loom::force_inline]]
void qwen3_embedding_query_key_ingress_root_attention(
    Bytes sequence_storage, loomt::offset origin, Bytes local_storage,
    loomt::offset result_offset, Bytes peer_storage, Bytes gather_storage,
    Bytes activations, ActivationStream activation_stream, loomt::index zero,
    loomt::index one, Bytes constants, Bytes embedding, Bytes state,
    BarrierStream incoming_arrival, BarrierStream outgoing_departure,
    FeatureStream projection_stream, Bytes output, WeightStream weight_stream,
    loomt::offset activation_offset, FeatureStream probability_stream,
    FeatureStream result_stream, BarrierStream phase_complete,
    loomt::index layer);

[[loom::symbol("qwen3_embedding.query_key_egress"), loom::force_inline]]
void qwen3_embedding_query_key_egress(
    Bytes sequence_storage, loomt::offset origin, Bytes local_storage,
    loomt::offset result_offset, Bytes peer_storage,
    ActivationStream activation_stream, loomt::index zero, loomt::index one,
    BarrierStream incoming_arrival, BarrierStream outgoing_arrival,
    BarrierStream incoming_departure, BarrierStream outgoing_departure,
    FeatureStream projection_stream, Bytes output, Bytes state,
    WeightStream weight_stream, loomt::offset activation_offset,
    FeatureStream probability_stream, FeatureStream result_stream,
    loomt::index column_offset, loomt::index row_column,
    loomt::offset finish_initial, loomt::offset normalized_initial,
    loomt::index skip_column, loomt::offset query_initial,
    loomt::offset key_initial, loomt::offset value_initial,
    loomt::offset mlp_initial, loomt::offset residual_initial,
    loomt::offset output_initial, loomt::index layer, loomt::index dense_group);

[[loom::symbol("qwen3_embedding.query_key_egress.attention"),
  loom::force_inline]]
void qwen3_embedding_query_key_egress_attention(
    Bytes sequence_storage, loomt::offset origin, Bytes local_storage,
    loomt::offset result_offset, Bytes peer_storage,
    ActivationStream activation_stream, loomt::index zero, loomt::index one,
    BarrierStream incoming_arrival, BarrierStream outgoing_arrival,
    BarrierStream incoming_departure, BarrierStream outgoing_departure,
    FeatureStream projection_stream, Bytes output, Bytes state,
    WeightStream weight_stream, loomt::offset activation_offset,
    FeatureStream probability_stream, FeatureStream result_stream,
    loomt::index column_offset, loomt::index row_column,
    loomt::offset finish_initial, loomt::offset normalized_initial,
    loomt::index skip_column, loomt::offset query_initial,
    loomt::offset key_initial, loomt::offset value_initial,
    loomt::offset mlp_initial, loomt::offset residual_initial,
    loomt::offset output_initial, loomt::index layer);

[[loom::symbol("qwen3_embedding.probability_value"), loom::force_inline]]
void qwen3_embedding_probability_value(
    bool normalize_peer_rows, Bytes dimension_storage, loomt::offset origin,
    Bytes local_storage, loomt::offset result_offset, Bytes peer_storage,
    ActivationStream activation_stream, loomt::index zero, loomt::index one,
    BarrierStream incoming_arrival, BarrierStream outgoing_arrival,
    BarrierStream incoming_departure, BarrierStream outgoing_departure,
    FeatureStream projection_stream, WeightStream weight_stream,
    loomt::offset activation_offset, FeatureStream probability_stream,
    FeatureStream result_stream, loomt::index column_offset,
    loomt::index dimension_row_base, loomt::index row_column,
    loomt::index skip_column, loomt::index layer, loomt::index dense_group);

[[loom::symbol("qwen3_embedding.probability_value.attention"),
  loom::force_inline]]
void qwen3_embedding_probability_value_attention(
    bool normalize_peer_rows, Bytes dimension_storage, loomt::offset origin,
    Bytes local_storage, loomt::offset result_offset, Bytes peer_storage,
    ActivationStream activation_stream, loomt::index zero, loomt::index one,
    BarrierStream incoming_arrival, BarrierStream outgoing_arrival,
    BarrierStream incoming_departure, BarrierStream outgoing_departure,
    FeatureStream projection_stream, WeightStream weight_stream,
    loomt::offset activation_offset, FeatureStream probability_stream,
    FeatureStream result_stream, loomt::index column_offset,
    loomt::index dimension_row_base, loomt::index row_column,
    loomt::index skip_column, loomt::index layer);

[[loom::symbol("qwen3_embedding.query_key_epilogue"), loom::force_inline]]
void qwen3_embedding_query_key_epilogue(
    Bytes sequence_storage, loomt::offset origin, Bytes dimension_storage,
    Bytes local_storage, loomt::offset result_offset, Bytes peer_storage,
    ActivationStream activation_stream, loomt::index zero, loomt::index one,
    BarrierStream incoming_arrival, BarrierStream outgoing_arrival,
    BarrierStream incoming_departure, BarrierStream outgoing_departure,
    FeatureStream normalization_stream, WeightStream weight_stream,
    loomt::offset activation_offset, FeatureStream projection_stream,
    FeatureStream probability_stream, loomt::index column_offset,
    loomt::index dimension_row_base, loomt::index row_column,
    loomt::index skip_column, loomt::index layer, loomt::index dense_group);

[[loom::symbol("qwen3_embedding.query_key_epilogue.attention"),
  loom::force_inline]]
void qwen3_embedding_query_key_epilogue_attention(
    Bytes sequence_storage, loomt::offset origin, Bytes dimension_storage,
    Bytes local_storage, loomt::offset result_offset, Bytes peer_storage,
    ActivationStream activation_stream, loomt::index zero, loomt::index one,
    BarrierStream incoming_arrival, BarrierStream outgoing_arrival,
    BarrierStream incoming_departure, BarrierStream outgoing_departure,
    FeatureStream normalization_stream, WeightStream weight_stream,
    loomt::offset activation_offset, FeatureStream projection_stream,
    FeatureStream probability_stream, loomt::index column_offset,
    loomt::index dimension_row_base, loomt::index row_column,
    loomt::index skip_column, loomt::index layer);

[[loom::symbol("qwen3_embedding.probability_value_ingress"),
  loom::force_inline]]
void qwen3_embedding_probability_value_ingress(
    Bytes dimension_storage, Bytes local_storage, loomt::offset origin,
    loomt::offset result_offset, Bytes peer_storage, Bytes gather_storage,
    Bytes activations, ActivationStream activation_stream, loomt::index zero,
    loomt::index one, Bytes constants, Bytes embedding, Bytes state,
    BarrierStream incoming_arrival, BarrierStream outgoing_arrival,
    BarrierStream incoming_departure, BarrierStream outgoing_departure,
    FeatureStream normalization_stream, Bytes output,
    WeightStream weight_stream, loomt::offset activation_offset,
    FeatureStream projection_stream, FeatureStream probability_stream,
    Bytes sequence_storage, FeatureStream result_stream,
    loomt::index column_offset, loomt::index gather_row_base,
    loomt::index row_column, loomt::offset row_input_initial,
    loomt::offset finish_initial, loomt::offset normalized_initial,
    loomt::offset dense_normalized_initial, loomt::offset dense_context_initial,
    loomt::offset dense_mlp_initial, loomt::offset skip_initial,
    loomt::offset rotary_address, loomt::index skip_column,
    loomt::offset query_initial, loomt::offset key_initial,
    loomt::offset value_initial, loomt::offset mlp_initial,
    loomt::offset residual_initial, loomt::offset output_initial,
    loomt::index layer, loomt::index dense_group);

[[loom::symbol("qwen3_embedding.probability_value_ingress.attention"),
  loom::force_inline]]
void qwen3_embedding_probability_value_ingress_attention(
    Bytes dimension_storage, Bytes local_storage, loomt::offset origin,
    loomt::offset result_offset, Bytes peer_storage, Bytes gather_storage,
    Bytes activations, ActivationStream activation_stream, loomt::index zero,
    loomt::index one, Bytes constants, Bytes embedding, Bytes state,
    BarrierStream incoming_arrival, BarrierStream outgoing_arrival,
    BarrierStream incoming_departure, BarrierStream outgoing_departure,
    FeatureStream normalization_stream, Bytes output,
    WeightStream weight_stream, loomt::offset activation_offset,
    FeatureStream projection_stream, FeatureStream probability_stream,
    Bytes sequence_storage, FeatureStream result_stream,
    loomt::index column_offset, loomt::index gather_row_base,
    loomt::index row_column, loomt::offset row_input_initial,
    loomt::offset finish_initial, loomt::offset normalized_initial,
    loomt::offset dense_normalized_initial, loomt::offset dense_context_initial,
    loomt::offset dense_mlp_initial, loomt::offset skip_initial,
    loomt::offset rotary_address, loomt::index skip_column,
    loomt::offset query_initial, loomt::offset key_initial,
    loomt::offset value_initial, loomt::offset mlp_initial,
    loomt::offset residual_initial, loomt::offset output_initial,
    loomt::index layer);

[[loom::symbol("qwen3_embedding.probability_value_egress"), loom::force_inline]]
void qwen3_embedding_probability_value_egress(
    Bytes dimension_storage, Bytes local_storage, loomt::offset origin,
    loomt::offset result_offset, Bytes peer_storage,
    ActivationStream activation_stream, loomt::index zero, loomt::index one,
    BarrierStream incoming_arrival, BarrierStream outgoing_arrival,
    BarrierStream incoming_departure, BarrierStream outgoing_departure,
    FeatureStream normalization_stream, Bytes output, Bytes state,
    WeightStream weight_stream, loomt::offset activation_offset,
    FeatureStream projection_stream, FeatureStream probability_stream,
    FeatureStream result_stream, loomt::index column_offset,
    loomt::index row_column, loomt::offset finish_initial,
    loomt::offset normalized_initial, loomt::index skip_column,
    loomt::offset query_initial, loomt::offset key_initial,
    loomt::offset value_initial, loomt::offset mlp_initial,
    loomt::offset residual_initial, loomt::offset output_initial,
    loomt::index layer, loomt::index dense_group);

[[loom::symbol("qwen3_embedding.probability_value_egress.attention"),
  loom::force_inline]]
void qwen3_embedding_probability_value_egress_attention(
    Bytes local_storage, loomt::offset origin, loomt::offset result_offset,
    Bytes peer_storage, ActivationStream activation_stream, loomt::index zero,
    loomt::index one, BarrierStream incoming_arrival,
    BarrierStream outgoing_arrival, BarrierStream incoming_departure,
    BarrierStream outgoing_departure, FeatureStream normalization_stream,
    Bytes output, Bytes state, WeightStream weight_stream,
    loomt::offset activation_offset, FeatureStream projection_stream,
    FeatureStream probability_stream, FeatureStream result_stream,
    loomt::index column_offset, loomt::index row_column,
    loomt::offset finish_initial, loomt::offset normalized_initial,
    loomt::index skip_column, loomt::offset query_initial,
    loomt::offset key_initial, loomt::offset value_initial,
    loomt::offset mlp_initial, loomt::offset residual_initial,
    loomt::offset output_initial, loomt::index layer);

[[loom::symbol("qwen3_embedding.query_key_ingress"), loom::force_inline]]
void qwen3_embedding_query_key_ingress(
    Bytes sequence_storage, loomt::offset origin, Bytes local_storage,
    loomt::offset result_offset, Bytes peer_storage, Bytes gather_storage,
    Bytes activations, ActivationStream activation_stream, loomt::index zero,
    loomt::index one, Bytes constants, Bytes embedding, Bytes state,
    BarrierStream incoming_arrival, BarrierStream outgoing_arrival,
    BarrierStream incoming_departure, BarrierStream outgoing_departure,
    FeatureStream projection_stream, Bytes output, WeightStream weight_stream,
    loomt::offset activation_offset, FeatureStream probability_stream,
    FeatureStream result_stream, loomt::index layer, loomt::index dense_group);

[[loom::symbol("qwen3_embedding.query_key_ingress.attention"),
  loom::force_inline]]
void qwen3_embedding_query_key_ingress_attention(
    Bytes sequence_storage, loomt::offset origin, Bytes local_storage,
    loomt::offset result_offset, Bytes peer_storage, Bytes gather_storage,
    Bytes activations, ActivationStream activation_stream, loomt::index zero,
    loomt::index one, Bytes constants, Bytes embedding, Bytes state,
    BarrierStream incoming_arrival, BarrierStream outgoing_arrival,
    BarrierStream incoming_departure, BarrierStream outgoing_departure,
    FeatureStream projection_stream, Bytes output, WeightStream weight_stream,
    loomt::offset activation_offset, FeatureStream probability_stream,
    FeatureStream result_stream, loomt::index layer);

[[loom::symbol("qwen3_embedding.query_key_egress_terminal"),
  loom::force_inline]]
void qwen3_embedding_query_key_egress_terminal(
    Bytes sequence_storage, loomt::offset origin, Bytes local_storage,
    loomt::offset result_offset, Bytes peer_storage,
    ActivationStream activation_stream, loomt::index zero, loomt::index one,
    BarrierStream outgoing_arrival, BarrierStream incoming_departure,
    FeatureStream projection_stream, Bytes output, Bytes state,
    WeightStream weight_stream, loomt::offset activation_offset,
    FeatureStream probability_stream, FeatureStream result_stream,
    loomt::index layer, loomt::index dense_group);

[[loom::symbol("qwen3_embedding.query_key_egress_terminal.attention"),
  loom::force_inline]]
void qwen3_embedding_query_key_egress_terminal_attention(
    Bytes sequence_storage, loomt::offset origin, Bytes local_storage,
    loomt::offset result_offset, Bytes peer_storage,
    ActivationStream activation_stream, loomt::index zero, loomt::index one,
    BarrierStream outgoing_arrival, BarrierStream incoming_departure,
    FeatureStream projection_stream, Bytes output, Bytes state,
    WeightStream weight_stream, loomt::offset activation_offset,
    FeatureStream probability_stream, FeatureStream result_stream,
    loomt::index layer);

[[loom::symbol("qwen3_embedding.replay_weights16"), loom::force_inline]]
void qwen3_embedding_replay_weights16(WeightStream source, WeightStream row0,
                                      WeightStream row1, WeightStream row2,
                                      WeightStream row3, loomt::index panels);

[[loom::symbol("qwen3_embedding.replay_weights24"), loom::force_inline]]
void qwen3_embedding_replay_weights24(WeightStream source, WeightStream row0,
                                      WeightStream row1, WeightStream row2,
                                      WeightStream row3, loomt::index panels);

#endif // QWEN3_EMBEDDING_ARRAY_WORKERS_H_
