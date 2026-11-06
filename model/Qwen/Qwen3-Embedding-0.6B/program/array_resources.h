// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

// Core-local storage and channel layouts retained across resident phases.

#ifndef QWEN3_EMBEDDING_ARRAY_RESOURCES_H_
#define QWEN3_EMBEDDING_ARRAY_RESOURCES_H_

#include <loomcxx/kernel.h>
#include <loomcxx/pipeline.h>
#include <loomcxx/target.h>

#include <stdfloat>

namespace loomt = loom::type;

using Bytes = loomt::buffer<unsigned char>;
using ActivationStream =
    loomt::channel<loomt::tile<loomt::shape<2048>, unsigned>>;
using BarrierStream = loomt::channel<loomt::tile<loomt::shape<16>, unsigned>>;
using FeatureStream =
    loomt::channel<loomt::tile<loomt::shape<4096>, std::bfloat16_t>>;
using QueryOutputSlots =
    loomt::view<loomt::shape<2, 4096>, std::bfloat16_t,
                loomt::encoding<loom::encoding::role::layout, 2>>;
using WeightStream = loomt::channel<loomt::tile<loomt::shape<4624>, unsigned>>;

struct CoreStorage {
  // Physical memory pool owning this core allocation.
  loomt::pool memory;
  // Core-local backing retained across all model phases.
  unsigned char *storage;
};

struct CoreStreams {
  // One packed projection record received by this core.
  WeightStream weights;
  // One packed token panel shared across the row.
  ActivationStream activations;
};

struct BarrierResources {
  // Upstream completion credit for the current shared-row phase.
  BarrierStream arrival;
  // Downstream permission to reuse the shared-row storage.
  BarrierStream departure;
};

struct QueryOutputResources {
  // Two nonoverlapping projection records in the owning core allocation.
  QueryOutputSlots slots;
  // FIFO credits guarding the projection records.
  FeatureStream stream;
};

// Every phase starts fresh FIFO credits over the retained memory-tile records.
template <unsigned Records>
[[loom::inline]] static WeightStream bind_weight_ingress(Bytes storage) {
  auto slots = loom::buffer::view<Records, 4624>(
      loom::buffer::cast<unsigned>(storage), {});
  return loom::channel::bind<loom::channel::discipline::fifo>(
      slots, loomt::index{Records});
}

[[loom::inline]] static CoreStorage allocate_core(loomt::pool memory) {
  auto *storage = loom::buffer::alloca<loom::memory_space::workgroup, 64>(
      loomt::offset{64448}, memory);
  return {.memory = memory, .storage = storage};
}

[[loom::inline]] static CoreStreams
bind_core_streams(unsigned char *storage) {
  auto weight_slots =
      loom::buffer::view<1, 4624>(reinterpret_cast<unsigned *>(storage), {});
  auto weights = loom::channel::bind<loom::channel::discipline::fifo>(
      weight_slots, loomt::index{1});
  auto activation_slots = loom::buffer::view<1, 2048>(
      reinterpret_cast<unsigned *>(
          storage + static_cast<unsigned long long>(loomt::offset{18816})),
      {});
  auto activations = loom::channel::bind<loom::channel::discipline::fifo>(
      activation_slots, loomt::index{1});
  return {.weights = weights, .activations = activations};
}

[[loom::inline]] static BarrierResources
make_barriers(loomt::pool memory) {
  auto *arrival_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64},
                                                              memory);
  auto arrival_slots = loom::buffer::view<1, 16>(
      reinterpret_cast<unsigned *>(arrival_storage), {});
  auto arrival = loom::channel::bind<loom::channel::discipline::fifo>(
      arrival_slots, loomt::index{1});
  auto *departure_storage =
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(loomt::offset{64},
                                                              memory);
  auto departure_slots = loom::buffer::view<1, 16>(
      reinterpret_cast<unsigned *>(departure_storage), {});
  auto departure = loom::channel::bind<loom::channel::discipline::fifo>(
      departure_slots, loomt::index{1});
  return {.arrival = arrival, .departure = departure};
}

[[loom::inline]] static FeatureStream
make_feature_stream(unsigned char *storage) {
  auto slots = loom::buffer::view<1, 4096>(
      reinterpret_cast<std::bfloat16_t *>(
          storage + static_cast<unsigned long long>(loomt::offset{10624})),
      {});
  return loom::channel::bind<loom::channel::discipline::fifo>(slots,
                                                              loomt::index{1});
}

[[loom::inline]] static QueryOutputResources
make_query_output(unsigned char *storage) {
  auto slots = loom::buffer::view<2, 4096>(
      reinterpret_cast<std::bfloat16_t *>(
          storage + static_cast<unsigned long long>(loomt::offset{39872})),
      {}, loom::encoding::layout::strided(loomt::index{8192}, loomt::index{1}));
  auto stream = loom::channel::bind<loom::channel::discipline::fifo>(
      slots, loomt::index{2});
  return {.slots = slots, .stream = stream};
}

#endif // QWEN3_EMBEDDING_ARRAY_RESOURCES_H_
