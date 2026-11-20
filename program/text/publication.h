// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

// Source interface for the native text-service publication callback. All
// buffers are borrowed. The caller joins work and feedback completion before
// invoking publication and retains every backing allocation through return.
#pragma once

#include <loomcxx/buffer.h>

namespace text_service {

using Buffer = loom::type::buffer<int>;

struct PublicationInputs {
  // Source-owned retained control record.
  Buffer control_state;
  // Original nine-word native request records, in publication order.
  Buffer requests;
  // Model-defined media records for recovering source positions.
  Buffer media;
  // Number of original request records to publish.
  int request_count;
  // Number of media features in the first cohort.
  int first_media_count;
  // Number of media features in the continuation cohort.
  int continuation_media_count;
};

struct CompletedEpoch {
  // Native i32 flag selecting verified feedback instead of ordinary output.
  int verified;
  // Native i32 flag indicating a completed continuation cohort.
  int has_continuation;
  // Continuation speculation flags indexed by original request owner.
  Buffer continuation_flags;
  // Densely packed ordinary output token IDs.
  Buffer ordinary_tokens;
  // Immutable first-cohort feedback, in the model's own format.
  Buffer first_feedback;
  // Immutable continuation feedback, including model-defined owner mapping.
  Buffer continuation_feedback;
};

struct PublicationOutputs {
  // Caller-owned 21-word semantic progress records in original request order.
  Buffer progress;
  // Caller-owned actual proposal counts in original request order.
  Buffer proposal_counts;
};

struct EpochPublication {
  // Original request context; source positions retain model-defined semantics.
  PublicationInputs inputs;
  // Completed source results; publication performs no wait or submission.
  CompletedEpoch completed;
  // Borrowed destinations consumed by the native caller after return.
  PublicationOutputs outputs;
};

} // namespace text_service

// Member-order flattening defines the fourteen-argument native publication signature.
// Implementations and C++ callers share these records without heap allocation.
void text_publish_epoch_v2(text_service::EpochPublication epoch);
