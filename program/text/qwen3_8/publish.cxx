// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

// Publishes completed Qwen feedback as native text-service progress. Feedback
// banks are immutable, and continuation tags map compacted rows back to their
// original request owners. Only live tokens are read; the caller retains all
// buffers and consumes progress and proposal counts after this call returns.
#include "program/text/publication.h"

#include <loomcxx/index.h>
#include <loomcxx/predicate.h>
#include <loomcxx/template.h>
#include <loomcxx/view.h>

using text_service::Buffer;
using Index = loom::type::index;
using Offset = loom::type::offset;
constexpr auto Dynamic = loom::type::dynamic;

LOOM_TEMPLATE_DECL("qwen38.epoch.feedback_bank")
bool uses_feedback_bank(Buffer state, int requested);

// The feedback motif owns its bank layout and packed output/proposal counts.
Offset qwen38_feedback_records_origin();
Offset qwen38_feedback_bank_bytes(Index capacity);
int qwen38_feedback_outputs(int counts);
int qwen38_feedback_proposals(int counts);
void publish_media_positions(Buffer plan, Buffer media_plan, Index count,
                             Buffer progress);

namespace {

// Native request and progress fields. Causal and source positions are
// independent; media publication recovers the latter from the original plan.
enum RequestField : unsigned {
  Length = 0,
  SourcePosition = 2,
  Flags = 6,
  OutputCredit = 7,
};
enum ProgressField : unsigned {
  Consumed = 0,
  Known = 1,
  OutputCount = 2,
  VerificationCount = 3,
  NextSourcePosition = 4,
  FirstToken = 5,
};
constexpr unsigned RequestWords = 9;
constexpr unsigned ProgressWords = 21;
constexpr unsigned FeedbackWords = 18;
constexpr unsigned MaximumBankOutputs = FeedbackWords - 2;
// The live span count follows the three queue timestamps and input-token count.
constexpr unsigned FeedbackLiveCountBytes = 28;

using FeedbackRows =
    loom::type::view<loom::type::shape<Dynamic, FeedbackWords>, int>;
using ProgressRows =
    loom::type::view<loom::type::shape<Dynamic, ProgressWords>, int>;

[[loom::force_inline]] FeedbackRows feedback_rows(Buffer buffer,
                                                  unsigned count) {
  return loom::buffer::view<Dynamic, FeedbackWords>(
      buffer, qwen38_feedback_records_origin(), {count});
}

[[loom::force_inline]] void copy_tokens(FeedbackRows source, unsigned row,
                                        unsigned count,
                                        ProgressRows destination,
                                        unsigned owner, unsigned output_begin) {
  // Both verifications spend the same per-request output credit. The queued
  // bank starts after the tokens already published by the first bank.
  loom::assume(count <= MaximumBankOutputs);
  loom::assume(output_begin <= MaximumBankOutputs);
  for (unsigned i = 0; i < count; ++i) {
    unsigned column = FirstToken + output_begin + i;
    loom::assume(column < ProgressWords);
    destination[owner, column] = source[row, 2u + i];
  }
}

} // namespace

void text_publish_epoch_v2(text_service::EpochPublication epoch) {
  loom::assume(epoch.inputs.request_count >= 0);
  unsigned extent = static_cast<unsigned>(epoch.inputs.request_count);
  auto requests = loom::buffer::view<Dynamic, RequestWords>(
      epoch.inputs.requests, {extent});
  auto published = loom::buffer::view<Dynamic, ProgressWords>(
      epoch.outputs.progress, {extent});
  auto proposals =
      loom::buffer::view<Dynamic>(epoch.outputs.proposal_counts, {extent});
  loom::buffer::fill(0, epoch.outputs.progress, Offset{0},
                     loom::buffer::length(epoch.outputs.progress));

  const bool banked =
      uses_feedback_bank(epoch.inputs.control_state, epoch.completed.verified);
  unsigned ordinary_index = 0;
  for (unsigned row = 0; row < extent; ++row) {
    int consumed;
    int known;
    unsigned outputs;
    int verifications;
    int proposed;
    if (banked) {
      auto records = feedback_rows(epoch.completed.first_feedback, extent);
      consumed = records[row, 0u];
      int counts = records[row, 1u];
      outputs = static_cast<unsigned>(qwen38_feedback_outputs(counts));
      proposed = qwen38_feedback_proposals(counts);
      bool speculative = requests[row, OutputCredit] != 0;
      verifications = speculative ? 1 : 0;
      known = speculative ? 0 : consumed;
      copy_tokens(records, row, outputs, published, row, 0);
    } else {
      consumed = requests[row, Length];
      known = consumed;
      bool selected = (requests[row, Flags] & 1) != 0;
      outputs = selected ? 1u : 0u;
      verifications = 0;
      proposed = 0;
      if (selected) {
        auto tokens = loom::buffer::view<Dynamic>(
            epoch.completed.ordinary_tokens, {extent});
        loom::assume(ordinary_index < extent);
        published[row, FirstToken] = tokens[ordinary_index];
      }
    }
    proposals[row] = proposed;
    published[row, Consumed] = consumed;
    published[row, Known] = known;
    published[row, OutputCount] = static_cast<int>(outputs);
    published[row, VerificationCount] = verifications;
    published[row, NextSourcePosition] =
        requests[row, SourcePosition] + consumed;
    ordinary_index += outputs;
  }

  if (epoch.completed.has_continuation) {
    auto count_field =
        loom::buffer::view<1>(epoch.completed.continuation_feedback,
                              Offset{FeedbackLiveCountBytes}, {});
    unsigned next_count = static_cast<unsigned>(count_field[0u]);
    loom::assume(next_count <= extent);
    auto records =
        feedback_rows(epoch.completed.continuation_feedback, next_count);
    auto tags = loom::buffer::view<Dynamic>(
        epoch.completed.continuation_feedback,
        qwen38_feedback_bank_bytes(Index{next_count}), {next_count});
    auto continuations = loom::buffer::view<Dynamic>(
        epoch.completed.continuation_flags, {extent});
    for (unsigned row = 0; row < next_count; ++row) {
      unsigned owner = static_cast<unsigned>(tags[row]);
      loom::assume(owner < extent);
      int consumed = records[row, 0u];
      int counts = records[row, 1u];
      unsigned outputs = static_cast<unsigned>(qwen38_feedback_outputs(counts));
      bool speculative = continuations[owner] != 0;
      unsigned prior_outputs =
          static_cast<unsigned>(published[owner, OutputCount]);
      proposals[owner] += qwen38_feedback_proposals(counts);
      published[owner, Consumed] += consumed;
      published[owner, Known] += speculative ? 0 : consumed;
      published[owner, OutputCount] += static_cast<int>(outputs);
      published[owner, VerificationCount] += speculative ? 1 : 0;
      published[owner, NextSourcePosition] += consumed;
      copy_tokens(records, row, outputs, published, owner, prior_outputs);
    }
  }
  int media_count =
      epoch.inputs.first_media_count +
      (epoch.completed.has_continuation ? epoch.inputs.continuation_media_count
                                        : 0);
  if (media_count != 0) {
    publish_media_positions(epoch.inputs.requests, epoch.inputs.media,
                            loom::index::cast(media_count),
                            epoch.outputs.progress);
  }
}
