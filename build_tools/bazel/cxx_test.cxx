// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

// Build-rule integration: C++ checks link a separately authored Loom provider
// through ordinary deps, and retain their benchmark for repository discovery.
#include <loomcxx/check.h>

int qwen38_feedback_counts(int outputs, int proposals);

LOOM_CHECK_CASE(feedback_counts) {
  loom::check::expect_equal(qwen38_feedback_counts(2, 3), 0x00030002);
}

LOOM_CHECK_BENCHMARK(feedback_publication, feedback_counts);
