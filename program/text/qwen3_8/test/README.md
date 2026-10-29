# Qwen text-program checks

These checks exercise shared chat, media, storage, and epoch policy without
checkpoint acquisition. The native request integration uses the real text
caller and a small tokenizer whose emitted tokens are independently inspectable.
VM cases check packet contents, ownership transitions, and exact upload ranges;
the pixel path joins CPU preparation to device filtering.

Storage and admission tests create small descriptor tables, including byte
extents beyond 32 bits, without allocating the represented model storage.
Complete-buffer comparisons preserve unsubmitted capacity, unrelated resident
rows, and the original media packet across compaction. Independent expectations
and any substituted native dependency are documented in each test source.

Numerical encoder checks belong to the [shared vision package](../../../../motif/vision/qwen3_8/test/).
Model tests own learned outputs and the complete serving lifetime. The
[development guide](../../../../docs/development.md#build-and-check) explains
ordinary execution and the source-SDK requirement of the native request test.
