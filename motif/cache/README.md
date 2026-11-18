# Cohort cache

`cohort.loom` reserves and publishes slots for immutable keyed payloads. One
execution owner processes ordered cohorts; requests within a cohort can reserve
concurrently. Payload representation, byte ranges, transfer scheduling, and
decoding belong to the caller. Flash-Next uses this directory for exact packed
PLE rows, while the Qwen3.8-27B embedding check exercises the same operations
with Q5_K rows and different cache geometries.

The caller supplies a zero-initialized directory of 64-bit tags, a zero initial
generation, a positive associativity, and a power-of-two set count. Keys range
from zero through 2147483646. Each tag contains a 32-bit generation, a pending
bit, and the key plus one; zero is empty. This implementation requires native
64-bit device atomics.

The ownership sequence is:

```
retire previous consumers -> begin -> reserve -> fill -> publish -> consume
```

Every lane of one workgroup calls `cohort_cache_begin` with its previously
loaded generation. Completion of that dispatch precedes reservations. Generation
rollover clears the directory before any new reservations; payload needs no
clear because only published entries are readable.

`cohort_cache_reserve` returns a slot and whether its caller owns a fill. A hit
or another request's current pending slot needs no duplicate fill. Older pending
entries cannot become hits: an abandoned transfer is retried in a later cohort.
Slots reserved by the current cohort remain stable until its consumers retire.
The search is bounded by associativity, including compare/exchange contention.
An exhausted set returns `capacity`, leaving the caller to assign private spill
space for that request instead of evicting an active row or waiting for it.

The unique fill owner calls `cohort_cache_publish` only after transfer readiness.
Consumers wait for all fills and publication. A failed cohort retires accepted
transfers before the next begin, skips publication for incomplete fills, and
never consumes them. The directory provides neither transfer fences nor an
independent lifetime for payload storage.
