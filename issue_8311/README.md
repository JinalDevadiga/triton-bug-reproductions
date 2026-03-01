# Issue #8311 — Incorrect Results from `warp_specialize=True`

## Source
- **GitHub Issue:** https://github.com/triton-lang/triton/issues/8311
- **Repo:** Triton
- **Reported on:** Triton commit 8ee5840 (between v3.2 and v3.3, Sep 2025)
- **GPU required:** RTX 5090 or H100 (sm90+) to observe wrong numerical output
- **Status:** Closed

## What is the Bug?

Triton's `warp_specialize=True` option splits the warps in a thread block
into two specialized teams:

- **Producer warps** — dedicated to loading data from global memory into
  shared memory using TMA (Tensor Memory Accelerator), an async bulk-transfer
  engine available only on sm90+ GPUs (H100, RTX 5090).
- **Consumer warps** — dedicated to computing (`tl.dot` / matrix multiply).

This is a performance optimization: producers load tile K+1 while consumers
compute tile K simultaneously, like a factory assembly line.

The bug: when `warp_specialize=True` is combined with
`tl.make_tensor_descriptor` (TMA-based async loads), the synchronization
between producer and consumer warps is broken. Consumer warps proceed to
`tl.dot` before the TMA load for the current tile has completed, reading
stale or partially-written shared memory. This is a classic
**producer-consumer race condition**.

Result on RTX 5090 (as reported):
```
warp_specialize=False  -->  PASS    (all warps work in lockstep, no race)
warp_specialize=True   -->  99.3% of output values are WRONG
```

## Key Evidence in the Compiled TTGIR

The bug is visible in the compiled Triton GPU IR (TTGIR). Comparing the
inner `scf.for` loop in both variants:

### CORRECT (`warp_specialize=False`):
```
%acc = scf.for %ko = ... iter_args(...) -> (...) {
  ...
  %y_tile_51 = tt.load %y_tile_45, ...   ← regular synchronous load
  ...
  %acc_58 = tt.dot %acc_54, %acc_57, ... ← dot follows load safely
  scf.yield %acc_58 ...
} loc(#loc42)                             ← no warp_specialize attribute
```

### BUGGY (`warp_specialize=True`):
```
%acc = scf.for %ko = ... iter_args(...) -> (...) {
  ...
  %y_tile_51 = tt.load %y_tile_45, ...   ← load tagged for async TMA
  ...
  %acc_58 = tt.dot %acc_54, %acc_57, ... ← dot proceeds WITHOUT waiting
  scf.yield %acc_58 ...
} {tt.warp_specialize} loc(#loc42)        ← BUG: warp_specialize attribute
                                             present but missing sync barrier
```

The `{tt.warp_specialize}` attribute on the `scf.for` loop tells Triton's
backend to split warps into producer/consumer roles. On sm90+, this triggers
async TMA loads — but the generated code is missing the barrier that would
make consumer warps wait for the TMA transfer to complete before calling
`tt.dot`. On sm75 (this machine), TMA hardware does not exist so the
attribute is accepted but the warp split never happens, masking the bug.

## Requirements

- Python 3.10+
- CUDA 12.x
- Any NVIDIA GPU (to compile and inspect IR)
- **NVIDIA H100 or RTX 5090 (sm90+)** to observe wrong numerical output at runtime

## Setup
```bash
conda create -n triton-bugs python=3.12 -y
conda activate triton-bugs
pip install torch==2.4.0
pip install triton
```

## How to Run
```bash
python reproduce.py
```

## Expected Output

### On sm75 (MX450) — this machine:
```
warp_specialize=False: PASSED
warp_specialize=True:  PASSED   ← passes for wrong reason (no TMA hardware)
```
Both pass because TMA is unavailable on sm75. The TTGIR dump shows identical
code for both variants except the `{tt.warp_specialize}` attribute on the
buggy version's inner loop.

### On sm90+ (H100 / RTX 5090) — required to trigger bug:
```
warp_specialize=False: PASSED
warp_specialize=True:  BUG CONFIRMED
  Mismatched elements: 1041126 / 1048576 (99.3%)
  Greatest absolute difference: 139.0 at index (673, 708)
```

## Note on Version

This bug was reported on Triton commit `8ee5840`. The issue is closed,
suggesting it was fixed in a later commit. The `{tt.warp_specialize}` IR
attribute is still present in Triton 3.6.0 compiled on sm75, confirming
the code path exists — the fix likely added the missing synchronization
barrier in the sm90+ backend codegen path.
