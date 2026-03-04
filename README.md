# Triton Bug Reproductions

This repository contains reproductions of bugs found in the Triton GPU kernel compiler,
including data races, synchronization issues, and incorrect code generation.

## Bugs

| Issue | Title | Bug Type | Status |
|-------|-------|----------|--------|
| [#8311](issue_8311/) | Incorrect results from `warp_specialize=True` | Producer-consumer race — missing sync barrier between TMA load and `tl.dot` | Closed |
| [#7402](issue_7402/) | `tl.atomic_add` return value wrong across threads | Layout mismatch — only thread 0 holds atomic return value, others get 0 | Fixed in PR #7460 |
| [#7264](issue_7264/) | Write-write data race in reduction (butterfly shuffle) | WAW hazard — threads write different FP values to same shared memory address | Open |
| [#4362](issue_4362/) | `tl.associative_scan` wrong results with `reverse=True` | Incorrect scan ordering — final value broadcast to all positions instead of scanning | Open |
| [#4736](issue_4736/) | Racecheck bug when `tl.min` used with `tl.sum` | WAW hazard in `tl.min` lowering (`standard.py:237`) — also causes wrong output on Triton 3.6.0 | Open |
| [#4233](issue_4233/) | `scatter_add` data race without `tl.atomic_add` | WAW race — non-atomic read-modify-write loses updates when duplicate indices are present | Open |

## Setup

Different issues require different Triton versions to trigger the bug.
See each issue's README for the specific version required.

```bash
# General environment (Triton 3.6.0) — for issues #8311, #4736, #4233
conda create -n triton-bugs python=3.12 -y
conda activate triton-bugs
pip install torch==2.4.0
pip install triton

# Older environment (Triton 3.0.0) — for issues #7402, #7264, #4362
conda create -n triton-7402 python=3.12 -y
conda activate triton-7402
pip install torch==2.4.0
pip install triton==3.0.0
```

## Repository Structure

```
triton-bug-reproductions/
├── README.md
├── issue_8311/
│   ├── README.md
│   └── reproduce.py
├── issue_7402/
│   ├── README.md
│   └── reproduce.py
├── issue_7264/
│   ├── README.md
│   └── reproduce.py
├── issue_4362/
│   ├── README.md
│   └── reproduce.py
├── issue_4736/
│   ├── README.md
│   └── reproduce.py
└── issue_4233/
    ├── README.md
    └── reproduce.py
```

## Notes

- Issue #8311 requires an sm90+ GPU (H100 or RTX 5090) to observe wrong output at
  runtime. On sm75 (MX450), `reproduce.py` compiles the kernel and dumps the TTGIR,
  which shows the `{tt.warp_specialize}` attribute on the inner loop without a
  synchronization barrier — confirming the buggy code path was generated.
- Issue #7402 requires Triton 3.0.0 to trigger. The bug is fixed in Triton 3.1.0+
  via PR #7460 (Atomic RMW Broadcasting). Run in the `triton-7402` conda environment.
- Issue #7264 is a silent data race — the output is often numerically close to correct
  but non-deterministic across runs. Use `compute-sanitizer --tool racecheck` to
  observe the WAW hazards explicitly.
- Issue #4362 requires Triton 3.0.0 to trigger. The `reverse=False` direction works
  correctly and is included as a baseline to isolate the bug to `reverse=True`.
- Issue #4736 fires on both Triton 3.0.0 and 3.6.0. On 3.6.0 the race also causes
  numerically wrong output, making it the stronger reproduction. Use
  `compute-sanitizer --tool racecheck` to observe the WAW hazards explicitly.
- Issue #4233 fires directly on Triton 3.6.0 with wrong numerical output — no
  sanitizer needed. The atomic version is included in the same script as a correct
  reference implementation.