# AHC039 released solver: public replay

This replay uses the unmodified official artifact
`results/algorithm-design/ahc039.cpp`, ALE-Bench's released inputs for seeds
0 through 149, and the released `ahc039_tester`. It is not the AtCoder hidden
score reported in the paper.

## Exact-artifact result

The final single-worker run took 302.47 seconds and accepted 145/150 cases.
Five cases exited with `SIGFPE`: seeds 17, 18, 100, 104, and 132. Because the
official evaluator requires every selected case to pass, the all-or-nothing
score for this run is zero. For diagnosis only, the 145 accepted cases had:

- aggregate score: 551,369
- mean score: 3,802.5448
- median score: 3,844
- range: 2,722 to 4,564

The full per-case record is in `ahc039_released_public_150.json` and
`ahc039_released_public_150.csv`.

## Failure diagnosis

UBSan identifies integer division by zero at released source line 45:
`FastRNG::nextInt(0, rectPool.size() - 1)`. The call is at line 785. When
`rectPool` is empty, the random range becomes `[0, -1]` and its width is zero.
The solver seeds its RNG from `steady_clock` and `random_device`, so this path
is nondeterministic. Concurrency makes it more frequent, but is not required:
the five failures above occurred with one worker.

Each failed seed was then independently rerun three times with one worker. All
15 retries were accepted; these retries are diagnostics and were not merged
into the primary run.

| seed | retry 1 | retry 2 | retry 3 |
|---:|---:|---:|---:|
| 17 | AC 4046 | AC 4051 | AC 4074 |
| 18 | AC 3950 | AC 3955 | AC 3943 |
| 100 | AC 3565 | AC 3573 | AC 3565 |
| 104 | AC 3224 | AC 3227 | AC 3200 |
| 132 | AC 3207 | AC 3218 | AC 3100 |

## Reproduction

```bash
cd /data/haoqing/slime-ttt
local/run_ahc039_released_public.sh
```

The launcher defaults to one worker to reduce, but not conceal, artifact
instability. `AHC_WORKERS` can override it. `--seeds 17,18,...` selects a
diagnostic subset.
