# Run sheet and results

The authoritative list of runs is `results/RUNS.tsv` (seat, name, features, commit, args, start time),
maintained by Achyuthan. Frozen settings for every scored run: `--rounds 8 --samples 4 --context 8192
--repeat 5`, model Qwen/Qwen3-8B on the seat's own server; runs differ only in `--features`, the level
selection and `--log`. Single-level runs are comparable with `--all` runs because each level is solved
independently.

## Baselines

| Seat | Name | Code | Runs complete | Files |
|---|---|---|---|---|
| 152 | baseline | organisers' `d11ccdf`, untouched | 5 of 5, all levels | `results/seat-152/baseline.jsonl`, `baseline-run.log` |
| 154 | baseline | team `0278dc1`, no features | L1-2: 5, L3-4: 4 (as of 13:50 pull) | `results/seat-154/baseline.*` |
| 158 | baseline2 | team `413e595`, no features | partial (as of 13:50 pull) | `results/seat-158/baseline2.*` |

## Results so far (13:50 pull; solved/runs, then mean and every run's best)

| Level | Baseline (152, untouched) | Baseline (154 + 158, team code) | locate (150) | state (153 / 156) | facts,origin (157 / 159) |
|---|---|---|---|---|---|
| 1 | 0/5, 0.30 all | 0/7, 0.30 all | | | |
| 2 | 2/5, mean 0.58 | 3/7 | | | |
| 3 | 0/5, 0.30 all | 0/6, 0.30 all | 0/5, 0.30 all | 0/5, 0.30 all | 0/4, 0.30 all |
| 4 | 0/5, 0.62 all | 0/5, 0.62 all | | **0/3, 0.75 all** | 0/4, 0.62 all |

### 14:00 pull: full checker, commit `39b8e5a` (features internal, locate, origin, state)

| Level | Baseline (152) | Full checker | Seat |
|---|---|---|---|
| 1 | 0/5, 0.30 all | 0/3, 0.30 all | 150 |
| 2 | 2/5, mean 0.58 | 1/3, mean 0.60 | 151 |
| 3 | 0/5, 0.30 all | **2/3 solved**, all=[1.0, 1.0, 0.3] | 153 |
| 4 | 0/5, 0.62 all | 0/2, 0.75 all (same as `state`) | 159 |

Level 3 had 0 solves in 25 earlier runs across every condition. Caveat for the note: runs 1 and 2 are the
**same trajectory** (identical failure sequence, solved on round 5 both times), so they are one path found
twice under near-deterministic decoding. Run 3 took a different first kernel and was 3 rounds in at the pull.

The solving trajectory (run 1), what the checker added each round, and what the model did:

| Round | Line the checker named | What it added | Model's next kernel |
|---|---|---|---|
| 0 | 16, `psum = nl.ndarray(shape=out.shape, ...)` | `out` is (64,); a tile needs 2 dimensions. **Still carries the old "(1, N) or (N, 1)" hint** | psum (64, 1): followed the hint |
| 1 | 18, `nc_matmul` | psum's 2nd dimension is too small, must equal `sbuf_rhs`'s 2nd; it got its shape on line 16, change that line | psum (128, 512) |
| 2 | 18, `nc_matmul` | psum's 1st dimension too large, must equal `sbuf_lhsT`'s 2nd; change line 16 | psum (64, 512): correct |
| 3 | 20, `tensor_copy(dst=sbuf_lhsT, src=psum)` | `sbuf_lhsT` already used by line 13 at its size: do not resize it, allocate a separate tile | new tile |
| 4 | 24, `dma_copy(dst=out, ...)` | `out` holds 64, needs 32768; it got its shape on line 8, change that line | fixed `out` |
| 5 | | Correct on every shape | solved |

Every message states a relation between two named tiles and where the wrong shape came from; none states the
shape to write. Round 0 is the old hint misleading the model, exactly as in the baseline; removing it would
probably save a round.

### 14:33 pull: complete 5-run results

| Level | Baselines (152 + 154 + 158) | Full checker `39b8e5a` | Seat |
|---|---|---|---|
| 1 | 0/14, 0.30 every run | 0/3 (partial) | 150 |
| 3 | 0/13, 0.30 every run | **4/5 solved**, all=[1.0, 1.0, 0.3, 1.0, 1.0], mean 0.86 | 153 |
| 3 (repeat) | | 2/3 so far, all=[1.0, 1.0, 0.3] (commit `ec306f1`) | 157 |
| 4 | 0/13, 0.62 every run | 0/5, **0.75 every run** | 159 |
| 4 (`state` only) | | 0/5, 0.75 every run | 156 |

Level 3 trajectories under the full checker (letters = failure each round, D = solved):
runs 1, 2, 4 `E B B C A D`; run 5 `A A B C A D`; run 3 `A A A A A` (not solved).
So the solves came by **two distinct paths**, not one path repeated: run 5 started from a different kernel
and still converged. The repeat on seat 157 reproduces the pattern on a second chip.

### 15:10 pull: all baselines complete, checker `d761cd0` ("pieces, shapes")

| Level | Baselines (152 / 154 / 158, 15 runs) | Checker `39b8e5a` | Checker `d761cd0` |
|---|---|---|---|
| 1 | 0/15, 0.30 every run | 0/5, 0.30 every run | 0/5, 0.30 every run |
| 2 | 6/15 solved (2/5, 1/5, 3/5) | 2/5 | 1/5 |
| 3 | 0/15, 0.30 every run | **4/5** | **4/5** |
| 4 | 0/15, 0.62 every run | **0.75 every run (5/5)** | 0.62 every run (4 runs) - regression |

16-round pair (`--rounds 16`, comparable only with each other): baseline 0/5, all 0.62 except one 0.30;
checker `d761cd0` 0.75, 0.75, 0.62 over 3 runs. Doubling the rounds does not help the baseline.

**Level 4 regression under `d761cd0`.** Under `39b8e5a` every run went partition -> contraction -> width
and the contraction message worked (next attempt scored higher). Under `d761cd0` runs 2-4 go
partition -> contraction -> tensor_copy into HBM -> contraction x3: the contraction message no longer moves
the model. Achyuthan's next commits (`1c47e56` "one limit at a time", `16dcf94` "speak about width unless the
error is the contraction") target this and have not been run yet. The reportable checker is `39b8e5a` unless
a later version beats it on both levels 3 and 4.

**Level 1 did not move under either checker, as predicted.** All 10 checker runs open with the same five
failures as the baseline (tile size, `transpose_moving`, psum placement, `nisa.multiply` x2) and still
compute the mean as a matmul; only the last three rounds differ. A checker cannot reach a wrong algorithm.

**No level-1 run so far contains prompt C.** Round-0 prompt length is 2,323 characters in the baseline,
`checker-l1` and `cand-l1` alike; the detailed hint would add about 400.

**Level 2 is noise**, 1/5 to 3/5 in every condition, including the three baselines.

### Prompt change C is now in `dev` (`ce61bac`, merged via PR #10)

Team decision (15:14): **not reverted**. So from now on every level-1 run launched from `dev` carries the
detailed hint, and must be reported as "prompt C (detailed)", never as a checker result.

It is always on for level 1 (no feature flag), and it is the **detailed** version: it tells the model
"Use this exact plan", `.ap(...)`, `nl.sum(..., axis=[3, 4])`, and `1.0 / (pool_size * pool_size)`.
No run in `RUNS.tsv` so far includes it (all commits predate it), but any level-1 run launched from `dev`
from now on does, including checker runs. Such runs must be labelled "checker + prompt C", and a level-1
result from them cannot be credited to the checker. If it stays, the note must quote the hint verbatim and
say it gives the model the reference algorithm.

### Prompt C, short hint (seat 155, `Perumal/Level1-Prompt` at `c4e5eb2`)

Code `e29cc01` (= `0278dc1` + the short hint), `--features '' --level 1`, frozen settings. Logs:
`projects/02-kernel-agent/results/seat-155/level1-route-first-prompt-repeat5{.log,-attempts.jsonl}` on that
branch. The jsonl holds 6 sessions; only the last (104 attempts, reps 0-4, round-0 prompt 2,484 chars vs
2,323 for seat-154's baseline on `0278dc1`) is the measured repeat. Result 0/5, always 0.30.
Algorithm moved: `nc_matmul` 0/104, `nl.sum` 103/104, `1/p^2` scale 79/104. Failures: fixed `(128, 128)` tile
80, missing `pool_size` argument 24. Not in `RUNS.tsv` (run outside `pod.sh run`).

### 15:33 pull (`e780bf9`): `ahead` and `pieces` on `16dcf94`, partial

| Log | Flags | Completed runs | Result |
|---|---|---|---|
| seat-153 `a-l3` | internal,origin,ahead | 4 of 5 | 4/4 solved, **each after 4 rounds** (all 10 earlier solves took 6) |
| seat-154 `p-l4` | internal,origin,pieces | 1 of 5 | 0.75 |
| seat-156 `p-r16-l4` | internal,origin,pieces, 16 rounds | 1 of 5 | 0.75 after 7 rounds (gave up on repeats) |
| seat-159 `p-r16-l4` | internal,origin,pieces, 16 rounds | 1 of 5 | 0.75 after 7 rounds |

Not yet pulled: `a-l2` (150), `pa-l3` (158), `pa-r16-l4` (151), and `a-l3` on 157 (`RUNS.tsv` lists it
on both 153 and 157). `pieces` alone holds level 4 at 0.75, so the `d761cd0` regression came from `shapes`,
the combination, or the wider version of `pieces` that `16dcf94` narrowed. Level 4 still stalls on the same rules (partition > 128 on the copy, then the
stationary free dimension > 128). Replays of feedback (`results/replay-*.txt`): 235 kernels, 0 differences
between `39b8e5a` and `16dcf94` under `internal,origin`, and 0 between the organisers' code and `16dcf94`
with no flags; both were run on the stand-in simulator.

### 15:40 pull (`30bd03e`, PR #15): rest of the `16dcf94` runs, and `ahead` narrowed (`11fbb86`)

| Log | Flags | Completed runs | Result (rounds to solve) |
|---|---|---|---|
| seat-157 `a-l3` | internal,origin,ahead | 4 of 5 | 4/4 solved (4, 7, 4, 4) |
| seat-158 `pa-l3` | internal,origin,pieces,ahead | 3 of 4 started | 3/3 solved (4, 7, 7) |
| seat-150 `a-l2` | internal,origin,ahead | 5 of 5 | 3/5 (0.30, 0.50, 1, 1, 1): noise range |
| seat-151 `pa-r16-l4` | internal,origin,pieces,ahead, 16 rounds | 1 of 2 started | **0.62**, regressed from 0.75 |

`ahead` on level 3, pooled over seats 153 and 157: 8/8 solved, 7 in 4 rounds, 1 in 7 (all 10 solves
without `ahead` took 6). `11fbb86` changes `ahead` so it counts only later writes, after the level 4
regression. Replay `16dcf94` vs `11fbb86` on the stand-in simulator, 303 kernels from 30 logs:
`internal,origin` differs on 0; `internal,origin,ahead` differs on 77 (L1 1, L2 47, L3 20, L4 9). So
`ahead` runs on `11fbb86` must be logged as a separate condition.

Other rows in `RUNS.tsv` since 14:29: `cand-*` on `d761cd0` (pieces, shapes) are in progress, and
`base-r16-l4` / `cand-r16-l4` use `--rounds 16`, a deliberate pair outside the frozen settings that is
only comparable with each other.

Reading rules: levels 1, 3 and 4 never varied in any baseline (17 runs at level 1 alone), so any change there
is a real effect of the condition. Because the model is near-deterministic, the failure *type* must also move
in the direction of the information added before a change is credited to that information. Level 2 can only
show a collapse; small differences there are noise.

## First moved wall: level 4 under `state` (seat-156)

Level 4 rose from 0.62 (1 of 4 shapes) to 0.75 (2 of 4 shapes) in 3 of 3 runs, and the failure type moved
in the direction of what `state` reports:

| | Baseline (152) | state (156) |
|---|---|---|
| Kernels that never tile (partition 256 on the whole-tensor copy) | 70 of 80 failed attempts | 6 of 64 |
| Failing kernels that contain a loop | 1 of 80 | most; 100% of the two main types |
| Sequence each run | stuck on "partition 256" for 4 rounds | partition 256 -> contraction > 128 (then **scored higher 12/12**) -> stationary free dimension > 128 |
| Identical code resent after a message | 60 of 60 | 28 of 40 (only on the new last failure) |

So it now tiles K correctly and stalls on the next rule, M per tile <= 128 (the stationary operand's free
dimension). That is the next target for the checker.
