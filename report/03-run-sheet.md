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

Still to pull: `state-all` (157), `facts-all` (159), `facts-l3` (151), and the rest of the `checker-*` runs.

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
