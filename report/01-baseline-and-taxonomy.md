# Baseline, failure taxonomy and hypothesis verdicts

Owner: John (Person A). Source data: `results/seat-152/baseline.jsonl` (416 attempts) and
`results/seat-152/baseline-run.log`. Counts below are over **every sample**, not only the best one per round.

## 1. Baseline record

| | |
|---|---|
| Seat | seat-152 |
| Code | organisers' unmodified `agent.py` / `nkibench.py` (commit `d11ccdf`; no `--features` flag exists in it) |
| Model | Qwen/Qwen3-8B, local server `http://localhost:8000/v1`, thinking off |
| Command | `python agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5 > run.log 2>&1` |
| Selftest | `nkibench.py --selftest` PASSED, nki 0.6.0, `nki.simulate` |
| Log | `attempts.jsonl` -> `baseline.jsonl`: 416 lines = 104 rounds x 4 samples, nothing from earlier runs |

| Level | Solved | Best | Worst | Mean | All runs |
|---|---|---|---|---|---|
| 1 average pooling | 0/5 | 0.30 | 0.30 | 0.30 | 0.3 0.3 0.3 0.3 0.3 |
| 2 transpose | 2/5 | 1.00 | 0.30 | 0.58 | 1.0 0.3 0.3 1.0 0.3 |
| 3 matmul, one tile | 0/5 | 0.30 | 0.30 | 0.30 | 0.3 0.3 0.3 0.3 0.3 |
| 4 matmul, tiled | 0/5 | 0.62 | 0.62 | 0.62 | 0.62 x5 |

### Property of the setup that affects every comparison

**The model is close to deterministic on this server.** Sampling is nominally on (temperature 0.6), but:

- Level 1: in all 40 rounds the 4 samples were byte-identical, and all 5 runs produced the **same 8-kernel
  sequence** character for character. Only 7 distinct kernels in 160 attempts.
- Level 4: 6 distinct kernels in 80 attempts; level 3: 8 in 84.
- Level 2 is the exception (18 distinct in 92), which is why it is the only level with spread.

Consequence: "zero variance on levels 1, 3, 4" does not mean the result is robust; it means the output is
nearly a function of the prompt. **Any** change to the feedback text changes the trajectory, whether or not
the new text is more useful. A moved score under change A or B is attributable only if the failure type also
moves in the direction of the information that was added (see section 2, re-done per condition).

### The model largely ignores repair instructions

After being sent a failure message, how often the very next attempt was the **same code** again:

| Level | Repair attempts | Identical code resent |
|---|---|---|
| 1 | 140 | 20 (14%) |
| 3 | 64 | 52 (81%) |
| 4 | 60 | **60 (100%)** |

On level 4 the prompt changed between rounds (a ledger of failed attempts was added) and the model still
returned the identical kernel every time.

## 2. Failure taxonomy (baseline)

Named by what the **kernel** did wrong, not by what the simulator said. Each level's counts sum to its
failed attempts.

### Level 1 - average pooling (160 failed attempts, 5/5 runs identical)

**Root type, 160/160: wrong algorithm.** Every kernel computes the average as
`nisa.nc_matmul(stationary=tile, moving=tile)` (the window multiplied by itself) and then scales by `0.5`.
That is not a mean, and `0.5` is not `1/p^2`. The `0.5` is copied from the `tensor_scalar` example in the
prompt's API card (`operand0=0.5`), and the API card contains no reduction example besides one `nl.sum` line.
The reference kernel uses one strided `.ap()` view plus `nl.sum` and has no loops. No attempt ever got far
enough to be checked numerically, so the checker never saw the algorithm.

The surface failures it cycled through, identical in every run (rounds 0-7: 1a 1a 1b 1c 1d 1d 1d 1e):

| Type | Attempts | What the kernel did | Message the model saw (start) |
|---|---|---|---|
| 1a Tile not sized to the window | 40 | allocates a fixed `(128, 128)` tile and copies a 2x2 window into it | `dma_copy requires src and dst to have the same number of elements, got src=4, dst=16384` |
| 1b Invented argument | 20 | `nc_matmul(..., transpose_moving=True)` | `nc_matmul() got an unexpected keyword argument 'transpose_moving'` |
| 1c Matmul result in SBUF | 20 | `nc_matmul(dst=tile2)` with `tile2` in `nl.sbuf` | `dst must be in ['psum'], got sbuf` |
| 1d Right function, wrong module | 60 | `op0=nisa.multiply`; the real name is `nl.multiply` | `` `nki.isa` has no `multiply`, and nothing similar exists `` - **false**: it exists in `nki.language` |
| 1e Invented name | 20 | `op=nisa.scalar_mul` | `` `nki.isa` has no `scalar_mul` `` |

Example (every run, round 4): `nisa.tensor_scalar(dst=tile, data=0.5, op0=nisa.multiply)` after
`nisa.nc_matmul(dst=tile2, stationary=tile, moving=tile, is_transpose=True)`.

### Level 3 - matmul, one tile `K=128 M=64 N=512` (84 failed attempts)

No attempt contains a loop (0/84). Every failure is about tile shapes and which tile holds what.

| Type | Attempts | What the kernel did | Message the model saw (start) |
|---|---|---|---|
| 3a Output shape dropped a dimension | 19 | `out = nl.ndarray(shape=lhsT.shape[1:])` gives `(64,)`; the PSUM tile copies that shape, so it is 1-D. The matmul result is `(M, N)` | `SBUF and PSUM tensors must have at least 2 dimensions` + hint "give a length-N vector the shape (1, N) or (N, 1)" |
| 3b One scratch tile for every operand | 35 | allocates one `(128, 512)` tile and DMAs `lhsT` `(128, 64)` and `rhs` into it | `got src=8192, dst=65536` + hint "If you want a 128x512 piece..." |
| 3c Output-shaped tile for an input, invented args | 14 | `tile = (M, N)`, copies `lhsT` into it; `dst_offset=`, `K=` arguments | `got src=8192, dst=32768` |
| 3d Followed the (1, N) hint | 16 | after 3a's hint, shrinks operands to `(1, M)` and `(K, 1)` | `got src=8192, dst=64` |

62 of 84 attempts contain `lhsT.shape[1:]`, so 3a's wrong output shape is underneath most of the others too.
**3d is checker-induced**: in run 4 the hint for 3a moved the model from a kernel whose only real bugs were the
output shape and the copy-out tile, to one with every tile wrong.

Same chain on seat-154 (team's code, no features, partial run): 16 attempts failed with
`cannot reshape array of size 32768 into shape (1,64)`. **0 of 16 contain `reshape`.** All allocate
`psum = (1, lhsT.shape[1])`, which is the (1, N) hint applied to the result tile; `nc_matmul` then fails
reshaping its `(64, 512)` result into it. The model was then told "Do not reshape".

### Level 4 - matmul, tiled (80 failed attempts)

| Type | Attempts | What the kernel did | Message the model saw (start) |
|---|---|---|---|
| 4a No tiling at all | 69 | whole `lhsT` `(K, M)` and whole `rhs` `(K, N)` in one SBUF tile each, one `nc_matmul`. Passes only `K=128 M=128 N=512`, the one shape that fits a single tile | `1 of 4 shapes passed. On K=256 M=256 N=1024: dma_copy dst partition dimension 256 exceeds maximum 128` + a loop instruction |
| 4b Result copied out through an operand's tile | 10 | `tensor_copy(dst=sbuf_lhsT, src=psum)`: the `(M, N)` result into the `(K, M)` operand tile. Fails even the single-tile shape (0/4) | `value array of shape (65536,) could not be broadcast to indexing result of shape (16384,)` |
| 4c Loops over K but stages whole tensors | 1 | loops `K` in 128s over views of a whole-tensor SBUF tile | same partition-256 error |

4a is one kernel, resent unchanged in 60 of 60 repair rounds. The message names the shape that failed, but
never says which shapes passed and what separates them (K <= 128 vs K > 128).

### Level 2 (not in scope, for reference)

Solved 2/5. The 3 failing runs went the same way each time: a tile sized `(P, F1, F2)` that does not match the
copy (40 attempts, `src=384, dst=512` then `dst=12`), then an index past the end of a dimension
(48 attempts, `Out-of-bound access ... index 3 exceed dimension size of 3`), resent unchanged 36/36 times.

## 3. Hypothesis verdicts

**H1 - Levels 1 and 3 do not fail because of tiling. CONFIRMED.** Level 3: 0 of 84 attempts contain a loop;
every failure is a tile's shape or role (3a-3d). Level 1: the kernels do loop (one 2x2 window per iteration,
which is unnecessary since every test shape has C <= 128), but no failure comes from a tile limit. They come
from API misuse on top of a wrong algorithm (type 1, 160/160). Example: level 1, any run, round 4.

**H2 - Level 3's reshape error comes from inside the simulator. CONFIRMED, with a correction.** It does not
occur in the seat-152 baseline at all (0/84). On seat-154 it hit 16 attempts and **0 of 16 contain
`reshape`**: `nc_matmul` reshapes its result into a `(1, 64)` PSUM tile the model allocated. "Do not reshape"
answered a question about code that was not there. More important: the `(1, 64)` tile was the model following
the checker's own 1-D hint ("give a length-N vector the shape (1, N)"). The hint causes the reshape error.
Example: seat-154, run 1, level 3, rounds 2-5.

**H3 - Level 4 passes only the square shape because the model swaps K and M. KILLED.** In every failing
kernel K is the partition dimension of both operands and the output is `(M, N)`: nothing is swapped. The
shape that passes, `K=128 M=128 N=512`, is the only one where whole tensors fit one tile. 69 of 80 attempts
do not tile at all (type 4a); 1 tiles K. Example: run 1, level 4, round 0 (identical in all runs).

**H4 - Discarding 3 of 4 samples wastes information. KILLED on this server.** Rounds in which the 4 samples
failed in more than one way: level 1 0/40, level 2 3/23, level 3 4/21, level 4 5/20; 12 of 104 overall.
In most rounds the 4 samples are the same kernel. Change D has little to choose between; the binding
constraint is the lack of diversity, not how it is used.
