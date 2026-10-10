# The agent was stuck on the checker's words, not on tiling

**Team:** John (measurement, write-up), Achyuthan (checker), Perumal (agent and prompt).
Full evidence: `report/01` (baseline, taxonomy, hypotheses), `report/03` (run sheet), `report/04` (long note),
`results/RUNS.tsv` (every run, seat, commit, flags).

## Claim

The organisers' agent sat at fixed scores: 0.30 on level 1, 0.30 on level 3, 0.62 on level 4. Their reading
was that the model does not understand tiling. On level 3 it was something else: **the checker described the
simulator's crash instead of the kernel's mistake, and twice told the model to do the wrong thing.** Changing
only what the checker says, never the answer, took level 3 from 0 of 15 runs to **[18:00: N of M]** runs, on
four chips. Level 4 did not truly move, and level 1 fails on the algorithm, which no checker message can reach.

## What we ran

Qwen3-8B on each seat's server, CPU simulator (nki 0.6.0). Every scored run used the same settings (8 rounds,
4 samples, context 8192, 5 repeats); only the checker flags differ. Baseline: 15 runs on 3 seats, including
the organisers' untouched code. Levels 1, 3 and 4 never varied across them.

| Level | Baseline | Checker (`internal,origin`) | Checker + `ahead` |
|---|---|---|---|
| 1 avg pooling | 0/15 (0.30) | 0/5 (0.30) | not run |
| 2 transpose | 6/15 | 2/5 | 3/5 |
| 3 matmul, one tile | **0/15** (0.30) | **[18:00: 6/7 + checker-l3c]** | **10/10**, faster (4 rounds, not 6) |
| 4 matmul, tiled | 0.62 | 0.75 as scored; **0.62 with the 128-row rule enforced** | not adopted |
| 8 attention (untuned) | [18:00] | [18:00] | not run |

A one-sentence prompt hint on level 1 ("a per-window sum followed by a scale; no matrix multiplication")
scored 0/5 but **fixed the algorithm**: matrix multiply in 0 of 104 attempts (baseline 160 of 160), `nl.sum`
in 103. The detailed recipe: **[18:00: l1-detailed-b]**.

## What made level 3 move

| Checker flags | What the model is also told | Level 3 |
|---|---|---|
| `locate` | the failing line | 0/5 |
| `state` | shapes and buffers on that line, and the rule broken | 0/5 |
| `facts,origin` | the line that gave the wrong tile its shape; rule replaces all advice | 0/5 |
| `internal,origin` | the same, but advice dropped only for numpy errors inside the simulator | **4/5** |

Naming the line was not enough; the model needed the line where the tile got its shape, and the misleading
advice removed. The failures moved where the new information pointed: the baseline never got past loading the
inputs (0 of 84 failed attempts); under the checker 64% got to the matmul or later.

## Why it was stuck (taxonomy of all 416 baseline attempts)

- **Level 3:** the result tile given one dimension (`lhsT.shape[1:]`, 62 of 84). The checker's hint said "give
  a vector the shape (1, N)"; the model obeyed, and got `cannot reshape` errors in kernels that contain no
  reshape (0 of 16), then was told "Do not reshape". Another hint's example tile, (128, 512), was the exact
  wrong tile the model had made.
- **Level 4:** one kernel in 69 of 80 attempts, never tiling, resent unchanged in 60 of 60 repairs.
- **Level 1:** wrong algorithm in 160 of 160: each window matrix-multiplied by itself, times 0.5.

## Caveats, all measured

- **Level 4's 0.75 is a loophole.** 22 of 23 kernels scoring 0.75 allocate tiles of 256 or 512 rows and pass
  128-row slices; the harness checks only the slice. With the rule enforced: 0.62 in 5 of 6 runs, 0.50 in 1.
  What moved is the failure (the model now tiles K), not the score.
- **The model is near-deterministic:** five baseline runs gave identical kernels. So we credit a change only
  when the failure type also moves toward what the new message said.
- **Every flag set is reported**, including the losers: `pieces`, `pieces,shapes`, fixed `ahead`, `names`
  [18:00], and the 16-round `example` group [18:00]. A flag is adopted only if it is better where it aims and
  no worse elsewhere.
- **Simulator only.** It also accepts a result returned from PSUM; we checked every solve, and all checker
  and first-`ahead` solves return from HBM.
- **Pooling across commits** only where a replay of every logged kernel gives identical feedback.
- **A grading bug in the agent, fixed:** candidates shared one file path, so importlib's bytecode cache could
  grade one sample as another; no logged score changed. The harness is the organisers' copy, byte for byte.

## Conclusion

One wall fell cleanly. Level 3 went from 0 of 15 to [18:00] of [18:00] runs, every solve within the chip's
rules, by changing only what the checker says. Level 4's kernels learned to tile, but their higher score
depends on a tile the harness fails to reject, so honestly it is still 0.62. Level 1 needs the algorithm, not
better errors: one sentence of prompt fixed the method, not the score. The largest lesson: the existing
feedback was not just vague but misleading in two places, and removing bad advice mattered as much as adding
precise facts.
