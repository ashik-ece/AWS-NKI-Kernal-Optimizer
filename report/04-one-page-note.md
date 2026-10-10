# The agent was stuck on the checker's words, not on tiling

**Team:** John (measurement, write-up), Achyuthan (checker), Perumal (agent and prompt).
**Draft 15:20.** Numbers marked TODO are runs still in progress.

## Claim

The organisers' kernel agent sat at fixed scores on three levels: 0.30 (level 1), 0.30 (level 3) and 0.62
(level 4). Their reading was that the model does not understand tiling. Ours: **on levels 3 and 4 the model
was stuck because the checker's feedback described the simulator's crash instead of the kernel's mistake, and
in two places the feedback told it to do the wrong thing.** Changing only what the checker says solved level 3
in 4 of 5 runs and lifted level 4 from 0.62 to 0.75 in 5 of 5. Level 1 did not move, because it fails on the
algorithm, which no checker message can reach.

## What we ran

- Model: Qwen/Qwen3-8B on each seat's own server, thinking off. Simulator: nki 0.6.0, CPU (no device timing).
- Frozen settings for every scored run: `--rounds 8 --samples 4 --context 8192 --repeat 5`. Runs differ only
  in `--features` and level selection; levels are solved independently, so single-level runs compare with
  `--all` runs. Every run, its seat, commit and flags: `results/RUNS.tsv`.
- **Baseline: 15 runs on 3 seats.** Seat 152 ran the organisers' untouched code (5 runs); seats 154 and 158
  ran ours with every feature off (10 runs). They agree: levels 1, 3 and 4 never varied.
- Checker: about 20 runs over 10 seats, one feature set per run, so each change can be credited separately.

## Results (solved / runs; every run's best score in brackets where it varies)

The reported checker is the flag set `internal,origin`. It has produced identical feedback at every commit
since `39b8e5a` (replayed on all 241 distinct failing kernels in our logs, zero differences; TODO: file in
`results/`), so its runs are pooled across commits.

| Level | Baseline (15 runs) | Checker `internal,origin` | TODO: `ahead` / `pieces` | Prompt C, short (level 1) |
|---|---|---|---|---|
| 1 average pooling | 0/15, always 0.30 | 0/5, always 0.30 | n/a | 0/5, always 0.30 (failure type moved; TODO verify) |
| 2 transpose | 6/15 (1/5 to 3/5 per seat) | 2/5 | TODO | n/a |
| 3 matmul, one tile | 0/15, always 0.30 | **6/8** on two chips [1, 1, 0.3, 1, 1 / 1, 1, 0.3] | TODO | n/a |
| 4 matmul, tiled | 0/15, always 0.62 | 0/5, **always 0.75** | TODO | n/a |

Level 3's solves came by **two different paths** (seat 153: runs 1, 2, 4 share one; run 5 starts from a
different kernel), and the same flags on a second chip (seat 157) solved 2 of 3. Level 2 moves between 1/5 and 3/5 in
every condition, baselines included: that is noise, not an effect.

**Which ingredient did it (level 3, 5 runs each unless noted):**

| Features | What the model is told, in addition | Level 3 | Level 4 |
|---|---|---|---|
| none (baseline) | the simulator's error plus advice keyed on its text | 0/15 | 0.62 |
| `locate` | the line of the kernel that failed | 0/5 | |
| `state` | plus the shape and buffer of every tile on that line, and the rule they break | 0/5 | **0.75** |
| `facts` | rule *replaces* the advice for every error | 0/4 | |
| `facts, origin` | plus the line that gave the wrong tile its shape | 0/5 | 0.62 (4 runs) |
| `internal, origin` | rule replaces the advice only for errors from *inside* the simulator | **4/5** | **0.75** |

Naming the line is not enough. What solved level 3 was tracing the wrong tile back to where it got its shape,
**and** dropping the advice only where it was keyed on a numpy error from inside the simulator. Dropping the
advice everywhere (`facts`) lost level 4, because for the simulator's own checks the advice carries the how-to.

## Why it was stuck: failure taxonomy of the baseline (416 attempts, every sample)

- **Level 3, 84 failed attempts, 0 contain a loop.** The common bug: the result tile given one dimension
  (`lhsT.shape[1:]`, in 62 of 84). The existing hint answered "give a vector the shape (1, N)"; the model
  followed it, and on another seat that produced 16 `cannot reshape` failures in kernels with **no reshape in
  them** (0 of 16), which were then told "Do not reshape". A second hint's example tile, (128, 512), was the
  exact wrong tile the model had allocated.
- **Level 4, 80 failed attempts: one kernel in 69 of them, never tiling.** It passes only the shape that fits a
  single tile, and was resent unchanged in 60 of 60 repair rounds. With `state` it starts tiling K, gets past
  the contraction limit (next attempt scored higher 12 of 12 times), and stalls on the next rule: M per tile
  at most 128.
- **Level 1, 160 failed attempts: the wrong algorithm in all 160.** The mean is computed as a matrix multiply
  of each window with itself, times 0.5, copied from the prompt's API card. It never reaches the numerical
  check, so the checker never sees the algorithm. All 10 checker runs on level 1 open with the same five API
  errors as the baseline.

Full taxonomy with examples: `report/01-baseline-and-taxonomy.md`.

## Caveats we measured

- **The model is near-deterministic** on this server: on level 1 all five baseline runs were the same kernels
  character for character, and the four samples per round differed in only 12 of 104 rounds. So any change to
  the feedback changes the trajectory. We credit a change only where the failure type also moved toward what
  the new message said (level 4: from "never tiles" to "tiles K, stalls on M"; level 3: each round's message
  names a tile and the model fixes that tile next).
- **Adding flags can regress.** `internal,origin` plus `pieces,shapes` kept level 3 at 4/5 but returned level 4
  to 0.62: at the contraction step it named three limits at once and the model stopped moving. Adoption rule:
  a flag is adopted only if it is better on the level it targets and no worse on any level it can change (found
  by replay), and on level 3, where the checker already varies, "better" needs more than one run.
- **Every flag set we tried is reported**, including the ones that lost, so the winner is not cherry-picked.
- **Simulator only.** Every number is CPU simulation; no kernel ran on the device and no latency was measured.
- **Prompt change C** (level 1) is reported separately. TODO: quote the hint verbatim. The version in `dev`
  names the `.ap()` view, the reduction axes and the `1/(p*p)` scale, i.e. the reference algorithm; a level-1
  solve with it would show the model can follow a recipe, not that it derived one.

## Conclusion

TODO (final wording after the last runs). Draft: two of the three walls moved, and they moved by changing only
what the checker tells the model, never what the answer is. Level 3 went from 0 of 15 to 4 of 5; level 4 from
0.62 to 0.75 in every run. The third wall, level 1, did not move under any checker change, and the taxonomy
says why: the model's algorithm is wrong before any rule is broken. The largest single lesson is that the
existing feedback was not only vague but in two places misleading, and removing misleading advice mattered as
much as adding precise facts.
