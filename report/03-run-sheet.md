# Run sheet and results

Frozen settings for every scored run: `--all --rounds 8 --samples 4 --context 8192 --repeat 5`, model
Qwen/Qwen3-8B on the seat's own server. Only `--log` / `--features` differ. One new log name per run.

## Runs

| Condition | Seat | Exact command | Code version | Started | Log files | Notes |
|---|---|---|---|---|---|---|
| Baseline | seat-152 | `python agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5 > run.log` | organisers' `d11ccdf` | ? (finished before 13:42) | `results/seat-152/baseline.jsonl`, `baseline-run.log` | 416 attempts, clean |
| Baseline, team code (side record) | seat-154 | `./pod.sh run baseline` (`--features ''`) | `17638cf` | ~12:40 | `results/seat-154/baseline.jsonl`, `.out` | partial: run 1 + start of run 2 |
| A (locate) | seat-152 | `./pod.sh run locate locate` | | | | |
| A+B (state) | seat-155 | `./pod.sh run state state` | | | | |
| C (stretch) | | | | | | |

## Results (solved/5, then best / worst / mean)

| Level | Baseline | With A | With A and B | With C |
|---|---|---|---|---|
| 1 | 0/5, 0.30 / 0.30 / 0.30 | | | |
| 2 | 2/5, 1.00 / 0.30 / 0.58 | | | |
| 3 | 0/5, 0.30 / 0.30 / 0.30 | | | |
| 4 | 0/5, 0.62 / 0.62 / 0.62 | | | |

Reading rules: levels 1, 3 and 4 never varied, so any change there is a real effect of the condition. But the
model is near-deterministic, so the failure *type* must also move (taxonomy re-done per run) before it can be
credited to the information added. Level 2 can only show a collapse; small differences there are noise.
