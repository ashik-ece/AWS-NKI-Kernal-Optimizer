# AWS NKI Kernel Optimizer

Team repository for **AWS Annapurna Labs / Trainium Agent Labs — Project 2: the kernel agent**.

An LLM writes Neuron Kernel Interface (NKI) kernels, runs them in the CPU simulator, reads failure feedback, and retries. This repo holds the team’s checker improvements, measured runs, failure taxonomy, and final write-up from the Oct 2026 hackathon (Team 31).

**Submission:** [yahavb/trainium-agent-labs#16](https://github.com/yahavb/trainium-agent-labs/pull/16)  
**Start with the one-page note:** [`report/05-one-page-final.md`](report/05-one-page-final.md)

---

## Claim

The organisers’ agent sat at fixed scores on levels 1, 3 and 4. The usual reading was that the model does not understand tiling. On level 3 the blocker was different: **the checker described the simulator’s crash instead of the kernel’s mistake, and in two places told the model to do the wrong thing.**

Changing only what the checker says (never the answer) took level 3 from **0/15** baseline runs to **16/17** completed runs on four seats, with every solve inside the chip’s rules.

| Level | Baseline | Reportable checker (`internal,origin`) | Notes |
|---|---|---|---|
| 1 average pooling | 0/15 (0.30) | 0/5 (0.30) | Prompt hint fixed the algorithm, not the score |
| 2 transpose | 6/15 | 2/5 | Noise range across conditions |
| 3 matmul, one tile | **0/15** (0.30) | **16/17** | Clean win; all solves HBM + ≤128 rows |
| 4 matmul, tiled | 0.62 | 0.75 as scored; **0.62 with `strict`** | Tall-tile loophole in the harness |
| 8 attention (untuned) | 0/5 (0.30) | 0/5 (0.30) | No transfer |

Full tables, caveats, and ingredient ladder: [`report/05-one-page-final.md`](report/05-one-page-final.md).

---

## What we changed

Checker behaviour is gated behind `--features` in `agent.py` / `diagnose.py`. With no flags set, behaviour matches the organisers’ agent except for two documented exceptions (level-1 prompt plan “C”, and writing each candidate to its own file to avoid a stale-bytecode grading bug).

| Flag set | Role |
|---|---|
| `locate` | Name the failing line |
| `state` | Tile shapes/buffers on that line and the rule broken |
| `origin` | Trace the wrong tile to the line that set its shape |
| `internal` | Drop misleading advice for numpy errors from *inside* the simulator |
| `ahead` | Look ahead to later writes / roles of a tile |
| `strict` | Enforce on-chip tile height ≤ 128 at allocation time |
| `pieces`, `shapes`, `names`, `example`, `layout` | Explored; reported; not adopted as the final checker |

The **reportable checker** is `internal,origin`.

`nkibench.py` (the organisers’ harness) is **byte-identical** to the upstream copy. Scoring gaps were reproduced with their own `--check` (see `projects/02-kernel-agent/findings/`).

---

## Repository layout

```
.
├── projects/02-kernel-agent/     # Agent, checker, harness, pod tooling
│   ├── agent.py                  # Repair loop + feature flags
│   ├── diagnose.py               # Rich failure feedback
│   ├── nkibench.py               # Organisers’ grader (unchanged)
│   ├── pod.sh                    # Laptop → seat pod: push / run / pull
│   ├── findings/                 # Repro kernels for harness holes
│   ├── reference_level*.py       # Reference kernels
│   └── README.md                 # Upstream project brief
├── report/                       # Measurement write-up
│   ├── 01-baseline-and-taxonomy.md
│   ├── 02-memo-achyuthan.md      # Internal checker memo (not for submission)
│   ├── 03-run-sheet.md
│   ├── 04-one-page-note.md       # Long working note
│   └── 05-one-page-final.md      # Final note → submitted as NOTE.md
├── results/                      # Logs, RUNS.tsv, replays, transcripts
│   ├── RUNS.tsv                  # Every scored run: seat, name, flags, commit, args
│   ├── level3-recovery-transcript.md
│   └── seat-150 … seat-159/
└── tools/
    ├── build_submission.sh       # Assemble PR to organisers’ repo
    └── copy_seat155.sh           # Copy Perumal’s seat-155 logs into results/
```

---

## How we measured

- **Model:** Qwen/Qwen3-8B on each seat’s server, thinking off  
- **Simulator:** nki 0.6.0 (`nki.simulate`), CPU only — no device latency  
- **Frozen settings** for scored runs: `--rounds 8 --samples 4 --context 8192 --repeat 5`  
- **Baseline:** 15 runs on 3 seats (organisers’ code + team code with features off)  
- **Pooling across commits** only where a feedback replay shows identical messages (`results/replay-*.txt`)

From a laptop with workshop credentials and `kubectl` access to a seat:

```bash
export SEAT=154   # your seat
cd projects/02-kernel-agent
./pod.sh serve
./pod.sh push
./pod.sh check
./pod.sh run checker-l3 internal,origin 3
./pod.sh status checker-l3
./pod.sh pull checker-l3
```

Replay feedback without calling the model:

```bash
./pod.sh replay checker-l3
```

Reproduce harness findings on the real simulator:

```bash
./pod.sh findings
```

Details: [`projects/02-kernel-agent/README.md`](projects/02-kernel-agent/README.md) and the header of [`pod.sh`](projects/02-kernel-agent/pod.sh).

---

## Reports

| Document | Contents |
|---|---|
| [`report/05-one-page-final.md`](report/05-one-page-final.md) | Final note (submission `NOTE.md`) |
| [`report/01-baseline-and-taxonomy.md`](report/01-baseline-and-taxonomy.md) | Baseline, failure taxonomy, hypotheses |
| [`report/03-run-sheet.md`](report/03-run-sheet.md) | Chronological run sheet |
| [`report/04-one-page-note.md`](report/04-one-page-note.md) | Long working note |
| [`results/level3-recovery-transcript.md`](results/level3-recovery-transcript.md) | One level-3 solve, messages verbatim |
| [`results/RUNS.tsv`](results/RUNS.tsv) | Master list of measured runs |

---

## Build the organisers’ PR

```bash
LEVEL3="16 of 17" tools/build_submission.sh "Team 31"          # dry-run locally
LEVEL3="16 of 17" tools/build_submission.sh "Team 31" --push   # fork, push, open/update PR
```

The script clones the upstream labs repo, copies code from `origin/dev`, the note from `origin/john-manage` (`report/05-one-page-final.md` → `NOTE.md`), and results; it refuses to push if credentials, laptop paths, AI-tool mentions, or a modified `nkibench.py` are detected.

---

## Team

| Person | Role |
|---|---|
| Ashik M. John | Measurement lead, taxonomy, write-up |
| Achyuthan Sivasankar | Checker engineer, features, findings, submission tooling |
| Perumal Marimuthu | Agent / prompt engineer (level-1 prompt conditions) |

Upstream challenge and harness: [yahavb/trainium-agent-labs](https://github.com/yahavb/trainium-agent-labs).

---

## Caveats

- All scores are from **CPU simulation**, not Trainium device timing.  
- Level 4’s harness score of 0.75 can rely on on-chip tiles taller than 128 rows; re-check solves with `--features strict`.  
- The model on this server is near-deterministic; credit a change only when the failure type also moves.  
- Every flag set tried is reported, including conditions that lost or were not adopted.
