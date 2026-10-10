#!/usr/bin/env bash
# copy_seat155.sh -- copy Perumal's seat-155 log files from his branch into results/seat-155/,
# as files. Not a merge: his branch carries an older agent.py.
set -euo pipefail
cd "$(dirname "$0")/.."
BR=origin/Perumal/Level1-Prompt
git fetch -q origin Perumal/Level1-Prompt
mkdir -p results/seat-155
n=0
# He has committed logs in two places: under the project folder and at the repo root.
for f in $(git ls-tree -r --name-only "$BR" projects/02-kernel-agent/results/seat-155/ results/seat-155/); do
  case "$(basename "$f")" in
    level1-route-first-prompt-repeat5*|l1-detailed.*|l1-detailed-b.*) ;;
    *) continue ;;
  esac
  git show "$BR:$f" > "results/seat-155/$(basename "$f")"
  echo "  copied results/seat-155/$(basename "$f")"; n=$((n + 1))
done
echo "$n file(s) from $BR ($(git log -1 --format=%h "$BR"))"
for want in level1-route-first-prompt-repeat5 l1-detailed. l1-detailed-b.; do
  ls results/seat-155/ | grep -q "^$want" || echo "  MISSING: nothing named $want* on $BR yet"
done
