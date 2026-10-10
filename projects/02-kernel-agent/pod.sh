#!/usr/bin/env bash
# pod.sh -- run from your LAPTOP, in a terminal that has the workshop credentials pasted.
# Moves this folder to your seat pod, starts measured runs with the same settings every time, and
# brings the logs back. The pod is only a runner: edit here, never in the pod.
#
#   export SEAT=154                  your own seat number, once per terminal
#
#   ./pod.sh push                    copy this folder's code into the pod
#   ./pod.sh check                   prove the checker in the pod: nkibench + diagnose selftests, probe
#   ./pod.sh run NAME [FEATURES]     start a measured run in the background, e.g.
#                                      ./pod.sh run baseline
#                                      ./pod.sh run locate locate
#   ./pod.sh status [NAME]           is a run going, and the last lines of its output
#   ./pod.sh pull NAME               copy NAME.jsonl and NAME.out into results/seat-$SEAT/
#   ./pod.sh get REMOTE [LOCAL]      copy any one file from the pod's project folder
#
# Every measured run uses the same command, so only the code and the features differ between two
# runs. Change RUN_ARGS and you can no longer compare against anything run before.
set -euo pipefail

RUN_ARGS=${RUN_ARGS:---all --rounds 8 --samples 4 --context 8192 --repeat 5}
REMOTE=${REMOTE:-/workspace/projects/02-kernel-agent}
HERE=$(cd "$(dirname "$0")" && pwd)
RESULTS=$(cd "$HERE/../.." && pwd)/results

[ -n "${SEAT:-}" ] || { echo "Set your seat first:  export SEAT=<your number>" >&2; exit 1; }
POD=seat-$SEAT
in_pod() { kubectl exec -i "$POD" -- bash -c "cd $REMOTE && $1"; }

case "${1:-}" in
  push)
    # The harness itself must be the one the organisers shipped, or scores stop being comparable
    # with theirs and with a baseline run on the untouched pod. Say so before overwriting it.
    mine=$(shasum -a 256 "$HERE/nkibench.py" | cut -d' ' -f1)
    theirs=$(kubectl exec "$POD" -- sha256sum "$REMOTE/nkibench.py" | cut -d' ' -f1)
    if [ "$mine" != "$theirs" ]; then
      echo "STOP: nkibench.py in $POD differs from yours. Nothing was copied." >&2
      echo "Bring the pod's copy here and look at the difference first:" >&2
      echo "  ./pod.sh get nkibench.py nkibench.pod.py" >&2
      [ "${FORCE:-}" = 1 ] || exit 1
    fi
    # tar over exec rather than kubectl cp, so logs and caches in either place are left alone.
    # devtools/ stays on the laptop: its stand-in simulator must never be importable in the pod.
    COPYFILE_DISABLE=1 tar --no-xattrs -cf - -C "$HERE" \
        --exclude '__pycache__' --exclude './devtools' --exclude '*.jsonl' --exclude '*.out' \
        --exclude '*.log' . \
      | kubectl exec -i "$POD" -- tar -xf - -C "$REMOTE" --no-same-owner
    in_pod "python -c 'import agent, diagnose; print(\"pushed to $POD; agent.py knows features:\", sorted(agent.KNOWN_FEATURES))'"
    ;;
  check)
    in_pod "python nkibench.py --selftest > check_nkibench.out 2>&1; echo \"nkibench selftest exit \$?\"; \
            python diagnose.py --selftest > check_diagnose.out 2>&1; echo \"diagnose selftest exit \$?\"; \
            python probe_sim.py > check_probe.out 2>&1; echo \"probe exit \$?\""
    mkdir -p "$RESULTS/$POD"
    for f in check_nkibench.out check_diagnose.out check_probe.out; do
      kubectl exec "$POD" -- cat "$REMOTE/$f" > "$RESULTS/$POD/$f"
    done
    echo "saved to $RESULTS/$POD/check_*.out"
    ;;
  run)
    NAME=${2:?give the run a name, e.g. ./pod.sh run locate locate}
    FEATURES=${3:-}
    in_pod "python3 -c 'import urllib.request; urllib.request.urlopen(\"http://localhost:8000/health\", timeout=5)' 2>/dev/null \
              || { echo 'The model is not answering on this seat yet. In the pod: cd /workspace && ./serve.sh, and wait for READY.' >&2; exit 1; }; \
            if [ -e $NAME.jsonl ]; then echo '$NAME.jsonl already exists in the pod. Pick another name: a log holding two runs cannot be told apart.' >&2; exit 1; fi; \
            if pgrep -fa 'python agent.py' | grep -qv 'bash -c'; then echo 'A run is already going on this seat. Two at once share one model and change each other. Wait, or use another seat.' >&2; exit 1; fi; \
            nohup python agent.py $RUN_ARGS --features '$FEATURES' --log $NAME.jsonl > $NAME.out 2>&1 < /dev/null & \
            sleep 2; echo \"started $NAME on $POD: agent.py $RUN_ARGS --features '$FEATURES'\"; head -3 $NAME.out"
    ;;
  status)
    in_pod "pgrep -fa 'python agent.py' | grep -v 'bash -c' || echo 'no run going'; \
            ls -la *.jsonl *.out 2>/dev/null | awk '{print \$5, \$9}'; \
            f=${2:+$2.out}; f=\${f:-\$(ls -t *.out 2>/dev/null | grep -v '^check_' | head -1)}; \
            if [ -n \"\$f\" ]; then echo \"---- \$f\"; tail -n 12 \"\$f\"; fi"
    ;;
  pull)
    NAME=${2:?which run? e.g. ./pod.sh pull baseline}
    mkdir -p "$RESULTS/$POD"
    for f in "$NAME.jsonl" "$NAME.out"; do
      if kubectl exec "$POD" -- test -e "$REMOTE/$f"; then
        kubectl exec "$POD" -- cat "$REMOTE/$f" > "$RESULTS/$POD/$f"
        echo "$RESULTS/$POD/$f  ($(wc -l < "$RESULTS/$POD/$f" | tr -d ' ') lines)"
      else
        echo "no $f in the pod"
      fi
    done
    ;;
  get)
    SRC=${2:?which file?}
    mkdir -p "$RESULTS/$POD"
    kubectl exec "$POD" -- cat "$REMOTE/$SRC" > "$RESULTS/$POD/${3:-$(basename "$SRC")}"
    echo "$RESULTS/$POD/${3:-$(basename "$SRC")}"
    ;;
  *)
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
