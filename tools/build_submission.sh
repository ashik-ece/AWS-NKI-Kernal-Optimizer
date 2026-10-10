#!/usr/bin/env bash
# build_submission.sh -- assemble the final pull request to the organisers' repo.
#
#   tools/build_submission.sh "Team 31"            build it locally and show what would be submitted
#   tools/build_submission.sh "Team 31" --push     also fork, push the branch and open the PR
#
# It starts from a fresh clone of yahavb/trainium-agent-labs at master, so none of this team repo's
# history comes along -- only files. Into that clone it copies, from this team repo as it is on GitHub:
#
#   projects/02-kernel-agent/                the code, from origin/dev
#   projects/02-kernel-agent/submission/
#       NOTE.md                              John's final note (report/05-one-page-final.md, or
#                                            report/04-one-page-note.md if there is no final yet)
#       report/                              his other report files, without the internal memo
#       results/                             every log and check, from origin/dev, plus the seat-152
#                                            baseline from origin/john-manage
#
# The note refers to results/... and report/..., so submission/ mirrors that layout and every path in
# it still resolves. Before anything is pushed it refuses to continue if the tree contains a
# credential, a laptop path, an AI-tool mention, or a harness that differs from the organisers'.
set -euo pipefail

TEAM=${1:?usage: tools/build_submission.sh "Team NN" [--push]}
PUSH=${2:-}
UPSTREAM=yahavb/trainium-agent-labs
BRANCH=${BRANCH:-kernel-agent-checker}
NOTE=${NOTE:-}
TEAM_REPO=$(cd "$(dirname "$0")/.." && pwd)
PROJ=projects/02-kernel-agent

echo "== fetching the team repo's dev and john-manage"
git -C "$TEAM_REPO" fetch -q origin dev john-manage

WORK=$(mktemp -d)
SRC=$WORK/src
mkdir -p "$SRC"
git -C "$TEAM_REPO" archive origin/dev | tar -x -C "$SRC"

echo "== fresh clone of $UPSTREAM"
git clone -q "https://github.com/$UPSTREAM.git" "$WORK/up"
cd "$WORK/up"
git checkout -q -b "$BRANCH" origin/master

echo "== copying files"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$SRC/$PROJ/" "$PROJ/"
SUB=$PROJ/submission
mkdir -p "$SUB/report" "$SUB/results"
rsync -a "$SRC/results/" "$SUB/results/"
for f in $(git -C "$TEAM_REPO" ls-tree --name-only origin/john-manage results/seat-152/); do
  [ -e "$SUB/$f" ] || git -C "$TEAM_REPO" show "origin/john-manage:$f" > "$SUB/$f"
done
for f in $(git -C "$TEAM_REPO" ls-tree --name-only origin/john-manage report/); do
  case "$f" in *memo*) continue ;; esac
  git -C "$TEAM_REPO" show "origin/john-manage:$f" > "$SUB/$f"
done
if [ -z "$NOTE" ]; then
  NOTE=report/05-one-page-final.md
  git -C "$TEAM_REPO" cat-file -e "origin/john-manage:$NOTE" 2>/dev/null || NOTE=report/04-one-page-note.md
fi
git -C "$TEAM_REPO" show "origin/john-manage:$NOTE" > "$SUB/NOTE.md"
echo "   NOTE.md is $NOTE"

echo "== checks"
fail=0
cmp -s "$PROJ/nkibench.py" <(git show "origin/master:$PROJ/nkibench.py") \
  && echo "   nkibench.py identical to the organisers' copy" \
  || { echo "   STOP: nkibench.py differs from the organisers' copy"; fail=1; }
hits=$(grep -rIlE 'ASIA[A-Z0-9]{14,}|IQoJ[A-Za-z0-9+/]{20,}|AWS_SECRET_ACCESS_KEY=.{8,}' . --exclude-dir=.git || true)
[ -z "$hits" ] && echo "   no credentials" || { echo "   STOP: credential-like strings in: $hits"; fail=1; }
hits=$(grep -rIl '/Users/' "$PROJ" || true)
[ -z "$hits" ] && echo "   no laptop paths" || { echo "   STOP: laptop paths in: $hits"; fail=1; }
hits=$(grep -rIliE 'claude|anthropic|cursoragent|chatgpt|copilot|co-authored-by' "$PROJ" || true)
[ -z "$hits" ] && echo "   no AI-tool mentions" || { echo "   STOP: AI-tool mentions in: $hits"; fail=1; }
python3 -m py_compile $(find "$PROJ" -name '*.py' -not -path '*/fake_nki/*') && echo "   all Python compiles"
find . -name '__pycache__' -prune -exec rm -rf {} +
[ "$fail" = 0 ] || { echo "Not submitting. Fix the above. Built tree left in $WORK/up"; exit 1; }

git add -A
git -c user.name="$(git -C "$TEAM_REPO" config user.name)" \
    -c user.email="$(git -C "$TEAM_REPO" config user.email)" \
    commit -q -m "$TEAM: kernel agent -- a checker that names the kernel's mistake, and holes found in the harness"
echo "== what would be submitted (against the organisers' master)"
git diff --stat origin/master HEAD | tail -15
echo "   $(git diff --name-only origin/master HEAD | wc -l | tr -d ' ') files; $(du -sh "$SUB/results" | cut -f1) of results"

cat > "$WORK/pr-body.md" <<EOF
**$TEAM, project 2: the kernel agent.** We changed what the checker tells the model, and measured it.
Start with [\`$SUB/NOTE.md\`]($SUB/NOTE.md); paths in it are relative to \`$SUB/\`.

- **Level 3 solved.** 0.30 in all 15 baseline runs. Solved by the checker in 6 of 7 completed runs, on two
  chips, with only the feedback changed. The round-by-round recovery: \`$SUB/results/level3-recovery-transcript.md\`.
- **Holes in the harness, each with a repro kernel** in \`$PROJ/findings/\`, run through \`nkibench.py --check\`
  on the real simulator (\`$SUB/results/seat-150/findings.out\`): on-chip tiles taller than 128 rows and
  kernels returning their output from PSUM both pass; the rule scan sees \`(256, M)\` but not \`(K, M)\`; two
  simulator messages point at the wrong thing; and the agent could grade one sample with another's code.
- **Level 4** moves from 0.62 to 0.75 as the harness scores it, but that 0.75 rests on tiles the chip does not
  have. With the 128-row rule enforced (\`--features strict\`) it is back to 0.62. Level 1 did not move.
- \`nkibench.py\` is the organisers' copy, unchanged. Every checker change is behind \`--features\`. With none
  set, the agent sends the original's prompts and feedback, with two exceptions stated in the note: level 1's
  first prompt carries a worked plan ("prompt C"), and each candidate is written to its own file (the bytecode
  fix). Every run, its seat, commit and flags: \`$SUB/results/RUNS.tsv\`.

Nothing here was timed on the device; all numbers come from \`nki.simulate\`.
EOF
echo "== PR description, editable before pushing: $WORK/pr-body.md"
sed 's/^/   | /' "$WORK/pr-body.md"

if [ "$PUSH" != "--push" ]; then
  echo
  echo "Built, not pushed. Look at $WORK/up, then rerun with --push."
  exit 0
fi

OWNER=$(gh api user --jq .login)
echo "== forking $UPSTREAM to $OWNER (no-op if the fork exists)"
gh repo fork "$UPSTREAM" --clone=false --remote=false >/dev/null 2>&1 || true
for i in 1 2 3 4 5 6; do gh api "repos/$OWNER/trainium-agent-labs" >/dev/null 2>&1 && break; sleep 5; done
git push -q -f "https://github.com/$OWNER/trainium-agent-labs.git" "$BRANCH"
echo "== opening the pull request"
gh pr create --repo "$UPSTREAM" --base master --head "$OWNER:$BRANCH" \
  --title "$TEAM: kernel agent -- a checker that names the kernel's mistake, and holes found in the harness" \
  --body-file "$WORK/pr-body.md"
