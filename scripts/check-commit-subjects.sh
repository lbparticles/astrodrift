#!/usr/bin/env bash
# Validate commit/PR subjects against the conventional-commit format that
# release-plz uses to build CHANGELOG.md (see README, "CHANGELOG").
#
# Reads subjects on stdin, one per line. Allowed:
#   type(scope)!: description   feat | fix | docs | style | refactor | perf |
#                               test | build | ci | chore | revert
#   Bump X from A to B          (dependabot; lands under "Dependencies")
#   Merge ...                   (merge commits; CI filters these separately)
set -uo pipefail

conventional='^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([A-Za-z0-9_./-]+\))?!?: .+'
dependabot='^Bump [^[:space:]]+ from [^[:space:]]+ to [^[:space:]]+'
merge='^Merge '

status=0
while IFS= read -r subject; do
  [[ -z $subject ]] && continue
  if [[ $subject =~ $merge || $subject =~ $conventional || $subject =~ $dependabot ]]; then
    continue
  fi
  printf 'invalid subject: %s\n' "$subject" >&2
  printf '  expected "type(scope)?: description" with type one of feat, fix, docs,\n' >&2
  printf '  style, refactor, perf, test, build, ci, chore, revert\n' >&2
  status=1
done
exit $status
