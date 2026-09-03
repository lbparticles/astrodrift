#!/usr/bin/env python3
"""SLOC cap checker -- JPL Power-of-Ten rule 4 companion (see tools/JPL10.md).

No source file may exceed 500 SLOC unless it is listed in
tools/sloc_exclusions.txt (one glob per line; blank lines and `# comments`
ignored; every entry must carry a reason).

SLOC counting rule (deliberately simple and deterministic):
  - blank lines do not count
  - comment-only lines do not count:
      .rs : `//` line comments and `/* ... */` block comments (stateful)
      .py : lines whose first non-whitespace character is `#`
  - everything else (including doc comments and string contents) counts
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from pathlib import Path

LIMIT = 500
CODE_GLOBS = ("*.rs", "*.py")
ROOT = Path(__file__).resolve().parent.parent
EXCLUSIONS = ROOT / "tools" / "sloc_exclusions.txt"


def tracked_code_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", *CODE_GLOBS],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(line for line in out.stdout.splitlines() if line)


def load_exclusions() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in EXCLUSIONS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            entries.append((line, ""))
    # keep the reasons recorded next to each glob on the same line
    for raw in EXCLUSIONS.read_text().splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            glob, _, reason = stripped.partition("#")
            glob, reason = glob.strip(), reason.strip()
            entries = [
                (g, reason if g == glob else r) for (g, r) in entries
            ]
    return entries


def excluded(paths: list[str], entries: list[tuple[str, str]]) -> dict[str, str]:
    hits: dict[str, str] = {}
    for path in paths:
        for glob, reason in entries:
            if fnmatch.fnmatch(path, glob):
                hits[path] = reason or "no reason recorded"
                break
    return hits


def count_sloc(path: str) -> int:
    """Count non-blank, non-comment-only lines (see module docstring)."""
    text = (ROOT / path).read_text(errors="replace")
    count = 0
    in_block = path.endswith(".rs") and False
    for line in text.splitlines():
        stripped = line.strip()
        if path.endswith(".rs"):
            if in_block:
                if "*/" in stripped:
                    in_block = False
                    stripped = stripped.split("*/", 1)[1].strip()
                else:
                    continue
            while stripped.startswith("/*"):
                if "*/" in stripped:
                    stripped = stripped.split("*/", 1)[1].strip()
                else:
                    in_block = True
                    stripped = ""
            if stripped.startswith("//"):
                stripped = ""
            if stripped:
                count += 1
        else:  # .py
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def main() -> int:
    paths = tracked_code_files()
    entries = load_exclusions()
    skips = excluded(paths, entries)

    violations: list[tuple[str, int]] = []
    rows: list[tuple[str, int]] = []
    for path in paths:
        sloc = count_sloc(path)
        rows.append((path, sloc))
        if sloc > LIMIT and path not in skips:
            violations.append((path, sloc))

    for path, sloc in sorted(rows, key=lambda r: -r[1]):
        if path in skips:
            print(f"  SKIP {sloc:5d}  {path}  [{skips[path]}]")
    for path, sloc in violations:
        print(f"  OVER {sloc:5d}  {path}  (limit {LIMIT})")

    if violations:
        print(
            f"\n{len(violations)} file(s) exceed the {LIMIT}-SLOC cap. "
            "Split the file, or add it to tools/sloc_exclusions.txt with a reason.",
            file=sys.stderr,
        )
        return 1
    print(f"SLOC OK: {len(paths)} files checked (limit {LIMIT}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
