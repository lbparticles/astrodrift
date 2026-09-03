# JPL Power-of-Ten — mapping to this repository

The ten rules from JPL's *The Power of Ten: Rules for Developing
Safety-Critical Code* (H. J. Holzmann, IEEE Computer, 2006), how each is (or
cannot be) enforced for this Rust/Python codebase, and where the enforcement
lives. The gate is the lefthook `pre-commit` hook (`lefthook.yml`); install
with `uv tool install lefthook && lefthook install`.

| # | Rule | Enforcement here |
|---|------|------------------|
| 1 | Avoid complex control flow (no `goto`, no direct/indirect recursion) | Rust has no `goto`. Recursion is not machine-checkable by clippy — manual review item. `clippy::while_immutable_condition`, `clippy::never_loop` active. |
| 2 | All loops must have fixed upper bounds | `clippy::infinite_loop` = **deny** (workspace lints). The two `while` loops inside the DOPRI5(4) transliteration are the reference control flow (excluded file, documented). |
| 3 | No heap allocation after initialization | Not statically checkable in general — manual review item. All allocation is confined to setup/builder paths; kernel and dispatch paths are allocation-free. |
| 4 | No function longer than one printed page; small files | `clippy::too_many_lines` (100 lines) = warn — allowed **only** in the excluded transliterations. Companion file-level cap: **≤ 500 SLOC per file**, enforced by `tools/check_sloc.py` with exemptions in `tools/sloc_exclusions.txt` (each entry requires a reason). |
| 5 | Minimum two assertions per function | Not lintable — covered by the permanent test suites (`cargo test`, galpy fixture corpora). |
| 6 | Data at the smallest possible scope | `clippy::items_after_statements`, `clippy::redundant_pub_crate` active (pedantic). |
| 7 | Check every return value | rustc `unused_must_use` (hard error), `clippy::let_underscore_must_use` = **deny**. |
| 8 | Preprocessor only for headers/simple macros | N/A — Rust has no preprocessor. |
| 9 | Restricted pointer use (single deref, no function pointers) | Only the transliterated integrators use raw pointers/function pointers, mirroring the reference C; they are on the SLOC exemption list and flagged for review. Everywhere else `clippy::borrow_as_ptr`, `clippy::ptr_as_ptr`, `clippy::ptr_cast_constness` rules apply (allowed only in the CUDA marshalling path with justification). |
| 10 | Compile with zero warnings at the strictest settings | Lefthook gate: `cargo clippy --workspace --all-targets -- -D warnings -D clippy::pedantic` — **any rustc or pedantic warning fails the commit** ([workspace.lints.clippy] enables pedantic for normal builds too); the Python surface is gated by `basedpyright` (zero diagnostics). |

Additional hard denials added for this alignment (workspace lints):
`clippy::exit` (no silent process termination in a library),
`clippy::unreachable`, `clippy::unimplemented`, `clippy::todo` (no
placeholder panics).

## Exemption register

`tools/sloc_exclusions.txt` is the only sanctioned file-size exemption list.
Every entry must state why the file cannot be split. Current exemptions:
the four DOPRI5(4)/DOP853 transliterations (host + device) and the generated
galpy fixture-comparison test corpus.
