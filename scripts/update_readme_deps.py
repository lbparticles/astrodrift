#!/usr/bin/env python3
import re
import subprocess
from pathlib import Path
import tomllib  # Python 3.11+

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
CARGO_ROOT = ROOT / "Cargo.toml"
CARGO_KERNELS = ROOT / "kernels" / "Cargo.toml"


# ------------------------
# Utility
# ------------------------
def run_cmd(cmd: list[str]) -> str:
    """Run a command and return its stdout, trimmed."""
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Command failed: {' '.join(cmd)}")
        print(e.output)
        return "(failed to generate output)"


def parse_cargo_sections(filepath: Path) -> dict[str, dict[str, str]]:
    """Extract [dependencies] and [build-dependencies] from a Cargo.toml file."""
    out: dict[str, dict[str, str]] = {
        "dependencies": {},
        "build-dependencies": {},
    }

    current = None
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("[dependencies]"):
                current = "dependencies"
                continue
            if stripped.startswith("[build-dependencies]"):
                current = "build-dependencies"
                continue
            if stripped.startswith("[") and current:
                current = None
                continue

            if current and stripped and not stripped.startswith("#"):
                # Keep the line verbatim, but split name= for merging unique
                key = stripped.split("=", 1)[0].strip()
                out[current][key] = stripped
    return out


def merge_rust_tomls(paths: list[Path]) -> str:
    """Merge multiple Cargo.toml dependencies and build-dependencies."""
    combined = {"dependencies": {}, "build-dependencies": {}}
    for path in paths:
        sections = parse_cargo_sections(path)
        for section in combined:
            combined[section].update(sections[section])

    out = ["```toml"]
    for section in ("dependencies", "build-dependencies"):
        if combined[section]:
            out.append(f"[{section}]")
            for key in sorted(combined[section].keys()):
                out.append(f"{combined[section][key]}")
            out.append("")
    out.append("```")
    return "\n".join(out)


# ------------------------
# Python dependencies
# ------------------------
def extract_python_deps(pyproject: Path) -> str:
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    deps = data.get("project", {}).get("dependencies", [])
    if not deps:
        deps = data.get("dependency-groups", {}).get("default", [])
    lines = ["```toml", "dependencies = ["]
    for d in sorted(deps):
        lines.append(f'    "{d}",')
    lines.append("]")
    lines.append("```")
    return "\n".join(lines)


def extract_python_tree() -> str:
    tree = run_cmd(["uv", "tree"])
    return f"<details>\n<summary>Python full dependency tree</summary>\n\n```\n{tree}\n```\n</details>"


# ------------------------
# Rust dependencies
# ------------------------
def extract_rust_tree() -> str:
    """Call `cargo tree` including build-deps for the entire workspace."""
    normal_tree = run_cmd(["cargo", "tree", "--workspace"])
    out = [
        "<details>",
        "    <summary>Rust full dependency tree</summary>",
        "",
        "```\n" + normal_tree + "\n```",
        "</details>",
    ]
    return "\n".join(out)


def replace_section(content: str, marker: str, new_block: str) -> str:
    pattern = re.compile(
        rf"(?<=<!-- {marker}_START -->)(.*?)(?=<!-- {marker}_END -->)",
        re.DOTALL,
    )
    return re.sub(pattern, f"\n{new_block}\n", content)


# ------------------------
# Main
# ------------------------
def main():
    readme_text = README.read_text()

    print("🧩 Generating Python sections...")
    python_block = "\n".join(
        [
            extract_python_deps(PYPROJECT),
            "",
            extract_python_tree(),
        ]
    )

    print("🧩 Merging Rust manifests...")
    rust_dep_block = merge_rust_tomls([CARGO_ROOT, CARGO_KERNELS])
    rust_tree_block = extract_rust_tree()

    rust_block = "\n".join([rust_dep_block, "", rust_tree_block])

    updated = replace_section(readme_text, "PYTHON_DEPS", python_block)
    updated = replace_section(updated, "RUST_DEPS", rust_block)

    if updated != readme_text:
        README.write_text(updated)
        print("✅ README.md updated.")
    else:
        print("ℹ️ README.md already current.")


if __name__ == "__main__":
    main()
