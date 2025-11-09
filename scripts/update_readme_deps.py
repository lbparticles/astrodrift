#!/usr/bin/env python3
import re
import subprocess
from pathlib import Path
import tomllib  # built into Python 3.11+

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
CARGO = ROOT / "Cargo.toml"


def run_cmd(cmd: list[str]) -> str:
    """Run a shell command and return its stdout as a string."""
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Command failed: {' '.join(cmd)}")
        print(e.output)
        return "(failed to generate dependency tree)"


def extract_python_deps(pyproject_path: Path) -> str:
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    # Adapt depending on your project’s layout:
    deps = data.get("project", {}).get("dependencies", [])
    if not deps:
        deps = data.get("dependency-groups", {}).get("default", [])

    dep_lines = ["```toml", "dependencies = ["]
    for d in sorted(deps):
        dep_lines.append(f'    "{d}",')
    dep_lines.append("]")
    dep_lines.append("```")
    return "\n".join(dep_lines)


def extract_rust_sections(cargo_path: Path) -> str:
    """Extract both [dependencies] and [build-dependencies] blocks."""
    blocks = {}
    current_block = None
    lines = []

    with open(cargo_path) as f:
        for line in f:
            l = line.strip()
            if l.startswith("[dependencies]"):
                if current_block:
                    blocks[current_block] = "\n".join(lines)
                    lines = []
                current_block = "dependencies"
                continue
            elif l.startswith("[build-dependencies]"):
                if current_block:
                    blocks[current_block] = "\n".join(lines)
                    lines = []
                current_block = "build-dependencies"
                continue
            elif l.startswith("[") and current_block:
                # Stop capturing when a new section begins
                blocks[current_block] = "\n".join(lines)
                current_block = None
                lines = []
                continue
            if current_block and line.strip():
                lines.append(line.rstrip())

    if current_block:
        blocks[current_block] = "\n".join(lines)

    out = ["```toml"]
    if "dependencies" in blocks:
        out.append("[dependencies]")
        out.append(blocks["dependencies"])
    if "build-dependencies" in blocks:
        out.append("")
        out.append("[build-dependencies]")
        out.append(blocks["build-dependencies"])
    out.append("```")
    return "\n".join(out)


def extract_python_tree() -> str:
    """Call `uv tree` to get an expanded dependency tree (markdown formatted)."""
    tree = run_cmd(["uv", "tree"])
    return f"<details>\n<summary>Python full dependency tree</summary>\n\n```\n{tree}\n```\n</details>"


def extract_rust_tree() -> str:
    """Call `cargo tree` and `cargo tree --build-deps` to get both runtime and build trees."""
    normal_tree = run_cmd(["cargo", "tree"])
    out = [
        "<details>",
        "    <summary>Rust full dependency tree</summary>",
        "",
        "```\n" + normal_tree + "```\n",
        "</details>",
    ]
    return "\n".join(out)


def replace_section(content: str, marker: str, new_block: str) -> str:
    """Replace section delimited by <!-- {marker}_START --> and <!-- {marker}_END -->."""
    pattern = re.compile(
        rf"(?<=<!-- {marker}_START -->)(.*?)(?=<!-- {marker}_END -->)",
        re.DOTALL,
    )
    return re.sub(pattern, f"\n{new_block}\n", content)


def main():
    readme = README.read_text()

    print("🧩 Extracting Python dependencies...")
    python_block = "\n".join(
        [
            extract_python_deps(PYPROJECT),
            "",
            extract_python_tree(),
        ]
    )

    print("🧩 Extracting Rust dependencies...")
    rust_block = "\n".join(
        [
            extract_rust_sections(CARGO),
            "",
            extract_rust_tree(),
        ]
    )

    updated = readme
    updated = replace_section(updated, "PYTHON_DEPS", python_block)
    updated = replace_section(updated, "RUST_DEPS", rust_block)

    print(updated)
    if updated != readme:
        README.write_text(updated)
        print("✅ README.md updated.")
    else:
        print("ℹ️ README.md already up to date.")


if __name__ == "__main__":
    main()
