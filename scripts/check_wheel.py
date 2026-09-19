#!/usr/bin/env python3
"""Validate the static contract of a Drift release wheel."""

from email.parser import Parser
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Never
from zipfile import ZipFile


EXPECTED_TAG = "cp313-abi3-manylinux_2_28_x86_64"
EXPECTED_LICENSE_EXPRESSION = "MIT AND BSD-2-Clause AND BSD-3-Clause"
EXPECTED_LICENSE_FILES = {
    "LICENSE",
    "LICENSES/README.md",
    "LICENSES/galpy-bovy-BSD-3-Clause.txt",
    "LICENSES/galpy-leung-BSD-3-Clause.txt",
    "LICENSES/hairer-unige-BSD-2-Clause.txt",
}
ALLOWED_LIBRARIES = {
    "ld-linux-x86-64.so.2",
    "libc.so.6",
    "libdl.so.2",
    "libm.so.6",
    "libpthread.so.0",
}
PTX_KIND = 0x100


def fail(message: str) -> Never:
    raise SystemExit(f"error: {message}")


def one(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        fail(f"expected one {suffix} member, found {len(matches)}")
    return matches[0]


def run(tool: str, *args: str) -> str:
    executable = shutil.which(tool) or fail(f"'{tool}' was not found on PATH")
    return subprocess.run(
        [executable, *args], check=True, capture_output=True, text=True
    ).stdout


def artifact_bundles(
    data: bytes,
) -> list[tuple[str, str, list[tuple[int, bytes]]]]:
    bundles = []
    offset = 0
    while offset < len(data) and any(data[offset:]):
        blob = data[offset:]
        if len(blob) < 32 or blob[:8] != b"OXIDEART":
            fail("the .oxart section contains an invalid artifact header")
        version, header_size = struct.unpack_from("<HH", blob, 8)
        total_size = struct.unpack_from("<I", blob, 12)[0]
        name_size, target_size, payload_count = struct.unpack_from(
            "<HHH", blob, 16
        )
        if (
            version not in {1, 2}
            or header_size != 32
            or total_size < header_size
            or total_size > len(blob)
        ):
            fail("the .oxart section contains an unsupported artifact record")

        blob = blob[:total_size]
        name_start = header_size
        target_start = name_start + name_size
        records_start = target_start + target_size
        if records_start + payload_count * 24 > len(blob):
            fail("the .oxart section contains truncated payload records")
        name = blob[name_start:target_start].decode()
        target = blob[target_start:records_start].decode()
        payloads = []
        for index in range(payload_count):
            record = records_start + index * 24
            kind = struct.unpack_from("<H", blob, record)[0]
            data_offset, data_size = struct.unpack_from("<II", blob, record + 4)
            if data_offset + data_size > len(blob):
                fail("the .oxart section contains a truncated payload")
            payloads.append((kind, blob[data_offset : data_offset + data_size]))
        bundles.append((name, target, payloads))
        offset += total_size
    return bundles


def check_wheel(wheel: Path) -> None:
    if not wheel.is_file():
        fail(f"wheel does not exist: {wheel}")
    expected_suffix = f"-{EXPECTED_TAG}.whl"
    if not wheel.name.startswith("astrodrift-") or not wheel.name.endswith(
        expected_suffix
    ):
        fail(f"unexpected wheel filename: {wheel.name}")

    with ZipFile(wheel) as archive:
        names = archive.namelist()
        wheel_metadata = Parser().parsestr(
            archive.read(one(names, ".dist-info/WHEEL")).decode()
        )
        metadata_member = one(names, ".dist-info/METADATA")
        package_metadata = Parser().parsestr(
            archive.read(metadata_member).decode()
        )
        if wheel_metadata.get_all("Tag") != [EXPECTED_TAG]:
            fail(f"unexpected wheel tags: {wheel_metadata.get_all('Tag')}")
        if package_metadata["Requires-Python"] != ">=3.13":
            fail(
                "unexpected Requires-Python: "
                f"{package_metadata['Requires-Python']}"
            )
        if (
            package_metadata["License-Expression"]
            != EXPECTED_LICENSE_EXPRESSION
        ):
            fail(
                "unexpected License-Expression: "
                f"{package_metadata['License-Expression']}"
            )
        license_files = set(package_metadata.get_all("License-File", []))
        if license_files != EXPECTED_LICENSE_FILES:
            fail(f"unexpected license files: {sorted(license_files)}")
        dist_info = metadata_member.removesuffix("/METADATA")
        missing_license_members = {
            f"{dist_info}/licenses/{path}" for path in EXPECTED_LICENSE_FILES
        } - set(names)
        if missing_license_members:
            fail(
                "missing wheel license files: "
                f"{sorted(missing_license_members)}"
            )
        extension = one(names, "/drift_rs.abi3.so")

        with tempfile.TemporaryDirectory(
            prefix="drift-wheel-check-"
        ) as directory:
            extension_path = Path(directory, "drift_rs.abi3.so")
            extension_path.write_bytes(archive.read(extension))

            dynamic = run("readelf", "-d", str(extension_path))
            libraries = set(re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic))
            unexpected_libraries = libraries - ALLOWED_LIBRARIES
            if unexpected_libraries or "libc.so.6" not in libraries:
                fail(f"unexpected ELF dependencies: {sorted(libraries)}")

            versions = run("readelf", "--version-info", str(extension_path))
            glibc_versions = {
                tuple(map(int, match))
                for match in re.findall(r"GLIBC_(\d+)\.(\d+)", versions)
            }
            if not glibc_versions or max(glibc_versions) > (2, 28):
                fail(
                    f"unexpected maximum glibc version: {max(glibc_versions, default=None)}"
                )

            artifact_path = Path(directory, "kernels.oxart")
            run(
                "objcopy",
                "--dump-section",
                f".oxart={artifact_path}",
                str(extension_path),
            )
            kernels = [
                bundle
                for bundle in artifact_bundles(artifact_path.read_bytes())
                if bundle[0] == "kernels"
            ]
            if len(kernels) != 1:
                fail(
                    f"expected one kernels artifact bundle, found {len(kernels)}"
                )
            _, target, payloads = kernels[0]
            if target != "sm_80":
                fail(f"unexpected CUDA target: {target}")
            if len(payloads) != 1 or payloads[0][0] != PTX_KIND:
                fail("the kernels bundle must contain exactly one PTX payload")

            ptx = payloads[0][1].decode()
            if not re.search(r"(?m)^\.version 7\.0$", ptx):
                fail("the embedded PTX does not declare PTX ISA 7.0")
            if not re.search(r"(?m)^\.target sm_80$", ptx):
                fail("the embedded PTX does not target sm_80")
            if re.search(r"(?m)^\s*\.extern\b", ptx):
                fail(
                    "the embedded PTX contains unresolved external declarations"
                )

    print(
        f"validated {wheel.name}: {EXPECTED_TAG}, glibc <= 2.28, "
        "self-contained sm_80 PTX, licenses present"
    )


if len(sys.argv) != 2:
    fail(f"usage: {sys.argv[0]} WHEEL")

check_wheel(Path(sys.argv[1]))
