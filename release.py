#!/usr/bin/env python3
"""Release build script for ffdas.

Builds core library, Python bindings, MATLAB bindings, and packages them
for distribution.

Environment variables:
    CUDA12_ROOT  Path to CUDA 12 toolkit
    CUDA13_ROOT  Path to CUDA 13 toolkit
    CUDA_ROOT    Fallback if the version-specific variable is not set

Usage:
    python release.py                     # build everything for CUDA 12 + 13
    python release.py --cuda 12           # CUDA 12 only
    python release.py --target python     # Python packages only
    python release.py --version 0.1.2.dev1  # a pre-release of the version in the sources
"""

import argparse
import contextlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
BUILD_DIR = ROOT / "_build"

# Every place the package version is written: (file, pattern whose group 1 is the version,
# number of occurrences).
VERSION_SITES = [
    ("bindings/python/pyproject.toml", r'(?m)^version = "([^"]+)"', 1),
    ("bindings/python/pyproject.toml", r'"ffdas-core-cuda\d+==([^"]+)"', 2),
    ("bindings/python/ffdas/__init__.py", r'(?m)^__version__ = "([^"]+)"', 1),
    ("packaging/core/setup.py", r'(?m)^[ \t]*version="([^"]+)"', 1),
]

CUDA_ARCHITECTURES = {
    "12": "53-real;60-real;70-real;75-real;80-real;86-real;89-real;90",
    "13": "75-real;80-real;86-real;89-real;90-real;100-real;120",
}


def run(*cmd, env=None, **kwargs):
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"  > {cmd_str}", flush=True)
    full_env = {**os.environ, **(env or {})}
    try:
        subprocess.run([str(c) for c in cmd], check=True, env=full_env, **kwargs)
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        raise e


def is_windows():
    return platform.system() == "Windows"


def lib_filename():
    return "ffdas.dll" if is_windows() else "libffdas.so"


def nvcc_path(cuda_root):
    name = "nvcc.exe" if is_windows() else "nvcc"
    return str(Path(cuda_root) / "bin" / name)


def find_lib(build_dir):
    """Find the shared library in the build tree (handles multi-config generators)."""
    direct = build_dir / lib_filename()
    if direct.exists():
        return direct
    release = build_dir / "Release" / lib_filename()
    if release.exists():
        return release
    raise FileNotFoundError(f"{lib_filename()} not found in {build_dir}")


def project_version():
    for line in (ROOT / "CMakeLists.txt").read_text().splitlines():
        if "project(ffdas VERSION" in line:
            return line.split("VERSION")[1].split(")")[0].strip().split()[0]
    raise RuntimeError("could not parse version from CMakeLists.txt")


def source_version():
    """The package version written in the sources; every site must agree."""
    found = set()
    for path, pattern, count in VERSION_SITES:
        matches = re.findall(pattern, (ROOT / path).read_text(encoding="utf-8"))
        if len(matches) != count:
            raise RuntimeError(
                f"{path}: expected {count} version string(s) matching {pattern!r}, "
                f"found {len(matches)}")
        found.update(matches)
    if len(found) != 1:
        raise RuntimeError(f"the sources disagree on the version: {sorted(found)}")
    return found.pop()


def check_version(version):
    """Refuse a version that is not normalised PEP 440, or not of the release the sources lead to."""
    from packaging.version import InvalidVersion, Version

    try:
        parsed = Version(version)
    except InvalidVersion:
        sys.exit(f"error: {version!r} is not a PEP 440 version (e.g. 0.1.2.dev1, 0.1.2rc1)")
    if str(parsed) != version:
        sys.exit(f"error: write the version in its normalised form: {parsed}")
    if parsed.base_version != project_version():
        sys.exit(f"error: {version} is not a version of {project_version()}, "
                 f"the release CMakeLists.txt describes")


@contextlib.contextmanager
def stamped_version(version):
    """Write ``version`` into every version site for the duration of the build.

    The files are restored byte for byte afterwards, also when the build fails, so the
    sources keep their own version.
    """
    originals = {}
    try:
        for path, pattern, count in VERSION_SITES:
            file = ROOT / path
            text = file.read_bytes()
            originals.setdefault(file, text)

            def splice(m):
                start, end = m.start(1) - m.start(0), m.end(1) - m.start(0)
                return m.group(0)[:start] + version.encode() + m.group(0)[end:]

            text, n = re.subn(pattern.encode(), splice, text)
            if n != count:
                raise RuntimeError(f"{path}: expected {count} version string(s), found {n}")
            file.write_bytes(text)
        yield
    finally:
        for file, text in originals.items():
            file.write_bytes(text)


def platform_tag():
    system = "win" if is_windows() else "linux"
    mach = platform.machine().lower()
    return f"{system}_{mach}"


# Phase 1: Core library

def build_core(cuda_ver, cuda_root):
    build_dir = BUILD_DIR / f"cuda{cuda_ver}"
    archs = CUDA_ARCHITECTURES[cuda_ver]

    cmake_args = [
        "cmake", "-S", str(ROOT), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_COMPILER={nvcc_path(cuda_root)}",
        f"-DCUDAToolkit_ROOT={cuda_root}",
        f"-DCMAKE_CUDA_ARCHITECTURES={archs}",
    ]
    if is_windows():
        cmake_args.extend(["-G", "Ninja"])

    run(*cmake_args)
    run("cmake", "--build", str(build_dir), "--config", "Release")

    find_lib(build_dir)
    return build_dir


# Phase 2: Core Python wheels

def package_core_wheel(cuda_ver, core_build_dir, out_dir):
    pkg_dir = ROOT / "packaging" / "core"
    core_mod = pkg_dir / "ffdas_core"

    shutil.copy2(find_lib(core_build_dir), core_mod / lib_filename())
    (core_mod / "CUDA_VERSION").write_text(cuda_ver)

    with tempfile.TemporaryDirectory() as tmp:
        try:
            run("python", "-m", "build", "--wheel",
                str(pkg_dir), "--outdir", tmp,
                env={"FFDAS_CUDA": cuda_ver})

            wheels = list(Path(tmp).glob("*.whl"))
            if not is_windows():
                for whl in wheels:
                    run("auditwheel", "repair", str(whl),
                        "--wheel-dir", str(out_dir),
                        "--exclude", "libcudart.so.*",
                        "--exclude", "libcublas.so.*",
                        "--exclude", "libcusolver.so.*",
                        "--exclude", "libcublasLt.so.*",
                        "--exclude", "libcusparse.so.*",
                        "--exclude", "libnvToolsExt.so.*",
                        "--exclude", "libcuda.so.*")
            else:
                for whl in wheels:
                    shutil.copy2(whl, out_dir / whl.name)
        finally:
            (core_mod / lib_filename()).unlink(missing_ok=True)
            (core_mod / "CUDA_VERSION").unlink(missing_ok=True)
            # setuptools leaves build artifacts in the source directory
            shutil.rmtree(pkg_dir / "build", ignore_errors=True)
            for d in pkg_dir.glob("*.egg-info"):
                shutil.rmtree(d, ignore_errors=True)


# Phase 3: Python bindings wheel

def build_bindings_wheel(core_build_dir, out_dir):
    py_dir = ROOT / "bindings" / "python"
    with tempfile.TemporaryDirectory() as tmp:
        run("python", "-m", "build", "--wheel",
            str(py_dir), "--outdir", tmp,
            env={"CMAKE_PREFIX_PATH": str(core_build_dir)})

        for whl in Path(tmp).glob("*.whl"):
            if is_windows():
                run("delvewheel", "repair", str(whl),
                    "--wheel-dir", str(out_dir),
                    "--exclude", "ffdas.dll")
            else:
                run("auditwheel", "repair", str(whl),
                    "--wheel-dir", str(out_dir),
                    "--exclude", "libffdas.so")

    shutil.rmtree(py_dir / "build", ignore_errors=True)
    for d in py_dir.glob("*.egg-info"):
        shutil.rmtree(d, ignore_errors=True)


# Phase 4: MATLAB

def build_matlab(core_build_dir):
    build_dir = BUILD_DIR / "matlab"

    cmake_args = [
        "cmake",
        "-S", str(ROOT / "bindings" / "matlab"),
        "-B", str(build_dir),
        f"-DCMAKE_PREFIX_PATH={core_build_dir}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if is_windows():
        cmake_args.extend(["-G", "Ninja"])

    run(*cmake_args)
    run("cmake", "--build", str(build_dir), "--config", "Release")
    return build_dir


def package_matlab_zip(cuda_ver, core_build_dir, matlab_build_dir, out_dir, version):
    staging = BUILD_DIR / "staging"
    if staging.exists():
        shutil.rmtree(staging)

    run("cmake", "--install", str(matlab_build_dir), "--prefix", str(staging))

    # Replace the shared library with the correct CUDA variant
    shutil.copy2(
        find_lib(core_build_dir),
        staging / "+ffdas" / "+core" / lib_filename(),
    )

    zip_name = f"ffdas-cuda{cuda_ver}-{version}-matlab-{platform_tag()}.zip"
    zip_path = out_dir / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in staging.rglob("*"):
            if f.is_file():
                zf.write(f, Path("ffdas-matlab") / f.relative_to(staging))

    shutil.rmtree(staging)
    return zip_path


# Orchestration

def main():
    parser = argparse.ArgumentParser(description="ffdas release builder")
    parser.add_argument(
        "--cuda", nargs="+", default=["12", "13"],
        help="CUDA versions to build (default: 12 13)")
    parser.add_argument(
        "--target", nargs="+", default=["python", "matlab"],
        help="Targets to build (default: python matlab)")
    parser.add_argument(
        "--version",
        help="package version to build, e.g. 0.1.2.dev1 for a pre-release; it is written "
             "into the Python packages for the build only (default: the sources' version)")
    parser.add_argument(
        "--clean", action="store_true",
        help="remove all build artifacts and exit")
    args = parser.parse_args()

    if args.clean:
        for d in [BUILD_DIR, ROOT / "dist"]:
            if d.exists():
                print(f"removing {d}")
                shutil.rmtree(d)
        return

    version = source_version()
    if args.version is not None and args.version != version:
        check_version(args.version)
        version = args.version
        stamp = stamped_version(version)
    else:
        stamp = contextlib.nullcontext()
    print(f"ffdas {version}")

    for ver in args.cuda:
        if ver not in CUDA_ARCHITECTURES:
            print(f"error: unsupported CUDA version {ver}")
            sys.exit(1)

    cuda_roots = {}
    for ver in args.cuda:
        root = os.environ.get(f"CUDA{ver}_ROOT") or os.environ.get("CUDA_ROOT")
        if root is None:
            print(f"error: set CUDA{ver}_ROOT or CUDA_ROOT")
            sys.exit(1)
        cuda_roots[ver] = root

    out_dir = ROOT / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)

    core_builds = {}
    for ver in args.cuda:
        print(f"\n{'=' * 60}")
        print(f"  Core library — CUDA {ver}")
        print(f"{'=' * 60}")
        core_builds[ver] = build_core(ver, cuda_roots[ver])

    if "python" in args.target:
        with stamp:
            for ver in args.cuda:
                print(f"\n{'=' * 60}")
                print(f"  Core Python wheel — CUDA {ver}")
                print(f"{'=' * 60}")
                package_core_wheel(ver, core_builds[ver], out_dir)

            print(f"\n{'=' * 60}")
            print("  Python bindings wheel")
            print(f"{'=' * 60}")
            build_bindings_wheel(core_builds[args.cuda[0]], out_dir)

    if "matlab" in args.target:
        print(f"\n{'=' * 60}")
        print("  MATLAB MEX build")
        print(f"{'=' * 60}")
        matlab_build = build_matlab(core_builds[args.cuda[0]])

        for ver in args.cuda:
            print(f"\n{'=' * 60}")
            print(f"  MATLAB archive — CUDA {ver}")
            print(f"{'=' * 60}")
            zip_path = package_matlab_zip(
                ver, core_builds[ver], matlab_build, out_dir, version)
            print(f"  -> {zip_path.name}")

    print(f"\n{'=' * 60}")
    print("  Build outputs")
    print(f"{'=' * 60}")
    for f in sorted(out_dir.iterdir()):
        if f.suffix in (".whl", ".zip"):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
