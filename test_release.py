#!/usr/bin/env python3
"""Test ffdas release wheels in isolated environments.

Creates temporary virtual environments, installs the built wheels,
runs the test suite and examples, and cleans up afterwards.

Usage:
    python test_release.py                     # test all CUDA versions found
    python test_release.py --cuda 12           # CUDA 12 only
    python test_release.py --skip-examples     # skip examples (faster)
    python test_release.py --dist-dir build/   # custom wheel location
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
TESTS_SRC = ROOT / "bindings" / "python" / "tests"
EXAMPLES_SRC = ROOT / "bindings" / "python" / "examples"

CUPY_PACKAGES = {
    "12": ["cupy-cuda12x", "nvidia-cufft-cu12"],
    "13": ["cupy-cuda13x", "nvidia-cufft"],
}


def find_wheels(dist_dir, prefix):
    return sorted(dist_dir.glob(f"{prefix}*.whl"))


def find_wheel(dist_dir, prefix):
    matches = find_wheels(dist_dir, prefix)
    if not matches:
        raise FileNotFoundError(f"no wheel matching '{prefix}*' in {dist_dir}")
    if len(matches) > 1:
        names = [m.name for m in matches]
        raise RuntimeError(
            f"multiple wheels matching '{prefix}*' in {dist_dir}: {names}"
        )
    return matches[0]


def find_cuda_versions(dist_dir):
    versions = set()
    for whl in dist_dir.glob("ffdas_core_cuda*-*.whl"):
        # ffdas_core_cuda12-0.1.0-... -> "12"
        rest = whl.name.split("ffdas_core_cuda")[1]
        ver = rest.split("-")[0]
        versions.add(ver)
    return sorted(versions)


class TestEnv:
    """Virtual environment in a temporary directory."""

    def __init__(self, base_dir, name):
        self.path = Path(base_dir) / name
        print(f"  creating venv at {self.path}")
        venv.create(str(self.path), with_pip=True)
        if sys.platform == "win32":
            self.python = str(self.path / "Scripts" / "python.exe")
        else:
            self.python = str(self.path / "bin" / "python")

    def pip_install(self, *args):
        self._run(self.python, "-m", "pip", "install", "-q", *args)

    def run_python(self, *args, check=True, **kwargs):
        return self._run(self.python, *args, check=check, **kwargs)

    def _run(self, *cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        display = " ".join(cmd)
        if len(display) > 120:
            display = display[:117] + "..."
        print(f"    $ {display}")
        return subprocess.run(cmd, **kwargs)


def heading(text):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def test_error_path(dist_dir, tmpdir):
    """Verify that importing without the core library gives a useful error."""
    heading("Error path: import without core library")

    env = TestEnv(tmpdir, "error-path")
    bindings = find_wheel(dist_dir, "ffdas-")
    env.pip_install(str(bindings))

    result = env.run_python(
        "-c", "import ffdas",
        check=False, capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("  FAIL — import succeeded, should have failed")
        return False

    if "pip install ffdas[cuda" in result.stderr:
        print("  PASS")
        return True

    print("  FAIL — error does not mention pip install ffdas[cudaXX]:")
    for line in result.stderr.strip().splitlines()[-5:]:
        print(f"    {line}")
    return False


def test_cuda_version(dist_dir, cuda_ver, tmpdir, run_examples):
    """Run all tests for one CUDA version. Returns dict of name -> bool."""
    results = {}

    heading(f"CUDA {cuda_ver}: setting up environment")
    env = TestEnv(tmpdir, f"cuda{cuda_ver}")

    core = find_wheel(dist_dir, f"ffdas_core_cuda{cuda_ver}")
    bindings = find_wheel(dist_dir, "ffdas-")
    env.pip_install(str(core), str(bindings))

    # smoke test
    heading(f"CUDA {cuda_ver}: smoke test")
    r = env.run_python("-c", "import ffdas; print('version:', ffdas.__version__)",
                       check=False)
    results["smoke"] = r.returncode == 0
    print(f"  {'PASS' if results['smoke'] else 'FAIL'}")
    if not results["smoke"]:
        return results

    # install dependencies
    heading(f"CUDA {cuda_ver}: installing dependencies")
    cupy_pkgs = CUPY_PACKAGES.get(cuda_ver, [])
    if cupy_pkgs:
        r = env.run_python("-m", "pip", "install", "-q", *cupy_pkgs, check=False)
        if r.returncode != 0:
            print(f"  WARNING: could not install {cupy_pkgs}")
            print("  GPU tests will likely fail")
    env.pip_install("pytest", "matplotlib", "Pillow")

    # copy tests to a neutral location so pytest doesn't add the source
    # tree to sys.path (which would shadow the installed package)
    test_dir = Path(tmpdir) / f"tests-cuda{cuda_ver}"
    shutil.copytree(TESTS_SRC, test_dir)

    heading(f"CUDA {cuda_ver}: pytest")
    r = env.run_python("-m", "pytest", str(test_dir), "-v", "--tb=short",
                       check=False)
    results["pytest"] = r.returncode == 0
    print(f"  {'PASS' if results['pytest'] else 'FAIL'}")

    # examples
    if run_examples:
        examples = sorted(EXAMPLES_SRC.glob("*.py"))
        if not examples:
            print("  WARNING: no example scripts found")

        for ex in examples:
            name = ex.stem
            heading(f"CUDA {cuda_ver}: example {name}")
            workdir = Path(tmpdir) / f"examples-cuda{cuda_ver}" / name
            workdir.mkdir(parents=True, exist_ok=True)
            r = env.run_python(str(ex), check=False, cwd=str(workdir))
            results[f"example:{name}"] = r.returncode == 0
            print(f"  {'PASS' if r.returncode == 0 else 'FAIL'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test ffdas release wheels")
    parser.add_argument("--cuda", nargs="+", default=None,
                        help="CUDA versions to test (default: auto-detect)")
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist",
                        help="directory containing wheels (default: dist/)")
    parser.add_argument("--skip-examples", action="store_true",
                        help="skip running examples")
    args = parser.parse_args()

    dist_dir = args.dist_dir.resolve()
    if not dist_dir.is_dir():
        print(f"error: {dist_dir} does not exist")
        sys.exit(1)

    cuda_versions = args.cuda or find_cuda_versions(dist_dir)
    if not cuda_versions:
        print(f"error: no ffdas_core_cuda* wheels in {dist_dir}")
        sys.exit(1)

    try:
        find_wheel(dist_dir, "ffdas-")
    except FileNotFoundError:
        print(f"error: no ffdas bindings wheel in {dist_dir}")
        sys.exit(1)

    print(f"dist dir:      {dist_dir}")
    print(f"CUDA versions: {', '.join(cuda_versions)}")
    print(f"examples:      {'skip' if args.skip_examples else 'run'}")

    all_results = {}

    with tempfile.TemporaryDirectory(prefix="ffdas_test_") as tmpdir:
        all_results["error_path"] = test_error_path(dist_dir, tmpdir)

        for ver in cuda_versions:
            try:
                all_results[f"cuda{ver}"] = test_cuda_version(
                    dist_dir, ver, tmpdir,
                    run_examples=not args.skip_examples,
                )
            except Exception as e:
                print(f"\n  EXCEPTION: {e}")
                all_results[f"cuda{ver}"] = {"exception": str(e)}

    # summary
    heading("Results")
    passed = 0
    failed = 0

    for group, val in all_results.items():
        if isinstance(val, bool):
            ok = val
            passed += ok
            failed += (not ok)
            print(f"  {'PASS' if ok else 'FAIL'}  {group}")
        elif isinstance(val, dict):
            for name, ok in val.items():
                if isinstance(ok, bool):
                    passed += ok
                    failed += (not ok)
                    print(f"  {'PASS' if ok else 'FAIL'}  {group}/{name}")
                else:
                    failed += 1
                    print(f"  FAIL  {group}/{name}: {ok}")

    print(f"\n  {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
