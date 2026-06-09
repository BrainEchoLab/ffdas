import glob
import os
import re
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))


def _find_build_dir():
    env = os.environ.get("FFDAS_BUILD_DIR")
    if env:
        d = os.path.abspath(env)
        if os.path.isfile(os.path.join(d, "CMakeCache.txt")):
            return d
        raise RuntimeError(f"FFDAS_BUILD_DIR={env} does not contain a cmake build")

    for name in ["_build", "build"]:
        d = os.path.join(REPO_ROOT, name)
        if os.path.isfile(os.path.join(d, "CMakeCache.txt")):
            return d

    for d in sorted(glob.glob(os.path.join(REPO_ROOT, "_build*"))):
        if os.path.isfile(os.path.join(d, "CMakeCache.txt")):
            return d

    return None


def _read_cuda_major(build_dir):
    path = os.path.join(build_dir, "ffdas_cuda_major.txt")
    return int(open(path).read().strip())


def _detect_cuda_major():
    build_dir = _find_build_dir()
    if build_dir:
        return build_dir, _read_cuda_major(build_dir)

    env = os.environ.get("FFDAS_CUDA_MAJOR")
    if env:
        return None, int(env)

    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True)
        m = re.search(r"release (\d+)", out)
        if m:
            return None, int(m.group(1))
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise RuntimeError(
        "Could not determine CUDA version. Build the core library first:\n"
        "  cmake -S <repo> -B _build -DFFDAS_BUILD_PYTHON=ON\n"
        "  cmake --build _build -j\n"
        "Or set FFDAS_BUILD_DIR or FFDAS_CUDA_MAJOR."
    )


BUILD_DIR, CUDA_MAJOR = _detect_cuda_major()


class CMakeBuildExt(build_ext):
    _LIB_PATTERNS = ["libffdas_cu*", "ffdas_cu*.dll", "libffdas_cu*.dylib"]

    def _bundled_libs(self):
        d = os.path.join(self.build_lib, "ffdas", "_core")
        libs = []
        for pattern in self._LIB_PATTERNS:
            libs.extend(glob.glob(os.path.join(d, pattern)))
        return libs

    def build_extension(self, ext):
        build_dir = BUILD_DIR
        if build_dir is None:
            build_dir = self.build_temp
            subprocess.check_call([
                "cmake", "-S", REPO_ROOT, "-B", build_dir,
                "-DFFDAS_BUILD_PYTHON=ON",
                "-DCMAKE_BUILD_TYPE=Release",
            ])
            subprocess.check_call([
                "cmake", "--build", build_dir,
                "--config", "Release", "-j",
            ])

        subprocess.check_call([
            "cmake", "--install", build_dir,
            "--component", "python",
            "--prefix", os.path.abspath(self.build_lib),
            "--config", "Release",
        ])

    def copy_extensions_to_source(self):
        super().copy_extensions_to_source()
        build_py = self.get_finalized_command("build_py")
        package_dir = build_py.get_package_dir("ffdas._core")
        for src in self._bundled_libs():
            dst = os.path.join(package_dir, os.path.basename(src))
            self.copy_file(src, dst)

    def get_outputs(self):
        outputs = super().get_outputs()
        outputs.extend(self._bundled_libs())
        return outputs


setup(
    name=f"ffdas-cu{CUDA_MAJOR}",
    version="0.1.0",
    description="GPU-accelerated delay-and-sum beamforming",
    python_requires=">=3.12",
    packages=["ffdas", "ffdas._core", "ffdas.utils"],
    package_dir={"": "."},
    package_data={"ffdas._core": ["*.pyi"]},
    install_requires=["array-api-compat"],
    extras_require={
        "cuda": [
            "cuda-pathfinder",
            f"cuda-toolkit[cudart,cublas,cusolver,nvtx]=={CUDA_MAJOR}.*",
        ],
        "dev": ["pytest>=6.0", "pytest-cov"],
        "examples": [
            f"cupy-cuda{CUDA_MAJOR}x",
            "numpy>=2.0",
            "matplotlib",
            "pillow",
        ],
        "docs": [
            "mkdocs>=1.6,<2",
            "mkdocs-material>=9.7",
        ],
    },
    ext_modules=[Extension("ffdas._core._ffdas", sources=[], py_limited_api=True)],
    cmdclass={"build_ext": CMakeBuildExt},
)
