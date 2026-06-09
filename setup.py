import glob
import os
import re
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _find_library_in(directory):
    """Find libffdas_cu*.so/dll/dylib in directory. Return (lib_dir, cuda_major) or None."""
    if not os.path.isdir(directory):
        return None
    pattern = re.compile(r"ffdas_cu(\d+)")
    for entry in os.listdir(directory):
        m = pattern.search(entry)
        if m and os.path.isfile(os.path.join(directory, entry)):
            return directory, int(m.group(1))
    return None


def _find_library(directory):
    """Search directory and multi-config subdirectories."""
    result = _find_library_in(directory)
    if result:
        return result
    for config in ["Release", "RelWithDebInfo", "Debug"]:
        result = _find_library_in(os.path.join(directory, config))
        if result:
            return result
    return None


def find_ffdas():
    """Locate pre-built libffdas_cu*.so and return (lib_dir, cuda_major)."""

    env_dir = os.environ.get("FFDAS_LIB_DIR")
    if env_dir:
        result = _find_library(os.path.abspath(env_dir))
        if result:
            return result
        raise RuntimeError(f"FFDAS_LIB_DIR={env_dir} does not contain libffdas_cu*")

    candidates = [os.path.join(REPO_ROOT, "build")]
    candidates += sorted(glob.glob(os.path.join(REPO_ROOT, "_build*")))
    candidates += sorted(glob.glob(os.path.join(REPO_ROOT, "build_*")))

    for candidate in candidates:
        result = _find_library(candidate)
        if result:
            return result

    raise RuntimeError(
        "Pre-built libffdas not found. Build it first:\n"
        "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build build -j"
    )


FFDAS_LIB_DIR, CUDA_MAJOR = find_ffdas()


class CMakeBuildExt(build_ext):
    _LIB_PATTERNS = ["libffdas_cu*", "ffdas_cu*.dll", "libffdas_cu*.dylib"]

    def _bundled_libs(self):
        d = os.path.join(self.build_lib, "ffdas", "_core")
        libs = []
        for pattern in self._LIB_PATTERNS:
            libs.extend(glob.glob(os.path.join(d, pattern)))
        return libs

    def build_extension(self, ext):
        # run() sets self.inplace = 0 before calling this, so
        # get_ext_fullpath() always returns a path inside build_lib
        ext_fullpath = self.get_ext_fullpath(ext.name)
        ext_dir = os.path.abspath(os.path.dirname(ext_fullpath))

        cmake_args = [
            f"-S{os.path.join(REPO_ROOT, 'bindings', 'python')}",
            f"-B{self.build_temp}",
            f"-DFFDAS_LIB_DIR={FFDAS_LIB_DIR}",
            f"-DFFDAS_INCLUDE_DIR={os.path.join(REPO_ROOT, 'include')}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        try:
            import nanobind
            cmake_args.append(f"-Dnanobind_DIR={nanobind.cmake_dir()}")
        except ImportError:
            cmake_args.append("-DFETCH_NANOBIND=ON")

        subprocess.check_call(["cmake"] + cmake_args)
        subprocess.check_call([
            "cmake", "--build", self.build_temp,
            "--config", "Release", "-j",
        ])
        subprocess.check_call([
            "cmake", "--install", self.build_temp,
            "--prefix", ext_dir, "--config", "Release",
        ])

    def copy_extensions_to_source(self):
        super().copy_extensions_to_source()
        build_py = self.get_finalized_command("build_py")
        package_dir = build_py.get_package_dir("ffdas._core")  # type: ignore
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
    package_dir={"": "bindings/python"},
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
