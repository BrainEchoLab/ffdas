import glob
import os
import re
import shlex
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

_LIB_PATTERN = re.compile(r"ffdas_cu(\d+)")


def _find_library_in(directory):
    """Find libffdas_cu*.so/dll/dylib in directory. Return (lib_dir, cuda_major) or None."""
    if not os.path.isdir(directory):
        return None
    for entry in os.listdir(directory):
        m = _LIB_PATTERN.search(entry)
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
    def build_extension(self, ext):
        ext_fullpath = self.get_ext_fullpath(ext.name)
        ext_dir = os.path.abspath(os.path.dirname(ext_fullpath))
        build_dir = os.path.join(REPO_ROOT, f"_build_python_cu{CUDA_MAJOR}")

        cmake_args = [
            f"-S{os.path.join(REPO_ROOT, 'bindings', 'python')}",
            f"-B{build_dir}",
            f"-DFFDAS_LIB_DIR={FFDAS_LIB_DIR}",
            f"-DFFDAS_INCLUDE_DIR={os.path.join(REPO_ROOT, 'include')}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        try:
            import nanobind
            cmake_args.append(f"-Dnanobind_DIR={nanobind.cmake_dir()}")
        except ImportError:
            cmake_args.append("-DFETCH_NANOBIND=ON")

        cuda_root = os.environ.get("CUDA_ROOT")
        if cuda_root:
            cmake_args.append(f"-DCUDAToolkit_ROOT={cuda_root}")

        generator = os.environ.get("CMAKE_GENERATOR")
        if generator:
            cmake_args.extend(["-G", generator])

        extra = os.environ.get("FFDAS_CMAKE_ARGS", "")
        if extra:
            cmake_args.extend(shlex.split(extra))

        subprocess.check_call(["cmake"] + cmake_args)
        subprocess.check_call([
            "cmake", "--build", build_dir,
            "--config", "Release", "-j",
        ])
        subprocess.check_call([
            "cmake", "--install", build_dir,
            "--prefix", ext_dir, "--config", "Release",
        ])


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
