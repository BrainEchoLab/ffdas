import os
import shlex
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

CUDA_MAJOR = int(os.environ.get("FFDAS_CUDA_MAJOR", "13"))
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


class CMakeBuildExt(build_ext):
    def build_extension(self, ext):
        build_dir = os.path.join(REPO_ROOT, f"_build_python_cu{CUDA_MAJOR}")
        install_dir = os.path.abspath(self.build_lib)

        cmake_args = [
            f"-S{REPO_ROOT}",
            f"-B{build_dir}",
            "-DBUILD_PYTHON=ON",
            "-DBUILD_MEX=OFF",
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
            nvcc = os.path.join(cuda_root, "bin", "nvcc")
            if sys.platform == "win32":
                nvcc += ".exe"
            if os.path.isfile(nvcc):
                cmake_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")

        cuda_archs = os.environ.get("CMAKE_CUDA_ARCHITECTURES")
        if cuda_archs:
            cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}")

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
            "--prefix", install_dir, "--config", "Release",
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
    ext_modules=[Extension("ffdas._core._ffdas", sources=[])],
    cmdclass={"build_ext": CMakeBuildExt},
)
