import os
import shutil
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_build_dir():
    env = os.environ.get("FFDAS_BUILD_DIR")
    if env:
        d = Path(env)
        if not (d / "CMakeCache.txt").exists():
            raise RuntimeError(f"FFDAS_BUILD_DIR={env} is not a cmake build directory")
        return d
    for d in [REPO_ROOT / "_build", REPO_ROOT / "build"]:
        if (d / "CMakeCache.txt").exists():
            return d
    for d in sorted(REPO_ROOT.glob("_build*")):
        if (d / "CMakeCache.txt").exists():
            return d
    raise RuntimeError(
        "No cmake build found. Build the library first, or set FFDAS_BUILD_DIR."
    )


BUILD_DIR = _find_build_dir()
CUDA_MAJOR = int((BUILD_DIR / "ffdas_cuda_major.txt").read_text().strip())


class PrebuiltExt(build_ext):
    def build_extension(self, ext):
        dest = Path(self.build_lib) / "ffdas" / "_core"
        dest.mkdir(parents=True, exist_ok=True)
        for line in (BUILD_DIR / "ffdas_python_files.txt").read_text().splitlines():
            shutil.copy2(line.strip(), dest)

    def get_outputs(self):
        outputs = super().get_outputs()
        dest = Path(self.build_lib) / "ffdas" / "_core"
        if dest.exists():
            lib_name = f"ffdas_cu{CUDA_MAJOR}"
            for f in dest.iterdir():
                if lib_name in f.name and str(f) not in outputs:
                    outputs.append(str(f))
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
    cmdclass={"build_ext": PrebuiltExt},
)
