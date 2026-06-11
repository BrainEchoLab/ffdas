import platform
from pathlib import Path

CUDA_VERSION = (Path(__file__).parent / "CUDA_VERSION").read_text().strip()
LIB_NAME = "ffdas.dll" if platform.system() == "Windows" else "libffdas.so"
