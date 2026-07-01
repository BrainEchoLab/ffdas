import os

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


cuda_ver = os.environ.get("FFDAS_CUDA", "12")

setup(
    name=f"ffdas-core-cuda{cuda_ver}",
    version="0.1.1",
    description=f"ffdas core library (CUDA {cuda_ver})",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=["ffdas_core"],
    package_data={"ffdas_core": ["*.so", "*.dll", "CUDA_VERSION"]},
    distclass=BinaryDistribution,
)
