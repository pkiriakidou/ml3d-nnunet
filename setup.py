import setuptools
from setuptools import find_packages
if __name__ == "__main__":
    setuptools.setup(
            packages=find_packages(
                where='nnunetv2'),
            package_dir={"":"nnunetv2"}
            )
