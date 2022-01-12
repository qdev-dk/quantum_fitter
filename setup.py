"""
Installs the quantum_fitter package
"""

from setuptools import setup, find_packages
from pathlib import Path

import versioneer

readme_file_path = Path(__file__).absolute().parent / "README.md"

required_packages = ['matplotlib>=3.0.0',
                     'numpy>=1.12.0',
                     'pyqtgraph>=0.10.0',
                     'h5py>=2.8.0',
                     'lmfit>=1.0']
package_data = {"quantum_fitter": ["conf/telemetry.ini"] }


setup(
    name="quantum_fitter",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    python_requires=">=3.9",
    install_requires=required_packages,
    author= "Kian Gao",
    author_email="shp593@alumni.ku.dk",
    description="'package for fitting quantum data'",
    long_description=readme_file_path.open().read(),
    long_description_content_type="text/markdown",
    license="",
    package_data=package_data,
    packages=find_packages(exclude=["*.tests", "*.tests.*"]),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
    ],
)
