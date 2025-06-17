import setuptools
from pathlib import Path

# Extract version from file
version_file = Path(__file__).parent / "src" / "shapeGMMTorch" / "version.py"
exec(version_file.read_text())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="shapeGMMTorch",
    version=__version__,
    author="Martin McCullagh",
    author_email="martin.mccullagh@okstate.edu",
    description="Gaussian Mixture Model clustering in size-and-shape space using PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mccullaghlab/shapeGMMTorch",
    project_urls={
        "Bug Tracker": "https://github.com/mccullaghlab/shapeGMMTorch/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    license="MIT",
    install_requires=[
        "numpy",
        "scipy",
        "torch>=2.4",
        "MDAnalysis",
        "matplotlib",
    ],
    include_package_data=True,
)
