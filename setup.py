import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="shapeGMMTorch",
    version="2.0.0",
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
    python_requires=">=3.7",
    license="MIT",
    install_requires=[
        "numpy",
        "scipy",
        "torch>=2.6",
        "MDAnalysis",
        "matplotlib",
        "pyemma"
    ],
    include_package_data=True,
)
