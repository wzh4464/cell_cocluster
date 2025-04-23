from setuptools import setup, find_packages

setup(
    name="cell_cocluster",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=[
        "numpy>=2.2.4",
        "pandas>=2.2.3",
        "scipy>=1.15.2",
        "scikit-learn>=1.6.1",
        "matplotlib>=3.10.1",
        "nibabel>=5.3.2",
        "open3d",
        "pyshtools>=4.13.1",
        "dash>=3.0.3",
        "plotly>=6.0.1",
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Cell co-clustering analysis toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cell_cocluster",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
