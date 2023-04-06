from setuptools import find_packages, setup

setup(
    name="graph_learn",
    packages=find_packages(),
    version="0.0.2",
    description="Extracting graphs from signals on nodes",
    author="William Cappelletti",
    license="MIT",
    url="https://github.com/LTS4/graph-learning",
    project_urls={
        "Bug Tracker": "https://github.com/LTS4/graph-learning/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.10",
    install_requires=[
        "networkx",
        "numpy>=1.2.1",
        "omegaconf",
        "scikit-learn",
        "seaborn",
        "tqdm",
    ],
)
