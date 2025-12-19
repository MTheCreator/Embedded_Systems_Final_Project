"""
Setup script for tiltrotor_sim package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tiltrotor-sim",
    version="2.0.0",
    author="UAV Simulation Team",
    author_email="",
    description="Modular simulation framework for tilt-rotor UAVs with H-infinity control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tiltrotor-sim",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "pybullet": ["pybullet>=3.2.0"],
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tiltrotor-sim=run_simulation:main",
        ],
    },
)
