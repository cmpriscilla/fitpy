#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["numpy>=1.19.4", "scipy>=1.5.4"]

setup_requirements = ["pytest-runner", "black", "flake8", "pylint"]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Gregory Loshkajian",
    author_email="gloshkajian3@gatech.edu",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
    ],
    description="FitPy fits distributions for you, from data! It can even decide which distribution is best for your data!",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="fitpy",
    name="fitpy",
    packages=find_packages(include=["fitpy", "fitpy.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.gatech.edu/fitpy/fitpy",
    version="0.1.0",
    zip_safe=False,
)
