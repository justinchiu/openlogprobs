from setuptools import find_packages, setup

setup(
    name="openlogprobs",
    version="0.0.2",
    description="extract log-probabilities from APIs",
    author="Justin Chiu & Jack Morris",
    author_email="jtc257@cornell.edu",
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines()
    # install_requires=[],
)