from setuptools import setup, find_packages

setup(
    name='semantics-pytorch',
    version='0.3.0',
    author='Jordan Madden',
    description='Utilities for implementing semantic communication workflows in PyTorch',
    packages=find_packages(),
    install_requires=open('requirements.txt').readlines(),
)