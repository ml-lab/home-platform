#!/usr/bin/env python

from setuptools import setup

setup(
    name = "HoME Platform",
    version = "0.1.0",
    author = "Simon Brodeur",
    author_email = "simon.brodeur@usherbrooke.ca",
    description = ("Househole multimodal environment (HoME) based on the SUNCG indoor scenes dataset."),
    license = "BSD 3-Clause License",
    keywords = "artificial intelligence, machine learning, reinforcement learning",
    url = "https://github.com/HoME-Platform/home-platform",
    packages=['home_platform'],
    setup_requires=['setuptools-markdown'],
    install_requires=[
        "setuptools-markdown",
        "numpy",
        "scipy",
        "matplotlib",
        "panda3d",
        "nltk"
    ],
    long_description_markdown_filename='README.md',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python",
        "License :: OSI Approved :: BSD License",
    ],
)
