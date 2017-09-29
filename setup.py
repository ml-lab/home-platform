#!/usr/bin/env python

import os
from setuptools import setup

setup(
    name = "Multimodal maze",
    version = "0.1.0",
    author = "Simon Brodeur",
    author_email = "Simon.Brodeur@USherbrooke.ca",
    description = ("Multimodal maze environment based on the SUNCG indoor scenes dataset."),
    license = "BSD 3-Clause License",
    keywords = "artificial intelligence, machine learning, reinforcement learning",
    url = "https://github.com/IGLU-CHISTERA/multimodal-maze",
    packages=['multimodalmaze'],
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
