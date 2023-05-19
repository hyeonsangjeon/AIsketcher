import setuptools
from pathlib import Path
from setuptools import setup, find_packages
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AIsketcher",
    version="0.0.0.1",
    author="hyeonsangjeon",
    author_email="wingnut0310@gmail.com",
    description="Text-to-image generation using Huggingface stable diffusion ControlNet conditioning and AWS Translate's prompt translation function package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hyeonsangjeon/AIsketcher",
    packages=find_packages(),
    install_requires=Path("requirements.txt").read_text().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
