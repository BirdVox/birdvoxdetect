import gzip
from importlib.machinery import SourceFileLoader
import os
from setuptools import setup, find_packages
import sys

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

model_dir = os.path.join("birdvoxdetect", "models")
model_names = [
    "birdvoxdetect-v03_T-1800_trial-37_network_epoch-023",
    "birdvoxdetect-v03_trial-12_network_epoch-068",
    "birdvoxdetect_empty",  # for unit tests
]
weight_files = ["birdvoxactivate.pkl"] + [
    "{}.h5".format(model_name) for model_name in model_names
]
base_url = "https://github.com/BirdVox/birdvoxdetect/raw/models/"

if len(sys.argv) > 1 and sys.argv[1] == "sdist":
    # exclude the weight files in sdist
    weight_files = []
else:
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    for weight_file in weight_files:
        weight_path = os.path.join(model_dir, weight_file)
        if not os.path.isfile(weight_path):
            compressed_file = weight_file + ".gz"
            compressed_path = os.path.join(model_dir, compressed_file)
            if not os.path.isfile(compressed_file):
                print("Downloading weight file {} ...".format(compressed_file))
                urlretrieve(base_url + compressed_file, compressed_path)
            print("Decompressing ...")
            with gzip.open(compressed_path, "rb") as source:
                with open(weight_path, "wb") as target:
                    target.write(source.read())
            print("Decompression complete")
            os.remove(compressed_path)
            print("Removing compressed file")


version = SourceFileLoader(
    "birdvoxdetect.version", os.path.join("birdvoxdetect", "version.py")
).load_module()

with open("README.md") as file:
    long_description = file.read()

setup(
    name="birdvoxdetect",
    version=version.version,
    description="Bioacoustic monitoring of nocturnal bird migration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BirdVox/birdvoxdetect",
    author="Vincent Lostanlen, Justin Salamon, Andrew Farnsworth, "
    + "Steve Kelling, and Juan Pablo Bello",
    author_email="vincent.lostanlen@nyu.edu",
    packages=find_packages(),
    entry_points={
        "console_scripts": ["birdvoxdetect=birdvoxdetect.cli:main"],
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
    ],
    keywords="tfrecord",
    project_urls={
        "Source": "https://github.com/BirdVox/birdvoxdetect",
        "Tracker": "https://github.com/BirdVox/birdvoxdetect/issues",
    },
    install_requires=[
        "birdvoxclassify>=0.3",
        "h5py>=2.7.0,<3.0.0",
        "librosa==0.7.0",
        "numba==0.48.0",
        "numpy==1.16.4",
        "pandas==0.25.1",
        "scikit-learn==0.21.2",
        "tensorflow>=2.2",
    ],
    extras_require={
        "docs": [
            "sphinx==1.2.3",  # autodoc was broken in 1.3.1
            "sphinxcontrib-napoleon",
            "sphinx_rtd_theme",
            "numpydoc",
        ],
        "tests": [],
    },
    package_data={
        "birdvoxdetect": [os.path.join("models", fname) for fname in weight_files]
    },
)
