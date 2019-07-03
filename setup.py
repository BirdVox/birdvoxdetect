import os
import sys
import gzip
import imp
from setuptools import setup, find_packages

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

model_dir = os.path.join('birdvoxdetect', 'models')
suffixes = [
    'pcen_cnn',
    'pcen_cnn_adaptive-threshold-T1800',
    'empty'  # for unit tests
]
weight_files = ['birdvoxdetect_{}.h5'.format(suffix) for suffix in suffixes]
base_url = 'https://github.com/BirdVox/birdvoxdetect/raw/models/'

if len(sys.argv) > 1 and sys.argv[1] == 'sdist':
    # exclude the weight files in sdist
    weight_files = []
else:
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # in all other cases, decompress the weights file if necessary
    for weight_file in weight_files:
        weight_path = os.path.join(model_dir, weight_file)
        if not os.path.isfile(weight_path):
            compressed_file = weight_file + '.gz'
            compressed_path = os.path.join(model_dir, compressed_file)
            if not os.path.isfile(compressed_file):
                print('Downloading weight file {} ...'.format(compressed_file))
                urlretrieve(base_url + compressed_file, compressed_path)
            print('Decompressing ...')
            with gzip.open(compressed_path, 'rb') as source:
                with open(weight_path, 'wb') as target:
                    target.write(source.read())
            print('Decompression complete')
            os.remove(compressed_path)
            print('Removing compressed file')


version = imp.load_source(
    'birdvoxdetect.version', os.path.join('birdvoxdetect', 'version.py'))

with open('README.md') as file:
    long_description = file.read()

setup(
    name='birdvoxdetect',
    version=version.version,
    description='Bioacoustic monitoring of nocturnal bird migration',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/BirdVox/birdvoxdetect',
    author='Vincent Lostanlen, Justin Salamon, Andrew Farnsworth, ' +\
        'Steve Kelling, and Juan Pablo Bello',
    author_email='vincent.lostanlen@nyu.edu',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['birdvoxdetect=birdvoxdetect.cli:main'],
    },
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='tfrecord',
    project_urls={
        'Source': 'https://github.com/BirdVox/birdvoxdetect',
        'Tracker': 'https://github.com/BirdVox/birdvoxdetect/issues'
    },
    install_requires=[
        'h5py>=2.9.0',
        'jams>=0.3.2',
        'librosa>=0.6.2',
        'mir-eval>=0.5',
        'numpy>=1.16.0',
        'PySoundFile>=0.9.0',
        'resampy>=0.2.1',
        'scikit-learn>=0.21.2',
        'scipy>=1.0.0',
        'pandas>=0.24',
        'pescador>=2.0.0',
        'tensorflow>=1.14.0'
    ],
    extras_require={
        'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
        'tests': [
        ]
    },
    package_data={
        'birdvoxdetect': [
            os.path.join('models', fname)
            for fname in weight_files
            ]
    }
)
