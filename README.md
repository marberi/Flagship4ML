[![Python application](https://img.shields.io/badge/python_application-passing-success)](https://github.com/lauracabayol/TEMPS/actions)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Mkdocs](https://img.shields.io/badge/mkdocs-passing-success)](https://github.com/lauracabayol/TEMPS/actions)
[![Docs](https://img.shields.io/badge/docs-passing-success)](https://lauracabayol.github.io/Flagship4ML/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# Flagship4ML

This repository contains the code for the Flagship4ML project, a package for generating simulated images of galaxies for machine learning.
For more information on the project, please visit the [docs website (TBD)](https://lauracabayol.github.io/Flagship4ML/).
## Description

Flagship4ML is a Python package that generates synthetic galaxy images based on cosmological simulations. It provides researchers and machine learning practitioners with a reliable source of labeled galaxy data for training and testing their models.

## Installation

You can install Flagship4ML using pip:

```bash
pip install -e git+https://github.com/lauracabayol/Flagship4ML.git
```

Or install directly from source:

```bash
git clone git+https://github.com/lauracabayol/Flagship4ML.git
cd Flagship4ML
pip install -e .
```

## Updates from Jiefeng

The order of the psf and the nosie addition.(The first update version)

Added a real PAUS background (cutout_background1.npy), it is optional in the generation.

Added real sampled zp (zpsamples.npy) using a fitted distribution of the PAUS zp (in the magnitude range [17,23] in the COSMOS field in 40 bands), it is optional in the generation.

Added 'cal_error' option, which generates many exposures for a galaxy to calculate the flux error.

Added image size modification, which cut the edges of the output images, to remove the 'drop flux' part. Originally we need to cut the edges by ourselves after the generation.

Added the 'add_thin' and 'add_bright' options, to make the galaxy brighter or fainter (more visiable or less visible in the image).

## Usage

To generate simulated images, you can use the `CreateSimulatedImages` class. Here's a basic example:
```python
from flagship4ml import CreateSimulatedImages
t0=time.time()
ImageSimulator = create_simulated_images(path_to_catalogue.parquet,
                                         Ngals=1_000,
                                         bands=bands,
                                         add_poisson=True,
                                         add_psf=True,
                                         add_constant_background=True,
                                         use_dask=False,
                                         num_exposures=3,
                                         output_dir='/path/to/output/',
                                        )
```

One can also use the `sims_generator.py` script to generate simulated images from a catalogue in parquet format.
```bash
python Flagship4ML/bin/create_sims.py --config path/to/config.yaml
```
