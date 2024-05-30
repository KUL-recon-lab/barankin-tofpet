# barankin-tofpet

Demo implementation of the Barankin Bound to estimate the variace of the detector time resolution.
We provide two version:
- `barankin.py` uses an analytical description of the pdf as input
- `barankin_lut.py` uses a lookup table of the pdf as input

## Run example with continous pdf on binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KUL-recon-lab/barankin-tofpet/main?labpath=barankin.ipynb)

## Run example with look up table based pdf on binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KUL-recon-lab/barankin-tofpet/main?labpath=barankin_lut.ipynb)

## Running the scripts offline

1. Install `conda` or `mamba` e.g. from [here](https://github.com/conda-forge/miniforge).
2. clone this repository `git clone https://github.com/KUL-recon-lab/barankin-tofpet.git` or downloand and unzip the code
3. create a `conda` / `mamba` environment containing all required packages: `mamba env create -f environment.yml`
4. Activate the environment: `mamba activate barankin-tofpet`
5. Run the examples: `python barankin.py` or `python barankin_lut.py`
