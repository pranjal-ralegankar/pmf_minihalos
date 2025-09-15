# Dark Matter Power Spectrum with Primordial Magnetic Fields

This repository contains code to compute the dark matter power spectrum for a given input of primordial magnetic field (PMF) initial values. The initial PMF is defined at the time when the coherence length of the magnetic field becomes equivalent to the photon Silk damping scale. For further details, see [arXiv:2303.11861](https://arxiv.org/abs/2303.11861). Note the code only works for PMF strengths between 1e-6 nG and 0.05 nG today. 
## Installation

To set up the environment and install dependencies:

1. ```bash
   conda env create -f environment.yml
   conda activate pmf_halos
   ```

2. ```bash
   pip install -r requirements.txt
   ```

## Code Structure

The main script is **dark_matter_spectrum.py**:

- **dark_matter_spectrum.py**  
  This script generates the dark matter power spectrum by importing and combining functions from the modules below.  
  It computes the evolution of the magnetic field, solves the perturbation equations, and outputs the power spectrum as a function of wavenumber.

- **background.py**  
  Solves the background cosmology evolution for standard cosmology.  
  Computes the evolution of the magnetic field from the time of phase transition until today.  
  Provides functions to obtain magnetic field values at the initial time (when coherence length equals Silk damping scale).

- **power_spectra.py**  
  Contains functions to determine the spectrum of the PMF source term that induces matter density perturbations.  
  This module is designed for non-helical magnetic fields.

- **PMF_perturbations.py**  
  Contains the equations and routines to solve for the evolution of cosmological perturbations, both with and without primordial magnetic fields.

## Module Interactions

- The **background** module provides cosmological background quantities and magnetic field evolution functions.  
  These are used by both the **power_spectra** and **PMF_perturbations** modules.

- The **power_spectra** module uses background quantities (such as the magnetic field strength and coherence scale) to compute the PMF source term spectrum.  
  Its main function, `find_S0`, is called by the main script and by the perturbation solver.

- The **PMF_perturbations** module uses the PMF source term from **power_spectra** and background cosmology from **background** to solve the evolution of density and velocity perturbations.  
  It provides routines to compute the evolution of dark matter and baryon perturbations with and without PMFs.

- The main script (`dark_matter_spectrum.py`) orchestrates the computation:  
  It initializes the background, computes the PMF source spectrum, solves the perturbation equations, and finally generates the dark matter power spectrum.

## Tests Folder and Jupyter Notebooks

The `tests` folder contains scripts and Jupyter notebooks to validate and demonstrate the functionality of each module. Notable `.ipynb` files include:

- **test_background_B_xi.ipynb**  
  Shows the evolution of magnetic field strength and cohrence scale from phase transition until today.  

- **test_PMF_perturbations.ipynb**  
  Shows the evolution of cosmological perturbations with and without PMFs.  
  Plots the growth of density perturbations.

- **test_dark_matter_spectrum.ipynb**  
  Runs the full pipeline: background evolution, PMF source spectrum calculation, perturbation evolution, and power spectrum generation.  
  Demonstrates how to use the main modules together and visualizes the resulting dark matter power spectrum.

Each notebook is self-contained and provides code, explanations, and plots for interactive exploration and validation.

## Reference

For theoretical background and details of the implementation, refer to:  
[Primordial magnetic fields and dark matter power spectrum](https://arxiv.org/abs/2303.11861)
