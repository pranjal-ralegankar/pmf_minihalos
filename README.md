This code requires CLASS https://github.com/lesgourg/class_public. 

The Pk_plot code generates the dark matter power spectrum for a given input of primordial magnetic field's initial values. Here initial is defined
by the time the coherence length of the magnetic field becomes equivalent to the photon silk damping scale (see https://arxiv.org/abs/2303.11861 for details).

The Pk_plot imports functions from several other modules defined below.

The background module solves the background cosmology evolution for standard cosmology.

The power_spectra module contains all the functions that determine the spectrum of the PMF source term that induces matter density perturbations.
The power_spectra module is for non-helical magnetic fields while power_spectra_helical is for maximally helical magnetic fields.

The pmf_perturbations contains the equations that soove for the evolution of perturbations, both with and without primordial magentic fields.

