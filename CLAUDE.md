# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Transfer Matrix Method (TMM) optical simulation project** for designing and optimizing multilayer "spaceplate" structures. The goal is to use genetic algorithms (GA) to find thin-film layer configurations that compress optical propagation distance while preserving beam phase and intensity characteristics.

**Core concept**: A spaceplate is a multilayer optical structure designed to mimic free-space propagation over a longer distance (e.g., 50 μm) using a much thinner physical structure (e.g., 10 μm), achieving a "compression ratio" R.

## Architecture

### Core TMM Engine
- **`tmm_core_test.py`**: Complete implementation of the Transfer Matrix Method for coherent and incoherent multilayer optical structures
  - Main functions: `coh_tmm()`, `inc_tmm()` for coherent/incoherent light propagation
  - Handles s- and p-polarization, complex refractive indices, angle-dependent transmission/reflection
  - Returns transmission coefficients (`t`), reflection coefficients (`r`), and power quantities (`T`, `R`)
  - Based on physics from https://arxiv.org/abs/1603.02720

### Optimization Scripts
The repository contains two main optimization approaches:

1. **Gaussian beam optimization** (`gaussian_opt_write.py`):
   - Optimizes for Gaussian beam input (characterized by Rayleigh range z_R)
   - Uses Angular Spectrum Method (ASM) for beam propagation
   - Fitness function combines:
     - Spatial phase matching (2D field at target plane)
     - Angular phase matching (transmission phase vs incident angle θ)
     - Intensity transmission threshold
   - Organized by z_R parameter in `ga_runs/zR_*/` directories

2. **Lens-based optimization** (`lens_opt_write_angle.py`):
   - Optimizes for spherical waves from a hyperbolic lens (characterized by NA or θ_max)
   - Similar ASM-based propagation and phase matching
   - Can organize runs by NA, θ_max, or both
   - Uses `RunLogger` class with flexible directory organization

### Material System
Both optimizers use alternating layers of:
- **a-Si** (amorphous silicon): n = 3.6211 + 1.26e-7j
- **SiO₂** (silicon dioxide): n = 1.4596 + 0.000011139j
- Layer thicknesses optimized in range [10, 1200] nm
- Total structure sandwiched between air (n=1)

### Result Analysis Scripts
- **`gaussian_result_fitting.py`**: Analyzes optimized Gaussian beam structures
  - Performs 2D propagation visualization (x-z intensity maps)
  - Fits x-profiles to Gaussian functions
  - Fits z-profiles to Rayleigh intensity distribution
  - Compares s-pol and p-pol performance

- **`gaussian_result_error_check.py`**: Similar analysis with fixed-center Rayleigh fitting

- **`lens_result.py`**: Analyzes lens-based optimizations with propagation maps

- **`layer.py`**: Simple visualization of layer structure profiles

### Quick Utility Scripts
- **`quick_plot_generation.py`** / **`quick_plot_generation2.py`**: Fast plotting for iteration
- **`quick_batch_fitness.py`**: Batch fitness evaluation

## Common Development Tasks

### Running an Optimization

**Gaussian beam optimization:**
```bash
python gaussian_opt_write.py
```
- Modify `z_R` parameter (line ~139) to change beam waist
- Modify `d_eff` (line ~136) to change target compression distance
- Results saved to `ga_runs/zR_<value>um/<timestamp>-<id>/`

**Lens optimization:**
```bash
python lens_opt_write_angle.py
```
- Modify `NA` (line ~154) to change numerical aperture
- Modify `theta_max_deg_for_loss` (line ~177) for angular phase matching range
- Results saved to `ga_runs/THETA_<value>deg/<timestamp>-<id>/` (or NA-based, see line 351)

### Key Parameters to Adjust

**Genetic Algorithm (both scripts):**
- `num_generations`: Default 500 (line ~397 in gaussian_opt, ~367 in lens_opt)
- `sol_per_pop`: Population size, default 500
- `num_genes`: Number of layers, default 25
- `gene_space`: Layer thickness range [10, 1200] nm

**Fitness weights (both scripts):**
- `W_SPACE`: Spatial phase error weight (default 0.1-0.5)
- `W_ANGLE`: Angular phase error weight (default 4.0)
- `weight_intensity`: Intensity penalty (default 0.1)
- `compression_ratio`: Minimum R value required (default 5)

**Thresholds:**
- `intensity_threshold_ratio`: Minimum transmission (0.5 = 50%)
- `phase_mse_threshold`: Spatial phase MSE cutoff
- `angle_mse_threshold`: Angular phase MSE cutoff

### Analyzing Results

After optimization completes, run the appropriate result script:
```bash
python gaussian_result_fitting.py
# or
python lens_result.py
```

Modify the `d_list` array in these files (around lines 17-86) with optimized layer thicknesses from the GA output.

### Output Structure

Each run creates:
- `meta.json`: Run parameters and configuration
- `fitness_per_generation.csv`: Fitness evolution over generations
- `best_d_list.json`: Optimized layer thicknesses [nm]
- `metrics.json`: Final performance metrics (R, T_rms, phase_mse, etc.)
- `runs_summary.csv`: Aggregated results across multiple runs (in parent directory)

## Important Implementation Details

### Phase Wrapping
The fitness function uses `angdiff()` (line ~274 in gaussian_opt) to compute phase differences in the [-π, +π] range, avoiding unwrapping artifacts when comparing phase profiles.

### Two-Domain Loss Function
Optimization considers both:
1. **Spatial domain**: Phase profile E(x) at target plane
2. **Angular domain**: Transmission phase φ(θ) vs incident angle

This dual approach ensures the spaceplate works correctly across the full angular spectrum of the beam.

### Angular Spectrum Method (ASM)
The `angular_spectrum_1d()` function (lines ~182-190 in gaussian_opt) propagates fields in Fourier space:
- Evanescent waves handled with `+0j` safety term
- Proper handling of `n_medium` for propagation in different materials

### TMM Integration
Each Fourier component (kx) is processed through the multilayer stack:
- `theta_kx = arcsin(kx/k0)` maps spatial frequency to incident angle
- `coh_tmm()` called for each angle with wavelength in **nm** (not m!)
- Transmission coefficient `t` applied to Fourier spectrum

### Coordinate Systems
- **Gaussian optimization**: z=0 is structure entrance, negative z means "distance to focus"
- **Lens optimization**: z=0 is lens plane, structure placed at z=z_pre

## Notes on TMM Module

The `tmm_core_test.py` implements standard TMM with:
- **Input units**: wavelength in **nm**, distances in **nm**, angles in **radians**
- Semi-infinite media must have `d_list[0] = d_list[-1] = inf`
- Returns complex amplitudes and real power quantities
- Handles absorbing media and evanescent waves
- No internal file I/O or side effects

## Wavelength

All simulations use **λ = 1550 nm** (telecom C-band). This is hardcoded in multiple places and should be changed consistently if needed.

## Dependencies

- `numpy`: Numerical operations and FFT
- `scipy`: Curve fitting, optimization
- `matplotlib`: Visualization
- `pygad`: Genetic algorithm library
- Standard library: `pathlib`, `json`, `csv`, `time`, `uuid`
