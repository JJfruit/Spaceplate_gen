# Inverse Design Spaceplate (Gradient Descent)

Alternative optimization approach using **gradient descent** instead of genetic algorithms.

## Authors

- Jordan Pagé (University of Ottawa) - jpage019@uottawa.ca
- Orad Reshef - orad@reshef.ca
- Last updated: May 2021

## Overview

This code generates multi-layer bi-material spaceplates using gradient descent optimization, where the parameter space consists of all layer thicknesses.

### Key Differences from GA Approach

| Aspect | Gradient Descent (This) | Genetic Algorithm (Main) |
|--------|------------------------|--------------------------|
| **Method** | Local hill-climbing | Global population-based |
| **Strategy** | Progressive layer addition | Fixed layer count |
| **Parallelization** | Swarm (100 devices) | Population (500 individuals) |
| **Target** | Fixed R=7 | R≥5 (constraint) |
| **Speed** | Fast convergence | Slower but robust |
| **Local Optima** | Vulnerable (restart mechanism) | Less likely |

## Files

```
Inverse_Design_Spaceplate-main/
├── gradient_descent.py      - Main optimization engine
├── TMM_subroutines.py       - TMM implementation with numba JIT
├── spaceplate.py            - Spaceplate-specific functions
├── TMM_sweep.py             - Parameter sweep utilities
├── plot_results.py          - Visualization tools
└── README.md                - This file
```

## Algorithm

### Progressive Layer Optimization

The algorithm starts with a small number of layers (default: 3) and progressively adds more layers to the optimization:

```
Start: 3 layers → optimize
Add 1 layer → 4 layers → optimize
Add 1 layer → 5 layers → optimize
...
Final: 13 layers → optimize
```

### Two-Stage Gradient Descent

**Stage 1: Coarse Search**
- Step size: 0.47 nm
- Goal: Find promising regions
- Threshold: FOM > 600

**Stage 2: Fine Tuning**
- Step size: 0.2 nm
- Goal: Refine to FOM > 700
- Only triggered if Stage 1 FOM > 600

### Swarm-Based Parallel Search

- Creates 100 random devices initially
- Optimizes all in parallel using multiple CPU cores
- Devices stuck at local maxima are deleted and replaced
- Successful devices (FOM > 700) are saved

## Physics

### Target Transfer Function

The ideal spaceplate phase profile:

```python
φ(θ) = -k · R · d_total · (1 - cos(θ))
```

Where:
- `k = 2π/λ` - wavevector
- `R` - compression ratio (target: 7)
- `d_total` - device physical thickness
- `θ` - incident angle

### Figure of Merit (FOM)

```python
RMSE = √(mean((φ_actual - φ_ideal)²))
FOM = 1 / RMSE
```

Higher FOM = better spaceplate performance

## Usage

### Basic Run

```bash
# Activate virtual environment
source venv/bin/activate

# Run optimization
python gradient_descent.py
```

### Key Parameters (Edit in main())

```python
# Device structure
num_layers = 13              # Number of alternating layers
n_low = 1.444                # SiO2 refractive index @ 1550nm
n_high = 3.47638             # Si refractive index @ 1550nm

# Simulation
wavelength = 1550e-9         # Wavelength in meters
max_angle = 10               # Maximum incident angle (degrees)
polarization = 'p'           # 'p' or 's'

# Layer thickness bounds
min_layer_thickness = 50     # nm
max_layer_thickness = 200    # nm

# Optimization targets
target_R = 7                 # Desired compression ratio
stage_2_threshold = 600      # FOM to trigger fine optimization
saving_threshold = 700       # FOM to save results

# Gradient descent parameters
step1 = 0.47e-9             # Stage 1 step size (nm)
step2 = 0.2e-9              # Stage 2 step size (nm)
derivative_step = 0.05e-9    # Numerical derivative dx

# Progressive optimization
default_start = 3            # Start with 3 layers
optimize_layers_increase = 1 # Add 1 layer at a time
```

### Parallelization

```python
num_computer_cores = 2       # Number of CPU cores to use
num_points_in_swarm = 100    # Total devices (must be divisible by cores)
```

## Output

Successful devices are saved to:
```
spaceplate_13layers_targetR7_theta10.txt
```

Contains:
- Optimized layer thicknesses
- Final FOM value
- Optimization metadata

## Dependencies

```bash
pip install numpy scipy matplotlib lmfit numba
```

- **numpy** - Numerical operations
- **scipy** - Scientific computing
- **matplotlib** - Visualization
- **lmfit** - Curve fitting for phase analysis
- **numba** - JIT compilation for TMM speedup

## Integration with Main Codebase

This gradient descent optimizer can be used in conjunction with the main GA optimizer:

**Suggested hybrid workflow:**
```
1. Run GA optimizer (global search)
   └─ Find promising parameter region

2. Use GA result as initial guess for GD
   └─ Fine-tune with gradient descent

Result: Combines global exploration + local refinement
```

## Notes

- Gradient descent is fast but can get stuck in local optima
- The swarm approach helps by exploring multiple starting points
- Progressive layer addition helps avoid getting trapped early
- Best used when you have a specific target R value
- GA is better for exploring unknown parameter spaces

## Reference

Based on work from University of Ottawa research group on inverse-designed optical spaceplates.

For comparison with genetic algorithm approach, see the main codebase documentation.
