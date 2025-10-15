# Setup Guide - TMM Genetic Optimization

## Environment Setup

This project uses a Python virtual environment to manage dependencies.

### Initial Setup (Already Done)

The virtual environment has been created and all dependencies are installed.

### Activating the Virtual Environment

**Every time you want to run the code**, activate the virtual environment first:

```bash
cd "/Users/ijun-o/Documents/PIE Lab/TMM/tmm_genetic"
source venv/bin/activate
```

You should see `(venv)` appear in your terminal prompt.

### Deactivating the Virtual Environment

When you're done:

```bash
deactivate
```

## Running the Code

### 1. Gaussian Beam Optimization

```bash
# Activate environment
source venv/bin/activate

# Run optimization
python gaussian_opt_write.py
```

**Results saved to**: `ga_runs/zR_<value>um/<timestamp>-<id>/`

### 2. Lens-based Optimization

```bash
# Activate environment
source venv/bin/activate

# Run optimization
python lens_opt_write_angle.py
```

**Results saved to**: `ga_runs/THETA_<value>deg/<timestamp>-<id>/` (or NA-based)

### 3. Analyzing Results

After optimization completes, edit the `d_list` in the result script with your optimized values, then run:

```bash
# For Gaussian results
python gaussian_result_fitting.py

# For lens results
python lens_result.py
```

### 4. Batch Runs

To run multiple optimizations with different random seeds:

```bash
python quick_batch_fitness.py
```

Edit the script to configure:
- `N_RUNS`: Number of runs
- `FITNESS_TARGET`: Success threshold
- `Z_R_TAG`: Which parameter folder to use

## Installed Packages

```
numpy       1.26.4      - Numerical computing
scipy       1.16.2      - Scientific computing & optimization
matplotlib  3.10.7      - Plotting and visualization
pygad       3.5.0       - Genetic algorithm framework
```

## Updating Dependencies

If you need to update packages:

```bash
source venv/bin/activate
pip install --upgrade <package_name>

# Or update all at once
pip install --upgrade -r requirements.txt
```

## Reinstalling from Scratch

If you need to recreate the environment:

```bash
# Remove old environment
rm -rf venv

# Create new environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Troubleshooting

### Issue: "command not found: python"
**Solution**: Use `python3` instead of `python`

### Issue: Packages not found when running scripts
**Solution**: Make sure you activated the virtual environment first:
```bash
source venv/bin/activate
```

### Issue: "externally-managed-environment" error
**Solution**: You must use the virtual environment. Never use `pip3 install` without activating `venv` first.

### Issue: Matplotlib doesn't show plots
**Solution**: On macOS, you may need to install the backend:
```bash
pip install PyQt5
```

## Python Version

This project is configured for **Python 3.13.3** (your current system version).

The code should work with Python 3.9+ but has been tested with 3.13.3.

## System Information

- **OS**: macOS (Apple Silicon / ARM64)
- **Python**: 3.13.3 at `/opt/homebrew/bin/python3`
- **Virtual Environment**: `./venv/`
