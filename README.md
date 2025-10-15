# TMM Genetic Optimization

Transfer Matrix Method (TMM) for designing multilayer optical spaceplates using genetic algorithms.

## Quick Start

### 1. Activate Environment

```bash
cd "/Users/ijun-o/Documents/PIE Lab/TMM/tmm_genetic"
source venv/bin/activate
```

Or use the helper script:
```bash
source activate.sh
```

### 2. Test Installation

```bash
python test_setup.py
```

### 3. Run Optimization

**Gaussian beam:**
```bash
python gaussian_opt_write.py
```

**Lens-based:**
```bash
python lens_opt_write_angle.py
```

## Project Structure

```
tmm_genetic/
├── tmm_core_test.py                    # TMM physics engine (1008 LOC)
├── gaussian_opt_write.py               # GA optimizer for Gaussian beams
├── lens_opt_write_angle.py             # GA optimizer for lens systems
├── gaussian_result_fitting.py          # Analysis with Gaussian/Rayleigh fitting
├── lens_result.py                      # Lens result analysis
├── layer.py                            # Layer structure visualization
├── quick_batch_fitness.py              # Batch optimization runner
├── quick_plot_generation.py            # Plotting utilities
├── Inverse_Design_Spaceplate-main/     # Gradient descent optimizer (alternative)
│   ├── gradient_descent.py             # GD optimization engine
│   ├── TMM_subroutines.py              # TMM with numba JIT
│   ├── spaceplate.py                   # Spaceplate functions
│   └── README.md                       # GD documentation
├── requirements.txt                    # Python dependencies
├── SETUP.md                            # Detailed setup guide
├── CLAUDE.md                           # AI assistant context
└── venv/                               # Virtual environment (do not commit)
```

## Dependencies

**Core (Genetic Algorithm):**
- **NumPy 1.26.4** - Numerical computing & FFT
- **SciPy 1.16.2** - Scientific computing & curve fitting
- **Matplotlib 3.10.7** - Visualization
- **PyGAD 3.5.0** - Genetic algorithm framework

**Additional (Gradient Descent):**
- **lmfit 1.3.4** - Curve fitting for phase analysis
- **numba 0.62.1** - JIT compilation for TMM speedup

## Documentation

- **SETUP.md** - Detailed setup and usage instructions
- **CLAUDE.md** - Architecture overview for AI assistants
- **requirements.txt** - Dependency specifications

## Output

Results are saved in `ga_runs/` organized by parameter:
- `ga_runs/zR_<value>um/` - Gaussian beam results
- `ga_runs/THETA_<value>deg/` - Lens results (angle-based)
- `ga_runs/NA_<value>/` - Lens results (NA-based)

Each run creates:
- `meta.json` - Configuration
- `fitness_per_generation.csv` - Evolution history
- `best_d_list.json` - Optimized layer thicknesses
- `metrics.json` - Performance metrics
- `runs_summary.csv` - Aggregated results (in parent dir)

## Physics

- **TMM**: Transfer Matrix Method for multilayer optics
- **ASM**: Angular Spectrum Method for beam propagation
- **GA**: Genetic Algorithm for optimization
- **Materials**: Alternating a-Si (n=3.62) / SiO₂ (n=1.46) layers
- **Wavelength**: 1550 nm (telecom C-band)

## License

Research code for optical metasurface design.

## Support

See `SETUP.md` for troubleshooting and detailed usage.
# Spaceplate_gen
