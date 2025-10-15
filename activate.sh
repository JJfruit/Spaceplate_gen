#!/bin/bash
# Quick activation script for TMM genetic optimization environment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "TMM Genetic Optimization Environment"
echo "=========================================="
echo ""
echo "Activating virtual environment..."
source "$SCRIPT_DIR/venv/bin/activate"

echo "✓ Virtual environment activated"
echo ""
echo "Installed packages:"
pip list | grep -E "(numpy|scipy|matplotlib|pygad)" | column -t
echo ""
echo "Quick commands:"
echo "  • Run Gaussian optimization:  python gaussian_opt_write.py"
echo "  • Run lens optimization:      python lens_opt_write_angle.py"
echo "  • Test environment:           python test_setup.py"
echo "  • Deactivate environment:     deactivate"
echo ""
