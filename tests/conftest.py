import sys
import os

# Non-interactive backend must be set before any pyplot import
import matplotlib
matplotlib.use("Agg")

# Make src/ importable from all test files
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
