"""
conftest.py — pytest configuration.
Adds src/ to sys.path so tests run without PYTHONPATH=src.
"""
import sys
from pathlib import Path

# Make `import lut_native` work regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
