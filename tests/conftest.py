"""Pytest configuration for SafeVoice tests."""
import sys
import os

# Add project root to path so `from src.X import Y` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
