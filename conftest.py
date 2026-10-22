import os
import sys

# Makes sure the project root is in sys.path, so that
# `from api.main import app` works whatever the directory pytest is
# started from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
