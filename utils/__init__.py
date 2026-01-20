# tests/__init__.py
import os, sys
from dotenv import load_dotenv

# Add the project root (parent directory) to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load .env once for all tests
load_dotenv(os.path.join(ROOT, ".env"))
