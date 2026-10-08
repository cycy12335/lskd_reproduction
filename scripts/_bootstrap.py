import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
