import os
import sys
from pathlib import Path

# tests import `app.*`, so the backend root has to be importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# never let a test reach the real API, whatever is in backend/.env
os.environ["LLM_MOCK"] = "1"
