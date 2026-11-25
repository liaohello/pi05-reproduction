import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# The candidate is imported as a package so tests can share support code.
sys.path.insert(0, str(_ROOT))
# The candidate source tree is imported directly: no editable install, so the
# locked environments stay untouched.
sys.path.insert(0, str(_ROOT / "src"))
