#!/usr/bin/env python3
"""V4 wrapper: run scene2_certify.py against the v4 narrowed-family adapter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene2_certify as v1
import scene2_common_v4 as s2
v1.s2 = s2
if __name__ == "__main__":
    sys.exit(v1.main())
