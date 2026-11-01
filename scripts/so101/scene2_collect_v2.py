#!/usr/bin/env python3
"""V2 wrapper: run scene2_collect.py against the v2 adapter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene2_collect as v1
import scene2_common_v2 as s2
v1.s2 = s2
if __name__ == "__main__":
    sys.exit(v1.main())
