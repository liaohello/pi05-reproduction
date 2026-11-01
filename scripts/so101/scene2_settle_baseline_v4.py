#!/usr/bin/env python3
"""V4 baseline wrapper: v3 baseline logic on the v4 narrowed family."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene2_settle_baseline_v3 as v3
import scene2_common_v4 as s2
v3.s2 = s2
if __name__ == "__main__":
    sys.exit(v3.main())
