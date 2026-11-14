"""Build kit entry point: rebuild DentUnity from your own upstream downloads.

    python -m unidental.rebuild --pano-root <panoramic download> --oral-root <intraoral download> --out dentunity_v1

This forwards to release/build_dentunity.py (standard library only), which matches every
image and label file to release/manifests/*.csv by the sha256 of its bytes.
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv[0] = "build_dentunity.py"
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "release" / "build_dentunity.py"), run_name="__main__")
