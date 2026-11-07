"""Rebuild the preregistered adaptive-gate experiment from a clean checkout."""
import hashlib
from pathlib import Path
import subprocess
import sys


OUTPUT_SHA256 = 'c72198bca88d91fafec88d442d3c6b029b7639d666a895e4ee90b039722f7442'


def main():
    subprocess.run([sys.executable, 'run_current_best.py'], check=True)
    subprocess.run([sys.executable, 'vacv_adaptive_gate.py'], check=True)
    path = Path('submission_v2_adaptive_gate.csv')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != OUTPUT_SHA256:
        raise ValueError(f'Candidate CSV hash changed: {actual}')
    print(f'Adaptive-gate candidate reproduced: {path}')


if __name__ == '__main__':
    main()
