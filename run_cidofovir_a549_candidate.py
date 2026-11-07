"""Rebuild the A549 publication candidate from a clean Kaggle data snapshot."""
import hashlib
from pathlib import Path
import subprocess
import sys


EXPECTED = '9dbf100872cb99818b765e382815db1a7edd7dcc65125ab2b1a88d0acd5dc263'


def main():
    subprocess.run([sys.executable, 'run_current_best.py'], check=True)
    subprocess.run([sys.executable, 'vacv_cidofovir_a549_publication.py'], check=True)
    path = Path('submission_v2_cidofovir_a549.csv')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED:
        raise ValueError(f'Candidate CSV changed: {actual}')
    print(f'Candidate reproduced: {path}; SHA256 {actual}')


if __name__ == '__main__':
    main()
