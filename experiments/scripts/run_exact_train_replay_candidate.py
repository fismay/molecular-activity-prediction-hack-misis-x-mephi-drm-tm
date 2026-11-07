"""Rebuild the exact train-structure replay candidate from the Kaggle files."""
import hashlib
from pathlib import Path
import subprocess
import sys


EXPECTED_SHA256 = 'b084f8031b0ac1aa207171abf0aa89795dce66d3de015bc97f6c3b5c60e57f83'


def main():
    subprocess.run([sys.executable, 'run_current_best.py'], check=True)
    subprocess.run([sys.executable, 'vacv_exact_train_replay.py'], check=True)
    path = Path('submission_v2_exact_train_replay.csv')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA256:
        raise ValueError(f'Unexpected candidate output: {actual}')
    print(f'Candidate reproduced: {path}; SHA256 {actual}')


if __name__ == '__main__':
    main()
