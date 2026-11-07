"""Rebuild the current best public Kaggle CSV from the three competition files."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


OUTPUT_SHA256 = '8a0d40cf37f6a4a3a6d99815e6961b7d89f9493c2e031880b506590dab0d2bf4'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/kaggle'))
    parser.add_argument('--output', type=Path, default=Path('submission_v2_cage_publication.csv'))
    args = parser.parse_args()
    base = Path('submission_v2_stereo_hurdle.csv')
    subprocess.run([sys.executable, 'run_best.py', '--data-dir', str(args.data_dir),
                    '--output', str(base)], check=True)
    subprocess.run([sys.executable, 'vacv_cage_publication.py',
                    '--train', str(args.data_dir / 'train_v2.csv'),
                    '--test', str(args.data_dir / 'test.csv'),
                    '--sample', str(args.data_dir / 'sample_submit.csv'),
                    '--base', str(base), '--output', str(args.output)], check=True)
    actual = hashlib.sha256(args.output.read_bytes()).hexdigest()
    if actual != OUTPUT_SHA256:
        raise ValueError(f'Unexpected final CSV hash: {actual}')
    print(f'Confirmed public-best submission reproduced: {args.output}')


if __name__ == '__main__':
    main()
