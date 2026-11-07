"""Rebuild the current best public Kaggle CSV from the three competition files."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


OUTPUT_SHA256 = '9dbf100872cb99818b765e382815db1a7edd7dcc65125ab2b1a88d0acd5dc263'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/kaggle'))
    parser.add_argument('--output', type=Path, default=Path('submission_v2_cidofovir_a549.csv'))
    args = parser.parse_args()
    base = Path('submission_v2_stereo_hurdle.csv')
    subprocess.run([sys.executable, 'run_best.py', '--data-dir', str(args.data_dir),
                    '--output', str(base)], check=True)
    cage = Path('submission_v2_cage_publication.csv')
    subprocess.run([sys.executable, 'vacv_cage_publication.py',
                    '--train', str(args.data_dir / 'train_v2.csv'),
                    '--test', str(args.data_dir / 'test.csv'),
                    '--sample', str(args.data_dir / 'sample_submit.csv'),
                    '--base', str(base), '--output', str(cage)], check=True)
    subprocess.run([sys.executable, 'vacv_cidofovir_a549_publication.py',
                    '--train', str(args.data_dir / 'train_v2.csv'),
                    '--test', str(args.data_dir / 'test.csv'),
                    '--sample', str(args.data_dir / 'sample_submit.csv'),
                    '--base', str(cage), '--output', str(args.output)], check=True)
    actual = hashlib.sha256(args.output.read_bytes()).hexdigest()
    if actual != OUTPUT_SHA256:
        raise ValueError(f'Unexpected final CSV hash: {actual}')
    print(f'Confirmed public-best submission reproduced: {args.output}')


if __name__ == '__main__':
    main()
