"""Recreate the confirmed Kaggle submission from a clean checkout."""
import argparse
import hashlib
from pathlib import Path

from vacv_zero_hurdle import TRAIN_V2_SHA256, main as train_main

SOURCE_SHA256 = '2ac50f4859e3d16a201eb97cc3bb4d17c7a973da21e1fa3d0aa183814d000f18'
TEST_SHA256 = '1b0c6a014bbe556c42ef2277e867d2e1a647ad31c157a9cd81bbc7e262481abd'
SAMPLE_SHA256 = '3eea4d23f5700361c09da24cd45be05c5dbbecf76cc29b022c3676835606884c'
OUTPUT_SHA256 = '0889724213860bd58db86e0b5ed0ed984160de4d67d9e943a0f199758ca96bae'


def check(path, expected):
    if not path.is_file():
        raise SystemExit(f'Missing file: {path}')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'SHA256 mismatch for {path}: {actual}; expected {expected}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/kaggle'))
    parser.add_argument('--output', type=Path, default=Path('submission_v2_stereo_hurdle.csv'))
    args = parser.parse_args()
    source = Path('artifacts/external/bornyl_amides_expanded_source.csv')
    train = args.data_dir / 'train_v2.csv'
    test = args.data_dir / 'test.csv'
    sample = args.data_dir / 'sample_submit.csv'
    for path, expected in ((source, SOURCE_SHA256), (train, TRAIN_V2_SHA256),
                           (test, TEST_SHA256), (sample, SAMPLE_SHA256)):
        check(path, expected)
    import sys
    original = sys.argv
    try:
        sys.argv = ['vacv_zero_hurdle.py', '--train', str(train), '--test', str(test),
                    '--sample', str(sample), '--source', str(source), '--calibration', 'raw',
                    '--conditional', 'source_stereo_only', '--model', 'artifacts/zero_v2_hurdle_nn.joblib',
                    '--output', str(args.output), '--report', 'zero_v2_release.json']
        train_main()
    finally:
        sys.argv = original
    check(args.output, OUTPUT_SHA256)
    print(f'Exact confirmed submission reproduced: {args.output}')


if __name__ == '__main__':
    main()
