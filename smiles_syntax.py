"""Features of the supplied SMILES serialization, computed without assay data."""
import re
import string
import difflib
import numpy as np
from rdkit import Chem

SYMBOLS = list(string.ascii_letters + string.digits + '@[]()=+#-./\\% ')
PATTERNS = ['[C@H]', '[C@@H]', '[C@]', '[C@@]', '[nH]', '[NH]',
            'c1', 'c2', 'c3', 'c4', 'C1', 'C2', 'C3', 'C4', '(=O)',
            'C(=O)', 'O=C', 'N(=O)', '=C', '[O-]', '[N+]']
FEATURE_NAMES = (['length', 'length_ratio_canonical', 'equals_rdkit_canonical',
                  'equals_rdkit_kekule', 'position_difference_canonical']
                 + ['count_' + c for c in SYMBOLS]
                 + ['fraction_' + c for c in SYMBOLS]
                 + ['pattern_' + s for s in PATTERNS]
                 + ['max_ring_digit', 'lowercase_count', 'bracketed_atom_count']
                 + [name + '_' + metric for name in ('atom_order', 'atom_order_kekule', 'no_stereo')
                    for metric in ('equals', 'length_ratio', 'sequence_similarity')])


def featurize_syntax(smiles):
    rows = []
    for value in smiles:
        s = str(value)
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError('Invalid SMILES')
        canonical = Chem.MolToSmiles(mol)
        kekule = Chem.MolToSmiles(mol, kekuleSmiles=True)
        row = [len(s), len(s)/max(1, len(canonical)), float(s == canonical),
               float(s == kekule), sum(a != b for a, b in zip(s, canonical))/max(1, len(s))]
        row += [s.count(c) for c in SYMBOLS]
        row += [s.count(c)/max(1, len(s)) for c in SYMBOLS]
        row += [s.count(pattern) for pattern in PATTERNS]
        row += [max([int(c) for c in s if c.isdigit()] + [0]),
                sum(c.islower() for c in s), len(re.findall(r'\[[^]]+\]', s))]
        for options in ({'canonical': False}, {'canonical': False, 'kekuleSmiles': True},
                        {'isomericSmiles': False}):
            alternative = Chem.MolToSmiles(mol, **options)
            row += [float(s == alternative), len(s)/max(1, len(alternative)),
                    difflib.SequenceMatcher(None, s, alternative, autojunk=False).ratio()]
        rows.append(row)
    return np.asarray(rows, dtype=float)
