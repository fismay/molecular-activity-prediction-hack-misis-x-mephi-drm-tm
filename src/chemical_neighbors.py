"""SMILES-only chemical similarities with explicit scaffold stereo gating."""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as F
from stereo_neighbors import scaffold


FEATURES = ['morgan_count_r1', 'morgan_count_r2', 'morgan_count_r3',
            'morgan_sparse_r2', 'morgan_sparse_r3', 'morgan_bit_r2',
            'rdkit_count', 'atom_pair_sparse', 'torsion_sparse']


def fingerprints(molecules, feature):
    if feature.startswith('morgan_'):
        generator = F.GetMorganGenerator(radius=int(feature[-1]), includeChirality=True)
    elif feature == 'rdkit_count':
        generator = F.GetRDKitFPGenerator()
    elif feature == 'atom_pair_sparse':
        generator = F.GetAtomPairGenerator(includeChirality=True, use2D=True)
    elif feature == 'torsion_sparse':
        generator = F.GetTopologicalTorsionGenerator(includeChirality=True)
    else:
        raise ValueError('Unknown molecular feature')
    if 'sparse' in feature:
        method = generator.GetSparseCountFingerprint
    elif '_bit_' in feature:
        method = generator.GetFingerprint
    else:
        method = generator.GetCountFingerprint
    return [method(molecule) for molecule in molecules]


def stereo_mask(reference_molecules, query_molecules):
    references = [scaffold(molecule) for molecule in reference_molecules]
    queries = [scaffold(molecule) for molecule in query_molecules]
    mask = np.ones((len(queries), len(references)), dtype=bool)
    compatibility = {}
    for i, (achiral, chiral, core) in enumerate(queries):
        for j, (other_achiral, other_chiral, other_core) in enumerate(references):
            if achiral != other_achiral or chiral == other_chiral:
                continue
            key = tuple(sorted((chiral, other_chiral)))
            if key not in compatibility:
                compatibility[key] = (core.HasSubstructMatch(other_core, useChirality=True)
                                      or other_core.HasSubstructMatch(core, useChirality=True))
            mask[i, j] = compatibility[key]
    return mask


def similarities(reference_smiles, query_smiles, feature, gate=True):
    references = [Chem.MolFromSmiles(s) for s in reference_smiles]
    queries = [Chem.MolFromSmiles(s) for s in query_smiles]
    if any(m is None for m in references + queries):
        raise ValueError('Invalid SMILES')
    reference_fps = fingerprints(references, feature)
    query_fps = fingerprints(queries, feature)
    result = np.asarray([DataStructs.BulkTanimotoSimilarity(fp, reference_fps) for fp in query_fps])
    if gate:
        result *= stereo_mask(references, queries)
    return result


def neighbor_values(similarity, y, k=1, power=2):
    k = min(int(k), similarity.shape[1])
    if k < 1:
        raise ValueError('No reference observations')
    cutoff = np.partition(similarity, -k, axis=1)[:, -k]
    weights = np.where(similarity >= cutoff[:, None]-1e-12, similarity**power, 0.)
    denominator = weights.sum(1)
    return np.divide(weights @ np.asarray(y, float), denominator,
                     out=np.zeros(len(similarity)), where=denominator > 1e-12)


def nearest_recipe(reference_smiles, y, query_smiles, feature='morgan_count_r2', k=1, power=2):
    return neighbor_values(similarities(reference_smiles, query_smiles, feature), y, k, power)
