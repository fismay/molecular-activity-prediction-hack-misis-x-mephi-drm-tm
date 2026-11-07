"""Chiral count neighbors with optional same-scaffold stereo compatibility."""
import numpy as np
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator as F
from rdkit.Chem.Scaffolds import MurckoScaffold

def scaffold(mol):
    core=MurckoScaffold.GetScaffoldForMol(mol)
    if not core.GetNumAtoms():core=mol
    return (Chem.MolToSmiles(core,isomericSmiles=False),Chem.MolToSmiles(core,isomericSmiles=True),core)

def nearest_stereo(smiles_train,y,smiles_query,enforce_scaffold=True,exclude_value=10000):
    keep=np.arange(len(y)) if exclude_value is None else np.flatnonzero(y!=exclude_value);strings=list(smiles_train);values=np.asarray(y,float)[keep]
    reference=[Chem.MolFromSmiles(strings[i]) for i in keep];query=[Chem.MolFromSmiles(s) for s in smiles_query]
    if any(m is None for m in reference+query):raise ValueError('Invalid SMILES')
    if not len(reference):raise ValueError('No ordinary-mode neighbor references')
    generator=F.GetMorganGenerator(radius=2,includeChirality=True);fps=[generator.GetCountFingerprint(m) for m in reference]
    ref_scaffolds=[scaffold(m) for m in reference] if enforce_scaffold else [];compatibility={};predictions=[]
    for mol in query:
        similarity=np.asarray(DataStructs.BulkTanimotoSimilarity(generator.GetCountFingerprint(mol),fps))
        if enforce_scaffold:
            achiral,chiral,core=scaffold(mol)
            for j,(ref_achiral,ref_chiral,ref_core) in enumerate(ref_scaffolds):
                if achiral!=ref_achiral or chiral==ref_chiral:continue
                key=tuple(sorted((chiral,ref_chiral)))
                if key not in compatibility:
                    # An unspecified stereocenter is not an asserted conflict:
                    # either directional substructure match permits the pair.
                    compatibility[key]=(core.HasSubstructMatch(ref_core,useChirality=True) or ref_core.HasSubstructMatch(core,useChirality=True))
                if not compatibility[key]:similarity[j]=0.
        maximum=similarity.max();weights=np.where(similarity>=maximum-1e-12,similarity**2,0.)
        predictions.append(float(weights@values/max(weights.sum(),1e-12)))
    return np.asarray(predictions)
