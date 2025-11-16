# ==========================================================
# QM9 FEATURE EXTRACTION SCRIPT
# ----------------------------------------------------------
# This script processes a subset of the QM9 molecular dataset
# and computes its SID – Sorted Interatomic Distances.
# The output is saved as a MATLAB .mat file for downstream use
# in clustering, regression, or quantum machine learning tasks.
# ==========================================================

import qm9pack
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import scipy.io as sio

# ========== SETTINGS ==========
N_MOLECULES = 100                    # Number of molecules to process from QM9
SAVE_PATH = r"QM9_features_SID_ONLY.mat"  # Output .mat file path

# ========== LOAD DATA ==========
# Load QM9 dataset (via qm9pack library)
qm9_data = qm9pack.get_data('qm9')
print("✅ Columns available:", qm9_data.columns.tolist())

# ========== DETECT ALL NUMERIC PROPERTY COLUMNS ==========
# Exclude non-numeric fields such as molecular identifiers
exclude_cols = ['SMILES', 'InChI', 'XYZ']

# Automatically identify all numerical property columns (target labels)
property_cols = [
    col for col in qm9_data.columns
    if col not in exclude_cols and pd.api.types.is_numeric_dtype(qm9_data[col])
]
label_names = property_cols
print(f"📊 Found {len(label_names)} numeric labels.")

# Extract molecular identifiers and numeric properties
smiles_list = qm9_data['SMILES'][:N_MOLECULES]  # SMILES strings for first N molecules
labels_matrix = qm9_data[label_names][:N_MOLECULES].to_numpy()  # Corresponding numeric labels

# ========== FEATURE STORAGE ==========
features_sid = []  # Sorted Interatomic Distances vectors

# ==========================================================
# ========== FEATURE COMPUTATION FUNCTIONS ==========
# ==========================================================

def compute_sid(positions, atom_numbers):
    """
    Compute Sorted Interatomic Distances (SID) descriptor.
    Removes hydrogen atoms and returns a sorted vector of all
    pairwise distances between remaining atoms.

    SID is efficient, rotation/translation invariant,
    and captures molecular geometry without explicit connectivity.
    """
    # Exclude hydrogen atoms (Z = 1)
    non_h = [i for i, z in enumerate(atom_numbers) if z != 1]
    positions = positions[non_h]

    # Compute pairwise Euclidean distances between atoms
    distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            distances.append(d)

    # Return sorted distances (ascending)
    return np.sort(distances)


def smiles_to_geometry(smiles):
    """
    Converts a SMILES string into 3D coordinates and atomic numbers.
    Uses RDKit’s ETKDG algorithm for molecular geometry embedding.
    Returns:
        positions     - numpy array (n_atoms × 3)
        atom_numbers  - list of atomic numbers
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # Add hydrogens for completeness
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    conf = mol.GetConformer()

    positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    atom_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return positions, atom_numbers


# ==========================================================
# ========== MAIN PROCESS LOOP ==========
# ==========================================================
for smiles in tqdm(smiles_list, desc="Processing molecules"):
    try:
        # Convert SMILES to atomic geometry
        R, Z = smiles_to_geometry(smiles)

        # Compute SID feature
        sid_vec = compute_sid(R, Z)
        features_sid.append(sid_vec)

    except Exception as e:
        # Continue even if a molecule fails (e.g., embedding issue)
        print(f"⚠️ Error with SMILES: {smiles} → {e}")
        continue


# ==========================================================
# ========== PADDING SID TO UNIFORM LENGTH ==========
# ==========================================================
# Molecules have different numbers of atoms → different vector lengths.
# To build a uniform matrix, pad shorter vectors with zeros.
max_sid_len = max(len(v) for v in features_sid)
features_sid = np.array([np.pad(v, (0, max_sid_len - len(v))) for v in features_sid])
# ==========================================================
# ========== NORMALIZATION STEP (to unit sphere) ==========
# ==========================================================
# Normalize each SID vector to have L2 norm = 1
norms = np.linalg.norm(features_sid, axis=1, keepdims=True)
norms[norms == 0] = 1  # avoid division by zero
features_sid = features_sid / norms

# ==========================================================
# ========== SAVE ALL RESULTS TO .MAT FILE ==========
# ==========================================================
sio.savemat(SAVE_PATH, {
    'features_sid': features_sid,                         # SID feature matrix
    'labels': labels_matrix,                              # QM9 numerical labels
    'label_names': np.array(label_names, dtype='object'), # Label names (properties)
})

print(f"💾 Saved {len(features_sid)} molecules to {SAVE_PATH}")

