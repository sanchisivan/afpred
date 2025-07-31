import numpy as np

# Diccionario de codificación simple one-hot para 20 aminoácidos naturales
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def preprocess_sequence(seq, max_len=50):
    seq = seq.upper()
    if any(a not in AMINO_ACIDS for a in seq):
        raise ValueError("Secuencia contiene aminoácidos no válidos.")
    if len(seq) > max_len:
        raise ValueError(f"La longitud máxima permitida es {max_len} aminoácidos.")
    tensor = np.zeros((1, max_len, len(AMINO_ACIDS)))
    for i, aa in enumerate(seq):
        tensor[0, i, AA_INDEX[aa]] = 1
    return tensor
