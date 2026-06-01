import itertools

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_STATE = 50

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
DIPEPTIDES = ["".join(pair) for pair in itertools.product(AMINO_ACIDS, repeat=2)]
