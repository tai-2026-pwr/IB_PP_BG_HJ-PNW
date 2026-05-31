from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = Path("data/raw")

POSITIVE_PATH = RAW_DIR / "cpp.csv"
UNIPROT_PATH = RAW_DIR / "uniprot.xlsx"
PEPTIPEDIA_PATH = RAW_DIR / "peptipedia.fasta"
