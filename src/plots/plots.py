from pathlib import Path
import matplotlib.pyplot as plt

def plot_length_distribution(df, title, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.hist(df["len"], bins=30)
    plt.title(title)
    plt.xlabel("Długość")
    plt.ylabel("Liczba")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_positive_vs_negative(pos, neg, neg_label, title, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.hist(pos["len"], bins=30, alpha=0.5, label="positive")
    plt.hist(neg["len"], bins=30, alpha=0.5, label=neg_label)
    plt.title(title)
    plt.xlabel("Długość")
    plt.ylabel("Liczba")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()