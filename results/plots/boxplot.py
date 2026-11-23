from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sns
import seaborn as sns

# Skrypt do generowania wykresów typu boxplot dla różnych metryk na podstawie wyników walidacji krzyżowej
# Nie dostosowany do aktualnej struktury katalogów!!! Docelowo należałoby zmienić ścieżki

SCRIPT_DIR = Path(__file__).resolve().parent


BASE_DIR = SCRIPT_DIR

input_files = {
    "AAC": BASE_DIR / "cv_folds_aac.csv",
    "DPC": BASE_DIR / "cv_folds_dpc.csv",
    "Cechy fizykochemiczne": BASE_DIR / "cv_folds_physicochemical.csv",
}


data_list = []
for repr_name, file_path in input_files.items():
    if file_path.exists():
        df = pd.read_csv(file_path)
        df["representation"] = repr_name
        data_list.append(df)
    else:
        print(f"Ostrzeżenie: Nie odnaleziono pliku pod ścieżką: {file_path}")

if not data_list:
    raise FileNotFoundError(
        f"Krytyczny błąd: Żaden z wymaganych plików CSV nie został odnaleziony w katalogu bazowym: {BASE_DIR}"
    )

df_all = pd.concat(data_list, ignore_index=True)

df_all["Konfiguracja"] = df_all["representation"] + " + " + df_all["model"]

metrics = ["accuracy", "balanced_accuracy", "f1", "mcc", "roc_auc"]
metric_titles = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "f1": "F1-Score",
    "mcc": "Matthews Correlation Coefficient (MCC)",
    "roc_auc": "ROC-AUC",
}


OUTPUT_DIR = SCRIPT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for metric in metrics:
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # Wykres dla zbioru zbalansowanego (balanced)
    df_bal = df_all[df_all["dataset"] == "balanced"]
    sns.boxplot(
        ax=axes[0],
        data=df_bal,
        x="Konfiguracja",
        y=metric,
        hue="model",
        palette="Set2",
        dodge=False,
    )
    axes[0].set_title(
        "Zbiór zbalansowany (1:1)", fontsize=12, fontweight="bold"
    )
    axes[0].set_ylabel(metric_titles[metric])
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].get_legend().remove()

    # Wykres dla zbioru niezbalansowanego (imbalanced)
    df_imbal = df_all[df_all["dataset"] == "imbalanced"]
    sns.boxplot(
        ax=axes[1],
        data=df_imbal,
        x="Konfiguracja",
        y=metric,
        hue="model",
        palette="Set2",
        dodge=False,
    )
    axes[1].set_title(
        "Zbiór niezbalansowany (1:2)", fontsize=12, fontweight="bold"
    )
    axes[1].set_ylabel(metric_titles[metric])
    axes[1].set_xlabel("Konfiguracja (Reprezentacja + Model)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    axes[1].legend(title="Algorytm", loc="lower left")
    plt.xticks(rotation=45, ha="right")

    plt.suptitle(
        f"Analiza statystyczna rozrzutu metryki {metric_titles[metric]} w walidacji krzyżowej",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"boxplot_{metric}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

print(f"Generowanie wykresów zakończone sukcesem. Pliki zapisano w: {OUTPUT_DIR}")