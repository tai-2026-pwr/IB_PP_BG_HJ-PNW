import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")


def generate_report(df: pd.DataFrame, file_prefix: str) -> None:
    df = df.copy()
    df["Configuration"] = df["model"] + " + " + df["representation"]
    config_order: np.ndarray = df["Configuration"].unique()

    # Identify datasets present
    available_datasets: list[str] = df["dataset"].unique().tolist()
    has_balanced: bool = "balanced" in available_datasets
    has_imbalanced: bool = "imbalanced" in available_datasets

    num_subplots: int = sum([has_balanced, has_imbalanced])
    if num_subplots == 0:
        raise ValueError(
            "DataFrame must contain 'balanced' or 'imbalanced' in the 'dataset' column."
        )

    # 2. Configure Matplotlib/Seaborn Settings
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "figure.titlesize": 24,
        }
    )

    # Professional scientific palette
    unique_models: list[str] = df["model"].unique().tolist()
    palette: dict[str, tuple] = dict(
        zip(unique_models, sns.color_palette("colorblind", n_colors=len(unique_models)))
    )

    # 3. Create Plots
    fig, axes = plt.subplots(
        nrows=num_subplots,
        ncols=1,
        figsize=(24, 10 * num_subplots),
        squeeze=False,
        sharex=True,
    )

    # Applied bold weight to main title
    fig.suptitle(
        "Analysis of deviation of Matthews Correlation Coefficient (MCC) in cross validation",
        y=0.98,
        fontweight="bold",
    )

    ax_idx: int = 0

    # Added is_bottom_plot boolean check
    def _plot_dataset(dataset_name: str, subtitle: str, ax: plt.Axes, is_bottom_plot: bool) -> None:
        subset = df[df["dataset"] == dataset_name]
        sns.boxplot(
            data=subset,
            x="Configuration",
            y="mcc",
            hue="model",
            order=config_order,
            palette=palette,
            dodge=False,  # Keeps boxes centered on x-ticks without spaces for missing hues
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
            ax=ax,
        )

        # Subtitle with bold font
        ax.set_title(subtitle, pad=10, fontweight="bold", fontsize=20)
        ax.set_ylabel("Matthews Correlation Coefficient (MCC)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.7, color="gray")

        # Handle shared X-axis labels to prevent overlap
        if is_bottom_plot:
            ax.set_xlabel("Configuration (Model + Representation)")
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

        # Legend cleanup (avoid duplicate labels from dodge=False)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            title="Model",
            loc="lower left",
        )

    # --- EXECUTE PLOTTING ---
    if has_balanced:
        # It's only the bottom plot if there is no imbalanced dataset
        is_bottom = not has_imbalanced
        _plot_dataset(
            "balanced", "Balanced dataset (1:1)", axes[ax_idx, 0], is_bottom_plot=is_bottom
        )
        ax_idx += 1

    if has_imbalanced:
        # If imbalanced exists, it's always placed at the bottom
        _plot_dataset(
            "imbalanced", "Imbalanced dataset (1:2)", axes[ax_idx, 0], is_bottom_plot=True
        )

    plt.tight_layout()
    plt.savefig(f"{file_prefix}.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 4. Generate Typst Summary Table
    def q1(x: pd.Series) -> float:
        return x.quantile(0.25)

    def q3(x: pd.Series) -> float:
        return x.quantile(0.75)

    summary: pd.DataFrame = (
        df.groupby(["dataset", "Configuration"], sort=False)["mcc"]
        .agg(Mean="mean", Median="median", Q1=q1, Q3=q3)
        .reset_index()
    )

    typst_content: str = ""

    for dataset in ["balanced", "imbalanced"]:
        if dataset in available_datasets:
            dataset_title: str = (
                "Balanced dataset (1:1)" if dataset == "balanced" else "Imbalanced dataset (1:2)"
            )
            dataset_summary: pd.DataFrame = summary[summary["dataset"] == dataset]

            # Enclose each table cleanly within its own #figure block
            typst_content += "#figure(\n"
            typst_content += f"  caption: [{dataset_title}],\n"
            typst_content += "  table(\n"
            typst_content += "    columns: (auto, auto, auto, auto, auto),\n"
            typst_content += "    align: (left, right, right, right, right),\n"
            typst_content += "    fill: (x, y) => if y == 0 { luma(230) },\n"
            typst_content += "    [*Configuration*], [*Mean*], [*Median*], [*Q1*], [*Q3*],\n"

            for _, row in dataset_summary.iterrows():
                typst_content += f"    [{row['Configuration']}], [{row['Mean']:.4f}], [{row['Median']:.4f}], [{row['Q1']:.4f}], [{row['Q3']:.4f}],\n"

            typst_content += "  ),\n"
            typst_content += "  kind: table\n"
            typst_content += ")\n\n"

    with open(f"{file_prefix}.txt", "w", encoding="utf-8") as f:
        f.write(typst_content)


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("input_csv_file.csv")
    generate_report(df, "output_files_name")
