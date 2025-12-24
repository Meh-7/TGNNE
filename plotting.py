from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_only(
    loss_csv: Path,
    out_path: Path,
) -> None:
    if not loss_csv.exists():
        return

    df = pd.read_csv(loss_csv)
    if df.empty:
        return

    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["train_loss"], label="Train loss", color="black")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_loss_and_MR(
    loss_csv: Path,
    metrics_csv: Path,
    out_path: Path,
) -> None:
    if not loss_csv.exists() or not metrics_csv.exists():
        return

    loss_df = pd.read_csv(loss_csv)
    metrics_df = pd.read_csv(metrics_csv)

    if loss_df.empty or metrics_df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Loss (left axis)
    ax1.plot(
        loss_df["epoch"],
        loss_df["train_loss"],
        color="black",
        label="Train loss",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # MR (right axis)
    ax2 = ax1.twinx()
    ax2.plot(
        metrics_df["epoch"],
        metrics_df["valid_MR"],
        color="tab:red",
        linestyle="--",
        marker="o",
        markersize=3,
        label="MR",
    )
    ax2.set_ylabel("Mean Rank (MR)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Training Loss and Mean Rank (MR)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_loss_and_metrics(
    loss_csv: Path,
    metrics_csv: Path,
    out_path: Path,
) -> None:
    if not loss_csv.exists() or not metrics_csv.exists():
        return

    loss_df = pd.read_csv(loss_csv)
    metrics_df = pd.read_csv(metrics_csv)

    if loss_df.empty or metrics_df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Loss (left axis)
    ax1.plot(
        loss_df["epoch"],
        loss_df["train_loss"],
        color="black",
        label="Train loss",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Metrics (right axis)
    ax2 = ax1.twinx()

    metric_colors = {
        "valid_MR": "tab:red",
        "valid_MRR": "tab:blue",
        "valid_Hits@1": "tab:green",
        "valid_Hits@3": "tab:orange",
        "valid_Hits@10": "tab:purple",
    }

    for metric, color in metric_colors.items():
        if metric in metrics_df.columns:
            ax2.plot(
                metrics_df["epoch"],
                metrics_df[metric],
                label=metric,
                color=color,
                linestyle="--",
                marker="o",
                markersize=3,
            )

    ax2.set_ylabel("Metrics")
    ax2.tick_params(axis="y")

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best",
        fontsize=8,
    )

    plt.title("Training Loss and Validation Metrics")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_loss_and_scaled_metrics(
    loss_csv: Path,
    metrics_csv: Path,
    out_path: Path,
) -> None:
    if not loss_csv.exists() or not metrics_csv.exists():
        return

    loss_df = pd.read_csv(loss_csv)
    metrics_df = pd.read_csv(metrics_csv)

    if loss_df.empty or metrics_df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Loss (left axis)
    ax1.plot(
        loss_df["epoch"],
        loss_df["train_loss"],
        color="black",
        label="Train loss",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # MRR + Hits@k (right axis)
    ax2 = ax1.twinx()

    metric_styles = {
        "valid_MRR": ("tab:blue", "MRR"),
        "valid_Hits@1": ("tab:green", "Hits@1"),
        "valid_Hits@3": ("tab:orange", "Hits@3"),
        "valid_Hits@10": ("tab:purple", "Hits@10"),
    }

    for col, (color, label) in metric_styles.items():
        if col in metrics_df.columns:
            ax2.plot(
                metrics_df["epoch"],
                metrics_df[col],
                label=label,
                color=color,
                linestyle="--",
                marker="o",
                markersize=3,
            )

    ax2.set_ylabel("MRR / Hits@k")
    ax2.set_ylim(0.0, 1.0)  # important: shared meaningful scale

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    plt.title("Training Loss and Ranking Quality Metrics")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
