import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_single_heatmap(df, metric, species_order, out_path, title=None, cmap="viridis"):
    pivot = df.pivot(index="train_species", columns="test_species", values=metric)

    # 按固定顺序重排
    pivot = pivot.reindex(index=species_order, columns=species_order)

    data = pivot.values

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_xticks(np.arange(len(species_order)))
    ax.set_yticks(np.arange(len(species_order)))
    ax.set_xticklabels(species_order, rotation=45, ha="right")
    ax.set_yticklabels(species_order)

    ax.set_xlabel("Test species")
    ax.set_ylabel("Train species")

    if title is None:
        title = f"Cross-species transfer heatmap ({metric})"
    ax.set_title(title)

    # 标注数值
    for i in range(len(species_order)):
        for j in range(len(species_order)):
            val = data[i, j]
            if np.isnan(val):
                text = "-"
            else:
                text = f"{val:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if not np.isnan(val) and val < 0.5 else "black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] saved: {out_path}")


def plot_combined_heatmaps(df, metrics, species_order, out_path, cmap="viridis"):
    n = len(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        pivot = df.pivot(index="train_species", columns="test_species", values=metric)
        pivot = pivot.reindex(index=species_order, columns=species_order)
        data = pivot.values

        im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap)

        ax.set_xticks(np.arange(len(species_order)))
        ax.set_yticks(np.arange(len(species_order)))
        ax.set_xticklabels(species_order, rotation=45, ha="right")
        ax.set_yticklabels(species_order)

        ax.set_xlabel("Test species")
        ax.set_ylabel("Train species")
        ax.set_title(metric)

        for i in range(len(species_order)):
            for j in range(len(species_order)):
                val = data[i, j]
                if np.isnan(val):
                    text = "-"
                else:
                    text = f"{val:.3f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=7,
                        color="white" if not np.isnan(val) and val < 0.5 else "black")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(metric)

    # 如果不足4张子图，空出来的隐藏
    for k in range(len(metrics), len(axes)):
        axes[k].axis("off")

    plt.suptitle("Cross-species transfer performance heatmaps", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="PhyloPromoter/results/vw_zcurve_species_loop/test_each_species_metrics.csv",
        help="Input CSV from species-loop evaluation"
    )
    parser.add_argument(
        "--out_dir",
        default="PhyloPromoter/results/vw_zcurve_species_loop/heatmaps",
        help="Output directory"
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=[
            "C_elegans",
            "Chicken",
            "Dog",
            "Fruit_fly",
            "Human",
            "Mouse",
            "Zebrafish",
        ],
        help="Species order for heatmap"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy", "mcc", "auc", "auprc"],
        help="Metrics to plot"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    print("[INFO] loaded:", args.input_csv)
    print(df.head())

    # 单独画每个热图
    for metric in args.metrics:
        out_path = os.path.join(args.out_dir, f"heatmap_{metric}.png")
        plot_single_heatmap(
            df=df,
            metric=metric,
            species_order=args.species,
            out_path=out_path,
            title=f"Cross-species transfer ({metric})"
        )

    # 组合图
    combined_out = os.path.join(args.out_dir, "heatmap_combined.png")
    plot_combined_heatmaps(
        df=df,
        metrics=args.metrics,
        species_order=args.species,
        out_path=combined_out
    )

    print(f"[OK] all heatmaps saved to: {args.out_dir}")


if __name__ == "__main__":
    main()