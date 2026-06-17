import os
import subprocess
import argparse
import pandas as pd
from tqdm.auto import tqdm


def run_one_job(vw_script, train_pos, train_neg, test_pos, test_neg, n_components, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python", vw_script,
        "--train_pos", train_pos,
        "--train_neg", train_neg,
        "--test_pos", test_pos,
        "--test_neg", test_neg,
        "--n_components", str(n_components),
        "--out_dir", out_dir,
    ]

    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def safe_read_metrics_csv(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None
        return df.iloc[0].to_dict()
    except Exception as e:
        return {"read_error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default="PhyloPromoter/data/promoter_dataset_final",
        help="Species dataset root",
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
        help="Species list",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=30,
        help="PLS n_components passed to vw_zcurve_pls.py",
    )
    parser.add_argument(
        "--vw_script",
        default="other_model/Promoter/vw_zcurve/vw_zcurve_pls.py",
        help="Path to existing vw_zcurve_pls.py",
    )
    parser.add_argument(
        "--result_root",
        default="other_model/Promoter/vw_zcurve/results/species_loop",
        help="Final results root",
    )
    args = parser.parse_args()

    os.makedirs(args.result_root, exist_ok=True)

    run_rows = []
    metrics_rows = []

    outer_bar = tqdm(args.species, desc="Train species loop", leave=True)

    for train_species in outer_bar:
        outer_bar.set_postfix(current_train=train_species)

        train_species_dir = os.path.join(args.data_root, train_species)
        train_pos_fa = os.path.join(train_species_dir, "positive.fa")
        train_neg_fa = os.path.join(train_species_dir, "negative.fa")

        if not os.path.exists(train_pos_fa) or not os.path.exists(train_neg_fa):
            print(f"[WARNING] Missing fasta for {train_species}, skip.")
            continue

        run_result_dir = os.path.join(args.result_root, train_species)
        os.makedirs(run_result_dir, exist_ok=True)

        # 每个测试物种单独测试
        each_species_root = os.path.join(run_result_dir, "test_each_species")
        os.makedirs(each_species_root, exist_ok=True)

        inner_species = [sp for sp in args.species if sp != train_species]
        inner_bar = tqdm(inner_species, desc=f"{train_species}: test species", leave=False)

        for test_species in inner_bar:
            inner_bar.set_postfix(test_species=test_species)

            test_species_dir = os.path.join(args.data_root, test_species)
            test_pos_fa = os.path.join(test_species_dir, "positive.fa")
            test_neg_fa = os.path.join(test_species_dir, "negative.fa")

            if not os.path.exists(test_pos_fa) or not os.path.exists(test_neg_fa):
                print(f"[WARNING] Missing fasta for test species {test_species}, skip.")
                continue

            out_sp_dir = os.path.join(each_species_root, test_species)

            run_one_job(
                vw_script=args.vw_script,
                train_pos=train_pos_fa,
                train_neg=train_neg_fa,
                test_pos=test_pos_fa,
                test_neg=test_neg_fa,
                n_components=args.n_components,
                out_dir=out_sp_dir,
            )

            run_rows.append({
                "train_species": train_species,
                "test_species": test_species,
                "n_components": args.n_components,
                "result_dir": out_sp_dir,
            })

            # 读取当前轮 test_metrics.csv，直接汇总
            test_metrics_path = os.path.join(out_sp_dir, "test_metrics.csv")
            test_metrics = safe_read_metrics_csv(test_metrics_path)

            if test_metrics is not None:
                row = {
                    "train_species": train_species,
                    "test_species": test_species,
                    "n_components": args.n_components,
                    **test_metrics,
                }
                metrics_rows.append(row)

    # 保存运行路径汇总
    run_summary_csv = os.path.join(args.result_root, "run_summary_each_species.csv")
    pd.DataFrame(run_rows).to_csv(run_summary_csv, index=False)

    # 保存逐物种测试指标总表
    metrics_summary_csv = os.path.join(args.result_root, "test_metrics_summary.csv")
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_summary_csv, index=False)

    # 再保存一个按训练物种聚合的平均结果
    if len(metrics_df) > 0:
        numeric_cols = [
            c for c in ["accuracy", "mcc", "auc", "auprc", "n_test"]
            if c in metrics_df.columns
        ]
        mean_df = (
            metrics_df.groupby("train_species")[numeric_cols]
            .mean()
            .reset_index()
        )
        mean_csv = os.path.join(args.result_root, "test_metrics_mean_by_train_species.csv")
        mean_df.to_csv(mean_csv, index=False)
        print(f"[OK] Mean summary saved to: {mean_csv}")

    print(f"\n[OK] All finished.")
    print(f"[OK] Run summary saved to: {run_summary_csv}")
    print(f"[OK] Test metrics summary saved to: {metrics_summary_csv}")


if __name__ == "__main__":
    main()