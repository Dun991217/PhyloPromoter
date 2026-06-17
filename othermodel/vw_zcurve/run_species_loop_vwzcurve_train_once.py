import os
import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

# 直接复用你现有 vw_zcurve_pls.py 里的函数/类
from vw_zcurve_pls import (
    build_dataset,
    VWZCurvePLSClassifier,
    evaluate_binary,
    save_prediction_table,
)


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
    parser.add_argument("--max_w", type=int, default=6)
    parser.add_argument("--n_components", type=int, default=30)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_scale", action="store_true")
    parser.add_argument(
        "--result_root",
        default="other_model/Promoter/vw_zcurve/results/species_loop_train_once",
        help="Final results root",
    )
    args = parser.parse_args()

    os.makedirs(args.result_root, exist_ok=True)

    run_rows = []
    val_rows = []
    test_rows = []

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

        # ======================================================
        # 1) 当前训练物种：只构建一次训练集
        # ======================================================
        print(f"\n[INFO] Building training dataset for {train_species} ...")
        X_full, y_full, full_names, full_seqs, dropped_train = build_dataset(
            train_pos_fa, train_neg_fa, max_w=args.max_w
        )

        train_idx, val_idx = train_test_split(
            np.arange(len(y_full)),
            test_size=args.val_ratio,
            random_state=args.seed,
            stratify=y_full,
        )

        X_train = X_full[train_idx]
        y_train = y_full[train_idx]

        X_val = X_full[val_idx]
        y_val = y_full[val_idx]
        val_names = [full_names[i] for i in val_idx]
        val_seqs = [full_seqs[i] for i in val_idx]

        print(
            f"[INFO] {train_species}: "
            f"full={X_full.shape}, train={X_train.shape}, val={X_val.shape}, dropped={dropped_train}"
        )

        # ======================================================
        # 2) 只训练一次
        # ======================================================
        clf = VWZCurvePLSClassifier(
            n_components=args.n_components,
            scale=(not args.no_scale),
        )

        print(f"[INFO] Training model once for {train_species} ...")
        clf.fit(X_train, y_train)

        # 验证集结果
        val_score = clf.predict_proba_like(X_val)
        val_pred = clf.predict(X_val)
        val_metrics = evaluate_binary(y_val, val_pred, val_score)

        val_metrics_simple = {
            "train_species": train_species,
            "accuracy": val_metrics["accuracy"],
            "balanced_accuracy": val_metrics["balanced_accuracy"],
            "mcc": val_metrics["mcc"],
            "auc": val_metrics["auc"],
            "auprc": val_metrics["auprc"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "sensitivity": val_metrics["sensitivity"],
            "specificity": val_metrics["specificity"],
            "f1": val_metrics["f1"],
            "confusion_matrix": str(val_metrics["confusion_matrix"]),
            "n_train": len(y_train),
            "n_val": len(y_val),
            "max_w": args.max_w,
            "n_components": args.n_components,
            "scale": not args.no_scale,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        }
        pd.DataFrame([val_metrics_simple]).to_csv(
            os.path.join(run_result_dir, "val_metrics.csv"),
            index=False
        )
        save_prediction_table(
            os.path.join(run_result_dir, "val_predictions.csv"),
            val_names,
            val_seqs,
            y_val,
            val_pred,
            val_score,
        )
        val_rows.append(val_metrics_simple)

        # ======================================================
        # 3) 依次测试其他物种，不再重新训练
        # ======================================================
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

            print(f"[INFO] Building test dataset for {train_species} -> {test_species} ...")
            X_test, y_test, test_names, test_seqs, dropped_test = build_dataset(
                test_pos_fa, test_neg_fa, max_w=args.max_w
            )

            y_score = clf.predict_proba_like(X_test)
            y_pred = clf.predict(X_test)
            test_metrics = evaluate_binary(y_test, y_pred, y_score)

            out_sp_dir = os.path.join(each_species_root, test_species)
            os.makedirs(out_sp_dir, exist_ok=True)

            test_metrics_simple = {
                "train_species": train_species,
                "test_species": test_species,
                "accuracy": test_metrics["accuracy"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "mcc": test_metrics["mcc"],
                "auc": test_metrics["auc"],
                "auprc": test_metrics["auprc"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "sensitivity": test_metrics["sensitivity"],
                "specificity": test_metrics["specificity"],
                "f1": test_metrics["f1"],
                "confusion_matrix": str(test_metrics["confusion_matrix"]),
                "n_test": len(y_test),
                "dropped_test": dropped_test,
                "max_w": args.max_w,
                "n_components": args.n_components,
                "scale": not args.no_scale,
            }

            pd.DataFrame([test_metrics_simple]).to_csv(
                os.path.join(out_sp_dir, "test_metrics.csv"),
                index=False
            )
            pd.DataFrame(test_metrics["classification_report"]).transpose().to_csv(
                os.path.join(out_sp_dir, "classification_report.csv")
            )
            save_prediction_table(
                os.path.join(out_sp_dir, "test_predictions.csv"),
                test_names,
                test_seqs,
                y_test,
                y_pred,
                y_score,
            )

            run_rows.append({
                "train_species": train_species,
                "test_species": test_species,
                "result_dir": out_sp_dir,
            })
            test_rows.append(test_metrics_simple)

    # ==========================================================
    # 4) 保存总表
    # ==========================================================
    pd.DataFrame(run_rows).to_csv(
        os.path.join(args.result_root, "run_summary_each_species.csv"),
        index=False
    )

    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(
        os.path.join(args.result_root, "val_metrics_summary.csv"),
        index=False
    )

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(
        os.path.join(args.result_root, "test_metrics_summary.csv"),
        index=False
    )

    if len(test_df) > 0:
        numeric_cols = [
            c for c in [
                "accuracy", "balanced_accuracy", "mcc", "auc", "auprc",
                "precision", "recall", "sensitivity", "specificity", "f1", "n_test"
            ]
            if c in test_df.columns
        ]
        mean_df = (
            test_df.groupby("train_species")[numeric_cols]
            .mean()
            .reset_index()
        )
        mean_df.to_csv(
            os.path.join(args.result_root, "test_metrics_mean_by_train_species.csv"),
            index=False
        )

    print(f"\n[OK] All finished. Results saved to: {args.result_root}")


if __name__ == "__main__":
    main()