import os
from typing import List, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_cohort_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    print(f"Loaded cohort csv: {df.shape}")
    return df


def check_columns_exist(cohort_df: pd.DataFrame, cols: Sequence[str]) -> None:
    for col in cols:
        if col in cohort_df.columns:
            print(f"Found column: {col}")
        else:
            print(f"Missing column: {col}")


def filter_by_test_keys(cohort_df: pd.DataFrame, test_csv_path: str) -> pd.DataFrame:
    test_data = pd.read_csv(test_csv_path)
    test_keys_df = test_data[["PATID", "ENCOUNTERID"]].drop_duplicates()
    merged = cohort_df.merge(test_keys_df, on=["PATID", "ENCOUNTERID"], how="inner")
    print(f"Filtered by test keys: {merged.shape}")
    return merged


def load_and_merge_labels(datafolder: str, site: str, cohort: pd.DataFrame) -> pd.DataFrame:
    site_path = os.path.join(datafolder, site)
    recover_path = os.path.join(site_path, "cohort_with_recover.csv")
    reverse_path = os.path.join(site_path, "cohort_with_reverse.csv")

    recover_df = pd.read_csv(recover_path, low_memory=False)
    reverse_df = pd.read_csv(reverse_path, low_memory=False)

    recover_sub = recover_df[["PATID", "ENCOUNTERID", "AKI_RCV"]]
    reverse_sub = reverse_df[["PATID", "ENCOUNTERID", "AKI_ERVRT"]]

    merged = cohort.merge(recover_sub, on=["PATID", "ENCOUNTERID"], how="left")
    merged = merged.merge(reverse_sub, on=["PATID", "ENCOUNTERID"], how="left")
    print(f"Merged labels rows: {len(merged)}")
    return merged


def drop_sparse_feature_cols(
    cohort: pd.DataFrame,
    zero_ratio_threshold: float = .,
    prefixes: Sequence[str] = ("."),
) -> pd.DataFrame:
    zero_ratio = (cohort == 0).sum() / len(cohort)
    high_zero_cols = zero_ratio[zero_ratio > zero_ratio_threshold].index.tolist()
    drop_cols = [col for col in high_zero_cols if col.startswith(prefixes)]
    if drop_cols:
        cohort = cohort.drop(columns=drop_cols)
    print(f"Dropped sparse cols: {len(drop_cols)}")
    return cohort


def run_xgboost_bootstrap(
    cohort_df: pd.DataFrame,
    label_col: str = ".",
    n_runs: int = .,
    random_state: int = .,
    unit_list: Sequence[int] = (.,),
    split_load_path: str = "./",
    split_seed: int = 1,
    save_dir: str = "./",
    prefix: str = ".",
    use_early_stopping: bool = True,
    n_estimators_es: int = .,
    learning_rate_es: float = .,
    early_stopping_rounds: int = .,
    max_depth: int = .,
    gamma: float = .,
    use_row_col_subsample: bool = True,
    subsample: float = .,
    colsample_bytree: float = .,
    colsample_bynode: float = .,
):
    drop_cols = ["PATID", "ENCOUNTERID", "FLAG", "death90", "AKI_RCV", "AKI_ERVRT"]
    all_rows: List[dict] = []

    def load_split(name: str, unit: int):
        path = os.path.join(split_load_path, f"Finetuning_{name}_splitseed{split_seed}_{unit}.csv")
        return pd.read_csv(path)

    def subset_by_keys(df: pd.DataFrame, keys: pd.DataFrame):
        keyset = set(map(tuple, keys[["PATID", "ENCOUNTERID"]].values))
        return df[df[["PATID", "ENCOUNTERID"]].apply(tuple, axis=1).isin(keyset)].copy()

    for unit in unit_list:
        train_keys = load_split("train", unit)[["PATID", "ENCOUNTERID"]]
        val_keys = load_split("val", unit)[["PATID", "ENCOUNTERID"]]
        test_keys = load_split("test", unit)[["PATID", "ENCOUNTERID"]]

        train_df = subset_by_keys(cohort_df, train_keys)
        val_df = subset_by_keys(cohort_df, val_keys)
        test_df = subset_by_keys(cohort_df, test_keys)

        X_train = train_df.drop(columns=drop_cols, errors="ignore").copy()
        y_train = train_df[label_col].astype(int)
        X_val = val_df.drop(columns=drop_cols, errors="ignore").copy()
        y_val = val_df[label_col].astype(int)
        X_test = test_df.drop(columns=drop_cols, errors="ignore").copy()
        y_test = test_df[label_col].astype(int)

        X_train = pd.get_dummies(X_train, drop_first=True)
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        for run in range(n_runs):
            seed = random_state + run
            params = dict(
                n_estimators=n_estimators_es,
                learning_rate=learning_rate_es,
                max_depth=max_depth,
                gamma=gamma,
                random_state=seed,
                eval_metric="logloss",
                use_label_encoder=False,
            )
            if use_row_col_subsample:
                params.update(
                    dict(
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        colsample_bynode=colsample_bynode,
                    )
                )

            model = xgb.XGBClassifier(**params)
            if use_early_stopping:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            row = {
                "unit": unit,
                "run": run + 1,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan,
                "pr_auc": average_precision_score(y_test, y_prob),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    os.makedirs(save_dir, exist_ok=True)
    for unit, g in df.groupby("unit"):
        mean_row = {"unit": unit}
        sd_row = {"unit": unit}
        metrics = ["accuracy", "precision", "recall", "roc_auc", "pr_auc", "f1"]
        for m in metrics:
            mean_row[m] = g[m].mean()
            sd_row[m] = g[m].std(ddof=1)
        pd.DataFrame([mean_row]).to_csv(os.path.join(save_dir, f"{prefix}_{unit}_mean.csv"), index=False)
        pd.DataFrame([sd_row]).to_csv(os.path.join(save_dir, f"{prefix}_{unit}_sd.csv"), index=False)

    return df
