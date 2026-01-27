import os
import pickle
import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import Transformer
from pyhealth.trainer import Trainer
from torch.utils.data import DataLoader


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_retain_ready_data(data_dir: str, filename: str = "./"):
    path = os.path.join(data_dir, filename)
    data = load_pickle(path)
    print(f"Loaded retain_ready_data: {len(data)}")
    return data


def load_sampled_keys_df(data_dir: str, filename: str = "./") -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    df = pd.read_pickle(path)
    print(f"Loaded sampled_keys_df: shape={df.shape}")
    return df


def filter_retain_ready_data_by_keys(retain_ready_data, sampled_keys_df: pd.DataFrame):
    valid_keys = set(
        (str(row["PATID"]).strip(), str(row["ENCOUNTERID"]).strip())
        for _, row in sampled_keys_df.iterrows()
    )

    filtered_data = []
    removed_count = 0
    for sample in retain_ready_data:
        pid = str(sample["pid"]).strip()
        enc_id = str(sample["enc_id"]).strip()
        if (pid, enc_id) in valid_keys:
            filtered_data.append(sample)
        else:
            removed_count += 1

    print(f"Retained {len(filtered_data)} samples; removed {removed_count}.")
    return filtered_data


def flatten_retain_ready_data(
    retain_ready_data,
    sampled_keys_df: pd.DataFrame,
    label_col: str = "FLAG",
    label_filter: bool = False,
    flag_col: str = "FLAG",
    keep_flag_value: int = 1,
):
    if label_filter:
        sampled_keys_df = sampled_keys_df[sampled_keys_df[flag_col] == keep_flag_value].copy()
        print(f"Filtered by {flag_col}={keep_flag_value}, rows={len(sampled_keys_df)}")

    label_dict = {
        (str(row["PATID"]).strip(), str(row["ENCOUNTERID"]).strip()): row[label_col]
        for _, row in sampled_keys_df.iterrows()
    }

    retain_samples = []
    for sample in retain_ready_data:
        pid = str(sample["pid"]).strip()
        vid = str(sample["enc_id"]).strip()
        key = (pid, vid)
        if key not in label_dict:
            continue

        visits = sample["visits"]
        flattened_codes = [token for visit in visits for token in visit]

        retain_samples.append(
            {
                "patient_id": f"{pid}_{vid}",
                "visit_id": f"{pid}_{vid}",
                "timestamp": 0,
                "code": flattened_codes,
                "label": int(label_dict[key]),
            }
        )

    print(f"Created {len(retain_samples)} samples")
    return retain_samples


def process_visit_ids(samples, parse_mode: str = "int") -> pd.DataFrame:
    def parse_visit_id(visit_id: str):
        parts = str(visit_id).split("_")
        enc_str = parts[-1] if len(parts) > 0 else None
        pid_str = "_".join(parts[:-1]) if len(parts) > 1 else None
        if parse_mode == "int":
            try:
                enc_val = int(enc_str)
            except Exception:
                enc_val = None
            try:
                pid_val = int(pid_str) if pid_str not in (None, "") else None
            except Exception:
                pid_val = None
            return pid_val, enc_val
        return pid_str, enc_str

    df = pd.DataFrame(
        [{"idx": i, "visit_id": s["visit_id"]} for i, s in enumerate(samples)]
    )
    if parse_mode == "int":
        df[["pid_int", "enc_int"]] = df["visit_id"].apply(lambda v: pd.Series(parse_visit_id(v)))
        bad_rows = df["enc_int"].isna().sum()
        if bad_rows > 0:
            print(f"Invalid enc_int rows: {bad_rows}")
        df = df.dropna(subset=["enc_int"]).copy()
        df["enc_int"] = df["enc_int"].astype(int)
    else:
        df[["pid_str", "enc_str"]] = df["visit_id"].apply(lambda v: pd.Series(parse_visit_id(v)))
    return df


def build_split_ids_from_csv(
    df_rs: pd.DataFrame,
    unit: int,
    split_load_path: str,
    require_pid_match: bool = False,
    parse_mode: str = "int",
) -> Tuple[set, set, set]:
    def load_split(name):
        path = os.path.join(split_load_path, f"Finetuning_{name}_splitseed4_{unit}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")
        if parse_mode == "int":
            return pd.read_csv(path)
        return pd.read_csv(path, dtype=str)

    train_keys = load_split("train")[["PATID", "ENCOUNTERID"]].dropna()
    val_keys = load_split("val")[["PATID", "ENCOUNTERID"]].dropna()
    test_keys = load_split("test")[["PATID", "ENCOUNTERID"]].dropna()

    if parse_mode == "int":
        for df in (train_keys, val_keys, test_keys):
            df["ENCOUNTERID"] = df["ENCOUNTERID"].astype(int)
            try:
                df["PATID"] = df["PATID"].astype(int)
            except Exception:
                pass

    if not require_pid_match:
        if parse_mode == "int":
            train_m = df_rs.merge(
                train_keys[["ENCOUNTERID"]].drop_duplicates(),
                left_on="enc_int",
                right_on="ENCOUNTERID",
                how="inner",
            )
            val_m = df_rs.merge(
                val_keys[["ENCOUNTERID"]].drop_duplicates(),
                left_on="enc_int",
                right_on="ENCOUNTERID",
                how="inner",
            )
            test_m = df_rs.merge(
                test_keys[["ENCOUNTERID"]].drop_duplicates(),
                left_on="enc_int",
                right_on="ENCOUNTERID",
                how="inner",
            )
        else:
            train_m = df_rs.merge(
                train_keys[["ENCOUNTERID"]].drop_duplicates(),
                left_on="enc_str",
                right_on="ENCOUNTERID",
                how="inner",
            )
            val_m = df_rs.merge(
                val_keys[["ENCOUNTERID"]].drop_duplicates(),
                left_on="enc_str",
                right_on="ENCOUNTERID",
                how="inner",
            )
            test_m = df_rs.merge(
                test_keys[["ENCOUNTERID"]].drop_duplicates(),
                left_on="enc_str",
                right_on="ENCOUNTERID",
                how="inner",
            )
    else:
        for df in (train_keys, val_keys, test_keys):
            df["_key"] = df.apply(lambda r: (r.get("PATID", None), r["ENCOUNTERID"]), axis=1)
        df_rs["_key"] = df_rs.apply(
            lambda r: (r.get("pid_int" if parse_mode == "int" else "pid_str", None),
                       r.get("enc_int" if parse_mode == "int" else "enc_str", None)),
            axis=1,
        )
        train_m = df_rs.merge(train_keys[["_key"]].drop_duplicates(), on="_key", how="inner")
        val_m = df_rs.merge(val_keys[["_key"]].drop_duplicates(), on="_key", how="inner")
        test_m = df_rs.merge(test_keys[["_key"]].drop_duplicates(), on="_key", how="inner")

    train_ids = set(train_m["idx"].tolist())
    val_ids = set(val_m["idx"].tolist())
    test_ids = set(test_m["idx"].tolist())

    print(f"unit={unit} -> train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    return train_ids, val_ids, test_ids


def collate_fn(batch):
    batch_dict = {k: [d[k] for d in batch] for k in batch[0]}
    batch_dict["code"] = [list(c) for c in batch_dict["code"]]
    return batch_dict


class BalancedTransformer(Transformer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def get_loss_function(self):
        if self.mode == "binary":
            if self.pos_weight is not None:
                return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))
            return nn.BCEWithLogitsLoss()
        if self.mode == "multiclass":
            return nn.CrossEntropyLoss()
        raise NotImplementedError("Only binary or multiclass supported")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_sd(series: pd.Series) -> pd.Series:
    mean = series.mean()
    sd = series.std(ddof=1)
    return pd.Series({"mean": mean, "sd": sd})


def train_transformer_earlystop(
    retain_samples: List[Dict],
    model_samples: List[Dict],
    unit_list: Sequence[int],
    seeds: Sequence[int],
    checkpoint_dir: str,
    checkpoint_name_fmt: str,
    embedding_dim: int,
    num_layers: int,
    dropout: float,
    device: Optional[str] = None,
) -> pd.DataFrame:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    full_dataset = SampleEHRDataset(samples=retain_samples, dataset_name="retain_full")
    full_loader = DataLoader(full_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    whole_dataset = SampleEHRDataset(samples=model_samples, dataset_name="retain_whole")

    all_rows = []
    for unit in unit_list:
        for seed in seeds:
            set_seed(seed)
            model = BalancedTransformer(
                dataset=whole_dataset,
                feature_keys=["code"],
                label_key="label",
                mode="binary",
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                pos_weight=None,
                dropout=dropout,
            ).to(device)

            trainer = Trainer(
                model=model,
                metrics=["accuracy", "precision", "recall", "roc_auc", "pr_auc"],
                device=device,
                output_path=None,
            )

            ckpt = os.path.join(checkpoint_dir, checkpoint_name_fmt.format(unit=unit, seed=seed))
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            trainer.model.load_state_dict(torch.load(ckpt, map_location=device))

            test_metrics = trainer.evaluate(full_loader)
            prec = float(test_metrics.get("precision", float("nan")))
            rec = float(test_metrics.get("recall", float("nan")))
            f1 = (2 * prec * rec) / (prec + rec + 1e-12)

            row = {
                "unit": unit,
                "seed": seed,
                "accuracy": float(test_metrics.get("accuracy", float("nan"))),
                "precision": prec,
                "recall": rec,
                "roc_auc": float(test_metrics.get("roc_auc", float("nan"))),
                "pr_auc": float(test_metrics.get("pr_auc", float("nan"))),
                "f1": f1,
            }
            all_rows.append(row)
            print(f"unit={unit} seed={seed} metrics={ {k: round(v, 4) for k, v in row.items() if k not in ['unit','seed']} }")

    return pd.DataFrame(all_rows)
