import os
import pickle
import random
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import RETAIN
from pyhealth.trainer import Trainer
from torch.utils.data import DataLoader

try:
    from pyhealth.tokenizer import Tokenizer
except Exception:  # fallback if tokenizer is unavailable
    Tokenizer = None


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_retain_ready_data(data_dir: str, filename: str = "."):
    path = os.path.join(data_dir, filename)
    data = load_pickle(path)
    print(f"Loaded retain_ready_data: {len(data)}")
    return data


def load_sampled_keys_df(data_dir: str, filename: str = ".") -> pd.DataFrame:
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


def load_token_dicts(file_path: str, token2id_name: str = ".", id2token_name: str = "."):
    token2id_path = os.path.join(file_path, token2id_name)
    id2token_path = os.path.join(file_path, id2token_name)

    token2id = load_pickle(token2id_path)
    id2token = load_pickle(id2token_path)

    print(f"token2id size: {len(token2id)}")
    return token2id, id2token


def build_tokenizer_from_id2token(id2token: Dict[int, str]):
    if Tokenizer is None:
        raise RuntimeError("Tokenizer is not available")
    tokens = [id2token[i] for i in range(len(id2token))]
    return Tokenizer(tokens=tokens)


def safe_convert_tokens_to_indices(tokenizer, tokens, unk_token: str = "[UNK]"):
    unk_id = tokenizer.convert_tokens_to_indices([unk_token])[0]
    indices = []
    for tok in tokens:
        try:
            idx = tokenizer.convert_tokens_to_indices([tok])[0]
        except ValueError:
            idx = unk_id
        indices.append(idx)
    return indices


def flatten_retain_ready_data(
    retain_ready_data,
    sampled_keys_df: pd.DataFrame,
    label_col: str = "FLAG",
    tokenizer=None,
):
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
        if tokenizer is not None:
            flattened_codes = safe_convert_tokens_to_indices(tokenizer, flattened_codes)

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


def process_visit_ids(samples) -> pd.DataFrame:
    def parse_visit_id(visit_id: str):
        parts = str(visit_id).split("_")
        enc_str = parts[-1]
        pid_str = "_".join(parts[:-1]) if len(parts) > 1 else ""
        try:
            enc_int = int(enc_str)
        except Exception:
            enc_int = None
        try:
            pid_int = int(pid_str) if pid_str != "" else None
        except Exception:
            pid_int = None
        return pid_int, enc_int

    df = pd.DataFrame([{"idx": i, "visit_id": s["visit_id"]} for i, s in enumerate(samples)])
    df[["pid_int", "enc_int"]] = df["visit_id"].apply(lambda v: pd.Series(parse_visit_id(v)))
    bad_rows = df["enc_int"].isna().sum()
    if bad_rows > 0:
        print(f"Invalid enc_int rows: {bad_rows}")
    df = df.dropna(subset=["enc_int"]).copy()
    df["enc_int"] = df["enc_int"].astype(int)
    return df


def build_split_ids_from_csv(
    df_rs: pd.DataFrame,
    unit: int,
    split_load_path: str,
    split_seed: int,
    require_pid_match: bool = False,
) -> Tuple[set, set, set]:
    def load_split(name):
        path = os.path.join(split_load_path, f"Finetuning_{name}_splitseed{split_seed}_{unit}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")
        return pd.read_csv(path)

    train_keys = load_split("train")[["PATID", "ENCOUNTERID"]].dropna()
    val_keys = load_split("val")[["PATID", "ENCOUNTERID"]].dropna()
    test_keys = load_split("test")[["PATID", "ENCOUNTERID"]].dropna()

    for df in (train_keys, val_keys, test_keys):
        df["ENCOUNTERID"] = df["ENCOUNTERID"].astype(int)
        try:
            df["PATID"] = df["PATID"].astype(int)
        except Exception:
            pass

    if not require_pid_match:
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
        for df in (train_keys, val_keys, test_keys):
            df["_key"] = df.apply(lambda r: (r.get("PATID", None), int(r["ENCOUNTERID"])), axis=1)
        df_rs["_key"] = df_rs.apply(lambda r: (r.get("pid_int", None), int(r["enc_int"])), axis=1)
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


class RETAINWithWeight(RETAIN):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def get_loss_function(self):
        if self.pos_weight is not None:
            return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        return nn.BCEWithLogitsLoss()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def only_my_print():
    import builtins
    import logging
    import tqdm as _tqdm

    builtins._orig_print = builtins.print
    orig_level = logging.getLogger().level
    orig_tqdm = _tqdm.tqdm
    try:
        builtins.print = lambda *a, **k: None
        logging.getLogger().setLevel(logging.CRITICAL)
        _tqdm.tqdm = lambda *a, **k: a[0] if a else []
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            yield
    finally:
        builtins.print = builtins._orig_print
        logging.getLogger().setLevel(orig_level)
        _tqdm.tqdm = orig_tqdm


def mean_sd(series: pd.Series) -> pd.Series:
    mean = series.mean()
    sd = series.std(ddof=1)
    return pd.Series({"mean": mean, "sd": sd})


def train_retain_earlystop(
    retain_samples: List[Dict],
    df_rs: pd.DataFrame,
    unit_list: Sequence[int],
    split_load_path: str,
    split_seed: int,
    base_seed: int = .,
    reruns: int = .,
    batch_size: int = .,
    embedding_dim: int = .,
    dropout: float = .,
    max_epochs: int = .,
    patience: int = .,
    lr: float = .,
    save_dir: str = "./",
    ckpt_name_fmt: str = "retain_best_seed{seed}.pt",
    require_pid_match: bool = False,
    device: Optional[str] = None,
) -> pd.DataFrame:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    whole_dataset = SampleEHRDataset(samples=retain_samples, dataset_name="retain_whole")
    all_rows = []

    for unit in unit_list:
        train_ids, val_ids, test_ids = build_split_ids_from_csv(
            df_rs=df_rs,
            unit=unit,
            split_load_path=split_load_path,
            split_seed=split_seed,
            require_pid_match=require_pid_match,
        )

        train_samples = [retain_samples[i] for i in sorted(train_ids)]
        val_samples = [retain_samples[i] for i in sorted(val_ids)]
        test_samples = [retain_samples[i] for i in sorted(test_ids)]

        train_ds = SampleEHRDataset(dataset_name="retain_flattened_admission", samples=train_samples)
        val_ds = SampleEHRDataset(dataset_name="retain_flattened_admission", samples=val_samples)
        test_ds = SampleEHRDataset(dataset_name="retain_flattened_admission", samples=test_samples)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        for rerun in range(reruns):
            seed = base_seed + rerun
            set_seed(seed)

            model = RETAINWithWeight(
                dataset=whole_dataset,
                feature_keys=["code"],
                label_key="label",
                mode="binary",
                embedding_dim=embedding_dim,
                pos_weight=None,
                dropout=dropout,
            ).to(device)

            with only_my_print():
                trainer = Trainer(
                    model=model,
                    metrics=["accuracy", "precision", "recall", "roc_auc", "pr_auc"],
                    device=device,
                    output_path=None,
                )

            best_val_loss = float("inf")
            stale = 0
            os.makedirs(save_dir, exist_ok=True)
            best_ckpt = os.path.join(save_dir, ckpt_name_fmt.format(unit=unit, seed=seed))

            for epoch in range(max_epochs):
                opt_kwargs = {"lr": lr} if epoch == 0 else None
                with only_my_print():
                    trainer.train(
                        train_dataloader=train_loader,
                        val_dataloader=val_loader,
                        test_dataloader=None,
                        epochs=1,
                        optimizer_params=opt_kwargs,
                    )

                with only_my_print():
                    val_metrics = trainer.evaluate(val_loader)

                val_loss = float(val_metrics.get("loss", float("inf")))
                if val_loss < best_val_loss - 1e-3:
                    best_val_loss = val_loss
                    stale = 0
                    torch.save(trainer.model.state_dict(), best_ckpt)
                else:
                    stale += 1
                    if stale >= patience:
                        break

            if os.path.exists(best_ckpt):
                trainer.model.load_state_dict(torch.load(best_ckpt, map_location=trainer.device))

            with only_my_print():
                test_metrics = trainer.evaluate(test_loader)

            prec = float(test_metrics.get("precision", float("nan")))
            rec = float(test_metrics.get("recall", float("nan")))
            f1 = (2 * prec * rec) / (prec + rec + 1e-12)

            row = {
                "unit": unit,
                "rerun": rerun + 1,
                "seed": seed,
                "accuracy": float(test_metrics.get("accuracy", float("nan"))),
                "precision": prec,
                "recall": rec,
                "roc_auc": float(test_metrics.get("roc_auc", float("nan"))),
                "pr_auc": float(test_metrics.get("pr_auc", float("nan"))),
                "f1": f1,
            }
            all_rows.append(row)
            print(f"unit={unit} rerun={rerun+1} seed={seed}")

    return pd.DataFrame(all_rows)
