import json
import os
import pickle
from collections import defaultdict
from typing import Dict, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # fallback when tqdm is not installed
    def tqdm(x, **kwargs):
        return x


def load_parameters(parameters_path: str = None):
    if parameters_path is None:
        parameters_path = os.path.join(
            os.path.dirname(__file__),
            "preprecessing_parameters.jason",
        )
    with open(parameters_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    return (
        params.get("LOAD_MAP", {}),
        params.get("TABLE_CONFIG", {}),
        set(params.get("EXCLUDE_DX_CODES", [])),
    )


def load_tables(datafolder2: str, load_map: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    missing = []
    for name, fname in load_map.items():
        fpath = os.path.join(datafolder2, fname)
        if not os.path.exists(fpath):
            missing.append(f"{name} -> {fpath}")
        else:
            print(f"Found: {name} at {fpath}")

    if missing:
        print("Missing files:")
        for m in missing:
            print(f"   - {m}")
    else:
        print("All files are present.")

    loaded_tables = {}
    for var_name, file_name in load_map.items():
        file_path = os.path.join(datafolder2, file_name)
        df = pd.read_pickle(file_path)
        loaded_tables[var_name] = df
        print(f"Loaded {var_name}, shape = {df.shape}")

    return loaded_tables


def load_medcode_description(datafolder2: str) -> pd.DataFrame:
    file_path = os.path.join(datafolder2, "./")
    return pd.read_csv(file_path)


def attach_module_id(loaded_tables: Dict[str, pd.DataFrame], medcode_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    medcode_df["code_no_prefix"] = medcode_df["code"].str.replace("^LAB::", "", regex=True)
    medcode_df["code_no_prefix"] = medcode_df["code"].str.replace("^LABC_", "", regex=True)

    loaded_tables["labcat"] = pd.merge(
        loaded_tables["labcat"],
        medcode_df[["code_no_prefix", "module_id"]],
        left_on="LAB_LOINC",
        right_on="code_no_prefix",
        how="left",
    )
    loaded_tables["labcat"] = loaded_tables["labcat"].dropna(subset=["module_id"]).reset_index(drop=True)

    loaded_tables["labnum"] = pd.merge(
        loaded_tables["labnum"],
        medcode_df[["code_no_prefix", "module_id"]],
        left_on="LAB_LOINC",
        right_on="code_no_prefix",
        how="left",
    )
    loaded_tables["labnum"] = loaded_tables["labnum"].dropna(subset=["module_id"]).reset_index(drop=True)

    loaded_tables["px"] = pd.merge(
        loaded_tables["px"],
        medcode_df[["code_no_prefix", "module_id"]],
        left_on="PX",
        right_on="code_no_prefix",
        how="left",
    )
    loaded_tables["px"] = loaded_tables["px"].dropna(subset=["module_id"]).reset_index(drop=True)

    loaded_tables["dx"] = pd.merge(
        loaded_tables["dx"],
        medcode_df[["code_no_prefix", "module_id"]],
        left_on="DX",
        right_on="code_no_prefix",
        how="left",
    )
    loaded_tables["dx"] = loaded_tables["dx"].dropna(subset=["module_id"]).reset_index(drop=True)

    return loaded_tables


def sample_loaded_tables_by_cohort(
    cohort_path: str,
    loaded_tables: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    print(f"Loading cohort from {cohort_path}")
    cohort = pd.read_csv(cohort_path)

    cohort_keys = cohort[["PATID", "ENCOUNTERID"]].drop_duplicates()
    sampled_keys = cohort_keys

    sampled_keys["key"] = sampled_keys["PATID"].astype(str) + "_" + sampled_keys["ENCOUNTERID"].astype(str)
    sampled_set = set(zip(sampled_keys["PATID"], sampled_keys["ENCOUNTERID"]))

    loaded_tables_sampled = {}
    for name, df in loaded_tables.items():
        if not {"PATID", "ENCOUNTERID"}.issubset(df.columns):
            print(f"Skipping {name} - missing PATID or ENCOUNTERID columns")
            continue

        original_shape = df.shape
        df_filtered = df[df[["PATID", "ENCOUNTERID"]].apply(tuple, axis=1).isin(sampled_set)].copy()
        loaded_tables_sampled[name] = df_filtered
        print(f"{name}: {original_shape} -> {df_filtered.shape}")

    return sampled_keys, loaded_tables_sampled


def filter_dx_latest_per_code(dx_df: pd.DataFrame, exclude_dx_codes) -> pd.DataFrame:
    dx_df = dx_df[~dx_df["DX"].isin(set(exclude_dx_codes))]
    dx_df = (
        dx_df.sort_values("DX_DATE")
        .groupby(["PATID", "ENCOUNTERID", "DX"], as_index=False)
        .tail(1)
    )
    return dx_df


def convert_to_retain_format(
    loaded_tables: Dict[str, pd.DataFrame],
    table_config: Dict[str, Dict[str, str]],
    demo_df: pd.DataFrame,
    maxlen: int = .,
):
    """
    Build RETAIN input with:
    - date ascending across visits
    - tie-breaker within same date: dx > vitals > px > labs > amed
    - always keep demo & dx; fill remaining budget (maxlen) with most-recent vitals/px/labs/amed
    """

    PRIORITY = {
        "dx": 0,
        "vitals": 1,
        "px": 2,
        "labs": 3,
        "amed": 4,
        "demo": -1,
    }

    def norm_kind(kind_or_tabname: str) -> str:
        s = str(kind_or_tabname).lower()
        if "dx" in s or "diag" in s:
            return "dx"
        if "vital" in s:
            return "vitals"
        if s in ("px",) or "proc" in s or "procedure" in s:
            return "px"
        if "lab" in s:
            return "labs"
        if "med" in s or "amed" in s or "drug" in s or "pharm" in s:
            return "amed"
        if "demo" in s:
            return "demo"
        return s

    patient_events = defaultdict(list)
    global_seq = 0

    for tab_name, cfg in table_config.items():
        df = loaded_tables[tab_name].copy()
        if df.empty:
            continue

        date_col = cfg["date_col"]
        code_col = cfg["code_col"]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col, code_col, "PATID", "ENCOUNTERID"])

        kind = norm_kind(cfg.get("kind", tab_name))

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {tab_name}"):
            pid, enc_id = row["PATID"], row["ENCOUNTERID"]
            date, code = row[date_col], row[code_col]
            patient_events[(pid, enc_id)].append(
                {"code": code, "date": date, "kind": kind, "seq": global_seq}
            )
            global_seq += 1

    if not demo_df.empty:
        demo_df = demo_df[["PATID", "ENCOUNTERID", "SEX", "RACE", "Age_label"]].copy()
        for _, row in tqdm(demo_df.iterrows(), total=len(demo_df), desc="Adding demo info"):
            pid, enc_id = row["PATID"], row["ENCOUNTERID"]
            key = (pid, enc_id)
            demo_tokens = [f"SEX_{row['SEX']}", f"RACE_{row['RACE']}", f"AGE_{row['Age_label']}"]
            for tok in demo_tokens:
                patient_events[key].append(
                    {
                        "code": tok,
                        "date": pd.Timestamp("1900-01-01"),
                        "kind": "demo",
                        "seq": global_seq,
                    }
                )
                global_seq += 1

    retain_input = []

    for (pid, enc_id), events in tqdm(patient_events.items(), desc="Consolidating patient sequences"):
        if not events:
            continue

        required = [e for e in events if e["kind"] in ("demo", "dx")]
        rest = [e for e in events if e["kind"] in ("vitals", "px", "labs", "amed")]

        budget_for_rest = max(0, maxlen - len(required))
        rest_sorted_by_recency = sorted(rest, key=lambda e: (e["date"], e["seq"]), reverse=True)
        selected_rest = rest_sorted_by_recency[:budget_for_rest]

        selected = required + selected_rest
        selected_sorted = sorted(
            selected, key=lambda e: (e["date"], PRIORITY.get(e["kind"], 999), e["seq"])
        )

        timeline = []
        cur_date = None
        day_tokens = []
        for e in selected_sorted:
            if cur_date is None or e["date"] != cur_date:
                if day_tokens:
                    timeline.append(day_tokens)
                cur_date = e["date"]
                day_tokens = []
            day_tokens.append(e["code"])
        if day_tokens:
            timeline.append(day_tokens)

        retain_input.append({"pid": str(pid), "enc_id": str(enc_id), "visits": timeline})

    return retain_input


def load_cohort_addlabel(datafolder: str, site: str) -> pd.DataFrame:
    save_path_base = os.path.join(datafolder, site, ".")
    pkl_path = save_path_base + ".pkl"

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File not found: {pkl_path}")

    cohort_addlabel = pd.read_pickle(pkl_path)
    print(f"Loaded: {pkl_path} (rows: {len(cohort_addlabel)})")
    return cohort_addlabel
