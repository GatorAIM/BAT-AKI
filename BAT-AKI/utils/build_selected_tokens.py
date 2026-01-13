import os
import json
import pickle
import re
import pandas as pd


def build_selected_tokens_df(file_path, static_dir):
    selectT_path = os.path.join(file_path, "AB_selectTopp.pickle")
    with open(selectT_path, "rb") as f:
        selected_tokens_df = pickle.load(f)

    focus_path = os.path.join(static_dir, "focus_tokens.json")
    with open(focus_path, "r") as f:
        focus_tokens = set(json.load(f)["focus_tokens"])

    qmap = {
        "Q1": "Q3", "Q2": "Q4", "Q3": "Q5",
        "Q4": "Q6", "Q5": "Q1", "Q6": "Q2"
    }

    def remap_q(token):
        if not isinstance(token, str):
            return token
        m = re.search(r"(Q[1-6])$", token)
        if m:
            q = m.group(1)
            return token[:-len(q)] + qmap[q]
        return token

    selected_tokens_df["opp_final"] = selected_tokens_df.apply(
        lambda r: remap_q(r["origin"])
        if re.search(r"(Q[1-6])$", str(r["origin"])) else r["opp_final"],
        axis=1
    )

    rows = []
    for _, row in selected_tokens_df.iterrows():
        origin = str(row["origin"])
        m = re.search(r"Q([1-6])$", origin)
        if m:
            base = origin[:-2]
            existing = {
                re.search(r"Q([1-6])$", str(o)).group(1)
                for o in selected_tokens_df["origin"]
                if re.search(r"Q([1-6])$", str(o)) and str(o).startswith(base)
            }
            for q in range(1, 7):
                if str(q) not in existing:
                    rows.append({
                        "origin": f"{base}Q{q}",
                        "opp_final": f"{base}{qmap[f'Q{q}']}"
                    })

    if rows:
        selected_tokens_df = pd.concat(
            [selected_tokens_df, pd.DataFrame(rows)],
            ignore_index=True
        )

    mask_focus = (
        selected_tokens_df["origin"].isin(focus_tokens) |
        selected_tokens_df["origin"].str.contains("SYSTOLIC", case=False, na=False) |
        selected_tokens_df["origin"].str.contains("DIASTOLIC", case=False, na=False)
    )

    selected_tokens_df = selected_tokens_df.loc[~mask_focus]

    new_pairs = [
        ("SYSTOLIC_0", "SYSTOLIC_2"),
        ("SYSTOLIC_1", "SYSTOLIC_3"),
        ("SYSTOLIC_2", "SYSTOLIC_4"),
        ("SYSTOLIC_3", "SYSTOLIC_5"),
        ("SYSTOLIC_4", "SYSTOLIC_0"),
        ("SYSTOLIC_5", "SYSTOLIC_1"),
        ("DIASTOLIC_0", "DIASTOLIC_2"),
        ("DIASTOLIC_1", "DIASTOLIC_3"),
        ("DIASTOLIC_2", "DIASTOLIC_4"),
        ("DIASTOLIC_3", "DIASTOLIC_5"),
        ("DIASTOLIC_4", "DIASTOLIC_0"),
        ("DIASTOLIC_5", "DIASTOLIC_1"),
    ]

    selected_tokens_df = pd.concat(
        [
            selected_tokens_df,
            pd.DataFrame(new_pairs, columns=["origin", "opp_final"])
        ],
        ignore_index=True
    )

    return selected_tokens_df.drop_duplicates()
