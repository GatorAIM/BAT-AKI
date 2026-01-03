import os
import pickle
import numpy as np
import pandas as pd


def load_prompt_embeddings(config, filename="prompt_Result_emb.csv"):
    path = os.path.join(config["file_path"], filename)
    prompt_df = pd.read_csv(path)
    prompt_df["embedding"] = prompt_df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ")
    )
    code2id = {code: idx for idx, code in enumerate(prompt_df["code"])}
    return prompt_df, code2id


def load_train_val_test_df(config, sample=None):
    base_path = config["file_path"]
    suffix = config.get("suffix", "")

    train_path = os.path.join(base_path, f"transformer_input_KUMC_train_{suffix}.pkl")
    val_path = os.path.join(base_path, f"transformer_input_KUMC_val_{suffix}.pkl")
    test_path = os.path.join(base_path, f"transformer_input_KUMC_test_{suffix}.pkl")

    train_df = pd.read_pickle(train_path)
    val_df = pd.read_pickle(val_path)
    test_df = pd.read_pickle(test_path)

    if sample is not None:
        train_df = train_df.sample(frac=sample, random_state=42)
        val_df = val_df.sample(frac=sample, random_state=42)
        test_df = test_df.sample(frac=sample, random_state=42)

    return train_df, val_df, test_df


def drop_first_token_flexibly(df):
    df = df.copy()

    if "TOKEN_SEQUENCE" in df.columns:
        df["TOKEN_SEQUENCE"] = df["TOKEN_SEQUENCE"].apply(
            lambda x: " ".join(x.split(" ")[1:]) if isinstance(x, str) else x
        )

    for field in ["MODULE_ID_SEQUENCE", "DELTA_T_SEQUENCE", "TYPE_SEQUENCE"]:
        if field in df.columns:
            df[field] = df[field].apply(
                lambda x: x[1:] if isinstance(x, list) and len(x) > 1 else x
            )

    return df


def load_token_dicts(file_path):
    token2id_path = os.path.join(file_path, "token2idUMSL.pkl")
    id2token_path = os.path.join(file_path, "id2tokenUMSL.pkl")

    with open(token2id_path, "rb") as f:
        token2id = pickle.load(f)

    with open(id2token_path, "rb") as f:
        id2token = pickle.load(f)

    print(f"token2id size: {len(token2id)}")
    print(f"id2token sample: {list(id2token.items())[:5]}")

    return token2id, id2token


def load_ontology_map_table(file_path, token2id, filename="df_tokens_final_UMSL.pkl"):
    pickle_path = os.path.join(file_path, filename)
    df = pd.read_pickle(pickle_path)

    df["medcode_id"] = df["medcode_origin"].map(token2id)
    df["ontology_id"] = df["final_ontology"].map(token2id).astype("Int64")
    df["minor_ontology_id"] = df["minor_ontology"].map(token2id).astype("Int64")

    print(f"rows: {len(df)}, columns: {list(df.columns)}")
    return df
