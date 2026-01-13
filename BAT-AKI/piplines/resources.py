from dataset.load_data import (
    load_token_dicts,
    load_ontology_map_table,
    load_prompt_embeddings,
    load_train_val_test_df,
    drop_first_token_flexibly,
)

def load_all_resources(config, drop_first_token=True):
    try:
        prompt_df, code2id = load_prompt_embeddings(config)
    except Exception:
        prompt_df, code2id = None, None

    train_df, val_df, test_df = load_train_val_test_df(config)

    if drop_first_token:
        train_df = drop_first_token_flexibly(train_df)
        val_df = drop_first_token_flexibly(val_df)
        test_df = drop_first_token_flexibly(test_df)

    for df in (train_df, val_df, test_df):
        df["DELTA_T_SEQUENCE"] = df["DELTA_T_SEQUENCE"].apply(
            lambda x: [v + 1 for v in x]
        )

    token2id, id2token = load_token_dicts(config["file_path"])
    ontology_map_table = load_ontology_map_table(config["file_path"], token2id)

    return train_df, val_df, test_df, token2id, ontology_map_table, prompt_df
