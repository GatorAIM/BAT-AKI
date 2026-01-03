from dataset.masked_ehr_dataset import MaskedEHRDataset, MaskedInputabj

def build_pretrain_datasets(
    train_df,
    val_df,
    token2id,
    token2ontoid,
    token2mrontoid,
    selected_tokens_map,
    config,
):
    train_base = MaskedEHRDataset(train_df, token2id, max_len=config["max_len"])
    val_base = MaskedEHRDataset(val_df, token2id, max_len=config["max_len"])

    train_ds = MaskedInputabj(
        train_base,
        token2id=token2id,
        token2ontoid=token2ontoid,
        token2mrontoid=token2mrontoid,
        selected_tokens_df_map=selected_tokens_map,
        inject_prob=0.3,
        mask_prob=config["mask_prob"],
    )

    val_ds = MaskedInputabj(
        val_base,
        token2id=token2id,
        token2ontoid=token2ontoid,
        token2mrontoid=token2mrontoid,
        selected_tokens_df_map=selected_tokens_map,
        inject_prob=0.3,
        mask_prob=config["mask_prob"],
    )

    return train_ds, val_ds
