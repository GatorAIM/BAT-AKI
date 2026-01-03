import os
import torch
from torch.utils.data import DataLoader

from configs.pretrain_config import config
from model.mlm_model import MaskedLanguageModel
from utils.build_selected_tokens import build_selected_tokens_df

from pipelines.resources import load_all_resources
from pipelines.semantic import build_semantic_resources
from pipelines.datasets import build_pretrain_datasets
from pipelines.train_mlm import train_mlm

def main():
    os.makedirs(config["save_dir"], exist_ok=True)

    selected_tokens_df = build_selected_tokens_df(
        file_path=config["file_path"],
        static_dir="./bat_aki/static_files",
    )

    train_df, val_df, _, token2id, ontology_map_table, prompt_df = load_all_resources(config)

    selected_tokens_map = {
        token2id[r["origin"]]: token2id.get(r["opp_final"])
        for _, r in selected_tokens_df.iterrows()
        if r["origin"] in token2id
    }

    token2ontoid, token2mrontoid, semantic_matrix = build_semantic_resources(
        ontology_map_table, prompt_df, token2id, token2id["[PAD]"]
    )

    train_ds, val_ds = build_pretrain_datasets(
        train_df,
        val_df,
        token2id,
        token2ontoid,
        token2mrontoid,
        selected_tokens_map,
        config,
    )

    model = MaskedLanguageModel(
        vocab_size=len(token2id),
        embedding_dim=config["embedding_dim"],
        max_len=config["max_len"],
        semantic_matrix=semantic_matrix,
        use_semantic_embedding=config["use_semantic_embedding"],
        freeze_semantic=config["freeze_semantic"],
        use_ontology=config["use_ontology"],
        use_mrontology=config["use_mrontology"],
    ).to(config["device"])

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])

    train_mlm(model, train_loader, val_loader, token2id, config)

if __name__ == "__main__":
    main()
