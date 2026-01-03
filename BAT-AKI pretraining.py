import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.mlm_model import MaskedLanguageModel
from dataset.masked_ehr_dataset import MaskedEHRDataset, MaskedInputabj
from dataset.load_data import (
    load_train_val_test_df,
    load_token_dicts,
    load_ontology_map_table,
    load_prompt_embeddings,
    drop_first_token_flexibly,
)
from dataset.handle_matrix import (
    build_token2ontoid,
    build_token2mrontoid,
    build_semantic_table,
    build_semantic_matrix_from_df,
    process_semantic_matrix,
)
from utils.evaluation import evaluate_loss

# -------------------------------------------------
# Configuration
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "file_path": "/..",
    "embedding_dim": 128,
    "max_len": 500,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "use_semantic_embedding": True,
    "freeze_semantic": False,
    "use_ontology": True,
    "use_mrontology": True,
    "num_ontology": 3000,
    "num_mrontology": 3000,
    "maxrange": 1000,
}

# -------------------------------------------------
# Load data
# -------------------------------------------------
train_df, val_df, _ = load_train_val_test_df(config)
train_df = drop_first_token_flexibly(train_df)
val_df = drop_first_token_flexibly(val_df)

token2id, _ = load_token_dicts(config["file_path"])
ontology_map = load_ontology_map_table(config["file_path"], token2id)
prompt_df, _ = load_prompt_embeddings(config)

# -------------------------------------------------
# Build semantic & ontology resources
# -------------------------------------------------
token2ontoid = build_token2ontoid(ontology_map)
token2mrontoid = build_token2mrontoid(ontology_map)

semantic_table = build_semantic_table(prompt_df, token2id)
semantic_matrix = build_semantic_matrix_from_df(semantic_table, target_dim=128)
semantic_matrix = process_semantic_matrix(semantic_matrix, padding_idx=token2id["[PAD]"])

# -------------------------------------------------
# Dataset & Dataloader
# -------------------------------------------------
train_base = MaskedEHRDataset(train_df, token2id, max_len=config["max_len"])
val_base = MaskedEHRDataset(val_df, token2id, max_len=config["max_len"])

train_dataset = MaskedInputabj(
    train_base,
    token2id=token2id,
    token2ontoid=token2ontoid,
    token2mrontoid=token2mrontoid,
    mask_prob=0.15,
)

val_dataset = MaskedInputabj(
    val_base,
    token2id=token2id,
    token2ontoid=token2ontoid,
    token2mrontoid=token2mrontoid,
    mask_prob=0.15,
)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

# -------------------------------------------------
# Model
# -------------------------------------------------
model = MaskedLanguageModel(
    vocab_size=len(token2id),
    embedding_dim=config["embedding_dim"],
    max_len=config["max_len"],
    use_semantic_embedding=config["use_semantic_embedding"],
    semantic_matrix=semantic_matrix,
    freeze_semantic=config["freeze_semantic"],
    use_ontology=config["use_ontology"],
    use_mrontology=config["use_mrontology"],
    num_ontology=config["num_ontology"],
    num_mrontology=config["num_mrontology"],
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
loss_fn = nn.CrossEntropyLoss(ignore_index=token2id["[PAD]"])

# -------------------------------------------------
# Training loop
# -------------------------------------------------
for epoch in range(config["maxrange"]):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()

        logits, abnormal_logits, _, _ = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["delta_t"].to(device),
            batch["segment_ids"].to(device),
            module_ids=batch["module_ids"].to(device),
            ontology_ids=batch["ontology_ids"].to(device),
            mrontology_ids=batch["mrontology_ids"].to(device),
        )

        mlm_loss = loss_fn(
            logits.view(-1, len(token2id)),
            batch["labels"].to(device).view(-1),
        )

        abnormal_loss = F.binary_cross_entropy_with_logits(
            abnormal_logits,
            batch["abnormal_flags"].to(device),
        )

        loss = mlm_loss + abnormal_loss
        loss.backward()
        optimizer.step()

    val_loss = evaluate_loss(
        model,
        val_loader,
        loss_fn,
        device,
        vocab_size=len(token2id),
        use_module_emb=True,
        use_ontology=True,
        use_abnormal_loss=True,
    )

    print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
