import os
import torch
from model.mlm_model import MaskedLanguageModel

def load_pretrained_mlm(
    checkpoint_path,
    token2id,
    semantic_matrix,
    config,
    device,
):
    model = MaskedLanguageModel(
        vocab_size=len(token2id),
        embedding_dim=config["embedding_dim"],
        max_len=config["max_len"],
        use_module_embedding=config["use_module_embedding"],
        use_semantic_embedding=config["use_semantic_embedding"],
        semantic_matrix=semantic_matrix,
        freeze_semantic=config["freeze_semantic"],
        use_ontology=config["use_ontology"],
        num_ontology=config["num_ontology"],
        use_mrontology=config["use_mrontology"],
        num_mrontology=config["num_mrontology"],
    )

    state_dict = torch.load(checkpoint_path, map_location=device)

    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model
