import torch

config = {
    "file_path": "./",
    "save_dir": "./outputs",
    "model_save_type": "ProposedAB",
    "suffix": "timintv",

    "vocab_size": 30522,
    "embedding_dim": 128,
    "hidden_dim": 256,
    "max_len": 500,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "max_timescale": 1e4,

    "use_module_embedding": False,
    "use_semantic_embedding": True,
    "freeze_semantic": False,
    "use_ontology": True,
    "num_ontology": 3000,
    "use_mrontology": True,
    "num_mrontology": 3000,

    "batch_size": 128,
    "num_epochs": 400,
    "learning_rate": 1e-4,
    "mask_prob": 0.15,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
