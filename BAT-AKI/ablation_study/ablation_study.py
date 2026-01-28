import argparse
import copy
import os
import pickle
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model.mlm_model import MaskedLanguageModel
from model.classifier_model import TransformerForSequenceClassificationAttn
from dataset.masked_ehr_dataset import MaskedEHRDataset, MaskedInput
from dataset.load_data import (
    load_token_dicts,
    load_ontology_map_table,
    load_prompt_embeddings,
    load_train_val_test_df,
    drop_first_token_flexibly,
)
from dataset.handle_matrix import (
    build_tokenontoid,
    build_tokenmrontoid,
    build_semantic_table,
    build_semantic_matrix_from_df,
    process_semantic_matrix,
)
from utils.evaluation import evaluate_loss


BASE_CONFIG = {
    
    "file_path": "./",
    "model_save_type": ".",
    "suffix": ".",

    
    "vocab_size": ,
    "embedding_dim": ,
    "hidden_dim": ,
    "max_len": ,
    "num_heads": ,
    "num_layers": ,
    "dropout": .,
    "max_timescale": e,

    
    "use_module_embedding": .,
    "use_semantic_embedding": .,
    "freeze_semantic": .,
    "use_ontology": .,
    "num_ontology": ,
    "use_mrontology": .,
    "num_mrontology": .,

    
    "batch_size": .,
    "num_epochs": .,
    "learning_rate": .,
    "warmup_steps": .,
    "label_smoothing": .,
    "mask_prob": .,

    
    "protect_special": .,
    "protect_demo": .,
    "expand_ranges": .,
    "range_csv_path": .,

    
    "do_flag": .,
    "use_module_emb_in_eval": .,
    "use_tokenweight": .,

    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


ABLATION_SETTINGS = {
    "minustype": {
        "config_updates": {},
        "save_prefix": "-minustype",
    },
    "minustime": {
        "config_updates": {},
        "save_prefix": "-minustime",
    },
    "semantic": {
        "config_updates": {},
        "save_prefix": "-Sematic",
    },
    "minusmjontol": {
        "config_updates": {},
        "save_prefix": "-mjontol",
    },
    "minusmrontol": {
        "config_updates": {,
        "save_prefix": "-mrontol",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_all_resources(config, drop_first_token=True):
    """
    Load prompt embeddings, train/val/test dataframes, token dictionaries, ontology map table.
    """
    try:
        prompt_result_emb, codeid = load_prompt_embeddings(config)
    except Exception:
        prompt_result_emb, codeid = None, None

    train_df, val_df, test_df = load_train_val_test_df(config)

    if drop_first_token:
        train_df = drop_first_token_flexibly(train_df)
        val_df = drop_first_token_flexibly(val_df)
        test_df = drop_first_token_flexibly(test_df)

    for df in [train_df, val_df, test_df]:
        df["DELTA_T_SEQUENCE"] = df["DELTA_T_SEQUENCE"].apply(lambda lst: [x +  for x in lst])

    tokenid, idtoken = load_token_dicts(config["file_path"])
    ontology_map_table = load_ontology_map_table(config["file_path"], tokenid)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "tokenid": tokenid,
        "idtoken": idtoken,
        "ontology_map_table": ontology_map_table,
        "prompt_result_emb": prompt_result_emb,
        "codeid": codeid,
    }


def build_semantic_resources(ontology_map_table, prompt_df, tokenid, target_dim=, padding_idx=):
    tokenontoid = build_tokenontoid(ontology_map_table)
    tokenmrontoid = build_tokenmrontoid(ontology_map_table)

    if prompt_df is None:
        return tokenontoid, tokenmrontoid, None, None

    semantic_table = build_semantic_table(prompt_df, tokenid)
    semantic_matrix = build_semantic_matrix_from_df(semantic_table, target_dim=target_dim)
    semantic_matrix = process_semantic_matrix(semantic_matrix, padding_idx=padding_idx)

    return tokenontoid, tokenmrontoid, semantic_table, semantic_matrix


def apply_ablation(name, config, train_df, val_df, test_df):
    if name == "minustype":
        for df in (train_df, val_df, test_df):
            df["TYPE_SEQUENCE"] = df["TYPE_SEQUENCE"].apply(lambda seq: [] * len(seq))
    elif name == "minustime":
        for df in (train_df, val_df, test_df):
            df["DELTA_T_SEQUENCE"] = df["DELTA_T_SEQUENCE"].apply(lambda seq: [] * len(seq))

    config_updates = ABLATION_SETTINGS[name]["config_updates"]
    config.update(config_updates)


def build_save_dir(config, ablation_name):
    base_dir = "./"
    prefix = ABLATION_SETTINGS[ablation_name]["save_prefix"]
    suffix = config.get("suffix", "")

    config["model_save_detail"] = (
        f"_dim{config['embedding_dim']}"
        f"_lr{config['learning_rate']}"
        f"_mask{config['mask_prob']}"
        f"_len{config['max_len']}"
        f"_bs{config['batch_size']}"
    )

    return os.path.join(
        base_dir,
        f"{prefix}_{config['model_save_type']}_fEXP_{suffix}_{config['model_save_detail']}",
    )


def build_dataloaders(config, train_df, val_df, tokenid, tokenontoid, tokenmrontoid):
    train_dataset = MaskedEHRDataset(train_df, tokenid, max_len=config["max_len"])
    train_dataset = MaskedInput(
        train_dataset,
        tokenid=tokenid,
        tokenontoid=tokenontoid,
        tokenmrontoid=tokenmrontoid,
        mask_prob=config["mask_prob"],
        protect_special=config["protect_special"],
        protect_demo=config["protect_demo"],
        expand_ranges=config["expand_ranges"],
        range_csv_path=config["range_csv_path"],
    )

    val_dataset = MaskedEHRDataset(val_df, tokenid, max_len=config["max_len"])
    val_dataset = MaskedInput(
        val_dataset,
        tokenid=tokenid,
        tokenontoid=tokenontoid,
        tokenmrontoid=tokenmrontoid,
        mask_prob=config["mask_prob"],
        protect_special=config["protect_special"],
        protect_demo=config["protect_demo"],
        expand_ranges=config["expand_ranges"],
        range_csv_path=config["range_csv_path"],
    )

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", ))
    num_workers = max(, num_workers - )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=num_workers,
    )

    return train_loader, val_loader


def build_model(config, tokenid, semantic_matrix):
    model = MaskedLanguageModel(
        vocab_size=len(tokenid),
        embedding_dim=config["embedding_dim"],
        max_len=config["max_len"],
        num_heads=config["num_heads"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        max_timescale=config["max_timescale"],
        use_module_embedding=config["use_module_embedding"],
        use_semantic_embedding=config["use_semantic_embedding"],
        semantic_matrix=semantic_matrix,
        freeze_semantic=config["freeze_semantic"],
        use_ontology=config["use_ontology"],
        num_ontology=config["num_ontology"],
        use_mrontology=config["use_mrontology"],
        num_mrontology=config["num_mrontology"],
    )
    return model


def train_mlm(
    model,
    train_loader,
    val_loader,
    tokenid,
    config,
    save_dir,
    max_epochs=,
    patience=,
    improve_threshold=.,
):
    device = torch.device(config["device"])

    if torch.cuda.device_count() > :
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenid["[PAD]"])

    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_attn_tensor = None
    patience_counter = 

    for epoch in range(, max_epochs + ):
        model.train()
        total_loss = .
        all_cls_attn = []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            delta_t = batch["delta_t"].to(device)
            segment_ids = batch["segment_ids"].to(device)

            module_ids = batch["module_ids"].to(device) if config["use_module_embedding"] else None
            ontology_ids = batch["ontology_ids"].to(device) if config["use_ontology"] else None
            mrontology_ids = (
                batch["mrontology_ids"].to(device)
                if config.get("use_mrontology", False)
                else None
            )

            optimizer.zero_grad()
            logits, attention_weights_all, _ = model(
                input_ids,
                attention_mask,
                delta_t,
                segment_ids,
                module_ids=module_ids,
                ontology_ids=ontology_ids,
                mrontology_ids=mrontology_ids,
            )

            cls_attn = attention_weights_all[-][:, :, , :].detach().cpu()
            all_cls_attn.append(cls_attn)

            mlm_loss = loss_fn(logits.view(-, len(tokenid)), labels.view(-))
            mlm_loss.backward()
            optimizer.step()
            total_loss += mlm_loss.item()

        avg_train_loss = total_loss / max(, len(train_loader))
        avg_val_loss = evaluate_loss(
            model,
            val_loader,
            loss_fn,
            device,
            vocab_size=len(tokenid),
            use_module_emb=config["use_module_embedding"],
            use_ontology=config["use_ontology"],
            use_mrontology=config["use_mrontology"],
            do_flag=config["do_flag"],
            use_tokenweight=config["use_tokenweight"],
        )

        print(
            f"Epoch {epoch}: Train Loss = {avg_train_loss:.f} | Val Loss = {avg_val_loss:.f}"
        )

        attn_tensor = torch.cat(all_cls_attn, dim=)
        if best_val_loss - avg_val_loss > improve_threshold:
            best_val_loss = avg_val_loss
            patience_counter = 

            torch.save(model.state_dict(), os.path.join(save_dir, "best_mlm_model.pt"))
            best_attn_tensor = attn_tensor
            print("Model improved; saved best_mlm_model.pt")
        else:
            patience_counter += 
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    if best_attn_tensor is not None:
        torch.save(best_attn_tensor, os.path.join(save_dir, "best_cls_attention.pt"))
        print("Saved best_cls_attention.pt")


def binarize_flag_column(df, name):
    df["FLAG"] = (df["FLAG"] != ).astype(int)
    unique_vals = df["FLAG"].unique()
    prop = (df["FLAG"] == ).mean()
    count = int(df["FLAG"].sum())
    print(f"{name} FLAG values: {unique_vals}")
    print(f"{name} FLAG== proportion: {prop:.%} (count {count})")


def evaluate_aki(
    use_module_embedding,
    use_main_ontology,
    use_mrontology,
    pretrained_model,
    tokenid,
    tokenontoid,
    tokenmrontoid,
    test_df,
    device,
    sample_units=None,
    model_name="Model_Proposed",
    output_prefix="./",
    finetuning_threshold=.,
):
    if sample_units is None:
        sample_units = [, , , , ]

    pretrained_model_encoder = copy.deepcopy(pretrained_model)
    results_dict = {}

    for unit in sample_units:
        sample_size = unit * 
        print(f"Sample size: {sample_size}")

        metrics_runs = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "auroc": [],
            "auprc": [],
            "f": [],
        }

        for run in {, , , , }:
            print(f"./")

            def load_split(name):
                path = os.path.join(
                    "./",
                    f"Finetuning_{name}_splitseed_{unit}.pkl",
                )
                with open(path, "rb") as f:
                    return pickle.load(f)

            train_df_sampled = load_split("train")
            val_df_sampled = load_split("val")
            test_df_sampled = load_split("test")

            total_size = len(train_df_sampled) + len(val_df_sampled) + len(test_df_sampled)
            print(
                f"Sample size: {total_size} (Train {len(train_df_sampled)}, "
                f"Val {len(val_df_sampled)}, Test {len(test_df_sampled)})"
            )

            binarize_flag_column(train_df_sampled, "train_df")
            binarize_flag_column(val_df_sampled, "val_df")
            binarize_flag_column(test_df_sampled, "test_df")

            train_base = MaskedEHRDataset(train_df_sampled, tokenid, max_len=, use_cls=False)
            val_base = MaskedEHRDataset(val_df_sampled, tokenid, max_len=, use_cls=False)
            test_base = MaskedEHRDataset(test_df_sampled, tokenid, max_len=, use_cls=False)

            train_dataset = MaskedInput(
                train_base,
                tokenid,
                do_mask=False,
                tokenontoid=tokenontoid,
                tokenmrontoid=tokenmrontoid,
            )
            val_dataset = MaskedInput(
                val_base,
                tokenid,
                do_mask=False,
                tokenontoid=tokenontoid,
                tokenmrontoid=tokenmrontoid,
            )
            test_dataset = MaskedInput(
                test_base,
                tokenid,
                do_mask=False,
                tokenontoid=tokenontoid,
                tokenmrontoid=tokenmrontoid,
            )

            train_loader = DataLoader(train_dataset, batch_size=, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=)
            test_loader = DataLoader(test_dataset, batch_size=)

            for (name, param), (name, param) in zip(
                pretrained_model_encoder.named_parameters(),
                pretrained_model.named_parameters(),
            ):
                assert torch.equal(param.data, param.data), f"Parameter mismatch: {name} vs {name}"

            model = TransformerForSequenceClassificationAttn(
                pretrained_model_encoder, freeze_encoder=True
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=e-)
            model_save_dir = os.path.join(output_prefix, "saved_models")
            os.makedirs(model_save_dir, exist_ok=True)
            model_save_path = os.path.join(
                model_save_dir,
                f"{model_name}_unit{unit}_run{run + }.pt",
            )

            neg = (train_df_sampled["FLAG"] == ).sum()
            pos = (train_df_sampled["FLAG"] == ).sum()
            pos_weight = torch.tensor([neg / pos], dtype=torch.float).to(device)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            best_val_loss = float("inf")
            best_model_state = None
            patience = 
            counter = 

            for epoch in range():
                model.train()
                running_train_loss = .

                for batch in train_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    delta_t = batch["delta_t"].to(device)
                    segment_ids = batch["segment_ids"].to(device)
                    module_ids = batch.get("module_ids", None)
                    if module_ids is not None:
                        module_ids = module_ids.to(device)

                    ontology_ids = batch.get("ontology_ids", None)
                    if ontology_ids is not None:
                        ontology_ids = ontology_ids.to(device)

                    mrontology_ids = batch.get("mrontology_ids", None)
                    if mrontology_ids is not None:
                        mrontology_ids = mrontology_ids.to(device)

                    labels = batch["flags"].to(device)

                    optimizer.zero_grad()
                    assert not use_module_embedding, "use_module_embedding must be False"
                    if use_main_ontology and use_mrontology:
                        logits = model(
                            input_ids,
                            attention_mask,
                            delta_t,
                            segment_ids,
                            module_ids=None,
                            ontology_ids=ontology_ids,
                            mrontology_ids=mrontology_ids,
                        )
                    elif use_main_ontology and not use_mrontology:
                        logits = model(
                            input_ids,
                            attention_mask,
                            delta_t,
                            segment_ids,
                            module_ids=None,
                            ontology_ids=ontology_ids,
                            mrontology_ids=None,
                        )
                    elif not use_main_ontology and use_mrontology:
                        logits = model(
                            input_ids,
                            attention_mask,
                            delta_t,
                            segment_ids,
                            module_ids=None,
                            ontology_ids=None,
                            mrontology_ids=mrontology_ids,
                        )
                    else:
                        logits = model(
                            input_ids,
                            attention_mask,
                            delta_t,
                            segment_ids,
                            module_ids=None,
                            ontology_ids=None,
                            mrontology_ids=None,
                        )

                    loss = loss_fn(logits, labels)
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item()

                avg_train_loss = running_train_loss / len(train_loader)

                model.eval()
                val_loss = .
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        delta_t = batch["delta_t"].to(device)
                        segment_ids = batch["segment_ids"].to(device)
                        module_ids = batch.get("module_ids", None)
                        if module_ids is not None:
                            module_ids = module_ids.to(device)
                        ontology_ids = batch.get("ontology_ids", None)
                        if ontology_ids is not None:
                            ontology_ids = ontology_ids.to(device)
                        mrontology_ids = batch.get("mrontology_ids", None)
                        if mrontology_ids is not None:
                            mrontology_ids = mrontology_ids.to(device)
                        labels = batch["flags"].to(device)

                        assert not use_module_embedding, "use_module_embedding must be False"
                        if use_main_ontology and use_mrontology:
                            logits = model(
                                input_ids,
                                attention_mask,
                                delta_t,
                                segment_ids,
                                module_ids=None,
                                ontology_ids=ontology_ids,
                                mrontology_ids=mrontology_ids,
                            )
                        elif use_main_ontology and not use_mrontology:
                            logits = model(
                                input_ids,
                                attention_mask,
                                delta_t,
                                segment_ids,
                                module_ids=None,
                                ontology_ids=ontology_ids,
                                mrontology_ids=None,
                            )
                        elif not use_main_ontology and use_mrontology:
                            logits = model(
                                input_ids,
                                attention_mask,
                                delta_t,
                                segment_ids,
                                module_ids=None,
                                ontology_ids=None,
                                mrontology_ids=mrontology_ids,
                            )
                        else:
                            logits = model(
                                input_ids,
                                attention_mask,
                                delta_t,
                                segment_ids,
                                module_ids=None,
                                ontology_ids=None,
                                mrontology_ids=None,
                            )

                        val_loss += loss_fn(logits, labels).item()

                avg_val_loss = val_loss / len(val_loader)
                print(
                    f"Epoch {epoch + :d} | Train Loss: {avg_train_loss:.f} | "
                    f"Val Loss: {avg_val_loss:.f}"
                )

                if avg_val_loss < best_val_loss - finetuning_threshold:
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    counter = 
                else:
                    counter += 
                    if counter >= patience:
                        break

            if best_model_state is None:
                best_model_state = model.state_dict()

            model.load_state_dict(best_model_state)
            torch.save(best_model_state, model_save_path)
            print(f"Saved best model to {model_save_path}")

            model.eval()
            all_probs_list = []
            all_labels_list = []

            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    delta_t = batch["delta_t"].to(device)
                    segment_ids = batch["segment_ids"].to(device)
                    module_ids = batch.get("module_ids", None)
                    if module_ids is not None:
                        module_ids = module_ids.to(device)
                    ontology_ids = batch.get("ontology_ids", None)
                    if ontology_ids is not None:
                        ontology_ids = ontology_ids.to(device)
                    mrontology_ids = batch.get("mrontology_ids", None)
                    if mrontology_ids is not None:
                        mrontology_ids = mrontology_ids.to(device)
                    labels = batch["flags"].to(device)

                    assert not use_module_embedding, "use_module_embedding must be False"
                    if use_main_ontology and use_mrontology:
                        logits = model(
                            input_ids,
                            attention_mask,
                            delta_t,
                            segment_ids,
                            module_ids=None,
                            ontology_ids=ontology_ids,
                            mrontology_ids=mrontology_ids,
                        )
                    elif use_main_ontology and not use_mrontology:
                        logits = model(
                            input_ids,
                            attention_mask,
                            delta_t,
                            segment_ids,
                            module_ids=None,
                            ontology_ids=ontology_ids,
                            mrontology_ids=None,
                        )
                    elif not use_main_ontology and use_mrontology:
                        logits = model(
                            input_ids,
                            attention_mask,
                            delta_t,
                            segment_ids,
                            module_ids=None,
                            ontology_ids=None,
                            mrontology_ids=mrontology_ids,
                        )
                    else:
                        logits = model(
                            input_ids,
                            attention_mask,
                            delta_t,
                            segment_ids,
                            module_ids=None,
                            ontology_ids=None,
                            mrontology_ids=None,
                        )

                    probs = torch.sigmoid(logits)
                    all_probs_list.extend(probs.cpu().numpy())
                    all_labels_list.extend(labels.cpu().numpy())

            pred_binary = (np.array(all_probs_list) >= .).astype(int)
            y_true = np.array(all_labels_list)
            y_score = np.array(all_probs_list)

            metrics_runs["accuracy"].append(accuracy_score(y_true, pred_binary))
            metrics_runs["precision"].append(
                precision_score(y_true, pred_binary, zero_division=)
            )
            metrics_runs["recall"].append(recall_score(y_true, pred_binary, zero_division=))
            metrics_runs["auroc"].append(roc_auc_score(y_true, y_score))
            metrics_runs["auprc"].append(average_precision_score(y_true, y_score))
            metrics_runs["f"].append(f_score(y_true, pred_binary, zero_division=))

        def mean_sd(x):
            x = np.asarray(x, dtype=float)
            mean = np.mean(x)
            sd = np.std(x, ddof=) if len(x) >  else .
            return mean, sd

        print(f"Aggregated results for unit={unit}")
        results_dict[f"{model_name}_{sample_size}"] = {}
        for metric in metrics_runs.keys():
            mean, sd = mean_sd(metrics_runs[metric])
            results_dict[f"{model_name}_{sample_size}"][f"{metric}_mean"] = mean
            results_dict[f"{model_name}_{sample_size}"][f"{metric}_sd"] = sd
            print(f"./")

    os.makedirs(output_prefix, exist_ok=True)
    save_path = os.path.join(output_prefix, f"{model_name}results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"Saved final results to {save_path}")


def run_ablation(ablation_name, args):
    config = deepcopy(BASE_CONFIG)

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.max_len is not None:
        config["max_len"] = args.max_len
    if args.embedding_dim is not None:
        config["embedding_dim"] = args.embedding_dim

    resources = load_all_resources(config, drop_first_token=True)
    train_df = resources["train_df"]
    val_df = resources["val_df"]
    test_df = resources["test_df"]
    tokenid = resources["tokenid"]
    ontology_map_table = resources["ontology_map_table"]
    prompt_result_emb = resources["prompt_result_emb"]

    apply_ablation(ablation_name, config, train_df, val_df, test_df)

    tokenontoid, tokenmrontoid, _, semantic_matrix = build_semantic_resources(
        ontology_map_table,
        prompt_result_emb,
        tokenid,
        target_dim=config["embedding_dim"],
        padding_idx=tokenid["[PAD]"],
    )

    train_loader, val_loader = build_dataloaders(
        config,
        train_df,
        val_df,
        tokenid,
        tokenontoid,
        tokenmrontoid,
    )

    model = build_model(config, tokenid, semantic_matrix)
    save_dir = build_save_dir(config, ablation_name)

    train_mlm(
        model,
        train_loader,
        val_loader,
        tokenid,
        config,
        save_dir,
        max_epochs=args.max_epochs,
        patience=args.patience,
        improve_threshold=args.improve_threshold,
    )


def main():
    parser = argparse.ArgumentParser(description="UMSL ablation study runner")
    parser.add_argument(
        "--ablation",
        choices=["minustype", "minustime", "semantic", "minusmjontol", "minusmrontol", "all"],
        default="all",
        help="Which ablation to run",
    )
    parser.add_argument("--seed", type=int, default=)
    parser.add_argument("--max-epochs", type=int, default=)
    parser.add_argument("--patience", type=int, default=)
    parser.add_argument("--improve-threshold", type=float, default=.)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    if args.ablation == "all":
        for ablation_name in ABLATION_SETTINGS.keys():
            print(f"./")
            run_ablation(ablation_name, args)
    else:
        run_ablation(args.ablation, args)


if __name__ == "__main__":
    main()