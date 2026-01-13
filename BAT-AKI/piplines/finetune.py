import os
import copy
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
)

from dataset.masked_ehr_dataset import MaskedEHRDataset, MaskedInput
from model.classifier_model import TransformerForSequenceClassificationAttn

def evaluate_aki(
    pretrained_model,
    token2id,
    token2ontoid,
    token2mrontoid,
    device,
    splits_dir,
    output_dir,
    sample_units,
    config,
):
    pretrained_encoder = pretrained_model.module if hasattr(pretrained_model, "module") else pretrained_model

    results = {}

    for unit in sample_units:
        metrics_runs = {k: [] for k in ["accuracy", "precision", "recall", "auroc", "auprc", "f1"]}

        for run in range(5):
            def load(name):
                path = os.path.join(splits_dir, f"Finetuning_{name}_splitseed1_{unit}.pkl")
                with open(path, "rb") as f:
                    return f.load()

            train_df, val_df, test_df = load("train"), load("val"), load("test")

            train_ds = MaskedInput(MaskedEHRDataset(train_df, token2id, 500), token2id, do_mask=False)
            val_ds   = MaskedInput(MaskedEHRDataset(val_df, token2id, 500), token2id, do_mask=False)
            test_ds  = MaskedInput(MaskedEHRDataset(test_df, token2id, 500), token2id, do_mask=False)

            train_loader = DataLoader(train_ds, 128, shuffle=True)
            val_loader   = DataLoader(val_ds, 128)
            test_loader  = DataLoader(test_ds, 128)

            model = TransformerForSequenceClassificationAttn(
                copy.deepcopy(pretrained_encoder),
                freeze_encoder=True,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=config["Finetunig_lr"])
            loss_fn = torch.nn.BCEWithLogitsLoss()

            best_state = None
            best_val = float("inf")
            patience = 10
            counter = 0

            for _ in range(1000):
                model.train()
                for batch in train_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    optimizer.zero_grad()
                    logits = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["delta_t"],
                        batch["segment_ids"],
                    )
                    loss = loss_fn(logits, batch["flags"])
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        logits = model(
                            batch["input_ids"],
                            batch["attention_mask"],
                            batch["delta_t"],
                            batch["segment_ids"],
                        )
                        val_loss += loss_fn(logits, batch["flags"]).item()

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        break

            model.load_state_dict(best_state)
            model.eval()

            y_true, y_score = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logits = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["delta_t"],
                        batch["segment_ids"],
                    )
                    y_score.extend(torch.sigmoid(logits).cpu().numpy())
                    y_true.extend(batch["flags"].cpu().numpy())

            y_true = np.array(y_true)
            y_score = np.array(y_score)
            y_pred = (y_score >= 0.5).astype(int)

            metrics_runs["accuracy"].append(accuracy_score(y_true, y_pred))
            metrics_runs["precision"].append(precision_score(y_true, y_pred, zero_division=0))
            metrics_runs["recall"].append(recall_score(y_true, y_pred, zero_division=0))
            metrics_runs["auroc"].append(roc_auc_score(y_true, y_score))
            metrics_runs["auprc"].append(average_precision_score(y_true, y_score))
            metrics_runs["f1"].append(f1_score(y_true, y_pred, zero_division=0))

        results[unit] = {
            m: (np.mean(v), np.std(v, ddof=1))
            for m, v in metrics_runs.items()
        }

    return results


def evaluate_death(
    use_module_embedding,
    use_ontology,
    pretrained_model,
    token2id,
    token2ontoid,
    token2mrontoid,
    device,
    sample_units,
    config,
    model_name="Model_Proposed",
    splits_dir="./splits",
    output_prefix="./eval_outputs",
    label_source_df=None,
    label_merge_keys=('PATID', 'ENCOUNTERID'),
    label_source_col='death90',
):
    pretrained_encoder = pretrained_model.module if hasattr(pretrained_model, "module") else pretrained_model
    pretrained_encoder_copy = copy.deepcopy(pretrained_encoder)
    
    results_dict = {}
    metrics_record = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'auroc': [],
        'auprc': [],
        'f1': []
    }
    
    for unit in sample_units:
        sample_size = unit * 25
        metrics_runs = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'auroc': [],
            'auprc': [],
            'f1': []
        }
        
        for run in range(5):
            def load_split(name):
                path = os.path.join(splits_dir, f"Finetuning_{name}_splitseed1_{unit}.pkl")
                with open(path, "rb") as f:
                    return pickle.load(f)
            
            train_df_sampled = load_split('train')
            val_df_sampled = load_split('val')
            test_df_sampled = load_split('test')
            
            train_df_sampled['FLAG'] = (train_df_sampled['FLAG'] != 0).astype(int)
            val_df_sampled['FLAG'] = (val_df_sampled['FLAG'] != 0).astype(int)
            test_df_sampled['FLAG'] = (test_df_sampled['FLAG'] != 0).astype(int)
            
            if label_source_df is not None:
                merge_keys = list(label_merge_keys)
                needed_cols = set(merge_keys + [label_source_col])
                miss_cols = needed_cols - set(label_source_df.columns)
                if miss_cols:
                    raise KeyError(f"label_source_df is missing required columns: {sorted(miss_cols)}")
                
                dup_mask = label_source_df.duplicated(subset=merge_keys, keep=False)
                if dup_mask.any():
                    raise ValueError("label_source_df has duplicate keys on merge_keys. Ensure 1 row per key in the label source.")
                
                source_df = label_source_df.copy()
                for k in merge_keys:
                    if not pd.api.types.is_integer_dtype(source_df[k]):
                        raise TypeError(f"label_source_df[{k}] must be int dtype, got {source_df[k].dtype}")
                
                def _merge_and_set_flag_minimal(
                    df, name,
                    merge_keys,
                    source_df,
                    label_source_col="death90",
                    target_flag_col="FLAG",
                    old_flag_col="FLAG"
                ):
                    missing_left = set(merge_keys) - set(df.columns)
                    if missing_left:
                        raise KeyError(f"{name}_df missing keys: {sorted(missing_left)}")
                    need_src = set(merge_keys + [label_source_col]) - set(source_df.columns)
                    if need_src:
                        raise KeyError(f"label source missing cols: {sorted(need_src)}")
                    if source_df.duplicated(subset=merge_keys).any():
                        raise ValueError("label source has duplicate keys; ensure 1 row per key.")
                    
                    df_local = df[df[old_flag_col] == 1].copy()
                    
                    df_local.drop(columns=[old_flag_col], inplace=True)
                    
                    for k in merge_keys:
                        df_local[k] = df_local[k].astype(int)
                    
                    src = source_df[merge_keys + [label_source_col]].copy()
                    for k in merge_keys:
                        src[k] = src[k].astype(int)
                    
                    merged = df_local.merge(
                        src,
                        on=merge_keys,
                        how='left',
                        validate='m:1'
                    )
                    if merged[label_source_col].isna().any():
                        n_miss = int(merged[label_source_col].isna().sum())
                        raise ValueError(f"{name}_df has {n_miss} missing '{label_source_col}' after merge.")
                    
                    merged[label_source_col] = merged[label_source_col].astype(int)
                    if set(merged[label_source_col].unique()) - {0, 1}:
                        raise ValueError(f"{name}_df '{label_source_col}' contains non-binary values.")
                    merged[target_flag_col] = merged[label_source_col].astype(int)
                    merged.drop(columns=[label_source_col], inplace=True)
                    
                    return merged
                
                train_df_sampled = _merge_and_set_flag_minimal(
                    train_df_sampled, "train",
                    merge_keys=list(label_merge_keys),
                    source_df=label_source_df,
                    label_source_col=label_source_col,
                    target_flag_col="FLAG",
                    old_flag_col="FLAG"
                )
                
                val_df_sampled = _merge_and_set_flag_minimal(
                    val_df_sampled, "val",
                    merge_keys=list(label_merge_keys),
                    source_df=label_source_df,
                    label_source_col=label_source_col,
                    target_flag_col="FLAG",
                    old_flag_col="FLAG"
                )
                
                test_df_sampled = _merge_and_set_flag_minimal(
                    test_df_sampled, "test",
                    merge_keys=list(label_merge_keys),
                    source_df=label_source_df,
                    label_source_col=label_source_col,
                    target_flag_col="FLAG",
                    old_flag_col="FLAG"
                )
            
            train_base = MaskedEHRDataset(train_df_sampled, token2id, max_len=500, use_cls=False)
            val_base = MaskedEHRDataset(val_df_sampled, token2id, max_len=500, use_cls=False)
            test_base = MaskedEHRDataset(test_df_sampled, token2id, max_len=500, use_cls=False)
            
            train_dataset = MaskedInput(train_base, token2id, do_mask=False, token2ontoid=token2ontoid, token2mrontoid=token2mrontoid)
            val_dataset = MaskedInput(val_base, token2id, do_mask=False, token2ontoid=token2ontoid, token2mrontoid=token2mrontoid)
            test_dataset = MaskedInput(test_base, token2id, do_mask=False, token2ontoid=token2ontoid, token2mrontoid=token2mrontoid)
            
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128)
            test_loader = DataLoader(test_dataset, batch_size=128)
            
            model = TransformerForSequenceClassificationAttn(
                copy.deepcopy(pretrained_encoder_copy),
                freeze_encoder=True
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=config["Finetunig_lr"])
            model_save_dir = os.path.join(output_prefix, "saved_models")
            os.makedirs(model_save_dir, exist_ok=True)
            model_save_path = os.path.join(
                model_save_dir,
                f"{model_name}_unit{unit}_run{run+1}.pt"
            )
            
            neg, pos = (train_df_sampled['FLAG'] == 0).sum(), (train_df_sampled['FLAG'] == 1).sum()
            pos_weight = torch.tensor([neg / pos], dtype=torch.float).to(device)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            best_val_loss, patience, counter = float('inf'), 10, 0
            best_model_state = None
            
            for epoch in range(1000):
                model.train()
                running_train_loss = 0.0
                
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    delta_t = batch['delta_t'].to(device)
                    segment_ids = batch['segment_ids'].to(device)
                    
                    module_ids = batch.get('module_ids', None)
                    if module_ids is not None:
                        module_ids = module_ids.to(device)
                    
                    ontology_ids = batch.get('ontology_ids', None)
                    if ontology_ids is not None:
                        ontology_ids = ontology_ids.to(device)
                    
                    mrontology_ids = batch.get('mrontology_ids', None)
                    if mrontology_ids is not None:
                        mrontology_ids = mrontology_ids.to(device)
                    
                    labels = batch['flags'].to(device)
                    
                    optimizer.zero_grad()
                    if use_module_embedding and use_ontology:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=module_ids, ontology_ids=ontology_ids, mrontology_ids=mrontology_ids)
                    elif use_module_embedding and not use_ontology:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=module_ids, ontology_ids=None, mrontology_ids=None)
                    elif not use_module_embedding and use_ontology:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=None, ontology_ids=ontology_ids, mrontology_ids=mrontology_ids)
                    else:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=None, ontology_ids=None, mrontology_ids=None)
                    
                    loss = loss_fn(logits, labels)
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item()
                
                avg_train_loss = running_train_loss / len(train_loader)
                
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        delta_t = batch['delta_t'].to(device)
                        segment_ids = batch['segment_ids'].to(device)
                        
                        module_ids = batch.get('module_ids', None)
                        if module_ids is not None:
                            module_ids = module_ids.to(device)
                        
                        ontology_ids = batch.get('ontology_ids', None)
                        if ontology_ids is not None:
                            ontology_ids = ontology_ids.to(device)
                        
                        mrontology_ids = batch.get('mrontology_ids', None)
                        if mrontology_ids is not None:
                            mrontology_ids = mrontology_ids.to(device)
                        
                        labels = batch['flags'].to(device)
                        
                        if use_module_embedding and use_ontology:
                            logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=module_ids, ontology_ids=ontology_ids, mrontology_ids=mrontology_ids)
                        elif use_module_embedding and not use_ontology:
                            logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=module_ids, ontology_ids=None, mrontology_ids=None)
                        elif not use_module_embedding and use_ontology:
                            logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=None, ontology_ids=ontology_ids, mrontology_ids=mrontology_ids)
                        else:
                            logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=None, ontology_ids=None, mrontology_ids=None)
                        
                        val_loss += loss_fn(logits, labels).item()
                
                avg_val_loss = val_loss / len(val_loader)
                
                threshold = config.get("Finetunig_threshold", 0.0)
                if avg_val_loss < best_val_loss - threshold:
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        break
            
            model.load_state_dict(best_model_state)
            torch.save(best_model_state, model_save_path)
            
            model.eval()
            all_probs_list, all_labels_list = [], []
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    delta_t = batch['delta_t'].to(device)
                    segment_ids = batch['segment_ids'].to(device)
                    
                    module_ids = batch.get('module_ids', None)
                    if module_ids is not None:
                        module_ids = module_ids.to(device)
                    
                    ontology_ids = batch.get('ontology_ids', None)
                    if ontology_ids is not None:
                        ontology_ids = ontology_ids.to(device)
                    
                    mrontology_ids = batch.get('mrontology_ids', None)
                    if mrontology_ids is not None:
                        mrontology_ids = mrontology_ids.to(device)
                    
                    labels = batch['flags'].to(device)
                    
                    if use_module_embedding and use_ontology:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=module_ids, ontology_ids=ontology_ids, mrontology_ids=mrontology_ids)
                    elif use_module_embedding and not use_ontology:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=module_ids, ontology_ids=None, mrontology_ids=None)
                    elif not use_module_embedding and use_ontology:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=None, ontology_ids=ontology_ids, mrontology_ids=mrontology_ids)
                    else:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=None, ontology_ids=None, mrontology_ids=None)
                    
                    probs = torch.sigmoid(logits)
                    all_probs_list.extend(probs.cpu().numpy())
                    all_labels_list.extend(labels.cpu().numpy())
            
            pred_binary = (np.array(all_probs_list) >= 0.5).astype(int)
            y_true = np.array(all_labels_list)
            y_score = np.array(all_probs_list)
            
            acc = accuracy_score(y_true, pred_binary)
            prec = precision_score(y_true, pred_binary, zero_division=0)
            rec = recall_score(y_true, pred_binary, zero_division=0)
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
            f1 = f1_score(y_true, pred_binary, zero_division=0)
            
            metrics_runs['accuracy'].append(acc)
            metrics_runs['precision'].append(prec)
            metrics_runs['recall'].append(rec)
            metrics_runs['auroc'].append(auroc)
            metrics_runs['auprc'].append(auprc)
            metrics_runs['f1'].append(f1)
        
        def mean_sd(x):
            x = np.asarray(x, dtype=float)
            mean = np.mean(x)
            sd = np.std(x, ddof=1) if len(x) > 1 else 0.0
            return mean, sd
        
        results_dict[f"{model_name}_{sample_size}"] = {}
        for metric in metrics_runs.keys():
            mean, sd = mean_sd(metrics_runs[metric])
            results_dict[f"{model_name}_{sample_size}"][f"{metric}_mean"] = mean
            results_dict[f"{model_name}_{sample_size}"][f"{metric}_sd"] = sd
        
        for k in metrics_record.keys():
            metrics_record[k].append(results_dict[f"{model_name}_{sample_size}"][f"{k}_mean"])
    
    os.makedirs(output_prefix, exist_ok=True)
    save_path = os.path.join(output_prefix, f"{model_name}results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)
    
    return results_dict


def evaluate_rcvrvrt(
    use_module_embedding,
    use_ontology,
    pretrained_model,
    token2id,
    token2ontoid,
    token2mrontoid,
    device,
    sample_units,
    config,
    model_name="Model_Proposed",
    splits_dir="./splits",
    output_prefix="./eval_outputs",
    label_source_df=None,
    label_merge_keys=('PATID', 'ENCOUNTERID'),
    label_source_col='death90',
):
    pretrained_encoder = pretrained_model.module if hasattr(pretrained_model, "module") else pretrained_model
    pretrained_encoder_copy = copy.deepcopy(pretrained_encoder)
    
    results_dict = {}
    metrics_record = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'auroc': [],
        'auprc': [],
        'f1': []
    }
    
    for unit in sample_units:
        sample_size = unit * 25
        metrics_runs = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'auroc': [],
            'auprc': [],
            'f1': []
        }
        
        for run in range(5):
            def load_split(name):
                path = os.path.join(splits_dir, f"Finetuning_{name}_splitseed1_{unit}.pkl")
                with open(path, "rb") as f:
                    return pickle.load(f)
            
            train_df_sampled = load_split('train')
            val_df_sampled = load_split('val')
            test_df_sampled = load_split('test')
            
            train_df_sampled['FLAG'] = (train_df_sampled['FLAG'] != 0).astype(int)
            val_df_sampled['FLAG'] = (val_df_sampled['FLAG'] != 0).astype(int)
            test_df_sampled['FLAG'] = (test_df_sampled['FLAG'] != 0).astype(int)
            
            if label_source_df is not None:
                merge_keys = list(label_merge_keys)
                needed_cols = set(merge_keys + [label_source_col])
                miss_cols = needed_cols - set(label_source_df.columns)
                if miss_cols:
                    raise KeyError(f"label_source_df is missing required columns: {sorted(miss_cols)}")
                
                dup_mask = label_source_df.duplicated(subset=merge_keys, keep=False)
                if dup_mask.any():
                    raise ValueError("label_source_df has duplicate keys on merge_keys. Ensure 1 row per key in the label source.")
                
                source_df = label_source_df.copy()
                for k in merge_keys:
                    if not pd.api.types.is_integer_dtype(source_df[k]):
                        raise TypeError(f"label_source_df[{k}] must be int dtype, got {source_df[k].dtype}")
                
                def _merge_and_set_flag_minimal(
                    df, name,
                    merge_keys,
                    source_df,
                    label_source_col="death90",
                    target_flag_col="FLAG",
                    old_flag_col="FLAG"
                ):
                    missing_left = set(merge_keys) - set(df.columns)
                    if missing_left:
                        raise KeyError(f"{name}_df missing keys: {sorted(missing_left)}")
                    need_src = set(merge_keys + [label_source_col]) - set(source_df.columns)
                    if need_src:
                        raise KeyError(f"label source missing cols: {sorted(need_src)}")
                    if source_df.duplicated(subset=merge_keys).any():
                        raise ValueError("label source has duplicate keys; ensure 1 row per key.")
                    
                    df_local = df[df[old_flag_col] == 1].copy()
                    
                    df_local.drop(columns=[old_flag_col], inplace=True)
                    
                    for k in merge_keys:
                        df_local[k] = df_local[k].astype(int)
                    
                    src = source_df[merge_keys + [label_source_col]].copy()
                    for k in merge_keys:
                        src[k] = src[k].astype(int)
                    
                    merged = df_local.merge(
                        src,
                        on=merge_keys,
                        how='left',
                        validate='m:1'
                    )
                    if merged[label_source_col].isna().any():
                        n_miss = int(merged[label_source_col].isna().sum())
                        raise ValueError(f"{name}_df has {n_miss} missing '{label_source_col}' after merge.")
                    
                    merged[label_source_col] = merged[label_source_col].astype(int)
                    if set(merged[label_source_col].unique()) - {0, 1}:
                        raise ValueError(f"{name}_df '{label_source_col}' contains non-binary values.")
                    merged[target_flag_col] = merged[label_source_col].astype(int)
                    merged.drop(columns=[label_source_col], inplace=True)
                    
                    return merged
                
                train_df_sampled = _merge_and_set_flag_minimal(
                    train_df_sampled, "train",
                    merge_keys=list(label_merge_keys),
                    source_df=label_source_df,
                    label_source_col=label_source_col,
                    target_flag_col="FLAG",
                    old_flag_col="FLAG"
                )
                
                val_df_sampled = _merge_and_set_flag_minimal(
                    val_df_sampled, "val",
                    merge_keys=list(label_merge_keys),
                    source_df=label_source_df,
                    label_source_col=label_source_col,
                    target_flag_col="FLAG",
                    old_flag_col="FLAG"
                )
                
                test_df_sampled = _merge_and_set_flag_minimal(
                    test_df_sampled, "test",
                    merge_keys=list(label_merge_keys),
                    source_df=label_source_df,
                    label_source_col=label_source_col,
                    target_flag_col="FLAG",
                    old_flag_col="FLAG"
                )
            
            train_base = MaskedEHRDataset(train_df_sampled, token2id, max_len=500, use_cls=False)
            val_base = MaskedEHRDataset(val_df_sampled, token2id, max_len=500, use_cls=False)
            test_base = MaskedEHRDataset(test_df_sampled, token2id, max_len=500, use_cls=False)
            
            train_dataset = MaskedInput(train_base, token2id, do_mask=False, token2ontoid=token2ontoid, token2mrontoid=token2mrontoid)
            val_dataset = MaskedInput(val_base, token2id, do_mask=False, token2ontoid=token2ontoid, token2mrontoid=token2mrontoid)
            test_dataset = MaskedInput(test_base, token2id, do_mask=False, token2ontoid=token2ontoid, token2mrontoid=token2mrontoid)
            
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128)
            test_loader = DataLoader(test_dataset, batch_size=128)
            
            model = TransformerForSequenceClassificationAttn(
                copy.deepcopy(pretrained_encoder_copy),
                freeze_encoder=True
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=config["Finetunig_lr"])
            model_save_dir = os.path.join(output_prefix, "saved_models")
            os.makedirs(model_save_dir, exist_ok=True)
            model_save_path = os.path.join(
                model_save_dir,
                f"{model_name}_unit{unit}_run{run+1}.pt"
            )
            
            loss_fn = torch.nn.BCEWithLogitsLoss()
            
            best_val_loss, patience, counter = float('inf'), 10, 0
            best_model_state = None
            
            for epoch in range(1000):
                model.train()
                running_train_loss = 0.0
                
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    delta_t = batch['delta_t'].to(device)
                    segment_ids = batch['segment_ids'].to(device)
                    
                    module_ids = batch.get('module_ids', None)
                    if module_ids is not None:
                        module_ids = module_ids.to(device)
                    
                    ontology_ids = batch.get('ontology_ids', None)
                    if ontology_ids is not None:
                        ontology_ids = ontology_ids.to(device)
                    
                    mrontology_ids = batch.get('mrontology_ids', None)
                    if mrontology_ids is not None:
                        mrontology_ids = mrontology_ids.to(device)
                    
                    labels = batch['flags'].to(device)
                    
                    optimizer.zero_grad()
                    if use_module_embedding and use_ontology:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=module_ids, ontology_ids=ontology_ids, mrontology_ids=mrontology_ids)
                    elif use_module_embedding and not use_ontology:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=module_ids, ontology_ids=None, mrontology_ids=None)
                    elif not use_module_embedding and use_ontology:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=None, ontology_ids=ontology_ids, mrontology_ids=mrontology_ids)
                    else:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=None, ontology_ids=None, mrontology_ids=None)
                    
                    loss = loss_fn(logits, labels)
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item()
                
                avg_train_loss = running_train_loss / len(train_loader)
                
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        delta_t = batch['delta_t'].to(device)
                        segment_ids = batch['segment_ids'].to(device)
                        
                        module_ids = batch.get('module_ids', None)
                        if module_ids is not None:
                            module_ids = module_ids.to(device)
                        
                        ontology_ids = batch.get('ontology_ids', None)
                        if ontology_ids is not None:
                            ontology_ids = ontology_ids.to(device)
                        
                        mrontology_ids = batch.get('mrontology_ids', None)
                        if mrontology_ids is not None:
                            mrontology_ids = mrontology_ids.to(device)
                        
                        labels = batch['flags'].to(device)
                        
                        if use_module_embedding and use_ontology:
                            logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=module_ids, ontology_ids=ontology_ids, mrontology_ids=mrontology_ids)
                        elif use_module_embedding and not use_ontology:
                            logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=module_ids, ontology_ids=None, mrontology_ids=None)
                        elif not use_module_embedding and use_ontology:
                            logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=None, ontology_ids=ontology_ids, mrontology_ids=mrontology_ids)
                        else:
                            logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=None, ontology_ids=None, mrontology_ids=None)
                        
                        val_loss += loss_fn(logits, labels).item()
                
                avg_val_loss = val_loss / len(val_loader)
                
                threshold = config.get("Finetunig_threshold", 0.0)
                if avg_val_loss < best_val_loss - threshold:
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        break
            
            model.load_state_dict(best_model_state)
            torch.save(best_model_state, model_save_path)
            
            model.eval()
            all_probs_list, all_labels_list = [], []
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    delta_t = batch['delta_t'].to(device)
                    segment_ids = batch['segment_ids'].to(device)
                    
                    module_ids = batch.get('module_ids', None)
                    if module_ids is not None:
                        module_ids = module_ids.to(device)
                    
                    ontology_ids = batch.get('ontology_ids', None)
                    if ontology_ids is not None:
                        ontology_ids = ontology_ids.to(device)
                    
                    mrontology_ids = batch.get('mrontology_ids', None)
                    if mrontology_ids is not None:
                        mrontology_ids = mrontology_ids.to(device)
                    
                    labels = batch['flags'].to(device)
                    
                    if use_module_embedding and use_ontology:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=module_ids, ontology_ids=ontology_ids, mrontology_ids=mrontology_ids)
                    elif use_module_embedding and not use_ontology:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=module_ids, ontology_ids=None, mrontology_ids=None)
                    elif not use_module_embedding and use_ontology:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=None, ontology_ids=ontology_ids, mrontology_ids=mrontology_ids)
                    else:
                        logits = model(input_ids, attention_mask, delta_t, segment_ids, module_ids=None, ontology_ids=None, mrontology_ids=None)
                    
                    probs = torch.sigmoid(logits)
                    all_probs_list.extend(probs.cpu().numpy())
                    all_labels_list.extend(labels.cpu().numpy())
            
            pred_binary = (np.array(all_probs_list) >= 0.5).astype(int)
            y_true = np.array(all_labels_list)
            y_score = np.array(all_probs_list)
            
            acc = accuracy_score(y_true, pred_binary)
            prec = precision_score(y_true, pred_binary, zero_division=0)
            rec = recall_score(y_true, pred_binary, zero_division=0)
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
            f1 = f1_score(y_true, pred_binary, zero_division=0)
            
            metrics_runs['accuracy'].append(acc)
            metrics_runs['precision'].append(prec)
            metrics_runs['recall'].append(rec)
            metrics_runs['auroc'].append(auroc)
            metrics_runs['auprc'].append(auprc)
            metrics_runs['f1'].append(f1)
        
        def mean_sd(x):
            x = np.asarray(x, dtype=float)
            mean = np.mean(x)
            sd = np.std(x, ddof=1) if len(x) > 1 else 0.0
            return mean, sd
        
        results_dict[f"{model_name}_{sample_size}"] = {}
        for metric in metrics_runs.keys():
            mean, sd = mean_sd(metrics_runs[metric])
            results_dict[f"{model_name}_{sample_size}"][f"{metric}_mean"] = mean
            results_dict[f"{model_name}_{sample_size}"][f"{metric}_sd"] = sd
        
        for k in metrics_record.keys():
            metrics_record[k].append(results_dict[f"{model_name}_{sample_size}"][f"{k}_mean"])
    
    os.makedirs(output_prefix, exist_ok=True)
    save_path = os.path.join(output_prefix, f"{model_name}results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)
    
    return results_dict
