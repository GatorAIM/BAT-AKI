import os
import torch
import torch.nn.functional as F
from utils.evaluation import evaluate_loss

def train_mlm(model, train_loader, val_loader, token2id, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=token2id["[PAD]"])

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    best_attn = None

    for _ in range(config["num_epochs"]):
        model.train()
        all_cls_attn = []

        for batch in train_loader:
            batch = {k: v.to(config["device"]) for k, v in batch.items()}
            optimizer.zero_grad()

            logits, abnormal_logits, attn_all, _ = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["delta_t"],
                batch["segment_ids"],
                module_ids=batch["module_ids"],
                ontology_ids=batch["ontology_ids"],
                mrontology_ids=batch.get("mrontology_ids"),
            )

            mlm_loss = loss_fn(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
            abnormal_loss = F.binary_cross_entropy_with_logits(
                abnormal_logits, batch["abnormal_flags"], reduction="mean"
            )

            loss = mlm_loss + abnormal_loss
            loss.backward()
            optimizer.step()

            all_cls_attn.append(attn_all[-1][:, :, 0, :].detach().cpu())

        val_loss = evaluate_loss(
            model,
            val_loader,
            loss_fn,
            config["device"],
            vocab_size=len(token2id),
            use_ontology=True,
            use_abnormal_loss=True,
        )

        if best_val_loss - val_loss > 1e-3:
            best_val_loss = val_loss
            patience_counter = 0
            best_attn = torch.cat(all_cls_attn, dim=0)
            torch.save(model.state_dict(), os.path.join(config["save_dir"], "best_mlm.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_attn is not None:
        torch.save(best_attn, os.path.join(config["save_dir"], "best_cls_attention.pt"))
