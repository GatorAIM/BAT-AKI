import torch

def evaluate_batch_mask_accuracy(
    model,
    dataloader,
    token2id,
    device,
    k_list=(1, 5),
):
    model.eval()
    k_list = sorted(set(k_list))
    max_k = max(k_list)

    correct = {k: 0 for k in k_list}
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            logits, _, _, _ = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["delta_t"],
                batch["segment_ids"],
                module_ids=batch["module_ids"],
                ontology_ids=batch["ontology_ids"],
                mrontology_ids=batch.get("mrontology_ids"),
            )

            mask = batch["labels"] != token2id["[PAD]"]
            topk_idx = torch.topk(logits, k=max_k, dim=-1).indices

            for i in range(mask.size(0)):
                pos = mask[i]
                if pos.sum() == 0:
                    continue

                true = batch["labels"][i][pos]
                pred = topk_idx[i][pos]
                unique = true.unique()

                for tok in unique:
                    tok_mask = true == tok
                    for k in k_list:
                        if pred[tok_mask, :k].eq(tok).any():
                            correct[k] += 1
                    total += 1

    return {f"Top@{k}": correct[k] / total for k in k_list}
