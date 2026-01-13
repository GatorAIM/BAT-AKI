import torch
import torch.nn as nn

class AttentionPoolingHead(nn.Module):
    def __init__(self, hidden_dim, mid_dim=64, use_single_layer=False):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

        if use_single_layer:
            self.classifier = nn.Linear(hidden_dim, 1)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, mid_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mid_dim, 1)
            )

    def forward(self, x, attention_mask):
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(~attention_mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)
        return self.classifier(pooled).squeeze(-1)
    
class TransformerForSequenceClassificationAttn(nn.Module):
    def __init__(self, pretrained_model, hidden_dim=128, freeze_encoder=False, mid_dim=64, use_single_layer=False):
        super().__init__()
        self.token_embedding = pretrained_model.token_embedding
        self.time_embedding = pretrained_model.time_embedding
        self.seg_embedding = pretrained_model.seg_embedding
        self.use_ontology = pretrained_model.use_ontology
        self.use_semantic_embedding = pretrained_model.use_semantic_embedding
        self.use_mrontology = getattr(pretrained_model, "use_mrontology", False)
        self.model_dim = pretrained_model.model_dim

        if hasattr(pretrained_model, "semantic_embedding"):
            self.semantic_embedding = pretrained_model.semantic_embedding
        else:
            self.semantic_embedding = None

        if hasattr(pretrained_model, "module_embedding"):
            self.module_embedding = pretrained_model.module_embedding
        else:
            self.module_embedding = None

        self.encoder_layers = pretrained_model.encoder_layers
        self.dropout = nn.Dropout(0.1)

        self.attn_head = AttentionPoolingHead(
            pretrained_model.model_dim,
            mid_dim=mid_dim,
            use_single_layer=use_single_layer
        )

        if freeze_encoder:
            for name, param in self.named_parameters():
                if not name.startswith("attn_head."):
                    param.requires_grad = False

            num_frozen = sum(int(not p.requires_grad) for p in self.parameters())
            num_total = sum(1 for _ in self.parameters())
            print(f"[freeze_encoder] Frozen params: {num_frozen}/{num_total}")

    def forward(self, input_ids, attention_mask, delta_t, segment_ids, module_ids=None, ontology_ids=None, mrontology_ids=None):
        if self.use_ontology or self.use_mrontology:
            if self.use_ontology:
                assert ontology_ids is not None
            if self.use_mrontology:
                assert mrontology_ids is not None
            x_token = self.token_embedding(
                input_ids,
                ontology_ids=ontology_ids if self.use_ontology else None,
                mrontology_ids=mrontology_ids if self.use_mrontology else None
            )
        else:
            x_token = self.token_embedding(input_ids)

        if self.use_semantic_embedding and self.semantic_embedding is not None:
            x_sem = self.semantic_embedding(input_ids)
            x_token = torch.cat([x_token, x_sem], dim=-1)

        x_time = self.time_embedding(delta_t)
        x_seg = self.seg_embedding(segment_ids)

        x = x_token + x_time + x_seg

        if module_ids is not None and self.module_embedding is not None:
            x_module = self.module_embedding(module_ids)
            x = x + x_module

        x = self.dropout(x)
        for layer in self.encoder_layers:
            x, _ = layer(x, src_key_padding_mask=~attention_mask)

        logits = self.attn_head(x, attention_mask)
        return logits

    
def get_embedding_dim(embedding_module):
    if hasattr(embedding_module, "word_embeddings"): 
        return embedding_module.word_embeddings.embedding_dim
    elif hasattr(embedding_module, "embedding"):  
        return embedding_module.embedding.embedding_dim
    elif hasattr(embedding_module, "embedding_dim"):  
        return embedding_module.embedding_dim
    else:
        raise ValueError(f"Unsupported embedding module: {type(embedding_module)}")
    

