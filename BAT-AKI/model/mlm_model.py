import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import pandas as pd
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False 
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src, attn_weights
    

    
class AKIEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        onto_dim=64,
        mronto_dim=64,
        dropout_prob=0.1,
        num_ontology=3000,
        num_mrontology=3000,
        use_ontology=False,
        use_mrontology=False,
    ):
        super().__init__()
        self.use_ontology = use_ontology
        self.use_mrontology = use_mrontology
        self.onto_dim = onto_dim if use_ontology else 0
        self.mronto_dim = mronto_dim if use_mrontology else 0
        self.token_dim = embedding_dim - self.onto_dim - self.mronto_dim

        self.word_embeddings = nn.Embedding(vocab_size, self.token_dim, padding_idx=0)
        if self.use_ontology:
            self.ontology_embeddings = nn.Embedding(num_ontology, onto_dim, padding_idx=0)

        if self.use_mrontology:
            self.module_ontology_embeddings = nn.Embedding(
                num_mrontology, mronto_dim, padding_idx=0
            )

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, ontology_ids=None, mrontology_ids=None):
        pieces = []
        x_token = self.word_embeddings(input_ids)
        pieces.append(x_token)

        if self.use_ontology:
            assert ontology_ids is not None
            x_onto = self.ontology_embeddings(ontology_ids)
            pieces.append(x_onto)

        if self.use_mrontology:
            assert mrontology_ids is not None
            x_mronto = self.module_ontology_embeddings(mrontology_ids)
            pieces.append(x_mronto)

        x = torch.cat(pieces, dim=-1)
        return self.dropout(x)
    
    
class AKITIMEmbeddings(nn.Module):
    def __init__(self, embedding_dim, max_timescale=1e4):
        super(AKITIMEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_timescale = max_timescale

    def forward(self, time_values, attention_mask=None):
        batch_size, seq_len = time_values.shape
        device = time_values.device

        dim_range = torch.arange(0, self.embedding_dim, 2, device=device).float()
        denom = self.max_timescale ** (dim_range / self.embedding_dim)

        time_values_expanded = time_values.unsqueeze(-1) / denom

        sin_part = torch.sin(time_values_expanded)
        cos_part = torch.cos(time_values_expanded)

        embeddings = torch.zeros(batch_size, seq_len, self.embedding_dim, device=device)
        embeddings[:, :, 0::2] = sin_part
        embeddings[:, :, 1::2] = cos_part

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings

class AKISEGEmbeddings(nn.Module):
    def __init__(self, embedding_dim, num_segments=60):
        super().__init__()
        self.embedding = nn.Embedding(num_segments, embedding_dim)

    def forward(self, segment_ids, attention_mask=None):
        embeddings = self.embedding(segment_ids)

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings

class AKIMODULEEmbeddings(nn.Module):
    def __init__(self, embedding_dim, num_modules=60):
        super().__init__()
        self.embedding = nn.Embedding(num_modules, embedding_dim)

    def forward(self, module_ids, attention_mask=None):
        embeddings = self.embedding(module_ids)

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings

class SemanticCodeEmbedding(nn.Module):
    def __init__(self, semantic_matrix: torch.Tensor, freeze: bool = False):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(semantic_matrix, freeze=freeze)

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        return embeddings
    
class BINPredictionHead(nn.Module):
    def __init__(self, embedding_dim=128, hidden_size=128):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.cls(x).squeeze(-1)
    
class MaskedLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        max_len=500,
        num_heads=4,
        hidden_dim=512,
        num_layers=2,
        dropout=0.1,
        max_timescale=1e4,
        use_module_embedding=True,
        use_semantic_embedding=False,
        semantic_matrix=None,
        freeze_semantic=True,
        use_ontology=False,
        num_ontology=3000,
        use_mrontology=False,
        num_mrontology=3000
    ):
        super().__init__()
        self.use_module_embedding = use_module_embedding
        self.use_semantic_embedding = use_semantic_embedding
        self.use_ontology = use_ontology
        self.num_ontology = num_ontology
        self.use_mrontology = use_mrontology
        self.num_mrontology = num_mrontology

        base_dim = embedding_dim
        if self.use_semantic_embedding:
            model_dim = base_dim * 2
        else:
            model_dim = base_dim

        self.model_dim = model_dim
        self.onto_dim = model_dim // 8
        self.mronto_dim = model_dim // 8
        self.token_dim = model_dim - self.onto_dim

        self.token_embedding = AKIEmbeddings(
            vocab_size=vocab_size,
            embedding_dim=base_dim,
            onto_dim=self.onto_dim,
            mronto_dim=self.mronto_dim,
            dropout_prob=dropout,
            num_ontology=self.num_ontology,
            num_mrontology=self.num_mrontology,
            use_ontology=self.use_ontology,
            use_mrontology=self.use_mrontology
        )

        self.time_embedding = AKITIMEmbeddings(model_dim)
        self.seg_embedding = AKISEGEmbeddings(num_segments=60, embedding_dim=model_dim)

        if self.use_semantic_embedding and semantic_matrix is not None:
            self.semantic_embedding = SemanticCodeEmbedding(
                semantic_matrix,
                freeze=freeze_semantic
            )

        if self.use_module_embedding:
            self.module_embedding = AKIMODULEEmbeddings(model_dim, num_modules=60)

        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            copy.deepcopy(
                CustomTransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    dropout=dropout
                )
            )
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(model_dim, vocab_size)
        self.flag_classifier = nn.Linear(model_dim, 1)
        self.abnormal_classifier = BINPredictionHead(model_dim, model_dim)

    def forward(
        self,
        input_ids,
        attention_mask,
        delta_t,
        segment_ids,
        module_ids=None,
        ontology_ids=None,
        mrontology_ids=None
    ):
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

        if self.use_semantic_embedding:
            x_sem = self.semantic_embedding(
                input_ids,
                attention_mask=attention_mask
            )
            x_token = torch.cat([x_token, x_sem], dim=-1)

        x_time = self.time_embedding(
            delta_t,
            attention_mask=attention_mask
        )
        x_seg = self.seg_embedding(
            segment_ids,
            attention_mask=attention_mask
        )
        x = x_token + x_time + x_seg

        if self.use_module_embedding and module_ids is not None:
            x_module = self.module_embedding(
                module_ids,
                attention_mask=attention_mask
            )
            x = x + x_module

        x = self.dropout(x)

        attention_weights_all = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(
                x,
                src_key_padding_mask=~attention_mask
            )
            attention_weights_all.append(attn_weights)

        logits = self.classifier(x)
        abnormal_logits = self.abnormal_classifier(x)

        return logits, abnormal_logits, attention_weights_all, x
    
