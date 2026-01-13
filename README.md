# BAT-AKI: Biomarker-Aware Transformer for Acute Kidney Injury Prediction

This repository contains the official implementation of **BAT-AKI**, a biomarker-aware transformer framework for early prediction and prognosis of Acute Kidney Injury (AKI) using longitudinal Electronic Health Records (EHRs).

BAT-AKI integrates temporal dynamics, clinical semantics, ontology structures, and biomarker abnormality–aware pretraining to improve predictive performance, robustness, and interpretability across multiple AKI-related tasks.

---

## 1. Introduction

Acute Kidney Injury (AKI) is a common and life-threatening complication among hospitalized patients. Although Electronic Health Records (EHRs) contain rich longitudinal information, their irregular temporal structure and heterogeneous clinical concepts pose challenges for conventional machine learning models.

BAT-AKI is designed to address these challenges by:
- Modeling long-range temporal dependencies using a Transformer backbone
- Incorporating structured medical knowledge through ontology-aware embeddings
- Enhancing sensitivity to biomarker abnormalities via tailored pretraining objectives
- Supporting flexible downstream fine-tuning across multiple AKI-related tasks

---

## 2. Model Overview

### 2.1 Input Representation

Each patient admission is serialized into a token sequence consisting of:
- Demographic tokens (e.g., age, sex, race)
- Time-ordered clinical event tokens
- Associated auxiliary sequences:
  - Time gaps (`delta_t`)
  - Segment identifiers (`segment_ids`)
  - Module identifiers (`module_ids`)
  - Ontology and minor-ontology identifiers
  - Biomarker abnormality flags (optional)

All sequences are padded or truncated to a fixed maximum length.

### 2.2 Embedding Components

BAT-AKI supports a modular embedding design, including:
- Token embeddings
- Continuous time embeddings (sinusoidal)
- Segment embeddings
- Module embeddings (optional)
- Ontology and minor-ontology embeddings (optional)
- Semantic embeddings derived from prompt-based medical code representations (optional)

These embeddings are combined and passed to the Transformer encoder.

### 2.3 Transformer Backbone

The backbone consists of multiple custom Transformer encoder layers with:
- Multi-head self-attention
- Feedforward networks
- Residual connections and layer normalization

Attention weights are preserved to support downstream interpretability analyses.

---

## 3. Repository Structure
```text
BAT-AKI/
├── BAT_AKI_evaluate.py              # Main evaluation script
├── BAT-AKI pretraining.py           # Main pretraining script
│
├── configs/                         # Configuration files
│   └── pretrain_config.py          # Pretraining configuration
│
├── dataset/                         # Dataset handling modules
│   ├── downstream_dataset.py       # Downstream task dataset
│   ├── handle_matrix.py            # Matrix handling utilities
│   ├── load_data.py                # Data loading functions
│   └── masked_ehr_dataset.py      # Masked EHR dataset implementation
│
├── model/                           # Model definitions
│   ├── classifier_model.py         # Classification model
│   └── mlm_model.py                # Masked Language Model
│
├── piplines/                        # Pipeline modules
│   ├── __init__.py
│   ├── datasets.py                 # Dataset utilities
│   ├── finetune.py                 # Fine-tuning functions
│   ├── load_pretrained.py          # Load pretrained models
│   ├── mask_eval.py                # Mask evaluation
│   ├── metrics.py                  # Metrics calculation
│   ├── resources.py                # Resource loading
│   ├── semantic.py                 # Semantic processing
│   └── train_mlm.py                # MLM training
│
├── static files/                    # Static configuration files
│   ├── focus_tokens.json           # Focus tokens configuration
│   └── ignore_prefixes.json        # Ignore prefixes configuration
│
└── utils/                           # Utility functions
    ├── build_selected_tokens.py    # Token selection builder
    ├── evaluation.py                # Evaluation utilities
    └── loss.py                      # Loss functions
```
