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
├── pretraining.py
│
├── model/
│   ├── mlm_model.py
│   ├── classifier_model.py
│   └── loss.py
│
├── dataset/
│   ├── masked_ehr_dataset.py
│   ├── downstream_dataset.py
│   └── load_data.py
│
├── utils/
│   ├── handle_matrix.py
│   ├── evaluation.py
│   └── lr_scheduler.py
│
├── notebooks/
│   ├── Step1_Pretrain_Proposed.ipynb
│   ├── Step2_Finetune_Proposed.ipynb
│   ├── Step3_Baseline.ipynb
│   └── ATT_visualization.ipynb
│
├── requirements.txt
└── README.md
