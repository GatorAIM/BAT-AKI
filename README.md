# BAT-AKI: Biomarker-Aware Transformer for Acute Kidney Injury Prediction

This repository contains the official implementation of **BAT-AKI**, a biomarker-aware transformer framework for early prediction and prognosis of Acute Kidney Injury (AKI) using longitudinal Electronic Health Records (EHRs).

BAT-AKI integrates temporal dynamics, clinical semantics, ontology structures, and biomarker abnormality–aware pretraining to improve both predictive performance and representation interpretability across multiple AKI-related tasks.

---

## 📌 Overview

BAT-AKI is designed to address key challenges in EHR-based AKI modeling:

- Longitudinal, irregularly sampled clinical events
- Heterogeneous medical concepts (labs, diagnoses, medications)
- Sensitivity to AKI-related biomarker abnormalities
- Transferability across institutions

The framework consists of:
1. A **Transformer-based backbone** with multi-channel embeddings
2. **Biomarker Abnormality–Aware Pretraining (BAP)** via masked language modeling
3. Flexible **downstream fine-tuning heads** for multiple AKI prediction tasks

---

## 🧠 Model Architecture

### Input Representation
Each patient admission is serialized into a token sequence containing:
- Demographics (e.g., age, sex, race)
- Time-ordered clinical events
- Associated metadata:
  - Time gaps (`delta_t`)
  - Segment IDs (visit-level)
  - Module IDs (lab / diagnosis / medication)
  - Ontology and minor-ontology IDs
  - Abnormal biomarker flags (optional)

### Embedding Components
- **Token Embedding**
- **Time Embedding** (continuous sinusoidal)
- **Segment Embedding**
- **Module Embedding** (optional)
- **Ontology / Minor Ontology Embedding** (optional)
- **Semantic Embedding** from prompt-based medical code embeddings (optional)

### Backbone
- Multi-layer Transformer encoder
- Attention weights preserved for interpretability

---

## 📂 Repository Structure

```text
.
├── masked_ehr_dataset.py      # Masked EHR dataset & masking strategy
├── downstream_dataset.py      # Inference / downstream datasets
├── handle_matrix.py           # Ontology & semantic embedding processing
├── load_data.py               # Data loading utilities
├── mlm_model.py               # BAT-AKI pretraining model (MLM + BAP)
├── classifier_model.py        # Downstream classification heads
├── loss.py                    # Label smoothing & auxiliary losses
├── lr_scheduler.py            # Transformer warmup LR scheduler
├── evaluation.py              # Training & evaluation utilities
├── notebooks/
│   ├── Step1_Pretrain.ipynb   # Pretraining (MLM + BAP)
│   ├── Step2_Finetune.ipynb   # Downstream fine-tuning
│   └── Step3_Attention.ipynb  # Attention visualization
└── README.md
