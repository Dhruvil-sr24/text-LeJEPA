# Text-LeJEPA
### A Research-Oriented Implementation of Latent Predictive Learning for Text

An experimental implementation of a **Joint Embedding Predictive Architecture (JEPA)** adapted for text, with deep diagnostics for representation collapse, spectral behavior, and embedding geometry.

This project investigates whether JEPA-style latent prediction — originally popularized in vision — can be effectively applied to language.

---

## 🔍 Why This Project Exists

Most language models optimize token prediction.

This project explores a different question:

> Can we learn strong text representations by predicting embeddings instead of tokens?

Instead of predicting discrete tokens, the model learns to:
- Encode one view of text
- Predict the embedding of another view
- Align representations in latent space

The focus is not just performance — but **understanding representation behavior**.

---

## 🧠 Core Contributions

This repository includes:

- ✅ Text-based JEPA implementation
- ✅ Adaptive Sigma Regularization (anti-collapse mechanism)
- ✅ Rank-ratio monitoring during training
- ✅ Full embedding diagnostics suite
- ✅ Linear probing for downstream evaluation
- ✅ Spectral analysis (eigenvalues, PCA)
- ✅ Geometry analysis (t-SNE, cosine similarity)
- ✅ Dimensional usage analysis

Most open-source JEPA-style projects stop at training.
This one investigates *why* it works (or fails).

---

## 🏗 Architecture Overview

The training objective:

1. Encode view A → embedding `z_a`
2. Encode view B → embedding `z_b`
3. Learn predictor `f(z_a)` → match `z_b`
4. Apply regularization to avoid trivial collapse

The model operates entirely in latent space.

Loss components:
- Embedding prediction loss (MSE / cosine-based)
- Adaptive Sigma Regularization
- Rank-based stability monitoring

---

## 📊 Representation Diagnostics

A dedicated debug suite generates:

### Spectral Analysis
- Eigenvalue distribution
- Rank ratio over training
- Collapse detection

### Geometric Analysis
- PCA projections
- t-SNE visualizations
- Cosine similarity matrices
- Correlation structure

### Dimensional Usage
- Per-dimension activation usage
- Imbalance detection
- Prediction vs target comparison

These tools help answer:

- Is the representation collapsing?
- Are only a few dimensions active?
- Is alignment trivial or meaningful?
- Does spectral decay indicate structure?

---

## 📁 Repository Structure

JSN.py # Core JEPA model
main.py # Training pipeline
lejeja_debug_suite.py # Embedding diagnostics
Jsn-linear-probe.py # Linear probing evaluation


## Diagnostic outputs:
- correlation_.png
- cosine_sim_.png
- dim_usage_.png
- eigenvalues_.png
- pca_.png
- tsne_.png


---

## 🏋️ Training

```bash
python main.py
```
Includes:

Adaptive regularization

Rank monitoring

Stability tracking
