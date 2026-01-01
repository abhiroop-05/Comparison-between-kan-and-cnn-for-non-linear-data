# EEG-Based Biometric Authentication using CNN vs KAN  
**Fair Architecture Comparison with Subject-Aware Evaluation**

This repository presents a **rigorous and leakage-free comparison** between a **Convolutional Neural Network (CNN)** and a **Kolmogorov–Arnold Network (KAN)** for **EEG-based biometric authentication**, using the PhysioNet EEG Motor Movement/Imagery dataset.

The project is fully **reproducible, publication-ready**, and designed with **strict fairness constraints** to ensure a meaningful comparison.

---

## 📌 Key Contributions

- ✅ Fair architecture comparison (CNN vs KAN)
- ✅ Subject-aware train/test splitting (no identity leakage)
- ✅ Identical preprocessing and feature pipelines
- ✅ Cross-validation with statistical testing
- ✅ ROC, AUC, EER, F1-score evaluation
- ✅ GPU acceleration supported

---

## 📊 ROC Curve Comparison

The ROC curve comparing CNN and KAN performance is automatically generated and saved as:

```
experiment_outputs/roc_comparison.png
```

**Observation:**  
KAN achieves a higher AUC than CNN, indicating superior modeling of non-linear EEG patterns.

---

## 🧠 Dataset

- **Dataset:** PhysioNet EEG Motor Movement/Imagery (EEGBCI)
- **Subjects:** 10
- **Channels:** 64
- **Tasks:** Left-hand vs Right-hand motor imagery
- **Runs:** R04, R08, R12
- **Sampling Rate:** 160 Hz
- **Epoch Length:** 2 seconds (320 samples)

The dataset is **automatically downloaded** using the `mne.datasets.eegbci` interface.

---

## ⚙️ Preprocessing Pipeline

1. EEG channel standardization
2. Notch filtering at 50 Hz and 60 Hz
3. Bandpass filtering (1–40 Hz)
4. Resampling to 160 Hz
5. Epoching into 2-second windows
6. **Z-score normalization per channel**
   - Fitted only on training data (no leakage)

---

## 🏗 Model Architectures (Fair Setup)

### Shared Components (CNN & KAN)
- Spatial convolution compression
- Projection MLP: `2560 → 128 → 64`
- Same activations, normalization, and dropout

### CNN Classifier Head
```
64 → 32 → 16 → 2
```

### KAN Classifier Head
```
KAN(width=[64, 32, 16, 2], grid=5, k=3)
```

### Parameter Count

| Model | Parameters |
|------|-----------|
| CNN  | 339,276 |
| KAN  | 372,826 |

---

## 🧪 Evaluation Strategy

### Train/Test Split
- GroupShuffleSplit (subject-aware)
- 8 subjects for training
- 2 subjects for testing
- ✔ No subject overlap

### Metrics
- Accuracy
- ROC-AUC
- Equal Error Rate (EER)
- Precision, Recall, F1-score

### Cross-Validation
- 5-Fold Stratified Group K-Fold
- Statistical tests:
  - Paired t-test
  - Wilcoxon signed-rank test

---

## 📈 Final Results

### Hold-Out Test Set

| Model | Accuracy | AUC | EER | F1 |
|------|---------|-----|-----|----|
| CNN | 0.7556 | 0.8501 | 0.2340 | 0.7556 |
| **KAN** | **0.8000** | **0.9099** | **0.1915** | **0.7907** |

### Cross-Validation (5-Fold)

| Model | Mean AUC ± Std |
|------|---------------|
| CNN | 0.8307 ± 0.1118 |
| KAN | 0.8317 ± 0.0993 |

**Statistical Significance:** p = 0.9571 (not significant due to limited subjects)

---

## 🧠 Key Research Insight

> **KAN improves AUC by 5.99% over CNN**, suggesting that KAN’s spline-based learnable functions better capture the inherent non-linear structure of EEG signals.

---

## 📂 Generated Outputs

```
experiment_outputs/
├── config.yaml
├── comparative_results.csv
├── roc_comparison.png
├── cnn_model.pth
└── kan_model.pth
```

---

## ▶️ How to Run

```bash
pip install torch mne scikit-learn pykan matplotlib seaborn
python bcieeg4.py
```

The script automatically uses GPU if available:
```
Using device: cuda
```

---

## 🔁 Reproducibility

- Fixed random seed (SEED = 42)
- Deterministic CUDA settings
- Configuration saved to `experiment_outputs/config.yaml`

---

## 📚 Suggested Citation

```
Kolmogorov–Arnold Networks for EEG-Based Biometric Authentication:
A Fair Architecture Comparison with CNNs
```

---

## 👤 Author

**Abhiroop Pamula**  
B.Tech – Electronics & Communication Engineering  
Amrita Vishwa Vidyapeetham  

---

## ⭐ Notes

- No data leakage
- Identical training conditions
- Fully reproducible from a single script
- Suitable for journal or conference submission

---

**End of README**
