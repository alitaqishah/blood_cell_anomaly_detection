# Blood Cell Anomaly Detection — CytoDiffusion Benchmark

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end data science project analysing **5,880 blood cell records** across **19 cell types** (7 normal, 12 abnormal) to detect morphological anomalies, classify cell types, and benchmark against the **CytoDiffusion** model published in *Nature Machine Intelligence (2025)*.

This repository demonstrates a complete data analyst / data scientist workflow from raw data loading, cleaning, and exploratory analysis to machine learning, model evaluation, interpretability, and clinical insight generation.

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Repository Structure](#-repository-structure)
4. [Notebook Walkthrough](#-notebook-walkthrough)
5. [Key Findings & Insights](#-key-findings--insights)
6. [Model Performance](#-model-performance)
7. [Feature Importance](#-feature-importance)
8. [Benchmark Comparison](#-benchmark-comparison)
9. [Clinical Implications](#-clinical-implications)
10. [Skills Demonstrated](#-skills-demonstrated)
11. [How to Run](#-how-to-run)
12. [Requirements](#-requirements)
13. [References](#-references)
14. [Connect with Me](#-connect-with-me)

---

## Project Overview

Blood cell morphology analysis is critical for diagnosing haematological diseases such as leukemia, anemia, and infections. Traditional manual microscopy is time-consuming and subject to inter-observer variability.

This project explores the **CytoDiffusion Benchmark Dataset** to:

- Perform **exploratory data analysis (EDA)** on morphological, colorimetric, and clinical features.
- Build **binary classification** models (Normal vs. Anomaly) using Random Forest, Gradient Boosting, and Logistic Regression.
- Build a **multi-class classifier** for 19 distinct cell types.
- Analyse **CytoDiffusion anomaly scores** and compare with published benchmarks.
- Extract **feature importance** to identify clinically interpretable signals.
- Provide **actionable clinical insights** for real-world deployment.

The notebook is designed as a portfolio piece for **Data Analyst / Data Scientist** roles, showcasing strong data manipulation, statistical analysis, machine learning, and storytelling skills.

---

## Dataset Description

| File | Description |
|------|-------------|
| `blood_cell_anomaly_detection.csv` | Main dataset — 5,880 cells × 36 features |
| `cell_type_reference.csv` | Clinical reference for each cell type (19 rows) |
| `cytodiffusion_benchmark_scores.csv` | Published benchmark metrics from the CytoDiffusion paper |

### Main Dataset Features (36 columns)

- **Identifiers**: `cell_id`
- **Targets**: `cell_type`, `anomaly_label` (0 = Normal, 1 = Anomaly), `disease_category`
- **Morphological**: `cell_diameter_um`, `nucleus_area_pct`, `chromatin_density`, `cytoplasm_ratio`, `circularity`, `eccentricity`, `granularity_score`, `lobularity_score`, `membrane_smoothness`, `cell_area_px`, `perimeter_px`
- **Colorimetric**: `mean_r`, `mean_g`, `mean_b`, `stain_intensity`
- **Clinical**: `patient_age_group`, `patient_sex`, `wbc_count_per_ul`, `rbc_count_millions_per_ul`, `hemoglobin_g_dl`, `hematocrit_pct`, `platelet_count_per_ul`, `mcv_fl`, `mchc_g_dl`
- **Technical**: `dataset_source`, `staining_protocol`, `microscope_model`, `magnification_x`, `image_resolution_px`
- **CytoDiffusion Scores**: `cytodiffusion_anomaly_score`, `cytodiffusion_classification_confidence`, `labeller_confidence_score`

### Cell Types & Disease Categories

| Category | Cell Types | Count |
|----------|------------|-------|
| **Normal WBC** | Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil | 2,800 |
| **Normal RBC** | Normal_RBC | 900 |
| **Normal Platelet** | Platelet | 300 |
| **Leukemia** | Blast_Cell, Prolymphocyte | 460 |
| **Anemia** | Elliptocyte, Schistocyte, Spherocyte, Target_Cell | 650 |
| **Sickle Cell Anemia** | Sickle_Cell | 140 |
| **Infection** | Hypersegmented_Neutrophil, Toxic_Granulation, Reactive_Lymphocyte | 450 |
| **Artefact** | Smudge_Cell, Artefact | 180 |

---

## 📁 Repository Structure

```text
blood-cell-anomaly-detection/
│
├── data/
│   ├── blood_cell_anomaly_detection.csv
│   ├── cell_type_reference.csv
│   └── cytodiffusion_benchmark_scores.csv
│
├── notebooks/
│   └── blood_cell_anomaly_detection_kaggle.ipynb
│
├── README.md
└── requirements.txt
```

---

## Notebook Walkthrough

The Jupyter notebook `blood_cell_anomaly_detection_kaggle.ipynb` is divided into 10 sections:

1. **Setup & Data Loading** – Import libraries, set visual style, load CSV files.
2. **Data Overview** – Shape, data types, null checks, descriptive statistics.
3. **EDA — Cell Type & Disease Distribution** – Bar charts, pie charts, cross-tabulation with patient age group.
4. **Morphological Feature Analysis** – Histograms of key features split by anomaly status, correlation heatmap.
5. **CytoDiffusion Anomaly Score Analysis** – Score distribution and mean score per cell type.
6. **ML — Binary Classification** – Train/test split, standard scaling, Random Forest, Gradient Boosting, Logistic Regression; ROC curves, confusion matrix, classification report.
7. **ML — Multi-class Cell Type Classification** – Label encoding, Random Forest classifier, confusion matrix, per-class metrics.
8. **Feature Importance & Explainability** – Random Forest feature importances, median split visualisation.
9. **Benchmark Comparison** – Comparison with CytoDiffusion and baseline models from the paper.
10. **Conclusions & Clinical Insights** – Summary of findings and clinical relevance.

---

## Key Findings & Insights

- **Blast cells are the most critical anomaly** — sensitivity must be ≥ 0.90 to avoid missed leukemia diagnoses.
- **Eccentricity is the strongest RBC anomaly signal** — sickle cells (0.92) and elliptocytes (0.78) far exceed normal RBC (0.18).
- **Lobularity separates neutrophil anomalies** — hypersegmented neutrophils (5.8 lobes) vs normal (3.5) indicate B12/folate deficiency.
- **Chromatin density drops in blasts** — blast cells show low density (0.35) compared to mature WBCs (0.55–0.80).
- **CytoDiffusion anomaly score reliably separates normal from abnormal** (paper AUC = 0.990).
- **10 expert haematologists could not distinguish synthetic from real cells** (accuracy = 0.523 ≈ random).

---

## Model Performance

Binary classification (Normal vs. Anomaly) results on the 20% hold-out test set:

| Model | AUC | Accuracy | Precision (Anomaly) | Recall (Anomaly) |
|-------|-----|----------|---------------------|------------------|
| Random Forest | **0.998** | **0.986** | 0.982 | 0.981 |
| Gradient Boosting | 0.997 | 0.984 | 0.979 | 0.978 |
| Logistic Regression | 0.992 | 0.972 | 0.965 | 0.961 |

Multi-class cell type classification (19 classes) using Random Forest:

- **Accuracy: 0.943**
- Macro-average F1-score: **0.928**

> These results are highly competitive and demonstrate the power of morphological features for automated blood cell analysis.

---

## Feature Importance

Top 10 most important features for anomaly detection (Random Forest):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `eccentricity` | 0.142 |
| 2 | `lobularity_score` | 0.128 |
| 3 | `chromatin_density` | 0.115 |
| 4 | `cell_diameter_um` | 0.098 |
| 5 | `nucleus_area_pct` | 0.091 |
| 6 | `circularity` | 0.087 |
| 7 | `granularity_score` | 0.079 |
| 8 | `cytoplasm_ratio` | 0.071 |
| 9 | `membrane_smoothness` | 0.064 |
| 10 | `stain_intensity` | 0.058 |

These features are clinically interpretable and align with known morphological changes in blood disorders.

---

## Benchmark Comparison

Published results from the CytoDiffusion paper (*Nature Machine Intelligence 2025*):

| Task | CytoDiffusion | Baseline |
|------|---------------|----------|
| Anomaly Detection (AUC) | **0.990** | 0.916 (ViT) |
| Domain Shift Accuracy | **0.854** | 0.738 (CNN) |
| Low-Data Balanced Accuracy | **0.962** | 0.924 (CNN) |
| Blast Detection Sensitivity | **0.905** | 0.281 (ViT) |
| Synthetic Image Authenticity (Human) | 0.523 | — |

Our Random Forest model achieves an **AUC of 0.998** for anomaly detection on this dataset, surpassing even the CytoDiffusion benchmark — highlighting the richness of the provided features.

---

## Clinical Implications

- **Generative models** like CytoDiffusion can detect rare, unseen anomalies by learning the full morphological distribution.
- **Interpretable features** (eccentricity, lobularity, chromatin density) allow clinicians to trust and validate model decisions.
- **Domain shift robustness** is essential for real-world deployment across different labs, microscopes, and staining protocols.
- **Automated screening** can prioritise urgent cases (e.g., blast cells) and reduce manual microscopy workload.

---

## Skills Demonstrated

- **Data Wrangling** – Loading, inspecting, and cleaning multi-feature datasets with pandas.
- **Exploratory Data Analysis** – Distribution analysis, correlation, cross-tabulation, visual storytelling.
- **Feature Engineering** – Selection of relevant morphological, colorimetric, and clinical features.
- **Machine Learning** – Classification (binary & multi-class), model training, hyperparameter basics, cross-validation.
- **Model Evaluation** – ROC-AUC, accuracy, precision, recall, F1-score, confusion matrices.
- **Interpretability** – Feature importance extraction and clinical mapping.
- **Benchmarking** – Comparing custom models against published research results.
- **Communication** – Translating technical findings into clinical insights.

---

## How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/blood-cell-anomaly-detection.git
   cd blood-cell-anomaly-detection
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**

   ```bash
   jupyter notebook notebooks/blood_cell_anomaly_detection_kaggle.ipynb
   ```

4. **Run all cells** – The notebook will load the data from the `data/` folder, perform EDA, train models, and display results.

---

## Requirements

```text
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
jupyter>=1.0
```

Install with:

```bash
pip install -r requirements.txt
```

---

## References

- Deltadahl et al. (2025). *Deep Generative Classification of Blood Cell Morphology.* **Nature Machine Intelligence.** DOI: [10.1038/s42256-025-01122-7](https://doi.org/10.1038/s42256-025-01122-7)
- Naouali & El Othmani (2025). *AI-Driven Automated Blood Cell Anomaly Detection.* **Journal of Imaging, 11(5), 157.**
- GitHub: [https://github.com/Deltadahl/CytoDiffusion](https://github.com/Deltadahl/CytoDiffusion)

---

## Connect with Me

I am actively seeking **Data Analyst / Data Scientist** opportunities. Feel free to reach out!

- **LinkedIn**: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)
- **GitHub**: [github.com/your-username](https://github.com/your-username)
- **Email**: your.email@example.com

---

*This project is part of my data science portfolio. If you find it useful, please ⭐ the repository!*