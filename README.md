# 🎯 Predicting 30-Day Readmissions in Diabetic Patients Using Ensemble Learning with AutoGluon

This repository presents an end-to-end pipeline for predicting 30-day hospital readmissions among diabetic patients using modern AutoML techniques. The study leverages [AutoGluon](https://github.com/autogluon/autogluon), a state-of-the-art ensemble-based AutoML framework, and evaluates it against traditional machine learning, deep learning, and transformer-based tabular models. The results highlight the performance advantages and clinical interpretability of ensemble learning in healthcare applications.

---

## 🧠 Overview

Hospital readmissions within 30 days are a major quality metric and financial burden, particularly for diabetic patients. This project builds and evaluates predictive models to assess readmission risk using structured clinical data from electronic health records (EHRs). The core contributions of this work include:

- Development of an AutoML pipeline based on AutoGluon  
- Comparison of ensemble methods with traditional ML, DL, and foundation models  
- Exploration of preprocessing techniques, feature importance, and subgroup performance  
 
The results consistently show that ensemble learning via AutoGluon outperforms other models, with LightGBM and CatBoost being strong individual contenders. Deep neural networks and transformer-based models (e.g., TabPFNMix) are competitive but underperform in this static tabular setting.

📄 Click [here](MBP1413_Readmission_Predictions.pdf) to access the full paper.

---

## 📊 Dataset

This project uses the publicly available dataset from the UCI Machine Learning Repository:  
[Diabetes 130-US hospitals for years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

---

## 📁 Repository Structure

```
.
├── Config.py            # Global configuration and constants
├── Prep.py              # Data cleaning, preprocessing, clustering
├── Train_model.py       # Training logic using AutoGluon
├── Utils.py             # Utility functions
├── Vis.py               # Visualizations (e.g., SHAP, performance plots)
├── train.ipynb          # Interactive notebook for training and evaluation
├── ag.yaml              # Conda environment file
├── README.md            # This file
```

---

## ⚙️ Installation

We recommend using a Conda environment for reproducibility:

```bash
conda env create -f ag.yaml
conda activate ag
```

---

## 🚀 Usage

You can run the full workflow interactively inside the Jupyter notebook:

1. **Launch the notebook**:
   ```bash
   jupyter notebook train.ipynb
   ```

2. **Run the full pipeline**:
   ```python
   # Inside train.ipynb
   from Train_model import TrainAutoGluon
   trainer = TrainAutoGluon(...)
   trainer.run_pipeline()
   ```

3. **Visualize results**:
   ```python
   from Vis import plot_feature_importance, shap_summary_plot, ...
   ```

This approach allows for step-by-step inspection, debugging, and comparison.

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{yuan2025readmission,
  title={Predicting 30-Day Readmissions in Diabetic Patients Using Ensemble Learning with AutoGluon},
  author={Yuan, Baijiang},
  year={2025},
  note={University of Toronto, Institute of Medical Science and University Health Network}
}
```

---

