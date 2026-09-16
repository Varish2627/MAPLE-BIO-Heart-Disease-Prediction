# Heart Disease Detection

This project implements a machine learning pipeline for heart disease prediction using multiple datasets, including a cardiovascular dataset and the Cleveland heart disease dataset. The workflow covers preprocessing, feature engineering, outlier handling, class balancing, model training, and result visualization.

## Overview

The repository contains:
- `Main.py` — main processing pipeline for the cardiovascular dataset
- `cleveland.py` — equivalent pipeline for the Cleveland dataset
- `Graph.py` — visualization script for plotting charts and model-performance outputs
- `metrics` — metrics and result metadata used by the graph generation logic
- `Datasets/` — source CSV datasets
- `results/` — generated plots, confusion matrices, and evaluation visuals

## Datasets

The project uses these datasets from the `Datasets/` folder:
- `Cardiovascular_Disease_Dataset.csv`
- `heart_cleveland_upload.csv`

## Requirements

Install the required Python libraries before running the project:

```bash
pip install numpy pandas tqdm matplotlib scikit-learn seaborn torch torchvision torchaudio
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/Varish2627/Heart-Disease-Detection.git
cd Heart-Disease-Detection
```

2. Install dependencies:

```bash
pip install numpy pandas tqdm matplotlib scikit-learn seaborn torch torchvision torchaudio
```

3. Run the main cardiovascular pipeline:

```bash
python Main.py
```

The scripts will preprocess the data, train the models, and save the generated outputs into the `results/` directory.

## Project Structure

```text
Heart-Disease-Detection/
├── Main.py
├── cleveland.py
├── Graph.py
├── metrics/
├── Datasets/
├── results/
├── README.md
├── .gitignore
└── Instructions.docx
```

## Outputs

The project produces:
- confusion matrices
- model performance charts
- feature importance plots
- age distribution and correlation plots
- normalization and attention-map visualizations
- classification metrics saved under `results/`

## Notes

This project focuses on experimental biomedical ML workflows and is designed to help analyze risk factors associated with heart disease detection using tabular clinical datasets.