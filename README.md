# Industrial Surface Defect Classification

## Overview

This project implements an end-to-end image classification pipeline designed for detecting and categorizing synthetic industrial surface defects. The system processes image inputs to accurately identify structural and surface anomalies across five predefined categories: crack, hole, normal, rust, and scratch.

## Approach

- **Model**: Transfer learning utilizing a pretrained ResNet18 architecture.
- **Training Setup**: Optimized using Cross-Entropy Loss and an Adam optimizer, with configurable hyperparameters for epochs and batch size execution.
- **Data Handling**: Streamlined ingestion pipeline leveraging `torchvision.datasets.ImageFolder` for efficient batch loading, automatic label encoding based on the directory structure, and standardized tensor transformations.

## Results

- **Final Accuracy**: 100% on the evaluation dataset.
- **Confusion Matrix**: Evaluation includes a generated confusion matrix demonstrating optimal class-wise performance with zero misclassifications.
- **Important Note**: The dataset utilized for this evaluation is purely synthetic. Consequently, while the model achieves perfect accuracy within this distribution, these results may not generalize to real-world, highly complex industrial environments.
  The dataset is synthetic and visually well-separated, which makes this level of performance plausible. In real-world scenarios, further validation would be required to ensure generalization.

## Project Structure

```text
.
├── data/
│   ├── train/
│   │   ├── crack/
│   │   ├── hole/
│   │   ├── normal/
│   │   ├── rust/
│   │   └── scratch/
│   └── test/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── report.py
├── requirements.txt
└── README.md
```

## How to Run

Execute the following commands sequentially from the project root to run the pipeline:

1. **Install requirements**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model**:

   ```bash
   python -m src/train.py
   ```

3. **Evaluate performance**:

   ```bash
   python -m src/evaluate.py
   ```

4. **Generate predictions**:

   ```bash
   python -m src/predict.py
   ```

5. **Generate report**:
   ```bash
   python -m src/generate_report.py
   ```

## Deliverables

- System Source Code
- `predictions.csv`
- `confusion_matrix.png`
- `results.html`

## Assumptions

- **Dataset Structure**: The data is strictly organized in an `ImageFolder`-compatible directory hierarchy layout.
- **Metadata**: Additional image metadata is considered optional and is not required for standard pipeline execution.
- **Synthetic Dataset Characteristics**: The synthetic nature of the dataset restricts complex real-world variance, which directly contributes to the baseline maximum accuracy observed.
