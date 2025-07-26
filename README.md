# Pneumonia Prediction using CNN (ResNet18 + PyTorch)

This project implements a Convolutional Neural Network (CNN) using **PyTorch** to classify chest X-ray images as either **Pneumonia** or **Normal**. It leverages **transfer learning** with the **ResNet18** architecture and includes data preprocessing, augmentation, and model evaluation strategies.

---

## Dataset

- **Name**: Chest X-Ray Images (Pneumonia)  
- **Source**: Kaggle — by Paul Timothy Mooney  
- **Link**: [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Contents:
- Over **5,800 X-ray images** categorized as `Pneumonia` or `Normal`.
- Structured into `train`, `val`, and `test` directories.
- Augmentations applied: Resize, RandomHorizontalFlip, RandomRotation.

---

## Model Architecture

- Based on **ResNet18** from `torchvision.models`
- **Transfer Learning**: Feature extractor layers frozen
- Custom classification head for binary classification

### Techniques Used:
- Dropout layers to prevent overfitting
- `CrossEntropyLoss` as the loss function
- Adam optimizer
- Learning rate scheduler
- Early stopping mechanism

---

## Results

| Metric              | Score              |
|---------------------|--------------------|
| Validation Accuracy | ~89%               |
| Test Accuracy       | ~78%               |
| Training/Val Loss   | Decreased steadily |

- Training was stable, with regularization and scheduling helping performance.
- GPU acceleration used to handle large image sizes efficiently.

---

## Challenges and Solutions

| Challenge            | Solution                                           |
|----------------------|----------------------------------------------------|
| Overfitting          | Data augmentation, Dropout, Early Stopping         |
| Limited Resources    | Batch size tuning, Learning rate scheduling        |
| No Internet Access   | Used untrained weights or uploaded pretrained ones |

---

## Key Takeaways

- Transfer learning significantly boosts performance for medical image classification tasks.
- Data augmentation and early stopping are critical when working with small datasets.
- Proper tuning of hyperparameters and regularization ensures generalization.

---

## File Structure
```bash
├── cnn-using-pytorch.ipynb # Main training and evaluation notebook
├── Pneumonia prediction using CNN.pdf # Project report with architecture & results
├── README.md # Project description and documentation
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- scikit-learn

### How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/pneumonia-prediction-cnn.git
cd pneumonia-prediction-cnn

# Open the notebook
jupyter notebook cnn-using-pytorch.ipynb
```

## Requirements

Install with:

```bash
pip install -r requirements.txt
```

## Author
Pandidharan.G.R
