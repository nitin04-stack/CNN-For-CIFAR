# CNN Image Classifier — CIFAR-10
Built a Convolutional Neural Network from scratch using PyTorch to classify images into 10 different categories. 
Trained on the CIFAR-10 benchmark dataset using a T4 GPU on Google Colab and achieved **76.03% test accuracy** in 10 epochs.

## What is CIFAR-10?
CIFAR-10 is one of the most well-known datasets in computer vision. It contains 60,000 color images (32×32 pixels) split across 10 classes:
`airplane` `automobile` `bird` `cat` `deer` `dog` `frog` `horse` `ship` `truck`
- 50,000 images for training
- 10,000 images for testing

## Model Architecture
Three convolutional blocks for feature extraction, followed by two fully connected layers for classification.
```
Input Image (3 × 32 × 32 — RGB)
        ↓
Conv2d(3 → 32, 3×3, pad=1) → ReLU → MaxPool2d(2×2)   →  32 × 16 × 16
        ↓
Conv2d(32 → 64, 3×3, pad=1) → ReLU → MaxPool2d(2×2)  →  64 × 8 × 8
        ↓
Conv2d(64 → 128, 3×3, pad=1) → ReLU → MaxPool2d(2×2) →  128 × 4 × 4
        ↓
Flatten (2048)
        ↓
Linear(2048 → 256) → ReLU
        ↓
Linear(256 → 10)   ← raw logits fed into CrossEntropyLoss
```
Each conv block doubles the number of filters while MaxPooling halves the spatial dimensions — a standard pattern that forces the network to learn increasingly abstract features as depth increases.

## Training Details

| Setting        | Value             |
|----------------|-------------------|
| Epochs         | 10                |
| Batch size     | 64                |
| Loss function  | CrossEntropyLoss  |
| Optimizer      | Adam (default lr) |
| Hardware       | NVIDIA T4 GPU     |
| Normalization  | Mean & Std = 0.5  |

Training and validation loss were tracked every epoch, and the best model weights were saved automatically based on the lowest validation loss.

**Loss over training:**

| Epoch | Train Loss |
|-------|------------|
| 1     | 1.3817     |
| 3     | 0.7567     |
| 5     | 0.5236     |
| 7     | 0.3489     |
| 10    | 0.1654     |

## Result

```
Test Accuracy: 76.03%
```
No data augmentation, no dropout, no batch normalization — just a clean baseline CNN.

## How to Run
The notebook is set up for Google Colab with Google Drive mounted. Just open it in Colab and it handles everything.
1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Mount your Google Drive when prompted
4. Run all cells — CIFAR-10 downloads automatically
To run locally instead:
```bash
pip install torch torchvision
jupyter notebook CNN_FOR_CIFAR.ipynb
```
Just update the dataset `root` path from the Google Drive path to a local folder like `./data`.

## Dependencies
- Python 3.x
- PyTorch
- torchvision

## What's Next
This is a clean baseline with no regularization techniques applied. Obvious next steps to push accuracy higher:
- Add **Batch Normalization** after each conv layer
- Add **Dropout** in the fully connected layers
- Apply **data augmentation** (random crops, horizontal flips)
- Experiment with **learning rate schedulers**
These changes alone can push a similar architecture past 85%+.
