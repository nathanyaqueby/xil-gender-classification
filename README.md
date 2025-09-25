# Explanatory Interactive Learning for Bias Mitigation in Gender Classification

This repository contains the complete implementation of the XIL (Explanatory Interactive Learning) framework for bias mitigation in visual gender classification, as described in the research paper.

## Overview

We propose a comprehensive framework that combines multiple state-of-the-art techniques for bias-aware machine learning:

### 🔬 **Core XIL Components**
- **CAIPI (Counterfactual Augmentation)** - Data augmentation with counterfactual transformations
- **BLA (Bounded Logit Attention)** - Self-explaining neural networks with attention mechanisms  
- **RRR (Right-for-Right-Reasons)** - Regularization using explanation-guided training
- **Hybrid Training** - Combined CAIPI + RRR approach for enhanced bias mitigation

### 📊 **Bias Evaluation Metrics**
- **FFP (Foreground Focus Proportion)** - Measures focus on relevant foreground features
- **BFP (Background Focus Proportion)** - Quantifies attention to spurious background features  
- **BSR (Background Saliency Ratio)** - Ratio of background to foreground saliency
- **DICE score** - Measures the alignment between two masks

## Project Structure

```
explanatory_gender_classification/
├── src/
│   ├── data/
│   │   ├── dataset.py          # Gender dataset with mask support
│   │   └── preprocessing.py    # Data preprocessing utilities
│   ├── models/
│   │   ├── architectures.py    # CNN model factory
│   │   ├── cnn_models.py      # Model compatibility wrappers
│   │   ├── rrr_model.py       # Right-for-Right-Reasons models
│   │   └── gender_classifier.py # Base gender classification models
│   ├── explainability/
│   │   ├── gradcam.py         # GradCAM explanations
│   │   ├── bla.py             # Bounded Logit Attention
│   │   └── lime_wrapper.py     # LIME explanations
│   ├── augmentation/
│   │   └── caipi.py           # CAIPI counterfactual augmentation
│   ├── training/
│   │   ├── trainer.py         # Base training classes
│   │   ├── rrr_trainer.py     # RRR-specific training
│   │   └── hybrid_trainer.py   # CAIPI + RRR hybrid training
│   ├── evaluation/
│   │   ├── bias_metrics.py    # FFP, BFP, BSR, DICE metrics
│   │   └── evaluator.py       # Model evaluation pipeline
│   └── utils/
│       ├── helpers.py         # Utility functions
│       └── settings.py        # Configuration constants
├── scripts/
│   ├── train_baseline.py      # Train baseline CNN models
│   ├── train_caipi.py         # Train CAIPI-augmented models
│   ├── train_rrr.py           # Train RRR models
│   ├── train_hybrid.py        # Train hybrid XIL models
│   ├── run_all_experiments.py # Complete experimental suite
│   ├── evaluate_models.py     # Model evaluation
│   └── generate_explanations.py # Generate visual explanations
├── experiments/               # Experimental results and analysis
├── tests/                    # Unit tests
└── notebooks/               # Jupyter notebooks for analysis
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd explanatory_gender_classification

# Install dependencies
pip install -r requirements.txt
```

### Test Installation

```bash
# Test all XIL components
python test_implementation.py
```

Expected output:
```
✓ All imports successful!
✓ BLA Model: PASSED
✓ CAIPI Augmentation: PASSED  
✓ Bias Metrics: PASSED
✓ Hybrid Training: PASSED
🎉 All tests passed! The XIL implementation is ready.
```

## Data Setup

This project expects your gender classification dataset to be organized in one of two ways:

### Option 1: Gender Dataset Folder (Recommended)
For the complete structured dataset with pre-splits and masks:
```
gender_dataset/
├── dataset_split/
│   ├── train_set.csv
│   ├── val_set.csv
│   └── test_set.csv
├── resized_female_images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── resized_male_images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── resized_female_masks/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── resized_male_masks/
    ├── image1.png
    ├── image2.png
    └── ...
```

### Option 2: Separate Directories
You can also provide separate paths to female and male image directories for simpler setups.

## 💻 Usage

### 1. Baseline CNN Training
```bash
# Train standard CNN without XIL
python scripts/train_baseline.py --gender_dataset_path ../gender_dataset --model efficientnet_b0 --epochs 20
```

### 2. CAIPI Augmentation Training  
```bash
# Train with counterfactual augmentation
python scripts/train_caipi.py --data_dir ../gender_dataset --k 3 --sampling uncertainty --explainer gradcam
```

### 3. RRR (Right-for-Right-Reasons) Training
```bash
# Train with explanation-guided regularization
python scripts/train_rrr.py --gender_dataset_path ../gender_dataset --model efficientnet_b0 --l2_grads 1000
```

### 4. Hybrid XIL Training (CAIPI + RRR)
```bash
# Train with combined CAIPI and RRR approach  
python scripts/train_hybrid.py --data_dir ../gender_dataset --k 3 --sampling uncertainty --explainer gradcam --rrr_lambda 10.0
```

### 5. Complete Experimental Suite (28 Experiments)
```bash
# Run all experiments from the paper
python scripts/run_all_experiments.py --data_dir ../gender_dataset
```

This runs:
- **6 Baseline experiments** (6 architectures)
- **6 CAIPI experiments** (uncertainty/confident × k={1,3,5})  
- **4 RRR experiments** (uncertainty/confident × GradCAM/BLA)
- **12 Hybrid experiments** (uncertainty/confident × k={1,3,5} × GradCAM/BLA)

### 6. Model Evaluation & Analysis
```bash
# Evaluate trained models with bias metrics
python scripts/evaluate_models.py --model_path experiments/hybrid/best_model.pt

# Generate visual explanations
python scripts/generate_explanations.py --method gradcam --model_path experiments/hybrid/best_model.pt
```

## ⚙️ Configuration Options

### Hybrid Training Parameters
```bash
--k                    # CAIPI counterexamples per sample (1, 3, 5)
--sampling            # Strategy: 'uncertainty' or 'high_confidence' 
--explainer           # Method: 'gradcam' or 'bla'
```

## 🔧 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Dataset Path**: Use absolute paths or correct relative paths
3. **Memory Issues**: Reduce batch size or use CPU for large models
4. **CUDA Errors**: Install compatible PyTorch version for your GPU

### Performance Tips
- Use GPU for faster training (`--device cuda`)
- Start with smaller experiments (fewer epochs, smaller k)
- Monitor GPU memory usage during hybrid training

## 📚 Research Paper

This implementation corresponds to the research paper:

**"Explanatory Interactive Learning for Bias Mitigation in Visual Gender Classification"**

### Abstract
*Explanatory interactive learning (XIL) enables users to guide model training in machine learning (ML) by providing feedback on the model’s explanations, thereby helping it to
focus on features that are relevant to the prediction from the user’s perspective. In this study, we explore the capability of this learning paradigm to mitigate bias and spurious correlations in visual classifiers, specifically in a scenario prone to data bias, such as gender classification. We investigate two methodologically different state-of-the-art XIL strategies, i.e., CAIPI and Right for the Right Reasons (RRR), as well as a novel hybrid approach that combines both strategies. The
results are evaluated quantitatively and qualitatively through visual inspection of local explanations provided via Gradient-weighted Class Activation Mapping (GradCAM) and Bounded
Logit Attention (BLA). Experimental results demonstrate the effectiveness of these methods in (i) guiding ML models to focus on relevant image features, particularly when CAIPI is used,
and (ii) reducing model bias (i.e., balancing the misclassification rates between male and female predictions). Our analysis further
supports the potential of XIL methods to improve fairness in gender classifiers. Overall, the increased transparency and fairness obtained by XIL leads to slight performance decreases
with an exception being CAIPI, which shows potential to even improving classification accuracy.*