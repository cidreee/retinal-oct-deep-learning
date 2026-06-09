# Retinal OCT Disease Classification with Deep Learning

This repository contains a Deep Learning final project comparing convolutional neural networks and a Vision Transformer for multiclass retinal Optical Coherence Tomography (OCT) disease classification using a Kermany2018-derived dataset.

The task is to classify retinal OCT images into four classes:

- CNV
- DME
- DRUSEN
- NORMAL

The project evaluates three main architectures:

1. ResNet-50
2. EfficientNet-B3
3. Vision Transformer Small (ViT-Small)

Each architecture was evaluated in two settings:

- Standard cross-entropy loss
- Weighted cross-entropy loss to address class imbalance

## Project Objective

The main objective is to compare predictive performance, generalization capacity, computational cost, and interpretability of different deep learning architectures applied to retinal OCT disease classification.

Special attention is given to class imbalance, since CNV and NORMAL have more samples than DME and DRUSEN. For this reason, Macro F1-score and per-class recall are emphasized in addition to accuracy.

## Dataset

The experiments use the following Kaggle dataset:

**Retinal OCT Images Splitted**  
https://www.kaggle.com/datasets/fabrizioravelli/retinal-oct-images-splitted

This dataset is based on the Kermany2018 retinal OCT dataset and is already divided into training, validation, and test sets.

Expected dataset structure in Kaggle:

```text
/kaggle/input/datasets/fabrizioravelli/retinal-oct-images-splitted/
├── training/
│   ├── train/
│   │   ├── CNV/
│   │   ├── DME/
│   │   ├── DRUSEN/
│   │   └── NORMAL/
│   └── val/
│       ├── CNV/
│       ├── DME/
│       ├── DRUSEN/
│       └── NORMAL/
└── test/
    ├── CNV/
    ├── DME/
    ├── DRUSEN/
    └── NORMAL/
```

## Repository Structure

```text
retinal-oct-deep-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_exploracion_kermany2018.ipynb
│   ├── 02_resnet50_kermany2018.ipynb
│   ├── 03_efficientnetb3_kermany2018.ipynb
│   ├── 04_vit_kermany2018.ipynb
│   ├── 05_comparacion_resultados.ipynb
│   ├── 06_resnet50_weighted_kermany2018.ipynb
│   ├── 07_efficientnetb3_weighted_kermany2018.ipynb
│   └── 08_vit_weighted_kermany2018.ipynb
└── results

```

## Notebooks

| Notebook | Description |
|---|---|
| `01_exploracion_kermany2018.ipynb` | Dataset exploration, class distribution, sample visualization |
| `02_resnet50_kermany2018.ipynb` | ResNet-50 baseline model |
| `03_efficientnetb3_kermany2018.ipynb` | EfficientNet-B3 baseline model |
| `04_vit_kermany2018.ipynb` | Vision Transformer baseline model |
| `05_comparacion_resultados.ipynb` | Comparative analysis of baseline models |
| `06_resnet50_weighted_kermany2018.ipynb` | ResNet-50 with weighted cross-entropy |
| `07_efficientnetb3_weighted_kermany2018.ipynb` | EfficientNet-B3 with weighted cross-entropy |
| `08_vit_weighted_kermany2018.ipynb` | Vision Transformer with weighted cross-entropy |

## Models Compared

### ResNet-50

ResNet-50 was used as a CNN baseline with ImageNet pretrained weights. The backbone was frozen and used as a feature extractor. A new classification head was trained for the four OCT classes.

### EfficientNet-B3

EfficientNet-B3 was selected because of its compound scaling strategy, which balances depth, width, and resolution. Like ResNet-50, the pretrained backbone was frozen and used as a feature extractor.

### Vision Transformer Small

ViT-Small was implemented using PyTorch and the `timm` library. Unlike the CNN models, ViT-Small was fine-tuned end-to-end, allowing the transformer representation to adapt more fully to the OCT domain.

## Class Imbalance Strategy

The dataset presents class imbalance, with CNV and NORMAL being more frequent than DME and DRUSEN. To address this, a second experimental phase used weighted cross-entropy.

For TensorFlow/Keras models, class weights were passed to the `class_weight` argument in `model.fit()`.

For the PyTorch ViT model, class weights were passed directly to:

```python
nn.CrossEntropyLoss(weight=class_weights_tensor)
```

## Evaluation Metrics

The following metrics were used:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted F1-score
- Per-class precision, recall, and F1-score
- Confusion matrix
- Training time
- Number of parameters
- Grad-CAM visualizations for CNN models

Macro F1-score was treated as the main metric because it gives equal importance to all classes and is more appropriate for imbalanced medical imaging datasets.

## Main Results

The best-performing model was:

**ViT-Small + Weighted Cross-Entropy**

Approximate test results:

| Model | Accuracy | Macro Recall | Macro F1 | DRUSEN Recall |
|---|---:|---:|---:|---:|
| ResNet-50 | 0.846 | 0.697 | 0.725 | 0.232 |
| ResNet-50 + WCE | 0.865 | 0.792 | 0.802 | 0.573 |
| EfficientNet-B3 | 0.874 | 0.790 | 0.810 | 0.513 |
| EfficientNet-B3 + WCE | 0.838 | 0.827 | 0.794 | 0.784 |
| ViT-Small | 0.966 | 0.951 | 0.952 | 0.890 |
| ViT-Small + WCE | 0.968 | 0.958 | 0.954 | 0.916 |

## Key Findings

- ViT-Small with weighted cross-entropy achieved the best overall performance.
- Weighted cross-entropy improved recall for minority classes, especially DRUSEN.
- The effect of weighted loss was architecture-dependent.
- EfficientNet-B3 with weighted loss increased DRUSEN recall but introduced more false positives.
- Grad-CAM showed that CNN models focused on retinal layer regions rather than empty background.
- Macro F1-score and per-class recall were more informative than accuracy alone.

## How to Reproduce

### Option 1: Kaggle

This project was developed primarily in Kaggle Notebooks.

1. Create a new Kaggle Notebook.
2. Add the dataset:

   ```text
   fabrizioravelli/retinal-oct-images-splitted
   ```

3. Enable GPU acceleration.
4. Upload or copy the notebooks from the `notebooks/` folder.
5. Run the notebooks in order.

Recommended execution order:

```text
01_exploracion_kermany2018.ipynb
02_resnet50_kermany2018.ipynb
03_efficientnetb3_kermany2018.ipynb
04_vit_kermany2018.ipynb
05_comparacion_resultados.ipynb
06_resnet50_weighted_kermany2018.ipynb
07_efficientnetb3_weighted_kermany2018.ipynb
08_vit_weighted_kermany2018.ipynb
```

### Option 2: Local Environment

A local environment is possible, but a GPU is strongly recommended.

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
# or
venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

Then open the notebooks using Jupyter:

```bash
jupyter lab
```

## Hardware Notes

The experiments were designed for GPU execution. ViT-Small is the most computationally expensive model and may require a smaller batch size than CNN models.

Typical settings:

| Model | Batch Size | Learning Rate |
|---|---:|---:|
| ResNet-50 | 32 | 1e-4 |
| EfficientNet-B3 | 32 | 1e-4 |
| ViT-Small | 16 | 2e-5 |

## Limitations

- ResNet-50 and EfficientNet-B3 were used as frozen feature extractors, while ViT-Small was fine-tuned end-to-end.
- External validation on another OCT dataset was not performed.
- ViT attention rollout was proposed as future work but not fully implemented.
- Repeated experiments with confidence intervals were not included due to computational constraints.

## Future Work

- Fine-tune the upper layers of ResNet-50 and EfficientNet-B3.
- Add ViT attention rollout visualizations.
- Evaluate the models on an external OCT dataset.
- Explore ensemble models combining CNN and transformer predictions.
- Add confidence intervals through repeated runs or bootstrapping.

## References

1. Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell, 172(5), 1122–1131.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
3. Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML.
4. Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR.
5. Wightman, R. (2019). *PyTorch Image Models (timm)*.
