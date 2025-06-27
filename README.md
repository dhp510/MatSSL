# MatSSL: Robust Self-Supervised Representation Learning for Metallographic Image Segmentation

A framework for Self-Supervised Learning (SSL) in materials science image analysis, including pre-training, fine-tuning, and evaluation of segmentation models.

## Overview

MatSSL is a comprehensive framework that enables:

1. **SSL Pre-training**: Train self-supervised models (MatSSL, DenseCL, MoCoV2) using unlabeled material science image data
2. **Fine-tuning**: Use pre-trained SSL models as encoders for segmentation tasks with labeled data
3. **Evaluation**: Quantitatively evaluate model performance with metrics like IoU, precision, and recall

## Requirements

python 3.8 or higher is required to run this project.

```
pip install -r requirements.txt
```

## Model Hub

### SSL Pre-trained Models

**Table: Pre-trained SSL Models Download Links**

| SSL Model | Training Dataset | Download Link |
|-----------|------------------|--------------|
| MatSSL    | Aachen + UHCS    | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/ssl-models/matssl) |
| MatSSL    | UHCS + MetalDAM  | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/ssl-models/matssl) |
| MatSSL    | Aachen + UHCS + MetalDAM | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/ssl-models/matssl) |
| DenseCL   | Aachen + UHCS    | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/ssl-models/densecl) |
| DenseCL   | UHCS + MetalDAM  | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/ssl-models/densecl) |
| DenseCL   | Aachen + UHCS + MetalDAM | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/ssl-models/densecl) |
| MoCoV2    | Aachen + UHCS    | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/ssl-models/mocov2) |
| MoCoV2    | UHCS + MetalDAM  | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/ssl-models/mocov2) |
| MoCoV2    | Aachen + UHCS + MetalDAM | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/ssl-models/mocov2) |

### EBC Dataset Performance

### Pretraining Strategy Comparison

**Table 1: Comparison of mIoU (%) for different pretraining strategies and finetune datasets**

| Fine-tune Dataset | Pretrain | SSL Dataset | mIoU (%) | Download Link |
|-------------------|----------|-------------|----------|--------------|
| MetalDAM          | super. ImageNet | - | 66.73 | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/finetune-models/imagenet) |
|                   | DenseCL  | Aachen + UHCS | 68.76 | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/finetune-models/densecl) |
|                   | MocoV2   | Aachen + UHCS | 67.18 | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/finetune-models/mocov2) |
|                   | **MatSSL** | **Aachen + UHCS** | **69.95** | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/finetune-models/matssl) |
|                   | DenseCL  | Aachen + UHCS + MetalDAM | 68.34 | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/finetune-models/densecl) |
|                   | MocoV2   | Aachen + UHCS + MetalDAM | 68.40 | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/finetune-models/matssl) |
|                   | **MatSSL** | **Aachen + UHCS + MetalDAM** | **69.02** | [Download](https://github.com/MatSSL/models/matssl-aachen-uhcs-metaldam-finetune-on-metaldam.pth) |
| Aachen            | super. ImageNet | - | 65.59 | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/finetune-models/imagenet) |
|                   | DenseCL  | UHCS + MetalDAM | 65.82 | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/finetune-models/densecl) |
|                   | MocoV2   | UHCS + MetalDAM | 65.90 | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/finetune-models/mocov2) |
|                   | **MatSSL** | **UHCS + MetalDAM** | **65.98** | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/finetune-models/matssl) |
|                   | DenseCL  | Aachen + UHCS + MetalDAM | 65.56 | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/finetune-models/densecl) |
|                   | MocoV2   | Aachen + UHCS + MetalDAM | 65.65 | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/finetune-models/matssl) |
|                   | **MatSSL** | **Aachen + UHCS + MetalDAM** | **65.86** | [Download](https://huggingface.co/dhp510/MatSSL/tree/main/model_hub/finetune-models/matssl) |

**Table 2: Comparison of segmentation performance on NASA EBC benchmark test sets**

| Test Set | Finetuning Method | Pretraining | Average mIoU (%) | Download Link |
|----------|-------------------|-------------|------------------|--------------|
| EBC-1    | Unet++            | MicroNet    | 95.17            | [Download](https://github.com/MatSSL/models/micronet-finetune-on-ebc1.pth) |
|          | Transformer       | MicroLite   | 93.01            | [Download](https://github.com/MatSSL/models/microlite-finetune-on-ebc1.pth) |
|          | CS-UNet           | MicroNet and MicroLite | 95.98 | [Download](https://github.com/MatSSL/models/csunet-finetune-on-ebc1.pth) |
|          | **Unet++**        | **MatSSL (Aachen + UHCS + MetalDAM)** | **96.79** | [Download](https://github.com/MatSSL/models/matssl-aachen-uhcs-metaldam-finetune-on-ebc1.pth) |
| EBC-2    | Unet++            | MicroNet    | 84.60            | [Download](https://github.com/MatSSL/models/micronet-finetune-on-ebc2.pth) |
|          | Transformer       | MicroLite   | 84.30            | [Download](https://github.com/MatSSL/models/microlite-finetune-on-ebc2.pth) |
|          | CS-UNet           | MicroNet and MicroLite | 86.73 | [Download](https://github.com/MatSSL/models/csunet-finetune-on-ebc2.pth) |
|          | **Unet++**        | **MatSSL (Aachen + UHCS + MetalDAM)** | **94.70** | [Download](https://github.com/MatSSL/models/matssl-aachen-uhcs-metaldam-finetune-on-ebc2.pth) |
| EBC-3    | Unet++            | MicroNet    | 42.58            | [Download](https://github.com/MatSSL/models/micronet-finetune-on-ebc3.pth) |
|          | Transformer       | MicroLite   | 56.72            | [Download](https://github.com/MatSSL/models/microlite-finetune-on-ebc3.pth) |
|          | CS-UNet           | MicroNet and MicroLite | 45.69 | [Download](https://github.com/MatSSL/models/csunet-finetune-on-ebc3.pth) |
|          | **Unet++**        | **MatSSL (Aachen + UHCS + MetalDAM)** | **84.53** | [Download](https://github.com/MatSSL/models/matssl-aachen-uhcs-metaldam-finetune-on-ebc3.pth) |

MatSSL (trained on Aachen + UHCS + MetalDAM) achieves the best mIoU across all sets, with notable improvement on EBC-3.

## Framework Workflow

### 1. Self-Supervised Learning (SSL) Pre-training

Train an SSL model on unlabeled material science images:

```bash
python train_ssl.py --model [MODEL_TYPE] --data_path [PATH_TO_UNLABELED_DATASET] [OPTIONS]
```

All training datasets are used for training SSL will be available soon

#### Example:

```bash
# Train a MatSSL model on a dataset combining Aachen and UHCS images
python train_ssl.py --model matssl --data_path datasets/ssl_datasets/aachen-uhcs 
```

The SSL models will be saved in the `experiments` directory.

### 2. Fine-tuning for Segmentation

Fine-tune a segmentation model using the pre-trained SSL model as an encoder:

```bash
python train_finetune.py --finetune_dataset [PATH_TO_LABELED_DATASET] --ssl_weights_path [PATH_TO_SSL_WEIGHTS] [OPTIONS]
```

#### Example:

```bash
# Fine-tune using MatSSL model pre-trained on Aachen-UHCS-MetalDAM dataset for segmenting Aachen images
python train_finetune.py --finetune_dataset datasets/finetune-datasets/EBC/nasa-EBC1 --ssl_weights_path model_hub/ssl-models/matssl/matssl-aachen-uhcs-metaldam.pth --gpu_id 0
```

Fine-tuned models will be saved in the `experiments` directory.
MetalDAM and Aachen processed dataset are used for training finetuning will be available soon
The EBC dataset is available at `datasets/finetune-datasets/EBC`

### 3. Evaluation

Evaluate the fine-tuned model on test data:

```bash
python eval_model.py --model_path [PATH_TO_FINETUNED_MODEL] --dataset [PATH_TO_TEST_DATASET] [OPTIONS]
```

#### Example:

```bash
# Evaluate a fine-tuned model on the MetalDAM dataset with visualization
python eval_model.py --model_path model_hub/finetune-models/matssl/matssl-aachen-uhcs-metaldam-finetune-on-ebc3.pth --dataset datasets/finetune-datasets/EBC/nasa-EBC3 --save_visualizations
```

Evaluation results will be saved in the `results_[DATASET_NAME]/` directory.




