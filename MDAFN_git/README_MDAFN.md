<!-- MDAFN - Mutual Distillation Attribute Fusion Network -->

<div align="center">
  
# MDAFN - Mutual Distillation Attribute Fusion Network for Multimodal Vehicle Object Detectio

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

## 📑 Overview

This repository contains the official implementation of the paper **"Mutual Distillation Attribute Fusion Network for Multimodal Vehicle Object Detection"**. Our research introduces an innovative attribute fusion network based on mutual distillation mechanisms for multimodal vehicle detection tasks.

<div align="center">
  <img src="path/to/architecture_diagram.png" alt="MDAFN Architecture" width="800px">
</div>

## 🔧 Requirements

Ensure your system meets the following specifications:
- Python 3.6+
- PyTorch 1.7+
- CUDA support (recommended)

## 🚀 Installation

```bash
git clone https://github.com/username/MDAFN.git
cd MDAFN
pip install -r requirements.txt
```

## 📊 Dataset Preparation

This project utilizes the M3FD multimodal dataset (visible light and infrared images). Organize your dataset according to the following structure:

```
dataset/M3FD/
  ├── ImageSets/
  │   └── Main/
  │       ├── trainval.txt    # Training image ID list
  │       ├── val.txt         # Validation image ID list
  │       └── label_list.txt  # Category label list
  ├── JPEGImages/             # Visible light images directory
  │   ├── image_id_1.png
  │   ├── image_id_2.png
  │   └── ...
  ├── JPEGImages_ir/          # Infrared images directory
  │   ├── image_id_1.png
  │   ├── image_id_2.png
  │   └── ...
  └── Annotations/            # XML format annotation directory
      ├── image_id_1.xml
      ├── image_id_2.xml
      └── ...
```

**Important Notes:**

1. Visible light and infrared images must correspond one-to-one with identical filenames
2. All images should be in PNG format
3. Annotations follow VOC format (XML files)
4. The M3FD dataset contains 6 categories of vehicle objects

## ⚙️ Model Training

Start training with a pre-trained model:

```bash
python tools/train.py --config configs/MDAFN/MDAFN.yml --load_pre path/to/pretrained.pth --device cuda:0
```

Train from scratch:

```bash
python tools/train.py --config configs/MDAFN/MDAFN.yml --device cuda:0
```

Resume training:

```bash
python tools/train.py --config configs/MDAFN/MDAFN.yml --resume path/to/checkpoint.pth --device cuda:0
```

## 📈 Model Evaluation

```bash
python tools/train.py --config configs/MDAFN/MDAFN.yml --load_pre path/to/model.pth --test_only --device cuda:0
```

## ⚙️ Configuration

Adjust model parameters and training strategies by modifying the `configs/MDAFN/MDAFN.yml` file.

In the configuration file, ensure you correctly specify the dataset path:

```yaml
dataset: 
  type: M3FDDetection
  root: /path/to/your/dataset/M3FD/ImageSets/Main/
  ann_file: trainval.txt  # or val.txt
  label_file: label_list.txt
```

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{author2023mutual,
  title={Mutual Distillation Attribute Fusion Network for Multimodal Vehicle Object Detection},
  author={},
  journal={},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License

## 📞 Contact

For any questions or issues, please contact: wanghaoyucumt@163.com