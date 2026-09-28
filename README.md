<div align="center">

# 🌊 USPCDenoise

### A Real-World 3D Sonar Point Cloud Dataset for Underwater Denoising

[![Dataset](https://img.shields.io/badge/Dataset-639%20Point%20Clouds-blue)](#dataset-composition)
[![Format](https://img.shields.io/badge/Format-HDF5-green)](#dataset-structure)
[![Task](https://img.shields.io/badge/Task-Point%20Cloud%20Denoising-orange)](#potential-applications)
[![Status](https://img.shields.io/badge/Status-Open%20Source-brightgreen)](#)

**Real 3D sonar measurements · Underwater terrain · Structural foundations**

</div>

---

## 📖 Overview

**USPCDenoise** is an open-source dataset of **real-world 3D sonar point clouds** designed for underwater point cloud denoising, with particular emphasis on **underwater terrain mapping** and **infrastructure inspection**.

Publicly available 3D sonar point cloud datasets remain limited, making it difficult to develop and systematically evaluate learning-based methods for underwater perception. To address this gap, we collected and processed real 3D sonar measurements and constructed **USPCDenoise** as a benchmark dataset for underwater sonar point cloud denoising.

The dataset covers two representative underwater measurement scenarios:

| Scenario | Target | Primary Purpose |
|:---:|:---:|:---|
| 🗺️ **Mapping** | Underwater terrain | Terrain mapping & reconstruction |
| 🔍 **Inspection** | Structural foundations | Underwater infrastructure inspection |

<br>

<p align="center">
  <img src="images/fig3_modify.jpg" width="90%">
</p>

<p align="center">
  <em>Overview of the USPCDenoise dataset and the data acquisition and construction process.</em>
</p>

---

## 📊 Dataset Composition

<div align="center">

| | Structural Foundation | Underwater Terrain | **Total** |
|:---|:---:|:---:|:---:|
| **Number of samples** | 173 | 466 | **639** |
| **Points per sample** | 2,048 | 3,072 | — |
| **Data source** | Real 3D sonar | Real 3D sonar | — |
| **Primary task** | Inspection | Mapping | Denoising |

</div>

### 🔹 Structural Foundation Point Clouds

The structural foundation subset is designed primarily for **underwater infrastructure inspection**.

- **173** point cloud samples
- **2,048 points** per sample
- `train_val_unsupervised.h5` — training and validation data
- `test_unsupervised.h5` — testing data

### 🔹 Underwater Terrain Point Clouds

The terrain subset is designed primarily for **underwater mapping and terrain reconstruction**.

- **466** point cloud samples
- **3,072 points** per sample
- `train_val_data.h5` — training and validation data
- `test_data.h5` — testing data

All point clouds are stored in **HDF5 (`.h5`) format** for efficient storage and convenient integration with deep learning frameworks such as **PyTorch** and **TensorFlow**.

---

## 📁 Dataset Structure

```text
USPCDenoise/
│
├── README.md
│
├── images/
│   └── overview.png
│
└── data/
    │
    ├── foundation/
    │   ├── train_val_unsupervised.h5
    │   └── test_unsupervised.h5
    │
    └── terrain/
        ├── train_val_data.h5
        └── test_data.h5
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🌊 **Real-world data** | Point clouds are obtained from real 3D sonar measurements rather than purely synthetic data |
| 🏗️ **Multiple scenarios** | Covers both underwater structural foundations and terrain surfaces |
| 🔬 **Denoising-oriented** | Specifically constructed for the development and evaluation of point cloud denoising methods |
| 📐 **Standardized samples** | Fixed numbers of points facilitate batch processing and deep learning applications |
| 💾 **HDF5 format** | Compact storage and straightforward integration into point cloud learning pipelines |

---

## 🚀 Potential Applications

USPCDenoise can support a broad range of underwater 3D perception and engineering applications, including:

- **3D sonar point cloud denoising**
- Underwater terrain mapping
- Underwater 3D reconstruction
- Infrastructure inspection
- Underwater target recognition
- Scour morphology analysis
- Structural condition assessment
- Learning-based underwater perception

The dataset provides a benchmark resource for research at the intersection of **computer vision**, **artificial intelligence**, **civil engineering**, and **marine surveying**.

---

## 📝 Citation

If you use **USPCDenoise** in your research, please cite the associated paper:

```bibtex
@article{huang_uspcdenoise,
  title   = {Deep Learning for Denoising 3D Sonar Point Clouds in Underwater Scour Investigation and Infrastructure Inspection},
  author  = { },
  journal = {...},
  year    = {...},
  doi     = {...}
}
```

> **Note:** Complete citation information will be updated upon publication of the associated paper.

---

## 📄 License

License information for **USPCDenoise** will be provided with the official dataset release.

Please refer to the repository license before using or redistributing the dataset.

---

<div align="center">

### USPCDenoise

**Real-world 3D sonar data for underwater point cloud research**

</div>
