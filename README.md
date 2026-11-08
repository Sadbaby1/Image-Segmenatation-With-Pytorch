# Image Segmentation with PyTorch & SAM-Geo

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![SAM-Geo](https://img.shields.io/badge/SAM--Geo-segment--geospatial-2b9348.svg)](https://samgeo.gishub.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive toolkit demonstrating both foundational deep learning semantic segmentation techniques (PyTorch, U-Net, FCN) and modern zero-shot geospatial segmentation using Meta's Segment Anything Model adapted for geospatial data (**SAM-Geo** / `segment-geospatial`).

---

## 📸 Showcase: Geospatial Segmentation (NAIP Aerial Imagery)

Automatic object segmentation and polygon boundary extraction on high-resolution National Agriculture Imagery Program (NAIP) aerial imagery using SAM-Geo:

![NAIP Segmentation Before and After](NAIP_before_after.gif)

---

## 📌 Features

- **Geospatial Foundation Model Segmentation (SAM-Geo)**
  - Automated mask generation for aerial and satellite remote sensing imagery using SAM (`vit_h` / `vit_b`).
  - Tuned inference parameters (`points_per_side`, `pred_iou_thresh`, `stability_score_thresh`, `min_mask_region_area`) for high-fidelity geographic feature extraction.
  - Multi-band GeoTIFF preprocessing with `rasterio` (converting multi-band NAIP imagery to optimized 3-band RGB GeoTIFFs).
  - Automated raster-to-vector pipeline: export segmented masks to ESRI Shapefile (`.shp`, `.dbf`, `.shx`, `.prj`) ready for GIS software like QGIS and ArcGIS.
  - Vector polygon visualization with `geopandas` and `matplotlib`.

- **Deep Learning Semantic Segmentation Fundamentals**
  - **U-Net Architecture**: Implementation and feature flow using `segmentation-models-pytorch` (`smp.Unet`) with ResNet backbones.
  - **Encoder-Decoder Mechanics**: Detailed breakdown of spatial downsampling, bottleneck latent representations, and decoder upsampling with skip connections.
  - **Transfer Learning**: Customizing pre-trained torchvision models (e.g. `fcn_resnet50`) for custom multi-class segmentation tasks.
  - **Loss Functions & Evaluation**: Multi-class Cross-Entropy Loss computation, logit thresholding, and `argmax` pixel-level prediction masks.

---

## 📂 Repository Structure

```text
Image-Segmenatation-With-Pytorch/
│
├── SAMGEO_fixed.ipynb          # End-to-end SAM-Geo geospatial segmentation notebook
├── semantic_segmentation.ipynb  # Interactive tutorial on PyTorch semantic segmentation
├── semantic_segmentation.py     # Standalone Python script for PyTorch U-Net & FCN models
├── NAIP_before_after.gif        # Animated before & after segmentation demo
├── niap1.tif / im.tif          # Sample high-resolution NAIP aerial GeoTIFF imagery
├── pyproject.toml               # Project metadata & Python packaging configuration
├── .python-version              # Python version specification (3.11)
└── README.md                    # Project documentation
```

---

## 🛠️ Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Image-Segmenatation-With-Pytorch.git
cd Image-Segmenatation-With-Pytorch
```

### 2. Create and Activate a Virtual Environment
```bash
# Using Python 3.11+
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies

Install PyTorch (CUDA recommended for SAM-Geo):
```bash
# Follow instructions at https://pytorch.org for your specific CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install standard deep learning & geospatial packages:
```bash
pip install segmentation-models-pytorch
pip install segment-geospatial leafmap localtileserver rasterio geopandas matplotlib numpy
```

---

## 🚀 Usage Guide

### 1. Geospatial Segmentation with SAM-Geo (`SAMGEO_fixed.ipynb`)

Open the notebook in Jupyter or Google Colab (GPU runtime recommended):

1. **Load and Preprocess GeoTIFF**:
   ```python
   import rasterio

   # Read multi-band NAIP imagery and export a clean 3-band 8-bit RGB GeoTIFF
   with rasterio.open("im.tif") as src:
       rgb = src.read([1, 2, 3])
       profile = src.profile

   profile.update(count=3, dtype="uint8", driver="GTiff")
   with rasterio.open("RGB.tif", "w", **profile) as dst:
       dst.write(rgb.astype("uint8"))
   ```

2. **Initialize SAM-Geo and Generate Masks**:
   ```python
   import torch
   from samgeo import SamGeo

   device = "cuda" if torch.cuda.is_available() else "cpu"

   sam = SamGeo(
       model_type="vit_h",  # 'vit_h' for highest accuracy, 'vit_b' for speed
       device=device,
       sam_kwargs={
           "points_per_side": 32,
           "pred_iou_thresh": 0.88,
           "stability_score_thresh": 0.92,
           "min_mask_region_area": 50,
       },
   )

   sam.generate("RGB.tif", "segment.tif", batch=True, unique=True)
   ```

3. **Convert Raster to GIS Vector Shapefile**:
   ```python
   sam.tiff_to_shp("segment.tif", "Result/segment.shp")
   ```

4. **Visualize Vector Overlays**:
   ```python
   import geopandas as gpd
   import matplotlib.pyplot as plt
   from rasterio.plot import show

   src_img = rasterio.open("RGB.tif")
   polygons = gpd.read_file("Result/segment.shp")

   fig, ax = plt.subplots(figsize=(12, 12))
   show(src_img, ax=ax)
   polygons.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=1)
   plt.axis("off")
   plt.show()
   ```

---

### 2. PyTorch Semantic Segmentation (`semantic_segmentation.py`)

Run the script directly to test model shapes and loss computations:
```bash
python semantic_segmentation.py
```

Key concepts demonstrated in the code:

- **Classification vs. Semantic Segmentation**:
  - Image classification assigns a single label to an entire image: `[B, C, H, W] -> [B, num_classes]`.
  - Semantic segmentation assigns a class label to every individual pixel: `[B, C, H, W] -> [B, num_classes, H, W]`.

- **U-Net with ResNet-34 Encoder**:
  ```python
  import segmentation_models_pytorch as smp
  import torch

  model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=5)
  images = torch.randn(2, 3, 256, 256)
  logits = model(images)  # Shape: [2, 5, 256, 256]
  predictions = logits.argmax(dim=1)  # Shape: [2, 256, 256]
  ```

- **Pretrained FCN for Fine-Tuning**:
  ```python
  import torch.nn as nn
  from torchvision.models.segmentation import fcn_resnet50

  model = fcn_resnet50(weights="DEFAULT")
  # Modify the final classification layer for custom class count (e.g. 5 classes)
  model.classifier[4] = nn.Conv2d(512, 5, kernel_size=1)
  ```

---

## ⚙️ SAM-Geo Parameter Tuning Guide

| Parameter | Recommended Value | Description |
| :--- | :--- | :--- |
| `model_type` | `"vit_h"` / `"vit_b"` | Vision Transformer backbone (`vit_h` = highest accuracy / heavier, `vit_b` = faster inference). |
| `points_per_side` | `32` (or `64` for fine features) | Density of sampling grid points for mask proposals. Higher values capture smaller objects. |
| `pred_iou_thresh` | `0.86` - `0.90` | Filtering threshold for predicted mask quality. Higher values reduce false positives. |
| `stability_score_thresh` | `0.90` - `0.95` | Threshold for mask stability across binarization cutoffs. |
| `min_mask_region_area` | `50` - `100` | Minimum pixel area required to keep a detected region (suppresses isolated noise). |
| `unique` | `True` | Assigns unique integer IDs to distinct segmented objects. |

---

## 🗺️ Geospatial Output Formats

The geospatial pipeline produces standard GIS outputs:
- **`segment.tif`**: Raster GeoTIFF with indexed object IDs for each detected segment.
- **`Result/segment.shp`**: ESRI Shapefile containing vector polygons with geospatial projections preserved, ready to drag-and-drop into QGIS, ArcGIS, or Google Earth Engine.

---

## 📚 References & Acknowledgments

- [Segment Anything (Meta AI)](https://github.com/facebookresearch/segment-anything)
- [Segment-Geospatial (SAM-Geo)](https://github.com/opengeos/segment-geospatial) by Prof. Qiusheng Wu
- [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) by Pavel Yakubovskiy
- [Rasterio](https://rasterio.readthedocs.io/) & [GeoPandas](https://geopandas.org/)
- [USDA NAIP (National Agriculture Imagery Program)](https://naip-usdaonline.hub.arcgis.com/)
