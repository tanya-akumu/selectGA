# SelectGA: Adaptive Frame Selection for Gestational Age Estimation from Blind Sweep Fetal Ultrasound Videos

[![Paper](https://papers.miccai.org/miccai-2025/paper/3136_paper.pdf)]


**SelectGA** is a novel AI framework designed to improve gestational age (GA) prediction from blind sweep ultrasound videos in low-resource healthcare settings. Our approach uses adaptive frame selection to identify the most clinically informative frames from ultrasound sweeps, achieving a **27% improvement** in prediction accuracy compared to existing methods.

## 🎯 Key Features

- **Adaptive Frame Selection**: Intelligently identifies the most informative frames from blind sweep videos
- **Anatomically-Guided Filtering**: Uses pretrained object detection to focus on frames containing fetal structures  
- **Diversity-Based Sampling**: Employs clustering to select diverse, non-redundant frames
- **Resource-Efficient**: Designed specifically for low-resource healthcare environments
- **Cross-Center Validation**: Tested across multiple geographical locations and equipment types

## 🚀 Performance Highlights

- **9.60 days** Mean Absolute Error (MAE) - 27% improvement over baselines
- **63.9%** of predictions within 7-day clinical threshold
- **R² = 0.906** correlation with ground truth gestational age
- Consistent performance across 2nd and 3rd trimesters
- Validated on multi-center dataset from diverse resource settings

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/tanya-akumu/selectGA.git 
cd SelectGA

# Create conda environment
conda create -n selectga python=3.10
conda activate selectga

# Install dependencies
pip install -r requirements.txt

```


## 📊 Dataset

Our framework was validated on a multi-center fetal ultrasound dataset:

- **1,314 blind sweep videos** from 162 study scans
- **Two geographical centers** with different equipment (Philips Lumify, GE Voluson v8)
- **245,048 total frames** across diverse gestational ages
- **Standardized blind sweep protocol** with 6-10 sweeps per patient

### Data Structure
```
data/
├── videos/
│   ├── center1/
│   │   ├── patient_001_sweep_001.mp4
│   │   └── ...
│   └── center2/
│       ├── patient_001_sweep_001.mp4
│       └── ...
├── annotations/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── metadata/
    └── patient_info.csv
```

**Note**: Due to privacy regulations, the original dataset cannot be shared publicly. Please use your own blind sweep ultrasound data following the same format.


## 🎓 Training

### Prepare Your Data

1. Organize your data following the structure above
2. Create annotation CSV files with columns: `sweep_video_path`, `gestational_age_days`, `patient_id`
3. Update the configuration file `configs/selectga_config.yaml`

### Train SelectGA Model

```bash

# Train with custom parameters
python train.py \
    --data_path /path/to/your/data \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 200 \
    --num_frames 16 \
    --confidence_threshold 0.25
```

### Training Configuration

Key parameters in `selectga_config.yaml`:

```yaml
# Data settings
data_path: "./data"
train_split: 0.6
val_split: 0.2
test_split: 0.2

# Model settings
num_frames: 16  # K frames selected per sweep video
confidence_threshold: 0.25  # Object detection threshold
resnet_pretrained: true

# Training settings
batch_size: 16
learning_rate: 1e-4
num_epochs: 200
early_stopping_patience: 5

# Hardware
device: "cuda"
num_workers: 4
```


### Performance Metrics

The evaluation script computes:
- **Mean Absolute Error (MAE)**
- **Root Mean Square Error (RMSE)** 
- **R² correlation coefficient**
- **Clinical accuracy** (% within 7 and 14 days)
- **Per-trimester performance**
- **Cross-center generalization**

## 🏗️ Model Architecture

SelectGA consists of two main stages:

### Stage 1: Adaptive Frame Selection

1. **Anatomically Guided (AG) Selector**
   - Pretrained Faster R-CNN detects fetal structures
   - Filters frames with confidence > threshold (α = 0.25)

2. **Diversity Guided (DS) Selector** 
   - Extracts feature embeddings of filtered anatomical frames using pretrained CNN
   - Applies K-means clustering (K=16) on feature space
   - Selects representative frame closest to each cluster centroid

### Stage 2: Gestational Age Prediction

1. **ResNet-50 Feature Extractor** (ImageNet pretrained)
2. **Weighted Average Attention (WAA) Module**
3. **Regression Head** for final GA prediction in days


## 📊 Results

### Quantitative Results

| Method | MAE (days) ↓ | RMSE (days) ↓ | R² ↑ | <7d (%) ↑ | <14d (%) ↑ |
|--------|-------------|---------------|------|-----------|------------|
| ResNet-50 | 12.90 | 17.73 | 0.851 | 41.7 | 66.7 |
| EchoNet | 11.49 | 15.86 | 0.881 | 47.2 | 69.4 |
| ViFi-CLIP | 12.41 | 19.38 | 0.822 | 52.8 | 69.4 |
| **SelectGA (Ours)** | **9.60** | **14.07** | **0.906** | **63.9** | **69.4** |

### Ablation Study

| Components | AG | DS | MAE ↓ | R² ↑ | <7d (%) ↑ |
|------------|----|----|-------|------|-----------|
| ResNet-50 + WAA | ✗ | ✗ | 12.90 | 0.851 | 41.7 |
| + AG Selector | ✓ | ✗ | 10.96 | 0.866 | 55.5 |
| **SelectGA (Full)** | ✓ | ✓ | **9.60** | **0.906** | **63.9** |

### Model checkpoints

SelectGA AG/DG blind sweep Selector: [https://drive.google.com/file/d/15HkFFgRYTKGqDKzVyOm4bkHZI_Ypb6Cc/view?usp=sharing](anatomy-detector-weights)
GA_predictor: [https://drive.google.com/file/d/1MT769lm2wxO6VowVOXvAjCZLueURo8sY/view?usp=sharing](gestation-age-predictor)


## 🔬 Research Impact

SelectGA addresses critical challenges in global healthcare:

- **Accessibility**: Enables GA estimation with minimal training data requirements
- **Resource Efficiency**: Reduces computational demands while improving accuracy
- **Clinical Relevance**: Achieves clinically acceptable accuracy thresholds
- **Scalability**: Framework designed for deployment in low-resource settings


## 📄 Citation

If you use SelectGA in your research, please cite our paper:

```bibtex
@InProceedings{AkuTan_Adaptive_MICCAI2025,
        author = { Akumu, Tanya AND Elbatel, Marawan AND Campello, Victor M. AND Osuala, Richard AND Martin-Isla, Carlos AND Valenzuela, Ignacio AND Li, Xiaomeng AND Khanal, Bishesh AND Lekadir, Karim},
        title = { { Adaptive Frame Selection for Gestational Age Estimation from Blind Sweep Fetal Ultrasound Videos } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
        year = {2025},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15973},
        month = {September},
        page = {3 -- 12}
}
```

## 📞 Contact

For questions about the code or collaboration opportunities:

- **Email**: [tanya.akumu@ub.edu]


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Research conducted across multiple international healthcare centers
- Special thanks to healthcare workers and patients who participated in data collection
- Built upon open-source contributions from the computer vision and medical AI communities

## ⚠️ Important Notes

- **Medical Device Regulation**: This research code is not intended for clinical use without proper validation and regulatory approval
- **Data Privacy**: Ensure compliance with local healthcare data regulations (HIPAA, GDPR, etc.)
- **Ethical Use**: Please use responsibly and consider the broader implications of AI in healthcare

---

**Made with ❤️ for improving global maternal and fetal healthcare accessibility**