# SelectGA: Adaptive Frame Selection for Gestational Age Estimation from Blind Sweep Fetal Ultrasound Videos


**SelectGA** is a novel AI framework designed to improve gestational age (GA) prediction from blind sweep ultrasound videos in low-resource healthcare settings. Our approach uses adaptive frame selection to identify the most informative frames from ultrasound sweeps, achieving a **27% improvement** in prediction accuracy compared to existing methods.

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
