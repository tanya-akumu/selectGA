import logging
import numpy as np
from scipy import stats
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import r2_score
import pandas as pd

def load_video(filename):
    '''
    Loads a video from file
    '''
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    cap = cv2.VideoCapture(filename)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)
    try:
        for count in range(frame_count):
            ret, frame = cap.read()
            # if not ret:
            #     raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video[count, :, :] = frame
    except Exception as e:
        print(f"Error loading video {filename}", flush=True)

    # video = video.transpose((3, 0, 1, 2))

    return video


def setup_logger(save_dir,model,fold, sampling):
    log_file = Path(save_dir) / f"{model}_{sampling}_fold_{fold}_training.log"
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S'
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(f"{model}_{sampling}_fold_{fold}_training.log")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

import numpy as np
from torchvision.transforms.functional import resize
from tqdm import tqdm  # For progress bar

def compute_mean_std(dataset, img_size=(112, 112)):
    """
    Compute mean and standard deviation of a video dataset.
    
    Args:
        dataset: FocusSweepDataset_ instance.
        img_size: Target size for resizing frames.
    
    Returns:
        mean: Mean of each channel (R, G, B).
        std: Standard deviation of each channel (R, G, B).
    """
    # Initialize accumulators
    pixel_sum = np.zeros(3)  # Sum of pixel values for each channel
    pixel_sq_sum = np.zeros(3)  # Sum of squared pixel values for each channel
    num_pixels = 0  # Total number of pixels processed
    
    # Iterate through the dataset
    for idx in tqdm(range(len(dataset))):
        frames, _ = dataset[idx]  # Get frames and label
        if frames is None:
            continue
        
        # Resize frames to (H, W)
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W) -> (T, C, H, W)
        resized_frames = torch.stack([resize(frame, img_size) for frame in frames])  # Resize each frame
        resized_frames = resized_frames.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
        
        # Accumulate pixel values
        pixel_sum += resized_frames.sum(dim=(1, 2, 3)).numpy()  # Sum over T, H, W
        pixel_sq_sum += (resized_frames ** 2).sum(dim=(1, 2, 3)).numpy()  # Sum of squares
        num_pixels += resized_frames.shape[1] * resized_frames.shape[2] * resized_frames.shape[3]  # T * H * W
    
    # Compute mean and std
    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sq_sum / num_pixels - mean ** 2)
    print(f"Mean: {mean}, STD: {std}", flush=True) # debug
    return mean, std

# def compute_metrics(predictions, ground_truth):
#     '''
#     Compute mean metrics with standard errors
#     '''
#     errors = predictions - ground_truth
#     abs_errors = np.abs(errors)
#     for idx,val in enumerate(ground_truth):
#         print(f"GT: {val}, Pred: {predictions[idx]}, Error: {abs_errors[idx]}")


#     mae = np.mean(abs_errors)
#     mae_std = np.std(abs_errors) / np.sqrt(len(abs_errors)) # standard error MAE

#     rmse = np.sqrt(np.mean(errors**2))
#     rmse_std = np.std(errors**2) / (2 * rmse * np.sqrt(len(errors))) # standard error RMSE

#     # % within error bounds
#     within_7_days = np.mean(abs_errors <= 7) * 100
#     within_14_days = np.mean(abs_errors <= 14) * 100

#     return {
#         'mae': mae,
#         'mae_std': mae_std,
#         'rmse': rmse,
#         'rmse_std': rmse_std,
#         'within_7_days': within_7_days,
#         'within_14_days': within_14_days,
#         'mean_error': np.mean(errors),
#         'std_error': np.std(errors)
#     }

def compute_metrics(predictions=None, ground_truth=None, csv_path=None, verbose=False):
    '''
    Compute mean metrics with standard errors for the whole dataset, 
    and aggregated by trimester and country if a CSV file is provided.
    
    Parameters:
    -----------
    predictions : np.array, optional
        Array of predictions for direct calculation
    ground_truth : np.array, optional
        Array of ground truth values for direct calculation
    csv_path : str, optional
        Path to CSV file containing predictions, labels, trimester, and country columns
    verbose : bool, optional
        Whether to print individual predictions and errors
        
    Returns:
    --------
    dict
        Dictionary containing metrics for the whole dataset and aggregated by trimester and country
    '''
    # Initialize results dictionary
    results = {
        'overall': {},
        'by_trimester': {},
        'by_country': {}
    }
    
    # If CSV path is provided, read the data from the CSV file
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        predictions = np.array(df['prediction'].tolist())
        ground_truth = np.array(df['label'].tolist())
    elif predictions is None or ground_truth is None:
        raise ValueError("Either provide predictions and ground_truth arrays or csv_path")
    
    # Convert inputs to numpy arrays if they aren't already
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Calculate metrics for the whole dataset using the original function logic
    results['overall'] = _calculate_single_metrics(predictions, ground_truth, verbose)
    
    # If using CSV, calculate metrics by trimester and country
    if csv_path is not None:
        # Calculate metrics by trimester
        for trimester in df['trimester'].unique():
            subset = df[df['trimester'] == trimester]
            if len(subset) > 0:
                subset_preds = np.array(subset['prediction'].tolist())
                subset_gt = np.array(subset['label'].tolist())
                
                results['by_trimester'][trimester] = _calculate_single_metrics(
                    subset_preds, subset_gt, False
                )
                # Add count of samples in this group
                results['by_trimester'][trimester]['count'] = len(subset)
        
        # Calculate metrics by country
        for country in df['country'].unique():
            subset = df[df['country'] == country]
            if len(subset) > 0:
                subset_preds = np.array(subset['prediction'].tolist())
                subset_gt = np.array(subset['label'].tolist())
                
                results['by_country'][country] = _calculate_single_metrics(
                    subset_preds, subset_gt, False
                )
                # Add count of samples in this group
                results['by_country'][country]['count'] = len(subset)
    
    return results

def _calculate_single_metrics(predictions, ground_truth, verbose=False):
    '''
    Helper function to calculate metrics for a single set of predictions and ground truth
    '''
    errors = predictions - ground_truth
    abs_errors = np.abs(errors)
    
    if verbose:
        for idx, val in enumerate(ground_truth):
            print(f"GT: {val}, Pred: {predictions[idx]}, Error: {abs_errors[idx]}")

    mae = np.mean(abs_errors)
    mae_std = np.std(abs_errors) / np.sqrt(len(abs_errors))  # standard error MAE

    rmse = np.sqrt(np.mean(errors**2))
    rmse_std = np.std(errors**2) / (2 * rmse * np.sqrt(len(errors)))  # standard error RMSE

    # % within error bounds
    within_7_days = np.mean(abs_errors <= 7) * 100
    within_14_days = np.mean(abs_errors <= 14) * 100
    
    # Calculate R-squared (coefficient of determination)
    r2 = r2_score(ground_truth, predictions)

    return {
        'mae': mae,
        'mae_std': mae_std,
        'rmse': rmse,
        'rmse_std': rmse_std,
        'within_7_days': within_7_days,
        'within_14_days': within_14_days,
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'r2': r2
    }
    
def plot_evaluation_results(predictions, ground_truth, save_dir):
    """Create evaluation plots"""
    save_dir = Path(save_dir)
    predictions = np.array(list(map(convert_days_to_weeks_decimal,predictions)), dtype=np.float32)
    ground_truth = np.array(list(map(convert_days_to_weeks_decimal,ground_truth)), dtype=np.float32)
    
    # 1. Prediction vs Ground Truth
    plt.figure(figsize=(10, 10))
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.plot([min(ground_truth), max(ground_truth)], 
             [min(ground_truth), max(ground_truth)], 'r--')
    plt.xlabel('Ground Truth GA (weeks)')
    plt.ylabel('Predicted GA (weeks)')
    plt.title('Resnet50+Attn Model Predictions vs Ground Truth')
    plt.savefig(save_dir / 'predictions_vs_truth.png', dpi=300)
    plt.close()
    
    # 2. Error vs Ground Truth
    errors = predictions - ground_truth
    print(f"Error length: {len(errors)}")
    try:
        plt.figure(figsize=(10, 10))
        plt.scatter(ground_truth, errors, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Ground Truth GA (weeks)')
        plt.ylabel('Error (weeks)')
        plt.title('Resnet50+Attn Model Error vs Ground Truth')
        plt.savefig(save_dir / 'error_vs_truth.png', dpi=300)
        plt.close()
    
    
        # 3. CDF of Absolute Error
        abs_errors = np.abs(errors)
        plt.figure(figsize=(10, 10))
        plt.hist(abs_errors, bins=50, density=True, cumulative=True, 
                histtype='step', label='Model')
        plt.xlabel('Absolute Error (weeks)')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution of Absolute Error')
        plt.grid(True)
        plt.legend()
        plt.savefig(save_dir / 'error_cdf.png', dpi=300)
        plt.close()
    except Exception as e:
        print(e)

def convert_days_to_weeks_decimal(total_days):
    total_weeks = total_days / 7
    
    # Separate the integer part (weeks) and the decimal part (fraction of a week)
    weeks = int(total_weeks)
    fractional_weeks = total_weeks - weeks
    
    # Convert the fractional part back to days and round to 1 decimal place
    days = round(fractional_weeks * 7, 1)
    
    # Combine weeks and days into the desired format
    return f"{weeks}.{int(days)}"


def custom_collate(batch):
    """
    Custom collate function to handle variable-length sequences.
    Args:
        batch: List of tuples (frames, label)
    Returns:
        Padded frames tensor and labels tensor
    """
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    frames, labels = zip(*batch)
    
    # Get max sequence length in this batch
    max_len = max(seq.shape[0] for seq in frames)
    
    # Pad all sequences to max length
    padded_frames = []
    for seq in frames:
        if seq.shape[0] < max_len:
            # Create padding
            padding = max_len - seq.shape[0]
            last_frame = seq[-1:]  # Keep last frame's dimensions
            padding_frames = last_frame.repeat(padding, 1, 1, 1)
            padded_seq = torch.cat([seq, padding_frames], dim=0)
            padded_frames.append(padded_seq)
        else:
            padded_frames.append(seq)
    
    # Stack all sequences and labels
    frames_tensor = torch.stack(padded_frames)
    labels_tensor = torch.stack(labels.to(torch.float32))
    
    return frames_tensor, labels_tensor