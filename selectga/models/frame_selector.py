import os
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from multiprocessing import Pool
import psutil
import gc
from tqdm import tqdm
import logging
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'video_processing_test{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Initialize the feature extractor
def initialize_model():
    """Initialize and prepare the ResNet model for feature extraction."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return model, transform, device

def extract_features_batch(frames, model, transform, device, batch_size=16):
    """Extract features from a batch of frames using the model."""
    features_list = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        batch_tensors = torch.stack([transform(frame) for frame in batch_frames]).to(device)
        
        with torch.no_grad():
            batch_features = model(batch_tensors)
            batch_features = batch_features.view(batch_features.size(0), -1).cpu().numpy()
            features_list.extend(batch_features)
            
        # Clear GPU memory
        del batch_tensors
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return np.array(features_list)

def select_frames_kmeans(features, seed=42,k=16):
    """Select K diverse frames using K-Means clustering."""
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(features)
    
    # Find frames closest to cluster centers
    distances = cdist(kmeans.cluster_centers_, features)
    selected_indices = [np.argmin(dist) for dist in distances]
    
    return sorted(selected_indices)

def process_video(args):
    """Process a single video file to extract and select diverse frames across the entire video."""
    try:
        video_path, frame_indices, output_folder = args
        logging.info(f"Processing video: {video_path}")
        
        # Initialize model and transform
        model, transform, device = initialize_model()
        
        # Create output directory and path
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, os.path.basename(video_path))
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"Video {video_path} has {total_frames} frames")
        
        # First pass: Collect all relevant frames
        frames = []
        frame_mapping = []
        idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Keep frame if we have ≤16 frame_indices or if this frame's index is in our list
            should_keep_frame = (
                len(frame_indices) <= 16 or 
                idx in frame_indices
            )
            
            if should_keep_frame:
                frames.append(frame)
                frame_mapping.append(idx)
                
            idx += 1
            
        cap.release()
        logging.info(f"Collected {len(frames)} frames for processing")
        
        # Now process all collected frames together
        if len(frames) > 0:
            # Extract features for all frames at once
            # batches to manage memory, but keep all features
            batch_size = 32
            all_features = []
            
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                batch_features = extract_features_batch(batch_frames, model, transform, device)
                all_features.extend(batch_features)
                
                # Clear GPU memory after each batch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            all_features = np.array(all_features)
            logging.info(f"Extracted features of shape {all_features.shape}")
            
            # Select most dissimilar frames from all clusters
            if len(frames) > 16:
                selected_indices = select_frames_kmeans(all_features,seed=60, k=16)
                final_frames = [frames[i] for i in selected_indices]
                final_frame_indices = [frame_mapping[i] for i in selected_indices]
            else:
                final_frames = frames
                final_frame_indices = frame_mapping
            
            logging.info(f"Selected frame indices: {final_frame_indices}")
            
            # Save output video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frames_written = 0
            for frame in final_frames:
                out.write(frame)
                frames_written += 1
            
            out.release()
            logging.info(f"Saved processed video to: {output_path} with {frames_written} frames")
        else:
            logging.warning(f"No frames collected for video: {video_path}")
            
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return output_path
        
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {str(e)}")
        return None

def process_row(row, output_folder):
    """Process a single row from the CSV file."""
    try:
        video_path = row[0]
        frame_indices = eval(row[2])
        return (video_path, frame_indices, output_folder)
    except Exception as e:
        logging.error(f"Error processing row {row}: {str(e)}")
        return None

def main():
    """Main function to process all videos."""
    try:
        # Load CSV file
        csv_path = "/dataset/data/test.csv" # Update with your CSV path of video paths
        output_folder = "/dataset/sampled_blindsweeps" # update with your desired output folder
        
        logging.info("Starting video processing pipeline")
        logging.info(f"Reading CSV from: {csv_path}")
        
        df = pd.read_csv(csv_path, header=None)
        total_videos = len(df)
        logging.info(f"Found {total_videos} videos to process")
        
        # Prepare arguments for multiprocessing
        args = [process_row(row, output_folder) for _, row in df.iterrows()]
        args = [arg for arg in args if arg is not None]
        
        # Calculate optimal number of processes based on system resources
        n_processes = min(32, os.cpu_count())
        logging.info(f"Using {n_processes} processes")
        
        # Process videos with progress tracking
        with Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.imap(process_video, args),
                total=len(args),
                desc="Processing videos"
            ))
        
        # Log results
        successful = len([r for r in results if r is not None])
        logging.info(f"Processing completed. Successfully processed {successful}/{total_videos} videos")
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()