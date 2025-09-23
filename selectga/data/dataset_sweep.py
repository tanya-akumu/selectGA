from decord import VideoReader
import decord
import torch 
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets import VisionDataset
# Set decord to use CPU for frame decoding
decord.bridge.set_bridge('torch')
import cv2
import csv
import torch
from torchvision import transforms as T
import numpy as np
from collections import defaultdict
import pandas as pd
import ast
from torchvision.io import read_video
# from utils import load_video
import traceback
from torchvision.io import read_video
import random

class BlindSweepDataset(Dataset):
    def __init__(self, datafile, mode='train', preprocess_params=None):
        """
        Dataset class for loading blindsweep videos with optimized decord video loading.
        Args:
            datafile: file containing list of paths to mp4 files and associated labels
            mode: 'train' or 'val'/'test' mode
            preprocess_params: dictionary with preprocessing parameters
        """
        # Initialize storage for preloaded frames
        self.preloaded_frames = {}
        self.mode = mode
        self.study_to_sweeps = defaultdict(list)
        
        # Set default parameters
        self.params = {
            # 'image_size': (256, 256),
            'image_size': (224, 224),
            'spacing': 0.75,
            'n_sample_frames': 50,
            'sampling_stride': 4  # Sample every 4th frame
        }
        if preprocess_params:
            self.params.update(preprocess_params)

        # Read dataset information
        self.video_paths = []
        self.labels = []
        self.study_ids = []
        
        # Read video paths and labels
        with open(datafile, 'r') as input_file:
            reader = csv.reader(input_file)
            for row in reader:
                row = row[0].split(' ')
                video_path = row[0]
                label = float(row[1])
                study_id = video_path.split('/')[-1].split('_')[0]
                
                self.study_ids.append(study_id)
                self.video_paths.append(video_path)
                self.labels.append(label)
                self.study_to_sweeps[study_id].append((video_path, label))
                
                # Preload and process video frames
                self.preload_video(video_path)

        self._setup_transforms()
        
        if mode == 'train':
            self._setup_sampling_weights()
        else:
            self.unique_ids = list(self.study_to_sweeps.keys())

    def preload_video(self, video_path):
        """
        Preload and process video frames with uniform sampling.
        Uses decord for efficient video loading and implements
        uniform temporal sampling.
        """
        try:
            # Load video using decord
            vr = VideoReader(video_path)
            total_frames = len(vr)
            
            # Calculate indices for uniform sampling with stride
            # We first get every nth frame based on stride
            strided_indices = list(range(0, total_frames, self.params['sampling_stride']))
            
            if len(strided_indices) >= self.params['n_sample_frames']:
                # If we have more frames than needed after stride sampling,
                # perform uniform sampling to get exactly n_sample_frames
                # This maintains temporal ordering and uniform coverage
                indices = np.linspace(
                    0, 
                    len(strided_indices) - 1, 
                    self.params['n_sample_frames'], 
                    dtype=int
                )
                final_indices = [strided_indices[i] for i in indices]
            else:
                # If we have fewer frames than needed, repeat the last frame
                final_indices = strided_indices
                while len(final_indices) < self.params['n_sample_frames']:
                    final_indices.append(final_indices[-1])
            
            # Read the selected frames
            frames = vr.get_batch(final_indices)
            # Convert to numpy for compatibility with transforms
            frames = frames.numpy()
            
            self.preloaded_frames[video_path] = frames
            
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
            # Create an empty frame as fallback
            # empty_frame = np.zeros((self.params['n_sample_frames'], 256, 256, 3), dtype=np.uint8)
            # self.preloaded_frames[video_path] = empty_frame

    def _setup_transforms(self):
        """Set up image transforms based on mode."""
        base_transforms = [
            T.ToPILImage(),
            T.Resize(self.params['image_size']),
        ]
        
        if self.mode == 'train':
            base_transforms.extend([
                T.Pad(16, padding_mode='constant'),
                T.RandomCrop(self.params['image_size'][0]),
            ])
        else:
            base_transforms.append(T.CenterCrop(self.params['image_size'][0]))
            
        base_transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485], std=[0.229])
        ])
        
        self.transforms = T.Compose(base_transforms)

    def _setup_sampling_weights(self):
        """Calculate weights for balanced sampling during training."""
        ga_weeks = np.array(self.labels).astype(int)
        unique_weeks, counts = np.unique(ga_weeks, return_counts=True)
        weight_per_week = 1.0 / counts
        week_to_weight = dict(zip(unique_weeks, weight_per_week))
        self.weights = torch.FloatTensor(
            [week_to_weight[int(label)] for label in self.labels]
        )

    def __getitem__(self, idx):
        """Get item implementation focusing on frame loading efficiency."""
        study_id = self.video_paths[idx].split('/')[-1].split('_')[0]
        if self.mode == 'train':
            frames = self.preloaded_frames[self.video_paths[idx]]
            transformed_frames = [self.transforms(frame) for frame in frames]
            frames_tensor = torch.stack(transformed_frames)
            
            return frames_tensor, torch.tensor(self.labels[idx], dtype=torch.float32), study_id
        else:
            
            sweep_videos = self.study_to_sweeps[study_id]
            all_study_frames = []
            
            for video, _ in sweep_videos:
                frames = self.preloaded_frames[video]
                frames = frames[::4]
                transformed_frames = [self.transforms(frame) for frame in frames]
                frames_tensor = torch.stack(transformed_frames)
                all_study_frames.append(frames_tensor)
            
            study_frames = torch.cat(all_study_frames, dim=0)
            return study_frames, torch.tensor(sweep_videos[0][1], dtype=torch.float32), study_id

    def __len__(self):
        return len(self.video_paths)
    

class FocusSweepDataset(VisionDataset):
    def __init__(self, data_csv, model='image', sampling = 'optimal', num_samples = 16, transform=None, seed=42):
        '''
        '''
        super().__init__(root=None, transform=transform)

        self.data = pd.read_csv(data_csv, header=None)
        self.data.columns = ['video_path', 'label', 'selected_indices']
        self.data['selected_indices'] = self.data['selected_indices'].apply(ast.literal_eval)

        # Convert to lists for faster access
        self.video_paths = self.data['video_path'].tolist()
        self.labels = self.data['label'].tolist()
        self.frame_indices = self.data['selected_indices'].tolist()

        self.seed = seed
        self.sampling = sampling
        self.transform = transform
        self.sampling = sampling
        self.k = num_samples
        self.model_type = model

        # set seed for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)


    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (frames, label) where frames are the selected video frames and 
                  label is the class label
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        indices = self.frame_indices[idx]

        # Load video and select frames
        try:
            # Read video frames
            video = load_video(video_path)  # (T, H, W, C)
            B = video.shape[0]

            if len(indices) >= self.num_samples:
                indices = sorted(np.random.choice(indices,size=self.k, replace=False))
            else:
            # if len(indices) < self.k:
                available_idxs = set(indices)
                remaining_idxs = self.k - len(indices)

                if self.sampling == 'random':
                    additional_idxs = np.random.choice([i for i in range(B) if i not in available_idxs], size=remaining_idxs, replace=False)
                elif self.sampling == 'uniform':
                    additional_idxs = np.linspace(0,B-1, self.k, dtype=int)
                    additional_idxs = [i for i in additional_idxs if i not in available_idxs]
                else:
                    raise ValueError(f"Unsuported sampling method: {self.sampling}")
                
                indices = sorted(list(indices) + list(additional_idxs))
            # if self.sampling == 'random':
            #     k = min(self.k, len(video))
            #     indices = sorted(np.random.choice(len(video), size=k, replace=False))

            assert len(indices) == self.k,f"Insufficient sampled indices. Expected {self.k} samples, got {len(indices)} samples"
            selected_frames = video[indices]  # (N, H, W, C)
            selected_frames = selected_frames / 255.0
            
            # Rearrange dimensions to (T, C, H, W) 
            if self.model_type == 'video':
                selected_frames = selected_frames.transpose(3, 0, 1, 2)
            else:
                selected_frames = selected_frames.transpose(0, 3, 1, 2) #  (T, C, H, W) for pretrained image models. change to (C,T,H,W) for video models
            selected_frames = torch.tensor(selected_frames, dtype=torch.float32)
            
            if self.transform:
                selected_frames = self.transform(selected_frames)
            
            return selected_frames, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading video {video_path}: {traceback.format_exc()}")
            return None

def load_video(video_path):
    video, _, _ = read_video(video_path, pts_unit='sec')
    return video.numpy()

class FocusSweepDataset_(VisionDataset):
    def __init__(self, data_csv, model='image', sampling = 'optimal', num_samples = 16, transform=None, seed=42):
        '''
        '''
        super().__init__(root=None, transform=transform)

        self.data = pd.read_csv(data_csv, header=None)
        self.data.columns = ['video_path', 'label', 'selected_indices']
        self.data['selected_indices'] = self.data['selected_indices'].apply(ast.literal_eval)

        # Convert to lists for faster access
        self.video_paths = self.data['video_path'].tolist()
        self.labels = self.data['label'].tolist()
        self.frame_indices = self.data['selected_indices'].tolist()

        # cache videos
        self.video_cache = {path: load_video(path) for path in self.video_paths}

        self.seed = seed
        self.sampling = sampling
        self.transform = transform
        self.sampling = sampling
        self.k = num_samples
        self.model_type = model

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (frames, label) where frames are the selected video frames and 
                  label is the class label
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        patient_id = video_path.split('/')[-1].split('sweep')[0]
        # indices = self.frame_indices[idx]

        # Load video and select frames
        try:
            # Read video frames
            # video = load_video(video_path)  # (T, H, W, C)
            video = self.video_cache[video_path]

               
            # if self.sampling == 'random':
            #     k = min(self.k, len(video))
            #     # indices = sorted(np.random.choice(len(video), size=k, replace=False))

            # selected_frames = video[indices]  # (N, H, W, C)
            selected_frames = video / 255.0 # normalize to 0,1
            
            # Rearrange dimensions to (T, C, H, W) 
            if self.model_type == 'video':
                selected_frames = selected_frames.transpose(3, 0, 1, 2)
            else:
                selected_frames = selected_frames.transpose(0, 3, 1, 2) #  (T, C, H, W) for pretrained image models. change to (C,T,H,W) for video models
            selected_frames = torch.tensor(selected_frames, dtype=torch.float32)
            
            if self.transform:
                selected_frames = self.transform(selected_frames)
            
            return selected_frames, torch.tensor(label, dtype=torch.float32), patient_id
            
        except Exception as e:
            print(f"Error loading video {video_path}: {traceback.format_exc()}")
            return None
