import cv2
import csv
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
import random
import os
from pathlib import Path
import numpy as np
from collections import defaultdict

class BlindSweepDataset(Dataset):
    def __init__(self, datafile,mode='train', preprocess_params=None):
        """
        Dataset class for loading blindsweep videos.
        Args:
            datafile: file containng list of paths to mp4 files and associated labels
            preprocess_params: dictionary with preprocessing parameters
        """

        reader = csv.reader(datafile)
        self.video_paths = []
        self.labels = []
        self.study_ids = []
        self.mode = mode
        self.study_to_sweeps = defaultdict(list)

        with open(datafile, 'r') as input_file:
            reader = csv.reader(input_file)

            for row in reader:
                row = row[0].split(' ')
                video_path = row[0]
                label= float(row[1])
                study_id = video_path.split('/')[-1].split('_')[0]

                self.study_ids.append(study_id)
                self.video_paths.append(video_path)
                self.labels.append(label)
                self.study_to_sweeps[study_id].append((video_path, label))

        if mode != 'train':
            self.unique_ids = list(self.study_to_sweeps.keys())

        # compute weights for balanced sampling
        if mode == 'train':
            ga_weeks = np.array(self.labels).astype(int)
            unique_weeks, counts = np.unique(ga_weeks, return_counts=True)
            weight_per_week = 1.0 / counts
            week_to_weight = dict(zip(unique_weeks, weight_per_week))

            self.weights = [week_to_weight[int(label)] for label in self.labels]
            self.weights = torch.FloatTensor(self.weights)

        self.params = {
            'image_size': (256, 256),
            'spacing': 0.75,
            'n_sample_frames': 50
        }

        if preprocess_params:
            self.params.update(preprocess_params)

        if self.mode == 'train':
            self.transforms = T.Compose([
                T.ToPILImage(),
                T.Resize(self.params['image_size']),
                T.Pad(16, padding_mode='constant'),
                T.RandomCrop(256),
                T.ToTensor(), # scales image to [0,1]
                T.Normalize(mean=[0.485], std=[0.229]) #imagenet normalization
            ])
        else:
            self.transforms = T.Compose([
                T.ToPILImage(),
                T.Resize(self.params['image_size']),
                T.CenterCrop(256),
                T.ToTensor(), # scales image to [0,1]
                T.Normalize(mean=[0.485], std=[0.229]) #imagenet normalization
            ])
    
    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames=[]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # frame = np.transpose(frame, (2,0,1))
            frames.append(frame)
        cap.release()

        frames = np.array(frames)

        if self.mode== 'train':
            if len(frames) >= self.params['n_sample_frames']:
                # np.random.seed(42)
                indices = np.random.choice(
                    len(frames),
                    self.params['n_sample_frames'],
                    replace=False,
                )
                indices.sort()
                frames= frames[indices]
            elif len(frames) < self.params['n_sample_frames']:
                # Repeat the last frame until we reach n_frames
                padding = self.params['n_sample_frames'] - len(frames)
                last_frame = frames[-1]
                frames = np.concatenate([frames] + [last_frame[None]] * padding)
        else:
            # use all frames
            # print(f"TOTal frames: {len(frames)}", flush=True)
            if len(frames) >= self.params['n_sample_frames']:
                # np.random.seed(42)
                indices = np.random.choice(
                    len(frames),
                    self.params['n_sample_frames'],
                    replace=False,
                )
                indices.sort()
                frames= frames[indices]
            
            # pass

        return frames
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            frames = self._load_video(self.video_paths[idx])
            transformed_frames = []

            for frame in frames:
                transformed = self.transforms(frame)
                transformed_frames.append(transformed)

            frames_tensor = torch.stack(transformed_frames)

            return frames_tensor, torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            study_id = self.video_paths[idx].split('/')[-1].split('_')[0]

            sweep_videos = self.study_to_sweeps[study_id]
            all_study_frames = []

            # Set a maximum number of frames per study to maintain consistent size
            # max_frames_per_study = self.params['n_sample_frames']  # e.g., 50
            # frames_per_sweep = max_frames_per_study // len(sweep_videos)

            for video,_ in sweep_videos:
                frames = self._load_video(video)
                tx_frames = []
                for frame in frames:
                    tx_frames.append(self.transforms(frame))
                frames_tensor = torch.stack(tx_frames)
                # frames_tensor = torch.tensor(frames)
                all_study_frames.append(frames_tensor)

            study_frames = torch.cat(all_study_frames, dim=0)
            # study_frames = self.transforms(study_frames)
            return study_frames, torch.tensor(sweep_videos[0][1],dtype=torch.float32) # all labels in study videos are the same

    def __len__(self):
        return len(self.video_paths)
    

def create_dataloaders(datafile, mode='train', batch_size = 24,collate_func=None, num_workers = 8):
    
    if mode=='train':
        dataset = BlindSweepDataset(
        datafile=datafile,
        mode=mode,
        )
        sampler = WeightedRandomSampler(
            weights=dataset.weights,
            num_samples=5000,
            replacement=True
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        dataset = BlindSweepDataset(
        datafile=datafile,
        mode=mode,
        preprocess_params={'n_sample_frames': 50}
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloader