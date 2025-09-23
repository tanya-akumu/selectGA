import torch
import torch.nn.functional as F
import torchvision.transforms as T 
import torchvision.transforms.functional as TF
import random
from typing import Tuple, Union, List
import numpy as np

class USAugmentation:
    def __init__(
        self, 
        img_size: Tuple[int, int],
        mean: Union[List[float], Tuple[float, ...]] = (0.485, 0.456, 0.406),
        std: Union[List[float], Tuple[float, ...]] = (0.229, 0.224, 0.225),
        padding: int = 16,
        p: float = 0.5,
        mode: str = 'train',
        seed: int = 42
    ):
        """
        Args:
            img_size: Target size (height, width) for resizing
            mean: Mean values for normalization, one per channel
            std: Standard deviation values for normalization, one per channel
            p: Probability of applying each augmentation
        """
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.p = p
        self.padding = padding
        self.seed = seed
        self.mode = mode

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        if self.mode == 'train':
            self.crop = T.RandomCrop(img_size)
        else:
            self.crop = T.CenterCrop(img_size)
        
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must each have 3 values (one per channel)")
        if not all(s > 0 for s in std):
            raise ValueError("std values must be positive")
        
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: Video frames in shape (C, T, H, W) or (T, C, H, W)
        Returns:
            torch.Tensor: Augmented frames in same shape as input
        """
        
        if not isinstance(frames, torch.Tensor):
            frames = torch.Tensor(frames)
            
        # Determine input format and reshape if needed for pretrained video models
        if frames.shape[1] > frames.shape[0]:  # (C, T, H, W) 
            needs_transpose = False
        else:  # (T, C, H, W) 
            frames = frames.transpose(0, 1)  # Convert to (C, T, H, W)
            needs_transpose = True
            
        C, T, H, W = frames.shape
        
        # resize image
        if (H, W) != self.img_size:
            frames = frames.reshape(C*T,H,W)
            frames = TF.resize(
                frames,  # reshape to (C*T, H, W)
                self.img_size,
                antialias=True
            )
            frames = frames.reshape(C, T, *self.img_size)  # reshape back to (C, T, H, W)


        # pad and crop
        if self.padding > 0:
            frames = TF.pad(frames.view(-1, *frames.shape[2:]), self.padding, padding_mode='constant')
            frames = frames.view(C, T, *frames.shape[-2:])
            frames = self.crop(frames)

        if self.mode == 'train':
        
            # Pre-determine which augmentations to apply
            apply_sharp = random.random() < self.p
            apply_blur = random.random() < self.p
            apply_speckle = random.random() < self.p
            apply_brightness = random.random() < self.p
            apply_contrast = random.random() < self.p
            apply_rotation = random.random() < self.p
            apply_hflip = random.random() < self.p
            apply_vflip = random.random() < self.p
            
            # Pre-generate random parameters
            if apply_blur:
                kernel = random.choice([(3,3), (3,5), (5,5)])
                sigma = random.uniform(0.1, 2.0)
            if apply_brightness:
                brightness_factor = random.uniform(0.8, 1.2)
            if apply_contrast:
                contrast_factor = random.uniform(0.8, 1.2)
                
            # Apply augmentations frame by frame
            for t in range(T):
                frame_t = frames[:, t]  # (C, H, W)
                
                if apply_sharp:
                    frame_t = TF.adjust_sharpness(frame_t, 2.0)
                    
                if apply_blur:
                    frame_t = TF.gaussian_blur(frame_t, kernel, [sigma, sigma])
                    
                if apply_brightness:
                    frame_t = TF.adjust_brightness(frame_t, brightness_factor)
                    
                if apply_contrast:
                    frame_t = TF.adjust_contrast(frame_t, contrast_factor)

                if apply_hflip:
                    frame_t = TF.hflip(frame_t)
                if apply_vflip:
                    frame_t = TF.vflip(frame_t)
                
                if apply_rotation:
                    angle = random.uniform(-45,45)
                    frame_t = TF.rotate(frame_t, angle)

                frame_t = TF.normalize(frame_t, mean=self.mean, std=self.std)
                    
                frames[:, t] = frame_t

            if apply_speckle: # speckle noise
                frames = frames + torch.randn_like(frames) * 0.05
        else:
            for t in range(T):
                frame_t = frames[:, t]
                frame_t = TF.normalize(frame_t, mean=self.mean, std=self.std)
                frames[:, t] = frame_t
      
        # Return to original format if needed
        if needs_transpose:
            frames = frames.transpose(0, 1)
            
        return frames