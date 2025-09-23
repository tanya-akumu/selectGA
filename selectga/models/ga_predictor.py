import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class WeightedAttention(nn.Module):
    """
    Implementation of attention meachanism based on Pokaprakan et. al. (DOI: 10.1056/EVIDoa2100058)
    This module helps the model to focus on input frames that contain important structures to estimate
    gestation age. It computes a weighted sum of the condensed input features.
    It consists of:
     - dense layer W which computes the attention weights followed by the hyperbolic tangent function
     - dense layer V which maps the weights to a single scalar score between 0 and 1 for each frame 
       using a sigmoid function.
    - dense layer Q reduces the dimension of the feature vector to 128
    Output:
     - weighted sum of the feature map
    
    """
    def __init__(self, input_dim=2048, hidden_dim=512, mode='video'):
        super().__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim,1)
        self.Q = nn.Linear(input_dim, 128)
        self.mode = mode

    def forward(self,x):
        # if self.mode == 'video':
        #     batch_size, C, T, H, W = x.shape

        #     # Transpose input to (batch_size, T, C, H, W)
        #     x = x.permute(0, 2, 1, 3, 4)  # Now shape: (batch_size, T, C, H, W)

        #     # Flatten spatial dimensions: (batch_size, T, C, H, W) -> (batch_size, T, C * H * W)
        #     x = x.reshape(batch_size, T, -1) 
        h = torch.tanh(self.W(x))
        weights = torch.sigmoid(self.V(h))
        weights = weights / weights.sum(dim=1, keepdim=True)
        features = self.Q(x)
        weighted_sum = (weights*features).sum(dim=1)

        return weighted_sum
        

class GAEstimator(nn.Module):
    def __init__(self,bckbn=None):
        super().__init__()

        resnet = models.resnet50(weights=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.attention = WeightedAttention()
        self.regressor = nn.Linear(128,1)
    
    def forward(self, x):
        bs, seq_len = x.shape[:2] # input shape x.shape: batch_size, seq_len, 3, height, width

        # extract features from each frame
        # x = x.view(-1,1,256,256) # combine batch and seq dim
        x = x.reshape(-1, *x.shape[2:])
        if self.which_bkbn == 'usfm':
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
            features = F.adaptive_avg_pool2d(features, 1)
        
        features = features.view(bs, seq_len, -1)
        features = self.attention(features)
        
        ga_pred = self.regressor(features)

        return ga_pred.squeeze(), features
    
