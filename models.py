import numpy as np
import cv2 as cv
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from pytorchcv.models.common import SEBlock


""" 
    PyTorch has the architecture for ResNet50 already available with the best pre-trained ImageNet weights. 
    We use Pyotrch's improved training weights instead of the original weights to better capture the features.
"""
class CustomResNet50(nn.Module):
    def __init__(self, device):
        super(CustomResNet50, self).__init__()
        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT).to(device)
        
        # Extract the first 5 blocks (Note that these are the conv1, conv2_x,..., conv_5x layers).
        self.get_blocks = nn.Sequential(*list(resnet50.children())[:8])
        
    
    def forward(self, x):
        return self.get_blocks(x)


"""
We now get the coordinates of the SR's. This takes as input the image batch and outputs a list of tensors of different sizes.
"""

def get_SRs_coords(image_batch, loc_device, kappa=8):
    batch_SRs = []
    region_counts = []

    B, C, H, W = image_batch.size()

    for k in range(B):
        # Convert a single image to numpy array of expected shape for cv inputs
        copy_tens = image_batch[k].clone().cpu()
        im2 = copy_tens.detach().numpy().transpose(1, 2, 0)
        # Rescale data for expected type for CV 
        im2 = (im2 * 255).astype(np.uint8)

        sift = cv.SIFT_create()
        kp, desc = sift.detectAndCompute(im2, None)

        # Sort keypoints by their response in descending order
        keypoints_lst = sorted(set(kp), key=lambda x: x.response, reverse=True)

        Secon_SRs = []
        # Regardless of how many kp are detected, we want to include the whole image.
        Secon_SRs.append(torch.tensor([0, H - 1, 0, W - 1]))

        if len(keypoints_lst) > 1:
            num_keypoints_to_keep = max(int(.5 * len(keypoints_lst)), 1)
            if len(keypoints_lst) < 9:
                num_keypoints_to_keep = len(keypoints_lst)
                kappa = 2

            keypoints = keypoints_lst[:num_keypoints_to_keep]
            pts = cv.KeyPoint_convert(keypoints)

            unique_pts = np.unique(pts, axis=0)
            kappa_loc = min(kappa, len(unique_pts))

            if kappa_loc > 1:  # Only fit GMM if there are at least 2 unique points
                gmm_pts = GMM(n_components=kappa_loc, covariance_type='full', init_params='kmeans').fit(unique_pts)
                # Note how we apply predict to the non-unique points, putting more weights onto locations with more than one kp.
                gmm_labels = gmm_pts.predict(pts)

                # Sort pts into clusters
                clusters = defaultdict(list)
                for pt, label in zip(pts, gmm_labels):
                    clusters[label].append(pt)

                # Compute min and max values for x and y coords of each cluster to get the bounding boxes
                Prim_SRs = {}
                for label, points in clusters.items():
                    xpts, ypts = zip(*points)
                    Prim_SRs[label] = {
                        'min_x': min(xpts),
                        'max_x': max(xpts),
                        'min_y': min(ypts),
                        'max_y': max(ypts),
                    }

                # Note that we can get the primary SR's via "diagonal" elements (when i = j )   
                for i, itemi in Prim_SRs.items():
                    for j, itemj in Prim_SRs.items():
                        if i <= j:
                            # Note that x1,x2 are computer from the "y"-values. This is NOT a typo, an implicit transpose operation is used because numpy image arrays and torch image tensors are transpositions along the H x W dimensions.
                            x1, x2 = min(itemi['min_y'], itemj['min_y']), max(itemi['max_y'], itemj['max_y'])
                            y1, y2 = min(itemi['min_x'], itemj['min_x']), max(itemi['max_x'], itemj['max_x'])

                            Secon_SRs.append(torch.tensor([x1, y1, x2, y2]))

        tens = torch.stack(Secon_SRs).to(loc_device)
        batch_SRs.append(tens)
        region_counts.append(tens.size(0))

    return batch_SRs, region_counts


""" Given the feature maps, we now want to begin the intra-attention mechanism. """

class IntraSelfAttn(nn.Module):
    """ 
        Self Attention Layer adapted from SAGAN Implementation available at https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    """
    def __init__(self, in_channels = 2048, B = 8):
        super(IntraSelfAttn,self).__init__()
        self.chanel_in = in_channels
        
        self.query_conv = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//8 , kernel_size= 1) # f(x)
        self.key_conv = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//8 , kernel_size= 1) # g(x)
        self.value_conv = nn.Conv2d(in_channels = in_channels , out_channels = in_channels , kernel_size= 1) # h(x)
        self.delta = nn.Parameter(torch.zeros(1)) # Delta initialized to be 0 as the paper suggests.

        self.softmax  = nn.Softmax(dim=-1) 
    def forward(self,x):
        """
            inputs :
                x : input feature maps (B x C x W x H)
            returns :
                out : self attention value + input feature (same size)
        """
        B,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(B,-1,width*height).permute(0,2,1) # B x C x W*H
        proj_key =  self.key_conv(x).view(B,-1,width*height) # B x C x W*H
        energy =  torch.bmm(proj_query,proj_key) 
        attention = self.softmax(energy) # B x W*H x W*H 
        proj_value = self.value_conv(x).view(B,-1,width*height) # B x C x W*H

        out = torch.bmm(proj_value,attention.permute(0,2,1) ).view(B,C,width,height)
        
        out = self.delta*out + x
        return out

"""  
Next, roi_align will take as input the Intra-self attention features and 
an input parameter boxes which takes in a list of box tensors [batch_idx, x1, y1, x2, y2]. 
obtained with get_SRs(). 
If a batch is passed, we must have that the first column of boxes contains the 
index of the corresponding element in the batch.


Returns a tensor of size sum(R_i + 1) x output_size x output_size, where R_i is the number of SR's detected in image i from a batch.
"""



class SE_Residual(nn.Module):

    """ Takes in batch of bilinearly pooled features. Outputs tensor of same size. Rp1 = R + 1."""
    def __init__(self, channels = 2048, Rp1 = 37):
        super(SE_Residual, self).__init__()
        self.SE = nn.ModuleList([SEBlock(channels=channels, reduction=16) for i in range(Rp1)])

    def forward(self, feature_tens, region_counts):
        
        B = len(region_counts)
        
        
        feature_list = torch.split(feature_tens, region_counts, dim = 0)
        outputs = []
        for i in range(B):
            for j in range(region_counts[i]):
                x = self.SE[j](feature_list[i][j])
                resid = x + feature_list[i][j].unsqueeze(0)
                outputs.append(resid)
                

        return torch.concat(outputs, dim=0)
    
class InterSelfAttn(nn.Module):
    """
    Takes in SE_Res features and applies an attention mechanism used to identify which SR's are most helpful for classification.
    """
    def __init__(self, in_channels = 2048, B = 8):
        super(InterSelfAttn, self).__init__()
        self.in_channels = in_channels
        
        # Shared 1x1 convolutions for parameter efficiency
        self.W_u = torch.nn.ModuleList([nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)  for i in range(B)])
        self.W_u_prime = torch.nn.ModuleList([nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)  for i in range(B)])
        self.W_m = torch.nn.ModuleList([nn.Conv2d(in_channels // 8, 1, kernel_size=1)  for i in range(B)])
        self.W_alpha = torch.nn.ModuleList([nn.Conv2d(in_channels, 1, kernel_size=1)  for i in range(B)])
        
    def forward(self, feature_tens, region_counts):
        B = len(region_counts)
        r = feature_tens.size(-1)

        feature_list = torch.split(feature_tens, region_counts)
        
        
        outputs = []
        for i in range(B):
            R = region_counts[i]
            regions = feature_list[i]  # R x C x H x W
            
            
            # Compute u_{r,r'}
            u_r = self.W_u[i](regions).unsqueeze(0)  # 1 x R x C/8 x H x W
            u_r_prime = self.W_u_prime[i](regions).unsqueeze(1)  # R x 1 x C/8 x H x W
            u_r_r_prime = torch.tanh(u_r + u_r_prime)  # R x R x C/8 x H x W
            
            # Compute m_{r,r'}
            m_r_r_prime = self.W_m[i](u_r_r_prime.view(R * R, -1, r, r)).view(R, R, 1, r, r)  # R x R x 1 x H x W 
            m_r_r_prime = torch.sigmoid(m_r_r_prime)
            
            # Aggregate attentional features
            alpha_r = torch.sum(m_r_r_prime * regions.unsqueeze(1), dim=0)  # R x C x H x W
            
            # Compute weights w_r
            w_r = self.W_alpha[i](alpha_r).view(R, -1)  
            w_r = F.softmax(w_r, dim=0).view(R, 1, r, r)  # R x 1 x H x W
            
            # Combine regional features
            f_hat = torch.sum(alpha_r * w_r, dim=0)  # C x H x W
            outputs.append(f_hat)
        
        outputs = torch.stack(outputs, dim=0)  # B x C x H x W
        
        return outputs
    
class Classify(nn.Module):
    def __init__(self, C = 2048, num_classes = 256):
        super(Classify, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        # Layers for summarizing
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        
        # Layers for generating weights of gmp
        self.Womega = nn.Linear(C, 1)

        # Note that this is redundant and will be removed in a later version after we retrain the model with the new number of parameters.
        self.bomega = nn.Parameter(torch.zeros(1))

        # Final dense layer which learns to take the summarized features and classify them.
        self.fc = nn.Linear(C, num_classes)

    def forward(self, x):
        """
            inputs:
                x: input batch of features (B x C x 7 x 7)
            returns:
                out: class probabilities (B x num_classes)
        """
        B, C, H, W = x.size()

        # Apply GAP and GMP
        f_gap = self.gap(x).view(B, C)
        f_gmp = self.gmp(x).view(B, C)

        # Calculate omega and combine features
        omega = self.softmax(self.Womega(f_gap) + self.bomega)
        F = omega * f_gmp + (1 - omega) * f_gap
        
        # Classification
        class_probs = self.fc(F)

        return class_probs

"""
Putting the pieces together.
"""

class AG_Net(nn.Module):
    def __init__(self, kappa, batch_size_loc, device_loc):
        super(AG_Net, self).__init__()
        self.dev = device_loc
        self.batch = batch_size_loc
        self.kap = kappa
        self.kappa_max = 1+kappa*(kappa+1)//2
        self.c_res = CustomResNet50(device_loc).to(device_loc)
        self.intra = IntraSelfAttn().to(device_loc)
        self.se_res = SE_Residual(channels= 2048, Rp1= self.kappa_max).to(device_loc)
        self.inter = InterSelfAttn(B= batch_size_loc).to(device_loc)
        self.classify = Classify().to(device_loc)
    
    def forward(self, x):
        device = self.dev
        res_feats = self.c_res(x)
        SR_list, region_counts = get_SRs_coords(x, self.dev, self.kap)
        x = self.intra(res_feats)
        x = roi_align(x, SR_list, output_size= 7, spatial_scale= 7/224).to(device)
        x = self.se_res(x, region_counts)
        x = self.inter(x, region_counts)
        x = self.classify(x)        
        return x