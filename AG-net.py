import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
import pandas as pd
from collections import defaultdict
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.ops import roi_align
from pytorchcv.models.common import SEBlock
import random
import itertools

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(.9, device=device)
else:
    device = torch.device('cpu')        

print(device)

class CopySingleChannels(object):
    ''' Does nothing to the image if it has three channels. Creates 3 copies if it has one channel. Returns assertion error if neither.'''
    
    def __call__(self, image):
        assert (image.shape[0] == 1 or image.shape[0] == 3)

        if image.shape[0] == 1:       
            return image.repeat(3, 1, 1)
        else:
            return image

transform = transforms.Compose([
    torchvision.transforms.RandomAffine(degrees=15, translate= (.15, .15), scale = (0.85, 1.15)),
    transforms.Resize((224,224)), #Resize data to be 224x224.
    transforms.ToTensor(),
    transforms.Lambda(CopySingleChannels()) #transforms both RGB and grayscale to same tensor size by stacking copies of grayscales along first axis.
    
    ])


# Get the data.
full_dataset = torchvision.datasets.Caltech256(root='./data', download=True, transform=transform)
# Get the indices of the data for splitting it later.
full_indices = full_dataset.index



# Generate breaks to generate a list of lists containing indices for each category.
brks = [i for i in range(1,len(full_indices)) if full_indices[i] < full_indices[i-1]]
split_indices = [full_indices[x:y] for x,y in zip([0]+brks,brks+[None])]


# We do not want to include the 257th category, clutter, because they don't use it in the paper, so 
split_indices = split_indices[:256]

# Get 60 random indices from each category.
rand_indices = [filter(random.sample(i, k = 60)) for i in split_indices]
# Concatenate the list of lists to a list.
train_indices = list(itertools.chain.from_iterable(rand_indices))

assert len(train_indices) == 256*60

# Get the indices not included in the train indices.
get_test_indices = [list(set(split_indices[i]) - set(rand_indices[i])) for i in range(len(split_indices))]
# Concatenate the list of lists to a list.
test_indices = list(itertools.chain.from_iterable(get_test_indices))


train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

batch_sze = 8
batched_train_data = torch.utils.data.DataLoader(train_dataset, batch_size = batch_sze, shuffle = True)




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
        self.delta = nn.Parameter(torch.zeros(1)) # Delta initialized to be 0

        self.softmax  = nn.Softmax(dim=-1) 
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature (same size)
        """
        B,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(B,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(B,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(B,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(B,C,width,height)
        
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
        r = 7

        feature_list = torch.split(feature_tens, region_counts)
        
        
        outputs = []
        for i in range(B):
            R = region_counts[i]
            regions = feature_list[i]  # (R, C, H, W)
            
            
            # Compute u_{r,r'}
            u_r = self.W_u[i](regions).unsqueeze(0)  # (1, R, C/8, H, W)
            u_r_prime = self.W_u_prime[i](regions).unsqueeze(1)  # (R, 1, C/8, H, W)
            u_r_r_prime = torch.tanh(u_r + u_r_prime)  # (R, R, C/8, H, W)
            
            # Compute m_{r,r'}
            m_r_r_prime = self.W_m[i](u_r_r_prime.view(R * R, -1, r, r)).view(R, R, 1, r, r)  # (R, R, 1, H, W)
            m_r_r_prime = torch.sigmoid(m_r_r_prime)
            
            # Aggregate attentional features
            alpha_r = torch.sum(m_r_r_prime * regions.unsqueeze(1), dim=0)  # (R, C, H, W)
            
            # Compute weights w_r
            w_r = self.W_alpha[i](alpha_r).view(R, -1)  # (R, num_features)
            w_r = F.softmax(w_r, dim=0).view(R, 1, r, r)  # (R, 1, H, W)
            
            # Combine regional features
            f_hat = torch.sum(alpha_r * w_r, dim=0)  # (C, H, W)
            outputs.append(f_hat)
        
        outputs = torch.stack(outputs, dim=0)  # (B, C, H, W)
        
        return outputs
    
class Classify(nn.Module):
    def __init__(self, batch_size = 8, in_dim = 37, C = 2048, num_classes = 256, r = 7):
        super(Classify, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        # Layers for classification
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        
        self.Womega = nn.Linear(C, 1)
        self.bomega = nn.Parameter(torch.zeros(1))
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
        self.c_res = CustomResNet50(device).to(device)
        self.intra = IntraSelfAttn().to(device_loc)
        self.se_res = SE_Residual(channels= 2048, Rp1= self.kappa_max).to(device_loc)
        self.inter = InterSelfAttn(B= batch_size_loc).to(device_loc)
        self.classify = Classify(batch_size= batch_size_loc, in_dim= self.kappa_max).to(device_loc)
    
    def forward(self, x):
        res_feats = self.c_res(x)
        SR_list, region_counts = get_SRs_coords(x, self.dev, self.kap)
        x = self.intra(res_feats)
        x = roi_align(x, SR_list, output_size= 7, spatial_scale= 7/224).to(device)
        x = self.se_res(x, region_counts)
        x = self.inter(x, region_counts)
        x = self.classify(x)        
        return x
    
    

kappa_global = 8
net = AG_Net(kappa = kappa_global, batch_size_loc=batch_sze, device_loc= device).to(device)


"""
Training the model.
"""

import os


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.99)


def load_checkpoint(model, optimizer, train_indices, test_indices, device, filename='AG_net_weights2.pth'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    loaded_flag = False
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_indices = checkpoint['train_indices']
        test_indices = checkpoint['test_indices']
        loaded_flag = True
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, train_indices, test_indices, loaded_flag

# Loading the saved model
 
net, optimizer, start_epoch, train_indices, test_indices, loaded_flag = load_checkpoint(net, optimizer, train_indices, test_indices, device)
net = net.to(device)

if loaded_flag:
    # now individually transfer the optimizer parts...
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    batched_train_data = torch.utils.data.DataLoader(train_dataset, batch_size = batch_sze, shuffle = True)


net.train()
end_epoch = 50

for epoch in range(start_epoch, end_epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    if (epoch > 23):
        optimizer = optim.SGD(net.parameters(), lr=1e-6, momentum=0.99)
    for i, batch in enumerate(batched_train_data):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch
        
        inputs, labels = inputs.to(device), labels.type(torch.LongTensor).to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs).type(torch.FloatTensor).to(device)
        #print(outputs, labels)
        loss = criterion(outputs, labels)
        #loss.requires_grad = True
        loss.backward()
        optimizer.step()

        # print statistics
        #"""
        running_loss += loss.item()
        print_num = 1920
        if i % print_num == (print_num - 1):    # print every print_num batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_num:.3f}')
            running_loss = 0.0
        #"""
print('Finished Training')


""" Save the model. """
# Get the current working directory
notebook_dir = os.getcwd()

# Define the filename for saving the model
filename = 'AG_net_weights2.pth'

# Construct the full path to save the model
save_path = os.path.join(notebook_dir, filename)

state = {'epoch': epoch + 1, 'state_dict': net.state_dict(),
             'optimizer': optimizer.state_dict(), 'train_indices': train_indices, 'test_indices': test_indices}
torch.save(state, filename)

# Define a DataLoader for the test dataset
batch_size_test = 8
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

# Set the model to evaluation mode
net.eval()

correct = 0
total = 0

# Disable gradient computation during evaluation
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = net(inputs)
        
        blah, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        
        correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
print(f'Accuracy on the test dataset: {accuracy:.2f}%')