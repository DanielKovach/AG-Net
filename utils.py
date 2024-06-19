import torch
import torchvision
import torchvision.transforms as transforms
import random
import itertools

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(.9, device=device)
elif torch.backends.mps.is_available(): device = torch.device("mps")
else:
    device = torch.device('cpu')    

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
rand_indices = [sorted(random.sample(i, k = 60)) for i in split_indices]
# Concatenate the list of lists to a list.
train_indices = list(itertools.chain.from_iterable(rand_indices))

assert len(train_indices) == 256*60

# Get the indices not included in the train indices.
get_test_indices = [set(split_indices[i]) - set(rand_indices[i]) for i in range(len(split_indices))]
# Concatenate the list of lists to a list.
test_indices = list(itertools.chain.from_iterable(get_test_indices))


train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

batch_sze = 8
batched_train_data = torch.utils.data.DataLoader(train_dataset, batch_size = batch_sze, shuffle = True)