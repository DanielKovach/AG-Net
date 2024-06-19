import torch
import torch.nn as nn
import torch.optim as optim
from models import AG_Net
from utils import full_dataset, test_dataset, train_dataset, train_indices, test_indices, device
import os

def load_checkpoint_for_training(model, optimizer, train_indices, test_indices, device, filename='AG_net_weights2.pth'):
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


class trainAGnet(object):
    def __init__(self, device, kappa_global = 8, batch_sze = 8, save_model = True, save_file_name: str = None):
        self.dev = device
        self.kap = kappa_global
        self.B = batch_sze
        self.save_model = save_model
        self.save_file_name = save_file_name

    def trainAGnet_method(self, train_indices, test_indices, full_dataset):
        kappa = self.kap
        B = self.B
        dev = self.dev
        save_model = self.save_model
        save_file_name = self.save_file_name

        net = AG_Net(kappa = kappa, batch_size_loc=B, device_loc= dev).to(device)


        """
        Training the model.
        """

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.99)

        # Loading the saved model
        
        net, optimizer, start_epoch, train_indices, test_indices, loaded_flag = load_checkpoint_for_training(net, optimizer, train_indices, test_indices, device)
        net = net.to(device)

        if loaded_flag:
            # now individually transfer the optimizer parts...
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

            train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
            batched_train_data = torch.utils.data.DataLoader(train_dataset, batch_size = B, shuffle = True)


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

        if save_model:
            """ Save the model. """
            # Get the current working directory
            notebook_dir = os.getcwd()
            
            # Define the filename for saving the model
            filename = 'AG_net_weights2.pth'
            if (save_file_name is not None):
                filename = save_file_name

            # Construct the full path to save the model
            save_path = os.path.join(notebook_dir, filename)

            state = {'epoch': epoch + 1, 'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(), 'train_indices': train_indices, 'test_indices': test_indices}
            torch.save(state, filename)

if __name__ == "__main__":
    trainAGnet(device= device, save_model= False).trainAGnet_method(train_indices, test_indices, full_dataset)