import torch
from models import AG_Net
from utils import full_dataset, test_dataset, test_indices, device
import os

def load_checkpoint_for_testing(model, test_indices, device, filename='AG_net_weights2.pth'):
        # Note: Input model should be pre-defined.  This routine only updates its state.
        loaded_flag = False
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            test_indices = checkpoint['test_indices']
            loaded_flag = True
            print("=> loaded checkpoint '{}'"
                    .format(filename))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        return model, test_indices, loaded_flag

class testAGnet(object):
    def __init__(self, device, kappa_global = 8, batch_sze = 8):
        self.dev = device
        self.kap = kappa_global
        self.B = batch_sze

    def testAGnet_method(self, test_indices):
        kap = self.kap
        dev = self.dev
        B = self.B
        net = AG_Net(kappa = kap, batch_size_loc= B, device_loc= dev).to(dev)

       

        # Loading the saved model
        
        net, test_indices, loaded_flag = load_checkpoint_for_testing(net, test_indices, dev)
        net = net.to(device)

        if loaded_flag:
            test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

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
                inputs, labels = inputs.to(dev), labels.to(dev)
                
                outputs = net(inputs)
                
                blah, predicted = torch.max(outputs, dim = 1)
                
                total += labels.size(0)
                
                correct += (predicted == labels).sum().item()

        # Calculate accuracy
        accuracy = 100 * correct / total
        print(f'Accuracy on the test dataset: {accuracy:.2f}%')

if __name__ == "__main__":
    testAGnet(device= device).testAGnet_method(test_indices)