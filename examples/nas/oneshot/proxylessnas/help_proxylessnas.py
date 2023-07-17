import torch 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import nni 
import nni.retiarii.nn.pytorch as nn 
from nni.retiarii import model_wrapper 
import numpy as np 

class TinyML8to2NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, index_labels) :
        super().__init__() 
        self.data = X 
        self.y = y 
        self.index_labels = index_labels 

    # in getitem, we retrieve one item based on the input index
    def __getitem__(self, index):
        data = self.data[index]
        y = self.y[index] 
        index_labels = self.index_labels[index]
        return data, y, index_labels 

    def __len__(self):
        return len(self.data) 
    
class TinyML2NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y) :
        super().__init__() 
        # https://pytorch.org/docs/stable/generated/torch.from_numpy.html 
        self.data = torch.from_numpy(X) 
        # # https://pytorch.org/docs/stable/generated/torch.Tensor.long.html 
        self.y = torch.from_numpy(y.reshape(-1)).long()  

    # in getitem, we retrieve one item based on the input index
    def __getitem__(self, index):
        data = self.data[index]
        y = self.y[index] 
        return data, y

    def __len__(self):
        return len(self.data) 