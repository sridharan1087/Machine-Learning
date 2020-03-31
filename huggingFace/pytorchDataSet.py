import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time

class CustomSet(Dataset):
    def __init__(self,fileName):
        with open(fileName) as f:
            lines = f.read().split('\n')
        
        X=[]
        y=[] 
         
        for line in lines:
            a = line.split('\t')
            time.sleep(10)
            X.append(a[0])
            y.append(a[1])
            
        self.x = X
        self.y = y
            
    def __len__(self):
        return(len(self.y))
    
    def __getitem__(self, index):
        return self.x[index],self.y[index]
            
            
obj = CustomSet(r'glue_data\SST-2\dev.tsv')
dataloader = DataLoader(obj, batch_size = 5, num_workers = 2)
for i,j in dataloader:
    print(i)
