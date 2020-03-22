import torch
import numpy as np

data = np.loadtxt(r'data\Bike-Sharing-Dataset\hour.csv',delimiter=",",skiprows=1,dtype=np.float32,
                  converters={1: lambda x: float(x[8:10])})


dataset = torch.from_numpy(data)
print(dataset.shape)  ##1
d = dataset[:17376,:]
bikes = d.view((-1,24,17))  ## Batch size is 1, 24 rows and 17 columns
print(bikes.shape)

