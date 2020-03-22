import torch
import numpy as np

data = np.loadtxt(r'data\Bike-Sharing-Dataset\hour.csv',delimiter=",",skiprows=1,dtype=np.float32,
                  converters={1: lambda x: float(x[8:10])})


dataset = torch.from_numpy(data)
print(dataset.shape)  ##1
d = dataset[:17376,:]


weather_onehot = d[:,9].to(torch.long)
print(weather_onehot.shape)
weather = torch.zeros(weather_onehot.shape[0],4)
print(weather.shape)

w = weather_onehot.unsqueeze(1)

w = weather.scatter_(1,weather_onehot.unsqueeze(1)-1,1.0)
print(w.shape)

w = torch.cat((w,d),1)
print(w)

bikes = w.view((-1,24,21))  ## Batch size is 1, 24 rows and 17 columns
print(bikes.shape)