import torch.nn as nn
import torch
import torch.optim as optim
#Linear(in_features=2, out_features=2, bias=True)

t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])

t_cSequeezed = t_c.unsqueeze(1)
print(t_cSequeezed)

linear_model = nn.Linear(1,1)
optimizer = optim.SGD(linear_model.parameters(),1e-4)


#Based on the Features, no of weights
print(linear_model.weight)
print(linear_model.bias)

t_c_unsqueeze = t_c.unsqueeze(1)
t_u_unsquueze = t_u.unsqueeze(1)

def training_loop(n_epochs,loss_fn,t_c_unsqueeze,t_u_unsquueze,model):
    for _ in range(n_epochs):
        t_p = model(t_u_unsquueze)
        loss = loss_fn(t_p,t_c_unsqueeze)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss)
        
        
        
training_loop(6, nn.MSELoss(), t_c_unsqueeze, t_u_unsquueze, linear_model)
        




