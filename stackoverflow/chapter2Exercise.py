import torch
import torch.optim as optim


t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])

def model(t_u,w1,w2,b):
    return w2 * t_u ** 2 + w1 * t_u + b

def loss_fn(tp,t_c):
    square_mean_error = (tp-t_c)**2
    return square_mean_error.mean()

params = torch.tensor([1.0,1.0,1.0],requires_grad=True)
learning_rate = 1e-4
optimizer = optim.SGD([params],learning_rate)  

def training_loop(n_epochs,t_c,t_u,params):
    for _ in range(n_epochs):
        loss = loss_fn(model(t_u,*params),t_c)
        loss.backward()
        print(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        
training_loop(6, t_c, t_u, params)
#The loss increases each epoch.
        
    