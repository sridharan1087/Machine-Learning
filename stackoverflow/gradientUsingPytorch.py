import torch

t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])

params = torch.tensor([1.0,1.0], requires_grad=True)
print(params)

def model(t_u,w,b):
    return w * t_u +b


def loss_fn(t_p,t_c):
    diff = (t_p-t_c)**2
    return diff.mean()


loss = loss_fn(model(t_u,*params),t_c)
loss.backward()
print(loss)

print(params.grad)

def training_loop(n_epoch,learning_rate,t_u,t_c,params):
    try:
        for i in range(n_epoch):
            print(i)
            loss = loss_fn(model(t_u,*params),t_c)
            loss.backward()
            params = (params - learning_rate * params.grad).detach().requires_grad_()
            print(loss)
            
    except Exception as e:
        print(e)
    
    
training_loop(6, 1e-4, t_u, t_c, params)
    
            
            
            
            
               
