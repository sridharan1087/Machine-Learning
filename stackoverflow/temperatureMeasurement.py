import torch
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


def model(w,t_u,b):
    return w * t_u +b


def loss(t_p,t_c):
    diff = (t_p-t_c)**2
    return diff.mean()

w = torch.ones(1)
# print(w)

b = torch.zeros(1)
# print(b)

t_p = model(w,t_u,b)
print(loss(t_p,t_c))

def loss_fn(t_p,t_c):
    return 2*(t_p-t_c)

def model_dw(w,t_u,b):
    return t_u

def model_db(w,t_u,b):
    return 1.0

def gradient(w,t_p,t_c,t_u,b):
    dw = loss_fn(t_p, t_c)* model_dw(w, t_u, b)
    db = loss_fn(t_p, t_c) * model_db(w, t_u, b)
    return torch.stack([dw.mean(),db.mean()])


def training_loop(n_epochs,learning_rate,params,t_u,t_c):
    try:
        for i in range(n_epochs):
            w,b = params
            t_p = model(w,t_u,b)
            grad = gradient(w, t_p, t_c, t_u, b) 
            params = params - learning_rate * grad
            print(loss(t_p,t_c))
    except:
        pass
    
    
training_loop(6, 1e-4, torch.tensor([1.0,1.0]), t_u, t_c)
    