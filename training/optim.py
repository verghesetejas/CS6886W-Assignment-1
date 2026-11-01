import torch

def make_optimizer(name, params, lr, wd, momentum):
    name = name.lower()
    if name == "sgd": return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    if name == "nesterov": return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    if name == "adam": return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adagrad": return torch.optim.Adagrad(params, lr=lr, weight_decay=wd)
    if name == "rmsprop": return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=wd)
    if name == "nadam": return torch.optim.NAdam(params, lr=lr, weight_decay=wd)
    raise ValueError(name)
