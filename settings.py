# 1. optimizer SGD/Adam
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=config.lr_init, 
                             betas=(0.9, 0.999), weight_decay=0.0002)
optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr,
                             momentum=0.9, dampening=0, weight_decay=0.001, nesterov=False)

# 2. learning rate 
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_list, gamma=0.9, last_epoch=-1)    
# lr_list is a epoch list, e.g. [60, 80], lr = lr * gamma when epoch is 60/80

# 3. weight_initialization
def weight_initialization(net, mode='normalization'):
    # 1. kaiming uniform
    if mode == 'uniform':
        for m in net.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    # 2. kaiming normalization
    elif mode == 'normalization':
        for m in net.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
    else:
        assert mode == 'uniform' or 'normalization', 'please assign a mode to initialize weights'
