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

# 4. a demo
# step 1: initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# step 2: load model
    model = Linnet2()
    if config.resume:
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint)
    else:
        weight_initialization(net=model)

# step 3: set loss
criterion = loss

# step 4: train/val loader
dataset_train = LinDataset(data_path=config.instance_train, label_path=config.label_path, test=False)
dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=config.batch_size, num_workers=1)
dataset_val = LinDataset(data_path=config.instance_val, label_path=config.label_path, test=False)
dataloader_val = DataLoader(dataset_val, shuffle=True, batch_size=config.batch_size, num_workers=1)

# step 5: optimizer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=config.lr_init, betas=(0.9, 0.999), weight_decay=0.0002)
                             
# step 6: lr
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_list, gamma=0.9, last_epoch=-1)

# train epoch
for epoch in range(config.num_epoch):
    lr_scheduler.step()
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    model.train().to(device=device)
    
    for i, (inputs, targets) in enumerate(dataloader_train):
        inputs = data.to(device).float()
        labels = label.to(device).float()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    model.eval().to(device)
    with torch.no_grad():
        for j, (inputs, targets) in enumerate(dataloader_val):
            inputs = data.to(device).float()
            labels = label.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
     torch.save(model, checkpoint_weight)
        
        
    
