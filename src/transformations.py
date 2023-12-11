from torchvision import datasets, transforms

# TINY VGG Transformations 
train_transform_tinyvgg = transforms.Compose([
    # 5% probability of random flip
    transforms.RandomHorizontalFlip(p=0.05),
    # 5% probability of random rotation
    transforms.RandomApply([transforms.RandomRotation(degrees=(-10, 10))], p=0.05),
    # 5% probability of a random colorjittter
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2,
                                                   contrast=0.2,
                                                   saturation=0.2,
                                                   hue=0.2)], p=0.05),
    # Resize the image
    transforms.Resize((300, 300)),
    # Turn Image into a torch.Tensor
    transforms.ToTensor(),
    ])

# Define Transformations for Test Data
test_transform_tinyvgg = transforms.Compose([
    # Resize the image
    transforms.Resize((300, 300)),
    # Turn Image into a torch.Tensor
    transforms.ToTensor(),
])


# ALEXNET TRANSFORMATIONS 
# TRAIN DATA TRANSFORMATIONS
train_transform_alexnet = transforms.Compose([
    # 5% probability of random flip
    transforms.RandomHorizontalFlip(p=0.05),
    # 5% probability of random rotation
    transforms.RandomApply([transforms.RandomRotation(degrees=(-10, 10))], p=0.05),
    # 5% probability of a random colorjittter
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2,
                                                   contrast=0.2,
                                                   saturation=0.2,
                                                   hue=0.2)], p=0.05),
    # Resize the image
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # Turn Image into a torch.Tensor
    transforms.ToTensor()
    ])

# Define Transformations for Test Data
# (No Data Augmentation)
test_transform_alexnet = transforms.Compose([
    # Resize the image
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # Turn Image into a torch.Tensor
    transforms.ToTensor(),
])


# VGG11 Transformations 

# TRAIN DATA TRANSFORMATIONS
train_transform_vgg11 = transforms.Compose([
    # 5% probability of random flip
    transforms.RandomHorizontalFlip(p=0.05),
    # 5% probability of random rotation
    transforms.RandomApply([transforms.RandomRotation(degrees=(-10, 10))], p=0.05),
    # 5% probability of a random colorjittter
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2,
                                                   contrast=0.2,
                                                   saturation=0.2,
                                                   hue=0.2)], p=0.05),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ])

# Define Transformations for Test Data
# (No Data Augmentation)
test_transform_vgg11 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])