import torchvision.transforms as transforms


data_transforms = transforms.Compose([
    transforms.Resize((256, 320)),
    transforms.ColorJitter(0.8, contrast=0.3),
    transforms.RandomAffine(160, scale=(0.8, 1.2), translate=(0.2, 0.2)),
    transforms.RandomHorizontalFlip(),            #flip transform
    transforms.ToTensor(),
    transforms.Normalize((0.48215605, 0.48855937, 0.50680548), (0.23369475, 0.23529582, 0.23668588))
])

validation_data_transforms = transforms.Compose([
    transforms.Resize((256, 320)),
    transforms.ToTensor(),
    transforms.Normalize((0.48215605, 0.48855937, 0.50680548), (0.23369475, 0.23529582, 0.23668588))
])
