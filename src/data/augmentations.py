from torchvision import transforms


def get_transforms(image_size: int = 224, strength: str = "medium"):
    if strength == "none":
        train_tfms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    elif strength == "light":
        train_tfms = transforms.Compose([
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
        ])
    else:  # medium
        train_tfms = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
        ])

    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    return train_tfms, val_tfms
