import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir), key=lambda x: x)
        self.images = []
        self.labels = []

        for label, class_dir in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = ImageDataset(root_dir="train", transform=transform)
test_dataset = ImageDataset(root_dir="test", transform=transform)

# train_size = int(0.8*len(dataset))
# test_size = len(dataset) - train_size

# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# dataset = ImageDataset(root_dir="RGB",transform=transform)
# data_loader = DataLoader(dataset,batch_size=32,shuffle=True,num_workers=4)

# # 下载和加载MNIST数据集
# train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, pin_memory=True, num_workers=4)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
